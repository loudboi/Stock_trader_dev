"""
bot/live_pullback.py
====================
Live execution for the phased trend-pullback strategy.

Safety invariants:
  * broker reads are authoritative; an API failure is not treated as "flat";
  * an accepted order is not treated as a fill -- state changes only after the
    broker position confirms the change;
  * while an order is uncertain/pending, no second strategy order is submitted;
  * a missing live price is a data stall, never a stale daily-price substitute;
  * managed positions may not silently disappear from the configured symbol set.
"""

import argparse
import csv
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone, timedelta

import config
from bot import indicators as ind
from bot import risk_manager as rm
from bot.notifier import Notifier
from bot.portfolio import Portfolio
from bot.backtest_pullback import _DAILY_WARMUP_DAYS
from bot.strategies.trend_pullback import TrendPullbackStrategy, PullbackParams

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("live_pullback")

PULLBACK_TRADES_CSV = "pullback_trades.csv"
PULLBACK_DAILY_PNL_CSV = "pullback_daily_pnl.csv"
PULLBACK_STATE_FILE = "pullback_state.json"
POLL_SECONDS = 60
STALL_ALERT_AFTER = 5
ORDER_CONFIRM_SECONDS = 10.0
ORDER_CONFIRM_INTERVAL = 0.25

_RUNNING = True


def _handle_sigterm(signum, frame):
    global _RUNNING
    log.info("Shutdown signal received; finishing current cycle.")
    _RUNNING = False


def configure_alpaca_runtime(is_live: bool) -> None:
    """Keep paper and real-money state/logs physically separate."""
    global PULLBACK_TRADES_CSV, PULLBACK_DAILY_PNL_CSV, PULLBACK_STATE_FILE
    if is_live:
        PULLBACK_TRADES_CSV = "pullback_live_trades.csv"
        PULLBACK_DAILY_PNL_CSV = "pullback_live_daily_pnl.csv"
        PULLBACK_STATE_FILE = "pullback_live_state.json"
    else:
        PULLBACK_TRADES_CSV = "pullback_trades.csv"
        PULLBACK_DAILY_PNL_CSV = "pullback_daily_pnl.csv"
        PULLBACK_STATE_FILE = "pullback_state.json"


class SingleInstanceLock:
    """Best-effort cross-platform process lock for one runtime state file."""
    def __init__(self, path):
        self.path = path
        self.fp = None

    def __enter__(self):
        self.fp = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                self.fp.seek(0)
                self.fp.write("0")
                self.fp.flush()
                self.fp.seek(0)
                msvcrt.locking(self.fp.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError) as e:
            self.fp.close()
            self.fp = None
            raise RuntimeError(
                f"Another live runner appears to be using {self.path}. "
                "Refusing to run two processes against the same state/account.") from e
        self.fp.seek(0)
        self.fp.truncate()
        self.fp.write(str(os.getpid()))
        self.fp.flush()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.fp:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self.fp.seek(0)
                msvcrt.locking(self.fp.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        finally:
            self.fp.close()


class PullbackLiveTrader:
    def __init__(self, pf, symbols, params: PullbackParams,
                 state_file=None, notifier: Notifier = None):
        self.pf = pf
        self.symbols = list(dict.fromkeys(symbols))
        self.params = params
        self.state_file = state_file or PULLBACK_STATE_FILE
        self.notifier = notifier or Notifier()
        self.instruments = {s: config.resolve_instrument(s) for s in self.symbols}
        missing = [s for s, i in self.instruments.items() if i is None]
        if missing:
            raise RuntimeError(
                f"Unknown symbol(s): {', '.join(missing)}. Add them to the configured universe.")
        self.strats = {s: TrendPullbackStrategy(self.instruments[s], params)
                       for s in self.symbols}
        self.state = self._load_state()
        managed = set(self.state["positions"]) | set(self.state["intents"]) | set(self.state["pending"])
        orphaned = managed - set(self.symbols)
        if orphaned:
            raise RuntimeError(
                "State contains managed symbol(s) omitted from this run: "
                f"{', '.join(sorted(orphaned))}. Include them, flatten them at the broker, "
                "or deliberately migrate the state file; refusing to leave them unmanaged.")
        self.state["runtime"] = {"symbols": sorted(self.symbols)}
        self._daily_cache = {}
        self._stall = {s: 0 for s in self.symbols}
        self._stall_alerted = set()

    def notify(self, message: str) -> None:
        log.info("ALERT: %s", message)
        self.notifier.notify(message)

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_state():
        return {"version": 2, "positions": {}, "intents": {}, "pending": {},
                "last_daily": {}, "daily": {"date": None, "realized": 0.0},
                "runtime": {}}

    def _load_state(self):
        state = self._default_state()
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    state.update(loaded)
                    for key in ("positions", "intents", "pending", "last_daily", "runtime"):
                        state.setdefault(key, {})
                    state.setdefault("daily", {"date": None, "realized": 0.0})
            except (json.JSONDecodeError, OSError) as e:
                raise RuntimeError(f"Could not safely read state file {self.state_file}: {e}") from e
        return state

    def save_state(self):
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.state_file)

    # ------------------------------------------------------------------ #
    # Broker helpers
    # ------------------------------------------------------------------ #
    def _latest_price(self, inst):
        """Live data only. Never substitute yesterday's close for a live stop."""
        return self.pf.latest_price(inst)

    def _broker_position(self, inst):
        r = self.pf.get_position_raw(inst)
        if r and r.get("side") != "long":
            raise RuntimeError(
                f"{inst.name}: broker holds a {r.get('side')} position. Strategy 4 is long-only; "
                "refusing to buy into/cover an externally managed short.")
        return r

    def _wait_pending(self, name, inst, seconds=ORDER_CONFIRM_SECONDS):
        deadline = time.monotonic() + seconds
        while name in self.state["pending"] and time.monotonic() < deadline:
            if self._confirm_pending(name, inst):
                return True
            time.sleep(ORDER_CONFIRM_INTERVAL)
        if name in self.state["pending"]:
            self.notify(f"ORDER UNCERTAIN: {name} order accepted but broker position change "
                        "was not confirmed yet; blocking further orders until reconciliation.")
            return False
        return True

    # ------------------------------------------------------------------ #
    # Startup / continuous reconciliation
    # ------------------------------------------------------------------ #
    def _adopt_long(self, name, inst, broker, fully_built=True):
        stop_dist = self._current_stop_dist(name, inst, broker["avg_entry"])
        pos = {
            "tranches": len(self.params.tranches) if fully_built else 1,
            "qty": broker["qty"], "avg_entry": broker["avg_entry"],
            "last_add_price": broker["avg_entry"], "stop_dist": stop_dist,
            "stop_order_id": None, "stop_level": None, "stop_qty": 0.0,
            "entry_time": datetime.now(timezone.utc).isoformat(),
        }
        self.state["positions"][name] = pos
        self._place_stop(name, inst, pos)
        return pos

    def reconcile(self):
        for name, inst in self.instruments.items():
            broker = self._broker_position(inst)
            pos = self.state["positions"].get(name)
            if broker:
                if pos is None:
                    pos = self._adopt_long(name, inst, broker, fully_built=True)
                    log.warning("Adopted existing %s long as fully built (qty=%s @%.4f).",
                                name, broker["qty"], broker["avg_entry"])
                else:
                    changed = (abs(float(pos.get("qty", 0)) - broker["qty"]) > 1e-9 or
                               abs(float(pos.get("avg_entry", 0)) - broker["avg_entry"]) > 1e-9)
                    pos["qty"], pos["avg_entry"] = broker["qty"], broker["avg_entry"]
                    if changed:
                        log.warning("%s broker position changed externally; refreshing stop.", name)
                        self._place_stop(name, inst, pos)
            elif pos:
                self._finalize_external_close(name, inst, pos)
            else:
                self.state["intents"].pop(name, None)
                self.state["pending"].pop(name, None)
        self.save_state()

    def _current_stop_dist(self, name, inst, price) -> float:
        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=_DAILY_WARMUP_DAYS + 30)
            daily = self.pf.get_historical_bars(inst, "1Day", start, end)
            if daily is None or daily.empty:
                return self.params.min_stop
            a = ind.atr(daily, self.params.atr_period).iloc[-1]
            atr_v = float(a) if a == a else 0.0
            return self.strats[name].stop_distance(atr_v, price)
        except Exception as e:  # noqa: BLE001
            log.warning("%s: could not compute current ATR stop (%s); using floor.", name, e)
            return self.params.min_stop

    # ------------------------------------------------------------------ #
    # Orders
    # ------------------------------------------------------------------ #
    def _buy_tranche(self, name, inst, price, tranche_index, stop_dist, how):
        if name in self.state["pending"]:
            return False
        before = self._broker_position(inst)
        before_qty = before["qty"] if before else 0.0
        pos = self.state["positions"].get(name)
        # Once a position exists, size every later tranche to the stop distance
        # that actually protects the position, not a newly sampled ATR distance.
        risk_dist = pos["stop_dist"] if pos else stop_dist
        equity = self.pf.get_equity()
        fraction = self.params.tranches[tranche_index]
        if price <= 0 or risk_dist <= 0 or equity <= 0:
            return False
        full_qty = (config.RISK_PER_TRADE * equity) / (price * risk_dist)
        qty = rm.round_qty(fraction * full_qty, inst.qty_decimals)
        if qty <= 0:
            log.warning("%s: tranche %d sized to 0, skipping.", name, tranche_index + 1)
            return False
        order_id = self.pf.submit_market_order(inst, qty, "buy")
        if not order_id:
            return False
        self.state["intents"].pop(name, None)
        self.state["pending"][name] = {
            "type": "buy", "order_id": str(order_id), "before_qty": before_qty,
            "tranche_index": tranche_index, "stop_dist": risk_dist,
            "signal_price": price, "requested_qty": qty, "how": how,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save_state()
        self._wait_pending(name, inst)
        return name not in self.state["pending"]

    def _confirm_pending(self, name, inst):
        pending = self.state["pending"].get(name)
        if not pending:
            return True
        broker = self._broker_position(inst)
        if pending["type"] == "buy":
            if not broker or broker["qty"] <= pending["before_qty"] + 1e-9:
                return False
            pos = self.state["positions"].get(name)
            if pos is None:
                pos = {
                    "tranches": 0, "stop_dist": pending["stop_dist"],
                    "entry_time": datetime.now(timezone.utc).isoformat(),
                    "stop_order_id": None, "stop_level": None, "stop_qty": 0.0,
                }
            pos["tranches"] = max(pos.get("tranches", 0), pending["tranche_index"] + 1)
            pos["qty"], pos["avg_entry"] = broker["qty"], broker["avg_entry"]
            fill_px = self.pf.recent_fill_price(inst, "buy", pending.get("order_id"))
            pos["last_add_price"] = fill_px or pending["signal_price"]
            self.state["positions"][name] = pos
            self.state["pending"].pop(name, None)
            protected = self._place_stop(name, inst, pos)
            self.save_state()
            self.notify(f"BUY {name} tranche {pos['tranches']}/{len(self.params.tranches)} "
                        f"qty={pending['requested_qty']} @~{(fill_px or broker['avg_entry']):.4f} "
                        f"({pending['how']})" + ("" if protected else " [broker stop unavailable]"))
            return True

        if pending["type"] == "close":
            if broker is not None:
                # A partial close is possible. Keep broker truth but do not clear
                # state until flat; no second strategy order is allowed meanwhile.
                pos = self.state["positions"].get(name)
                if pos:
                    pos["qty"], pos["avg_entry"] = broker["qty"], broker["avg_entry"]
                return False
            pos = pending["position"]
            exit_px = self.pf.recent_fill_price(inst, "sell", pending.get("order_id"))
            exit_px = exit_px or pending["requested_exit_price"]
            pnl = (exit_px - pos["avg_entry"]) * pos["qty"]
            self._log_trade(name, pos, exit_px, pnl, pending["reason"])
            self._accrue_daily(pnl)
            self.state["positions"].pop(name, None)
            self.state["intents"].pop(name, None)
            self.state["pending"].pop(name, None)
            self.save_state()
            self.notify(f"CLOSE {name} @~{exit_px:.4f} pnl={pnl:.2f} ({pending['reason']})")
            return True
        raise RuntimeError(f"Unknown pending order type for {name}: {pending['type']}")

    def _place_stop(self, name, inst, pos):
        old_id = pos.get("stop_order_id")
        old_level = pos.get("stop_level")
        old_qty = pos.get("stop_qty", pos.get("qty", 0.0))
        if old_id and not self.pf.cancel_order(old_id):
            self.notify(f"STOP REPLACE FAILED: could not confirm cancellation of {name} stop "
                        f"{old_id}; keeping old stop id and not creating a duplicate.")
            return False

        stop_level = pos["avg_entry"] * (1 - pos["stop_dist"])
        new_id = self.pf.submit_stop_order(inst, pos["qty"], stop_level)
        if new_id:
            pos["stop_order_id"] = str(new_id)
            pos["stop_level"] = stop_level
            pos["stop_qty"] = pos["qty"]
            return True

        # Crypto and the IBKR adapter intentionally return None. If an equity stop
        # existed and replacement failed, attempt to restore the previous cover.
        pos["stop_order_id"] = None
        pos["stop_level"] = None
        pos["stop_qty"] = 0.0
        if old_id and old_level and old_qty:
            restored = self.pf.submit_stop_order(inst, old_qty, old_level)
            if restored:
                pos["stop_order_id"] = str(restored)
                pos["stop_level"] = old_level
                pos["stop_qty"] = old_qty
                self.notify(f"STOP REPLACE FAILED for {name}; previous stop coverage was restored.")
                return False
            self.notify(f"CRITICAL: {name} broker stop replacement and restore both failed; "
                        "only the in-process stop is active.")
        return False

    def _close(self, name, inst, exit_price, reason):
        if name in self.state["pending"]:
            return False
        pos = self.state["positions"].get(name)
        if not pos:
            return False
        old_stop = pos.get("stop_order_id")
        if old_stop and not self.pf.cancel_order(old_stop):
            self.notify(f"CLOSE BLOCKED: could not confirm cancellation of {name} protective stop; "
                        "refusing to risk a double sell.")
            return False
        pos["stop_order_id"] = None
        order_id = self.pf.close_position_raw(inst)
        if not order_id:
            # Restore protection if close submission itself was rejected.
            self._place_stop(name, inst, pos)
            return False
        self.state["pending"][name] = {
            "type": "close", "order_id": str(order_id), "reason": reason,
            "requested_exit_price": exit_price, "position": dict(pos),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        self.state["intents"].pop(name, None)
        self.save_state()
        self._wait_pending(name, inst)
        return name not in self.state["pending"]

    def _finalize_external_close(self, name, inst, pos):
        exit_px = self.pf.recent_fill_price(inst, "sell")
        if exit_px is None:
            exit_px = pos["avg_entry"] * (1 - pos.get("stop_dist", self.params.min_stop))
        if pos.get("stop_order_id"):
            self.pf.cancel_order(pos["stop_order_id"])
        pnl = (exit_px - pos["avg_entry"]) * pos["qty"]
        self._log_trade(name, pos, exit_px, pnl, "broker stop / external close")
        self._accrue_daily(pnl)
        self.state["positions"].pop(name, None)
        self.state["intents"].pop(name, None)
        self.state["pending"].pop(name, None)
        self.save_state()
        self.notify(f"CLOSE {name} @~{exit_px:.4f} pnl={pnl:.2f} (broker stop/external)")

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    def _log_trade(self, name, pos, exit_price, pnl, reason):
        new = not os.path.exists(PULLBACK_TRADES_CSV)
        with open(PULLBACK_TRADES_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp", "instrument", "direction", "entry_price",
                            "exit_price", "pnl", "position_size", "tranches", "reason"])
            w.writerow([datetime.now(timezone.utc).isoformat(), name, "long",
                        round(pos["avg_entry"], 4), round(exit_price, 4),
                        round(pnl, 2), pos["qty"], pos.get("tranches", 0), reason])

    def _accrue_daily(self, pnl):
        today = datetime.now(timezone.utc).date().isoformat()
        d = self.state["daily"]
        if d.get("date") != today:
            if d.get("date") is not None:
                self._write_daily(d["date"], d["realized"])
            d["date"] = today
            d["realized"] = 0.0
        d["realized"] += pnl

    def _write_daily(self, date_str, realized):
        try:
            equity = self.pf.get_equity()
        except Exception:  # noqa: BLE001
            equity = ""
        rows = {}
        if os.path.exists(PULLBACK_DAILY_PNL_CSV):
            try:
                with open(PULLBACK_DAILY_PNL_CSV, newline="") as f:
                    for row in csv.DictReader(f):
                        if row.get("date"):
                            rows[row["date"]] = row
            except (OSError, csv.Error):
                rows = {}
        rows[date_str] = {"date": date_str, "realized_pnl": round(realized, 2),
                          "equity_snapshot": equity}
        tmp = PULLBACK_DAILY_PNL_CSV + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["date", "realized_pnl", "equity_snapshot"])
            w.writeheader()
            for key in sorted(rows):
                w.writerow(rows[key])
        os.replace(tmp, PULLBACK_DAILY_PNL_CSV)

    def flush_daily(self):
        d = self.state["daily"]
        if d.get("date"):
            self._write_daily(d["date"], d["realized"])
        self.save_state()

    # ------------------------------------------------------------------ #
    # Per-symbol processing
    # ------------------------------------------------------------------ #
    def _process(self, name):
        inst = self.instruments[name]
        strat = self.strats[name]

        # First settle any previously accepted order. Until broker state confirms
        # it, the symbol is blocked from all other strategy actions.
        if name in self.state["pending"]:
            if not self._confirm_pending(name, inst):
                return

        if not self.pf.is_tradable_now(inst):
            return

        today = datetime.now(timezone.utc).date()
        cached = self._daily_cache.get(name)
        if cached and cached[0] == today:
            daily_full = cached[1]
        else:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=_DAILY_WARMUP_DAYS + 30)
            daily_full = self.pf.get_historical_bars(inst, "1Day", start, end)
            if daily_full is None or daily_full.empty:
                self._mark_data(name, ok=False)
                return
            self._daily_cache[name] = (today, daily_full)

        daily = daily_full[[d.date() < today for d in daily_full.index]]
        if len(daily) < strat.warmup():
            self._mark_data(name, ok=False)
            return

        latest = self._latest_price(inst)
        if latest is None:
            self._mark_data(name, ok=False)
            return
        self._mark_data(name, ok=True)

        ma_f, ma_s = strat.moving_averages(daily)
        atr_series = ind.atr(daily, self.params.atr_period)
        i = len(daily) - 1
        price_i = float(daily["close"].iloc[i])
        a = atr_series.iloc[i]
        atr_i = float(a) if a == a else 0.0
        stop_dist = strat.stop_distance(atr_i, price_i)

        broker = self._broker_position(inst)
        pos = self.state["positions"].get(name)
        if pos and broker is None:
            self._finalize_external_close(name, inst, pos)
            return
        if broker and pos is None:
            pos = self._adopt_long(name, inst, broker, fully_built=True)
            self.state["intents"].pop(name, None)
            self.save_state()
            self.notify(f"ADOPTED {name}: broker had an untracked long; treating it as fully built.")
        elif broker and pos:
            changed = abs(pos["qty"] - broker["qty"]) > 1e-9
            pos["qty"], pos["avg_entry"] = broker["qty"], broker["avg_entry"]
            if changed and abs(float(pos.get("stop_qty", 0)) - broker["qty"]) > 1e-9:
                self._place_stop(name, inst, pos)
                self.save_state()

        if pos:
            stop_level = pos["avg_entry"] * (1 - pos["stop_dist"])
            if latest <= stop_level:
                self._close(name, inst, latest, "volatility stop max(5%,2xATR)")
                return

        last_ts = self.state["last_daily"].get(name)
        cur_ts = str(daily.index[i])
        if cur_ts != last_ts:
            self.state["last_daily"][name] = cur_ts
            trend = strat.trend_ok(daily, ma_f, ma_s, i)

            if pos:
                te = strat.trend_exit(daily, ma_f, i)
                if te:
                    self.state["intents"].pop(name, None)
                    self._close(name, inst, latest, te[1])
                    return

            intent = self.state["intents"].get(name)
            if intent:
                valid = trend and (
                    (intent["type"] == "enter" and pos is None) or
                    (intent["type"] == "add" and pos is not None
                     and pos["tranches"] < len(self.params.tranches)))
                if valid:
                    self._buy_tranche(name, inst, latest, intent["tranche_index"],
                                      stop_dist, "fallback")
                else:
                    self.state["intents"].pop(name, None)
                if name in self.state["pending"]:
                    return

            pos = self.state["positions"].get(name)
            if pos is None and trend:
                ok, _ = strat.entry_signal(daily, ma_f, i)
                if ok:
                    self.state["intents"][name] = {
                        "type": "enter", "tranche_index": 0,
                        "limit": price_i * (1 - self.params.improve_pct)}
            elif pos and trend and pos["tranches"] < len(self.params.tranches):
                add, _ = strat.should_add(daily, ma_f, i, pos["last_add_price"])
                if add:
                    self.state["intents"][name] = {
                        "type": "add", "tranche_index": pos["tranches"],
                        "limit": price_i * (1 - self.params.improve_pct)}
            self.save_state()

        intent = self.state["intents"].get(name)
        if intent and latest <= intent["limit"]:
            self._buy_tranche(name, inst, latest, intent["tranche_index"],
                              stop_dist, "limit dip")

    def _mark_data(self, name, ok: bool) -> None:
        if ok:
            self._stall[name] = 0
            self._stall_alerted.discard(name)
            return
        self._stall[name] = self._stall.get(name, 0) + 1
        if self._stall[name] >= STALL_ALERT_AFTER and name not in self._stall_alerted:
            self._stall_alerted.add(name)
            self.notify(f"DATA STALL: no fresh usable price/bars for {name} "
                        f"({self._stall[name]} cycles).")

    def step(self):
        for name in self.symbols:
            try:
                self._process(name)
            except Exception as e:  # noqa: BLE001
                log.exception("%s processing error (continuing): %s", name, e)
                self.notify(f"ERROR processing {name}: {e}")
        self.save_state()

    def run(self):
        log.info("Strategy 4 live runner started. Symbols: %s", ", ".join(self.symbols))
        self.notify("Strategy 4 live runner started: " + ", ".join(self.symbols))
        signal.signal(signal.SIGINT, _handle_sigterm)
        signal.signal(signal.SIGTERM, _handle_sigterm)
        while _RUNNING:
            self.step()
            for _ in range(POLL_SECONDS):
                if not _RUNNING:
                    break
                time.sleep(1)
        self.flush_daily()
        log.info("Strategy 4 live runner stopped cleanly.")
        self.notify("Strategy 4 live runner stopped.")


def main():
    ap = argparse.ArgumentParser(description="Live runner for Strategy 4 (trend-pullback).")
    ap.add_argument("--symbols", nargs="+", default=config.PULLBACK_SYMBOLS)
    ap.add_argument("--ema", action="store_true", help="Use EMAs instead of SMAs.")
    ap.add_argument("--live", action="store_true",
                    help="Required to run against the LIVE (real-money) endpoint.")
    args = ap.parse_args()

    config.validate_config()
    is_live_endpoint = "paper" not in config.ALPACA_BASE_URL.lower()
    if is_live_endpoint and not args.live:
        log.error("ALPACA_BASE_URL points at the LIVE endpoint. Re-run with --live to confirm.")
        return 1
    if not is_live_endpoint and args.live:
        log.error("--live was supplied but ALPACA_BASE_URL is a paper endpoint; refusing ambiguity.")
        return 1
    configure_alpaca_runtime(is_live_endpoint)
    if is_live_endpoint:
        log.warning("RUNNING AGAINST REAL MONEY (live endpoint).")

    params = PullbackParams(use_ema=args.ema)
    pf = Portfolio()
    with SingleInstanceLock(PULLBACK_STATE_FILE + ".lock"):
        trader = PullbackLiveTrader(pf, args.symbols, params, state_file=PULLBACK_STATE_FILE)
        trader.reconcile()
        trader.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
