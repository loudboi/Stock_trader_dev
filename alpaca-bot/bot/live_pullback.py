"""
bot/live_pullback.py
====================
Live execution for the phased trend-pullback strategy.

Safety invariants:
  * broker reads are authoritative; an API failure is not treated as "flat";
  * accepted orders are not fills -- the bot order's own terminal broker state and
    fill quantity are required before pending strategy state is resolved;
  * a symbol with an uncertain order is blocked from additional strategy orders;
  * missing live prices are data stalls, never stale daily-price substitutes;
  * broker position reconciliation still runs when the market is closed so manual
    changes and external closes do not wait for the next session;
  * managed positions cannot silently disappear from the configured symbol set;
  * untracked broker longs require explicit --adopt-existing consent;
  * externally/manual-adjusted managed positions are frozen as fully built because
    tranche intent cannot be reconstructed safely from broker quantity alone;
  * unknown broker fill prices are logged as unknown and are never booked as
    estimated realized P&L;
  * per-position sizing is additionally bounded by managed-book stop-risk and
    gross-exposure limits from config.py.
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
CANCEL_RETRY_SECONDS = 2.0
_RUNNING = True


def _handle_sigterm(signum, frame):
    global _RUNNING
    log.info("Shutdown signal received; finishing current cycle.")
    _RUNNING = False


def configure_alpaca_runtime(is_live: bool) -> None:
    global PULLBACK_TRADES_CSV, PULLBACK_DAILY_PNL_CSV, PULLBACK_STATE_FILE
    prefix = "pullback_live" if is_live else "pullback"
    PULLBACK_TRADES_CSV = f"{prefix}_trades.csv"
    PULLBACK_DAILY_PNL_CSV = f"{prefix}_daily_pnl.csv"
    PULLBACK_STATE_FILE = f"{prefix}_state.json"


class SingleInstanceLock:
    def __init__(self, path):
        self.path, self.fp = path, None

    def __enter__(self):
        self.fp = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                self.fp.seek(0); self.fp.write("0"); self.fp.flush(); self.fp.seek(0)
                msvcrt.locking(self.fp.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError) as e:
            self.fp.close(); self.fp = None
            raise RuntimeError(
                f"Another live runner appears to be using {self.path}; refusing duplicate execution.") from e
        self.fp.seek(0); self.fp.truncate(); self.fp.write(str(os.getpid())); self.fp.flush()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.fp:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self.fp.seek(0); msvcrt.locking(self.fp.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        finally:
            self.fp.close()


class PullbackLiveTrader:
    def __init__(self, pf, symbols, params: PullbackParams, state_file=None,
                 notifier: Notifier = None, adopt_existing=False):
        self.pf = pf
        self.symbols = list(dict.fromkeys(symbols))
        self.params = params
        self.state_file = state_file or PULLBACK_STATE_FILE
        self.notifier = notifier or Notifier()
        self.adopt_existing = bool(adopt_existing)
        self.instruments = {s: config.resolve_instrument(s) for s in self.symbols}
        missing = [s for s, i in self.instruments.items() if i is None]
        if missing:
            raise RuntimeError(f"Unknown live symbol(s): {', '.join(missing)}")
        self.strats = {s: TrendPullbackStrategy(self.instruments[s], params) for s in self.symbols}
        self.state = self._load_state()
        managed = set(self.state["positions"]) | set(self.state["intents"]) | set(self.state["pending"])
        orphaned = managed - set(self.symbols)
        if orphaned:
            raise RuntimeError(
                "State contains managed symbol(s) omitted from this run: "
                f"{', '.join(sorted(orphaned))}. Include/flatten/migrate them before restart.")
        self.state["runtime"] = {"symbols": sorted(self.symbols)}
        self._daily_cache = {}
        self._stall = {s: 0 for s in self.symbols}
        self._stall_alerted = set()
        # Retry throttling is intentionally process-local. A restart must be allowed
        # to retry a cancellation that previously failed rather than trusting a stale
        # persisted "cancel_requested" flag forever.
        self._cancel_retry_after = {}

    def notify(self, message: str) -> None:
        log.info("ALERT: %s", message)
        self.notifier.notify(message)

    @staticmethod
    def _default_state():
        return {"version": 2, "positions": {}, "intents": {}, "pending": {},
                "last_daily": {}, "daily": {"date": None, "realized": 0.0}, "runtime": {}}

    def _load_state(self):
        state = self._default_state()
        if not os.path.exists(self.state_file):
            return state
        try:
            with open(self.state_file) as f:
                loaded = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise RuntimeError(f"Could not safely read state file {self.state_file}: {e}") from e
        if not isinstance(loaded, dict):
            raise RuntimeError(f"Invalid state file {self.state_file}: expected JSON object")
        state.update(loaded)
        for key in ("positions", "intents", "pending", "last_daily", "runtime"):
            state.setdefault(key, {})
        state.setdefault("daily", {"date": None, "realized": 0.0})
        return state

    def save_state(self):
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=2, default=str)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, self.state_file)

    def _broker_position(self, inst):
        r = self.pf.get_position_raw(inst)
        if r and r.get("side") != "long":
            raise RuntimeError(
                f"{inst.name}: broker holds a {r.get('side')} position; Strategy 4 is long-only.")
        return r

    def _order_status(self, order_id):
        fn = getattr(self.pf, "order_status", None)
        if fn is None:
            return None
        try:
            return fn(order_id)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not read broker order status %s: %s", order_id, e)
            return None

    @staticmethod
    def _filled_qty(status):
        if not status:
            return None
        try:
            return max(0.0, float(status.get("filled_qty", 0) or 0))
        except (TypeError, ValueError):
            return None

    def _pop_pending(self, name):
        pending = self.state["pending"].pop(name, None)
        if pending:
            self._cancel_retry_after.pop(str(pending.get("order_id", "")), None)
        return pending

    def _request_pending_cancel(self, pending):
        """Attempt cancellation with a short retry throttle; failed attempts are retryable."""
        order_id = str(pending["order_id"])
        if pending.get("cancel_confirmed"):
            return True
        now = time.monotonic()
        if now < self._cancel_retry_after.get(order_id, 0.0):
            return False
        pending["cancel_requested"] = True
        pending["cancel_attempts"] = int(pending.get("cancel_attempts", 0) or 0) + 1
        pending["last_cancel_attempt_at"] = datetime.now(timezone.utc).isoformat()
        confirmed = bool(self.pf.cancel_order(order_id))
        pending["cancel_confirmed"] = confirmed
        self._cancel_retry_after[order_id] = now + CANCEL_RETRY_SECONDS
        self.save_state()
        return confirmed

    def _managed_capacity(self, equity):
        gross = risk = 0.0
        for p in self.state["positions"].values():
            qty = float(p.get("qty", 0) or 0)
            entry = float(p.get("avg_entry", 0) or 0)
            dist = float(p.get("stop_dist", self.params.min_stop) or self.params.min_stop)
            mark = max(entry, float(p.get("last_price", entry) or entry))
            gross += qty * mark
            risk += qty * entry * dist
        return {
            "gross_room": max(0.0, config.MAX_GROSS_EXPOSURE * equity - gross),
            "risk_room": max(0.0, config.MAX_PORTFOLIO_RISK * equity - risk),
        }

    def _wait_pending(self, name, inst, seconds=ORDER_CONFIRM_SECONDS):
        deadline = time.monotonic() + seconds
        while name in self.state["pending"] and time.monotonic() < deadline:
            if self._confirm_pending(name, inst):
                return True
            time.sleep(ORDER_CONFIRM_INTERVAL)
        if name not in self.state["pending"]:
            return True

        pending = self.state["pending"][name]
        status = self._order_status(pending["order_id"])
        if not (status and status.get("terminal")):
            self._request_pending_cancel(pending)
            cancel_deadline = time.monotonic() + 3.0
            while name in self.state["pending"] and time.monotonic() < cancel_deadline:
                if self._confirm_pending(name, inst):
                    return True
                time.sleep(ORDER_CONFIRM_INTERVAL)
        if name in self.state["pending"] and self._confirm_pending(name, inst):
            return True
        if name in self.state["pending"]:
            self.notify(f"ORDER UNCERTAIN: {name} could not be terminally reconciled; blocking symbol.")
            return False
        return True

    def _settle_terminal_buy(self, name, inst, pending, broker, filled_qty):
        requested = float(pending["requested_qty"])
        before = float(pending["before_qty"])
        fill_px = self.pf.recent_fill_price(inst, "buy", pending["order_id"])
        if filled_qty <= 1e-9:
            self._pop_pending(name)
            self.save_state()
            self.notify(f"BUY ENDED WITHOUT BOT FILL: {name} order {pending['order_id']}.")
            return True

        if broker is None:
            known_entry = fill_px or pending["signal_price"]
            transient = {
                "qty": filled_qty, "avg_entry": known_entry,
                "tranches": pending["tranche_index"] + 1,
                "entry_time": pending["submitted_at"],
                "entry_orders": [{
                    "order_id": pending["order_id"],
                    "requested_qty": requested, "filled_qty": filled_qty,
                    "fill_price": fill_px, "fill_price_source": ("broker_order" if fill_px else "unavailable"),
                    "submitted_at": pending["submitted_at"],
                    "tranche": pending["tranche_index"] + 1}],
            }
            self._log_trade(name, transient, None, None,
                            "bot buy filled, but broker was flat at reconciliation; "
                            "exit fill unavailable; PnL not accrued", None)
            self._pop_pending(name)
            self.state["positions"].pop(name, None)
            self.save_state()
            self.notify(f"BUY/FLAT RACE {name}: bot order filled {filled_qty}, but broker is flat; reconcile statement.")
            return True

        broker_delta = broker["qty"] - before
        exact_overlap = abs(broker_delta - filled_qty) > 1e-8
        fill_source = "broker_order" if fill_px is not None else "unavailable"
        resolved_fill = fill_px
        if resolved_fill is None and not exact_overlap:
            before_avg = float(pending.get("before_avg_entry", 0.0) or 0.0)
            inferred = (broker["qty"] * broker["avg_entry"] - before * before_avg) / filled_qty
            if inferred > 0 and inferred == inferred:
                resolved_fill = inferred
                fill_source = "inferred_from_position"

        pos = self.state["positions"].get(name)
        if pos is None:
            pos = {"tranches": 0, "stop_dist": pending["stop_dist"],
                   "entry_time": pending["submitted_at"], "stop_order_id": None,
                   "stop_level": None, "stop_qty": 0.0, "entry_orders": []}
        terminal_partial = filled_qty + 1e-8 < requested
        if terminal_partial or exact_overlap:
            pos["tranches"] = len(self.params.tranches)
        else:
            pos["tranches"] = max(pos.get("tranches", 0), pending["tranche_index"] + 1)
        pos["qty"], pos["avg_entry"] = broker["qty"], broker["avg_entry"]
        pos["last_add_price"] = resolved_fill or broker["avg_entry"]
        pos["last_price"] = broker["avg_entry"]
        if terminal_partial:
            pos["partial_fill_frozen"] = True
        if exact_overlap:
            pos["external_overlap"] = True
            pos["tranches"] = len(self.params.tranches)
        pos.setdefault("entry_orders", []).append({
            "order_id": pending["order_id"], "requested_qty": requested,
            "filled_qty": filled_qty, "fill_price": resolved_fill,
            "fill_price_source": fill_source,
            "submitted_at": pending["submitted_at"],
            "tranche": pending["tranche_index"] + 1,
            "terminal_partial": terminal_partial})
        self.state["positions"][name] = pos
        self._pop_pending(name)
        protected = self._place_stop(name, inst, pos)
        self.save_state()
        if terminal_partial:
            label = (f"PARTIAL BUY {name}: bot filled {filled_qty}/{requested}; "
                     "position frozen as fully built")
        elif exact_overlap:
            label = (f"BUY {name}: bot filled {filled_qty}, but broker quantity changed by "
                     f"{broker_delta}; external overlap detected and position frozen")
        else:
            px_text = f"@~{resolved_fill:.4f}" if resolved_fill is not None else "@ fill unavailable"
            label = (f"BUY {name} tranche {pos['tranches']}/{len(self.params.tranches)} "
                     f"qty={filled_qty} {px_text} ({pending['how']})")
        self.notify(label + ("" if protected else " [protective stop unavailable]"))
        return True

    def _record_close_piece(self, name, pos, qty, exit_px, reason, order_id):
        piece = dict(pos)
        piece["qty"] = qty
        if exit_px is None:
            self._log_trade(name, piece, None, None,
                            reason + " (fill unavailable; PnL not accrued)", order_id)
            return None
        pnl = (exit_px - pos["avg_entry"]) * qty
        self._log_trade(name, piece, exit_px, pnl, reason, order_id)
        self._accrue_daily(pnl)
        return pnl

    def _settle_terminal_close(self, name, inst, pending, broker, filled_qty):
        original = pending["position"]
        original_qty = float(original["qty"])
        exit_px = self.pf.recent_fill_price(inst, "sell", pending["order_id"])

        if filled_qty <= 1e-9:
            self._pop_pending(name)
            if broker is None:
                pos = self.state["positions"].get(name) or dict(original)
                self._finalize_external_close(name, inst, pos)
                return True
            pos = self.state["positions"].get(name) or dict(original)
            changed = (abs(float(pos.get("qty", 0)) - broker["qty"]) > 1e-9 or
                       abs(float(pos.get("avg_entry", 0)) - broker["avg_entry"]) > 1e-9)
            if changed:
                self._apply_external_position_change(name, inst, pos, broker)
            else:
                self.state["positions"][name] = pos
                protected = self._place_stop(name, inst, pos)
                self.notify(f"CLOSE ENDED WITHOUT BOT FILL: {name}; protection restored" +
                            ("." if protected else " [protective stop unavailable]."))
            self.save_state()
            return True

        if filled_qty + 1e-8 >= original_qty:
            pnl = self._record_close_piece(name, original, original_qty, exit_px,
                                           pending["reason"], pending["order_id"])
            self.state["positions"].pop(name, None)
            self.state["intents"].pop(name, None)
            self._pop_pending(name)
            self.save_state()
            detail = (f"@~{exit_px:.4f} pnl={pnl:.2f}"
                      if exit_px is not None else "fill/PnL unavailable")
            suffix = (" Broker now has a separate long; it will require explicit adoption."
                      if broker is not None else "")
            self.notify(f"CLOSE {name} {detail} ({pending['reason']}).{suffix}")
            return True

        pnl = self._record_close_piece(
            name, original, filled_qty, exit_px,
            pending["reason"] + " (terminal partial close)", pending["order_id"])
        expected_remaining = max(0.0, original_qty - filled_qty)
        self._pop_pending(name)
        if broker is None:
            remainder = dict(original)
            remainder["qty"] = expected_remaining
            if expected_remaining > 1e-9:
                self._log_trade(
                    name, remainder, None, None,
                    "external close of remainder after terminal partial bot close; "
                    "fill unavailable; PnL not accrued", None)
            self.state["positions"].pop(name, None)
            self.save_state()
            self.notify(f"PARTIAL CLOSE {name}: bot sold {filled_qty}; broker is now flat. Reconcile external remainder.")
            return True

        pos = self.state["positions"].get(name) or dict(original)
        pos["qty"], pos["avg_entry"] = broker["qty"], broker["avg_entry"]
        pos["tranches"] = len(self.params.tranches)
        pos["last_price"] = broker["avg_entry"]
        pos["partial_close_frozen"] = True
        if abs(broker["qty"] - expected_remaining) > 1e-8:
            pos["external_overlap"] = True
        self.state["positions"][name] = pos
        protected = self._place_stop(name, inst, pos)
        self.save_state()
        detail = f"realized pnl={pnl:.2f}" if pnl is not None else "fill/PnL unavailable"
        self.notify(f"PARTIAL CLOSE {name}: bot sold {filled_qty}, broker has {broker['qty']} long; "
                    f"{detail}; position frozen as fully built" +
                    ("." if protected else " [protective stop unavailable]."))
        return True

    def _adopt_long(self, name, inst, broker):
        stop_dist = self._current_stop_dist(name, inst, broker["avg_entry"])
        pos = {
            "tranches": len(self.params.tranches), "qty": broker["qty"],
            "avg_entry": broker["avg_entry"], "last_add_price": broker["avg_entry"],
            "last_price": broker["avg_entry"],
            "stop_dist": stop_dist, "stop_order_id": None, "stop_level": None,
            "stop_qty": 0.0, "entry_time": datetime.now(timezone.utc).isoformat(),
            "entry_orders": [], "adopted": True,
        }
        self.state["positions"][name] = pos
        self._place_stop(name, inst, pos)
        return pos

    def _handle_untracked_broker_long(self, name, inst, broker):
        if not self.adopt_existing:
            raise RuntimeError(
                f"{name}: broker has an existing long but state does not. Refusing to guess tranche/"
                "ownership. Re-run with --adopt-existing only if this bot should take it over.")
        pos = self._adopt_long(name, inst, broker)
        self.notify(f"ADOPTED {name}: existing broker long explicitly taken over as fully built.")
        return pos

    def _apply_external_position_change(self, name, inst, pos, broker):
        """Reconcile a manual/external broker edit without guessing tranche intent."""
        old_qty = float(pos.get("qty", 0) or 0)
        old_avg = float(pos.get("avg_entry", 0) or 0)
        pos["qty"], pos["avg_entry"] = broker["qty"], broker["avg_entry"]
        pos["last_price"] = broker["avg_entry"]
        pos["tranches"] = len(self.params.tranches)
        pos["last_add_price"] = broker["avg_entry"]
        pos["externally_adjusted"] = True
        pos["external_adjustment_at"] = datetime.now(timezone.utc).isoformat()
        self.state["intents"].pop(name, None)
        protected = self._place_stop(name, inst, pos)
        self.notify(
            f"EXTERNAL POSITION CHANGE {name}: broker qty/avg {old_qty}@{old_avg:.4f} -> "
            f"{broker['qty']}@{broker['avg_entry']:.4f}; treating as fully built and "
            "blocking further strategy adds" +
            ("." if protected else " [protective stop replacement not confirmed]."))
        return pos

    def _sync_broker_state(self, name, inst):
        """Reconcile strategy state to broker truth independent of market hours."""
        broker = self._broker_position(inst)
        pos = self.state["positions"].get(name)
        if pos and broker is None:
            self._finalize_external_close(name, inst, pos)
            return None, None, True
        if broker and pos is None:
            pos = self._handle_untracked_broker_long(name, inst, broker)
            self.state["intents"].pop(name, None)
            self.save_state()
        elif broker and pos:
            changed = (abs(float(pos.get("qty", 0)) - broker["qty"]) > 1e-9 or
                       abs(float(pos.get("avg_entry", 0)) - broker["avg_entry"]) > 1e-9)
            if changed:
                log.warning("%s broker position changed outside known strategy orders.", name)
                self._apply_external_position_change(name, inst, pos, broker)
                self.save_state()
        return broker, pos, False

    def reconcile(self):
        for name, inst in self.instruments.items():
            if name in self.state["pending"]:
                if not self._confirm_pending(name, inst):
                    log.warning("%s has an unresolved pending broker order; leaving symbol blocked.", name)
                    continue
            self._sync_broker_state(name, inst)
            if (name not in self.state["positions"] and
                    self._broker_position(inst) is None):
                self.state["intents"].pop(name, None)
        self.save_state()

    def _current_stop_dist(self, name, inst, price):
        try:
            end = datetime.now(timezone.utc)
            daily = self.pf.get_historical_bars(
                inst, "1Day", end - timedelta(days=_DAILY_WARMUP_DAYS + 30), end)
            if daily is None or daily.empty:
                return self.params.min_stop
            a = ind.atr(daily, self.params.atr_period).iloc[-1]
            return self.strats[name].stop_distance(float(a) if a == a else 0.0, price)
        except Exception as e:  # noqa: BLE001
            log.warning("%s: ATR stop unavailable (%s); using floor.", name, e)
            return self.params.min_stop

    def _buy_tranche(self, name, inst, price, tranche_index, stop_dist, how):
        if name in self.state["pending"]:
            return False
        before = self._broker_position(inst)
        before_qty = before["qty"] if before else 0.0
        before_avg = before["avg_entry"] if before else 0.0
        pos = self.state["positions"].get(name)
        risk_dist = pos["stop_dist"] if pos else stop_dist
        equity = self.pf.get_equity()
        if price <= 0 or risk_dist <= 0 or equity <= 0:
            return False
        fraction = self.params.tranches[tranche_index]
        desired = fraction * (config.RISK_PER_TRADE * equity) / (price * risk_dist)
        room = self._managed_capacity(equity)
        max_gross = room["gross_room"] / price
        max_risk = room["risk_room"] / (price * risk_dist)
        qty = rm.round_qty(min(desired, max_gross, max_risk), inst.qty_decimals)
        if qty <= 0:
            log.warning("%s: tranche %d blocked by size/portfolio limits.", name, tranche_index + 1)
            return False
        order_id = self.pf.submit_market_order(inst, qty, "buy")
        if not order_id:
            return False
        submitted_at = datetime.now(timezone.utc).isoformat()
        self.state["intents"].pop(name, None)
        self.state["pending"][name] = {
            "type": "buy", "order_id": str(order_id), "before_qty": before_qty,
            "before_avg_entry": before_avg,
            "tranche_index": tranche_index, "stop_dist": risk_dist, "signal_price": price,
            "requested_qty": qty, "how": how, "submitted_at": submitted_at,
        }
        self.save_state(); self._wait_pending(name, inst)
        return name not in self.state["pending"]

    def _confirm_pending(self, name, inst):
        pending = self.state["pending"].get(name)
        if not pending:
            return True
        broker = self._broker_position(inst)
        status = self._order_status(pending["order_id"])
        terminal = bool(status and status.get("terminal"))
        filled_qty = self._filled_qty(status)

        if pending["type"] == "buy":
            before = float(pending["before_qty"])
            broker_delta = max(0.0, (broker["qty"] if broker else 0.0) - before)
            if not terminal:
                if broker_delta > 1e-9:
                    self._request_pending_cancel(pending)
                    status = self._order_status(pending["order_id"])
                    terminal = bool(status and status.get("terminal"))
                    filled_qty = self._filled_qty(status)
                if not terminal:
                    return False
            if filled_qty is None:
                return False
            return self._settle_terminal_buy(name, inst, pending, broker, filled_qty)

        if pending["type"] == "close":
            original_qty = float(pending["position"]["qty"])
            broker_remaining = broker["qty"] if broker else 0.0
            broker_changed = (broker is None or
                              abs(broker_remaining - original_qty) > 1e-9)
            if not terminal:
                if broker_changed:
                    self._request_pending_cancel(pending)
                    status = self._order_status(pending["order_id"])
                    terminal = bool(status and status.get("terminal"))
                    filled_qty = self._filled_qty(status)
                if not terminal:
                    return False
            if filled_qty is None:
                return False
            return self._settle_terminal_close(name, inst, pending, broker, filled_qty)
        raise RuntimeError(f"Unknown pending order type for {name}: {pending['type']}")

    def _place_stop(self, name, inst, pos):
        old_id, old_level = pos.get("stop_order_id"), pos.get("stop_level")
        old_qty = pos.get("stop_qty", pos.get("qty", 0.0))
        if old_id and not self.pf.cancel_order(old_id):
            self.notify(f"STOP REPLACE FAILED: could not confirm cancellation of {name} stop {old_id}.")
            return False
        stop_level = pos["avg_entry"] * (1 - pos["stop_dist"])
        new_id = self.pf.submit_stop_order(inst, pos["qty"], stop_level)
        if new_id:
            pos["stop_order_id"], pos["stop_level"], pos["stop_qty"] = str(new_id), stop_level, pos["qty"]
            return True
        pos["stop_order_id"], pos["stop_level"], pos["stop_qty"] = None, None, 0.0
        if old_id and old_level and old_qty:
            restored = self.pf.submit_stop_order(inst, old_qty, old_level)
            if restored:
                pos["stop_order_id"], pos["stop_level"], pos["stop_qty"] = str(restored), old_level, old_qty
                self.notify(f"STOP REPLACE FAILED for {name}; previous coverage restored.")
                return False
            self.notify(f"CRITICAL: {name} broker stop replacement and restore both failed.")
        return False

    def _close(self, name, inst, exit_price, reason):
        if name in self.state["pending"]:
            return False
        pos = self.state["positions"].get(name)
        if not pos:
            return False
        old_stop = pos.get("stop_order_id")
        if old_stop and not self.pf.cancel_order(old_stop):
            self.notify(f"CLOSE BLOCKED: could not cancel {name} protective stop; avoiding double sell.")
            return False
        pos["stop_order_id"] = None
        order_id = self.pf.close_position_raw(inst)
        if not order_id:
            self._place_stop(name, inst, pos)
            return False
        self.state["pending"][name] = {
            "type": "close", "order_id": str(order_id), "reason": reason,
            "requested_exit_price": exit_price, "position": dict(pos),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        self.state["intents"].pop(name, None)
        self.save_state(); self._wait_pending(name, inst)
        return name not in self.state["pending"]

    def _finalize_external_close(self, name, inst, pos):
        exit_px = self.pf.recent_fill_price(inst, "sell", since=pos.get("entry_time"))
        if pos.get("stop_order_id"):
            self.pf.cancel_order(pos["stop_order_id"])
        if exit_px is None:
            self._log_trade(name, pos, None, None,
                            "broker stop / external close (fill unavailable; PnL not accrued)", None)
        else:
            pnl = (exit_px - pos["avg_entry"]) * pos["qty"]
            self._log_trade(name, pos, exit_px, pnl, "broker stop / external close", None)
            self._accrue_daily(pnl)
        pending = self.state["pending"].get(name)
        if pending:
            self._cancel_retry_after.pop(str(pending.get("order_id", "")), None)
        for bucket in ("positions", "intents", "pending"):
            self.state[bucket].pop(name, None)
        self.save_state()
        if exit_px is None:
            self.notify(f"CLOSE {name}: broker is flat but fill/PnL unavailable; reconcile statement (broker stop/external).")
        else:
            self.notify(f"CLOSE {name} @~{exit_px:.4f} pnl={pnl:.2f} (broker stop/external)")

    def _log_trade(self, name, pos, exit_price, pnl, reason, exit_order_id=None):
        new = not os.path.exists(PULLBACK_TRADES_CSV)
        fields = ["timestamp", "instrument", "direction", "entry_price", "exit_price", "pnl",
                  "position_size", "tranches", "reason", "entry_orders_json", "exit_order_id"]
        exit_value = "" if exit_price is None else round(exit_price, 4)
        pnl_value = "" if pnl is None else round(pnl, 2)
        with open(PULLBACK_TRADES_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(fields)
            w.writerow([datetime.now(timezone.utc).isoformat(), name, "long",
                        round(pos["avg_entry"], 4), exit_value, pnl_value,
                        pos["qty"], pos.get("tranches", 0), reason,
                        json.dumps(pos.get("entry_orders", []), separators=(",", ":")),
                        exit_order_id or ""])

    def _accrue_daily(self, pnl):
        today = datetime.now(timezone.utc).date().isoformat()
        d = self.state["daily"]
        if d.get("date") != today:
            if d.get("date") is not None:
                self._write_daily(d["date"], d["realized"])
            d["date"], d["realized"] = today, 0.0
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
                    rows = {r["date"]: r for r in csv.DictReader(f) if r.get("date")}
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

    def _process(self, name):
        inst, strat = self.instruments[name], self.strats[name]
        if name in self.state["pending"] and not self._confirm_pending(name, inst):
            return

        broker, pos, closed = self._sync_broker_state(name, inst)
        if closed:
            return
        if not self.pf.is_tradable_now(inst):
            return

        today = datetime.now(timezone.utc).date()
        cached = self._daily_cache.get(name)
        if cached and cached[0] == today:
            daily_full = cached[1]
        else:
            end = datetime.now(timezone.utc)
            daily_full = self.pf.get_historical_bars(
                inst, "1Day", end - timedelta(days=_DAILY_WARMUP_DAYS + 30), end)
            if daily_full is None or daily_full.empty:
                self._mark_data(name, False); return
            self._daily_cache[name] = (today, daily_full)
        daily = daily_full[[d.date() < today for d in daily_full.index]]
        if len(daily) < strat.warmup():
            self._mark_data(name, False); return
        latest = self.pf.latest_price(inst)
        if latest is None:
            self._mark_data(name, False); return
        self._mark_data(name, True)

        # Broker state may have changed while data was being fetched.
        broker, pos, closed = self._sync_broker_state(name, inst)
        if closed:
            return

        ma_f, ma_s = strat.moving_averages(daily)
        atr_series = ind.atr(daily, self.params.atr_period)
        i = len(daily) - 1
        price_i = float(daily["close"].iloc[i])
        a = atr_series.iloc[i]
        stop_dist = strat.stop_distance(float(a) if a == a else 0.0, price_i)

        if pos:
            pos["last_price"] = latest
        if pos and latest <= pos["avg_entry"] * (1 - pos["stop_dist"]):
            self._close(name, inst, latest, "volatility stop max(5%,2xATR)"); return

        cur_ts = str(daily.index[i])
        if cur_ts != self.state["last_daily"].get(name):
            self.state["last_daily"][name] = cur_ts
            trend = strat.trend_ok(daily, ma_f, ma_s, i)
            if pos:
                te = strat.trend_exit(daily, ma_f, i)
                if te:
                    self.state["intents"].pop(name, None)
                    self._close(name, inst, latest, te[1]); return
            intent = self.state["intents"].get(name)
            if intent:
                valid = trend and ((intent["type"] == "enter" and pos is None) or
                    (intent["type"] == "add" and pos is not None and
                     pos["tranches"] < len(self.params.tranches)))
                if valid:
                    self._buy_tranche(name, inst, latest, intent["tranche_index"], stop_dist, "fallback")
                else:
                    self.state["intents"].pop(name, None)
                if name in self.state["pending"]:
                    return
            pos = self.state["positions"].get(name)
            if pos is None and trend:
                ok, _ = strat.entry_signal(daily, ma_f, i)
                if ok:
                    self.state["intents"][name] = {"type": "enter", "tranche_index": 0,
                        "limit": price_i * (1 - self.params.improve_pct)}
            elif pos and trend and pos["tranches"] < len(self.params.tranches):
                add, _ = strat.should_add(daily, ma_f, i, pos["last_add_price"])
                if add:
                    self.state["intents"][name] = {"type": "add",
                        "tranche_index": pos["tranches"],
                        "limit": price_i * (1 - self.params.improve_pct)}
            self.save_state()

        intent = self.state["intents"].get(name)
        if intent and latest <= intent["limit"]:
            self._buy_tranche(name, inst, latest, intent["tranche_index"], stop_dist, "limit dip")

    def _mark_data(self, name, ok):
        if ok:
            self._stall[name] = 0; self._stall_alerted.discard(name); return
        self._stall[name] = self._stall.get(name, 0) + 1
        if self._stall[name] >= STALL_ALERT_AFTER and name not in self._stall_alerted:
            self._stall_alerted.add(name)
            self.notify(f"DATA STALL: no fresh usable price/bars for {name} ({self._stall[name]} cycles).")

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
        signal.signal(signal.SIGINT, _handle_sigterm); signal.signal(signal.SIGTERM, _handle_sigterm)
        while _RUNNING:
            self.step()
            for _ in range(POLL_SECONDS):
                if not _RUNNING:
                    break
                time.sleep(1)
        self.flush_daily()
        self.notify("Strategy 4 live runner stopped.")
        close = getattr(self.notifier, "close", None)
        if close:
            close()


def main():
    ap = argparse.ArgumentParser(description="Live runner for Strategy 4 (trend-pullback).")
    ap.add_argument("--symbols", nargs="+", default=config.PULLBACK_SYMBOLS)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--live", action="store_true", help="Required for the real-money endpoint.")
    ap.add_argument("--adopt-existing", action="store_true",
                    help="Explicitly take over untracked existing long positions as fully built.")
    args = ap.parse_args()
    config.validate_config()
    is_live = "paper" not in config.ALPACA_BASE_URL.lower()
    if is_live != args.live:
        log.error("Runtime mode and --live confirmation disagree; refusing ambiguous execution.")
        return 1
    configure_alpaca_runtime(is_live)
    if is_live:
        log.warning("RUNNING AGAINST REAL MONEY.")
    pf = Portfolio()
    with SingleInstanceLock(PULLBACK_STATE_FILE + ".lock"):
        trader = PullbackLiveTrader(pf, args.symbols, PullbackParams(use_ema=args.ema),
                                    state_file=PULLBACK_STATE_FILE,
                                    adopt_existing=args.adopt_existing)
        trader.reconcile(); trader.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
