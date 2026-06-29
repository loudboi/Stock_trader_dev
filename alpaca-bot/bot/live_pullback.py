"""
bot/live_pullback.py
====================
LIVE execution for Strategy 4 (phased trend-pullback).

Strategy 4 is multi-timeframe (daily signal) and pyramids into a position in
30/30/40 tranches with a volatility stop. It reuses the SAME decision logic as
the backtester (the methods on TrendPullbackStrategy), so live behavior matches
bot/backtest_pullback.py:

  - SIGNAL on the daily chart: trend gate, pullback/breakout entry, tranche adds,
    and the MA/structural exit are all decided on completed daily bars.
  - EXECUTION in real time: instead of replaying 1h/4h candles, the live runner
    polls the latest price every loop (finer than the backtest's intraday proxy).
    An armed entry/add fills when price dips to the daily-close * (1 - improve_pct)
    limit; if the dip doesn't come during the session, it fills at market on the
    next daily evaluation while the daily trend still holds (so intraday never
    vetoes a valid daily setup).
  - STOP: max(5%, 2 x ATR(14) / price) below the average entry, monitored in real
    time and closed at market when breached.

Outputs go to pullback_trades.csv / pullback_daily_pnl.csv / pullback_state.json.

Run from the project root (paper trading by default):

    python -m bot.live_pullback
    python -m bot.live_pullback --symbols SPY GLD
    python -m bot.live_pullback --symbols IWM ETH/USD     # after adding to config

SAFETY: one Alpaca account holds one position per symbol. Don't run two copies of
this runner on the same symbol against the same account at the same time.
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

_RUNNING = True


def _handle_sigterm(signum, frame):
    global _RUNNING
    log.info("Shutdown signal received; finishing current cycle.")
    _RUNNING = False


class PullbackLiveTrader:
    def __init__(self, pf: Portfolio, symbols, params: PullbackParams,
                 state_file=PULLBACK_STATE_FILE):
        self.pf = pf
        self.symbols = symbols
        self.params = params
        self.state_file = state_file
        self.instruments = {s: config.resolve_instrument(s) for s in symbols}
        missing = [s for s, i in self.instruments.items() if i is None]
        if missing:
            raise RuntimeError(
                f"Unknown symbol(s): {', '.join(missing)}. Add them to "
                "PULLBACK_UNIVERSE in config.py and to PULLBACK_SYMBOLS.")
        self.strats = {s: TrendPullbackStrategy(self.instruments[s], params)
                       for s in symbols}
        self.state = self._load_state()
        # Daily bars only change once per session, so cache them per UTC day and
        # refetch on rollover. The realtime stop still uses latest_price() each
        # loop; this just spares a historical-bars API call every cycle.
        self._daily_cache = {}   # name -> (utc_date, DataFrame)

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Could not read state (%s); starting fresh.", e)
        return {"positions": {}, "intents": {}, "last_daily": {},
                "daily": {"date": None, "realized": 0.0}}

    def save_state(self):
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=2, default=str)
        os.replace(tmp, self.state_file)

    # ------------------------------------------------------------------ #
    # Broker helpers
    # ------------------------------------------------------------------ #
    def _latest_price(self, inst, daily):
        px = self.pf.latest_price(inst)
        if px is not None:
            return px
        return float(daily["close"].iloc[-1]) if len(daily) else None

    def _broker_long(self, inst):
        """Return (qty, avg_entry) if a long position exists, else None."""
        r = self.pf.get_position_raw(inst)
        if not r or r["side"] != "long":
            return None
        return r["qty"], r["avg_entry"]

    # ------------------------------------------------------------------ #
    # Startup reconciliation (broker is source of truth)
    # ------------------------------------------------------------------ #
    def reconcile(self):
        for name, inst in self.instruments.items():
            bl = self._broker_long(inst)
            if bl:
                qty, avg = bl
                if name not in self.state["positions"]:
                    # Adopt: assume fully built so we don't keep pyramiding into it.
                    self.state["positions"][name] = {
                        "tranches": len(self.params.tranches),
                        "qty": qty, "avg_entry": avg,
                        "last_add_price": avg, "stop_dist": self.params.min_stop,
                        "entry_time": datetime.now(timezone.utc).isoformat(),
                    }
                    log.info("Adopted existing %s long (qty=%s @%.4f).", name, qty, avg)
            else:
                if name in self.state["positions"]:
                    log.info("State had %s but broker is flat; clearing.", name)
                    self.state["positions"].pop(name, None)
                self.state["intents"].pop(name, None)
        self.save_state()

    # ------------------------------------------------------------------ #
    # Orders
    # ------------------------------------------------------------------ #
    def _buy_tranche(self, name, inst, price, tranche_index, stop_dist, how):
        equity = self.pf.get_equity()
        fraction = self.params.tranches[tranche_index]
        if price <= 0 or stop_dist <= 0:
            return
        full_qty = (config.RISK_PER_TRADE * equity) / (price * stop_dist)
        qty = rm.round_qty(fraction * full_qty, inst.qty_decimals)
        if qty <= 0:
            log.warning("%s: tranche %d sized to 0, skipping.", name, tranche_index + 1)
            return
        if not self.pf.submit_market_order(inst, qty, "buy"):
            return

        bl = self._broker_long(inst)
        pos = self.state["positions"].get(name) or {}
        if not pos:
            pos = {"tranches": 0, "stop_dist": stop_dist,
                   "entry_time": datetime.now(timezone.utc).isoformat()}
        pos["tranches"] = pos.get("tranches", 0) + 1
        pos["last_add_price"] = price
        if bl:
            pos["qty"], pos["avg_entry"] = bl       # broker truth
        else:                                       # fallback if read lags
            pos["qty"] = pos.get("qty", 0.0) + qty
            pos["avg_entry"] = price
        self.state["positions"][name] = pos
        self.save_state()
        log.info("BUY %s tranche %d/%d qty=%s @~%.4f (%s) stop_dist=%.3f",
                 name, pos["tranches"], len(self.params.tranches), qty, price,
                 how, pos["stop_dist"])

    def _close(self, name, inst, exit_price, reason):
        pos = self.state["positions"].get(name)
        if not pos:
            return
        if not self.pf.close_position_raw(inst):
            return
        pnl = (exit_price - pos["avg_entry"]) * pos["qty"]
        self._log_trade(name, pos, exit_price, pnl, reason)
        self._accrue_daily(pnl)
        self.state["positions"].pop(name, None)
        self.state["intents"].pop(name, None)
        self.save_state()
        log.info("CLOSE %s @~%.4f pnl=%.2f (%s)", name, exit_price, pnl, reason)

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
        new = not os.path.exists(PULLBACK_DAILY_PNL_CSV)
        with open(PULLBACK_DAILY_PNL_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["date", "realized_pnl", "equity_snapshot"])
            w.writerow([date_str, round(realized, 2), equity])

    def flush_daily(self):
        d = self.state["daily"]
        if d.get("date"):
            self._write_daily(d["date"], d["realized"])
        self.save_state()

    # ------------------------------------------------------------------ #
    # Per-symbol processing (one loop iteration)
    # ------------------------------------------------------------------ #
    def _process(self, name):
        inst = self.instruments[name]
        strat = self.strats[name]
        if not self.pf.is_tradable_now(inst):
            return  # market closed for this asset; act next session

        today = datetime.now(timezone.utc).date()
        cached = self._daily_cache.get(name)
        if cached and cached[0] == today:
            daily_full = cached[1]
        else:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=_DAILY_WARMUP_DAYS + 30)
            daily_full = self.pf.get_historical_bars(inst, "1Day", start, end)
            if daily_full is None or daily_full.empty:
                return
            self._daily_cache[name] = (today, daily_full)
        # Only act on COMPLETED daily bars (exclude today's still-forming bar).
        daily = daily_full[[d.date() < today for d in daily_full.index]]
        if len(daily) < strat.warmup():
            return

        ma_f, ma_s = strat.moving_averages(daily)
        atr_series = ind.atr(daily, self.params.atr_period)
        i = len(daily) - 1
        price_i = float(daily["close"].iloc[i])
        a = atr_series.iloc[i]
        atr_i = float(a) if a == a else 0.0
        stop_dist = strat.stop_distance(atr_i, price_i)
        latest = self._latest_price(inst, daily)
        if latest is None:
            return

        pos = self.state["positions"].get(name)

        # 1. Real-time volatility stop.
        if pos:
            stop_level = pos["avg_entry"] * (1 - pos["stop_dist"])
            if latest <= stop_level:
                self._close(name, inst, latest, "volatility stop max(5%,2xATR)")
                return

        # 2. New completed daily bar -> daily decisions.
        last_ts = self.state["last_daily"].get(name)
        cur_ts = str(daily.index[i])
        if cur_ts != last_ts:
            self.state["last_daily"][name] = cur_ts
            trend = strat.trend_ok(daily, ma_f, ma_s, i)

            # 2a. Daily trend-break exit.
            if pos:
                te = strat.trend_exit(daily, ma_f, i)
                if te:
                    self.state["intents"].pop(name, None)
                    self._close(name, inst, latest, te[1])
                    return

            # 2b. Fallback-fill a stale unfilled intent (don't miss the trend).
            intent = self.state["intents"].get(name)
            if intent and not intent.get("filled"):
                valid = trend and (
                    (intent["type"] == "enter" and pos is None) or
                    (intent["type"] == "add" and pos is not None
                     and pos["tranches"] < len(self.params.tranches)))
                if valid:
                    self._buy_tranche(name, inst, latest,
                                      intent["tranche_index"], stop_dist, "fallback")
                self.state["intents"].pop(name, None)

            # 2c. Arm a fresh intent from this daily bar.
            pos = self.state["positions"].get(name)
            if pos is None and trend:
                ok, _ = strat.entry_signal(daily, ma_f, i)
                if ok:
                    self.state["intents"][name] = {
                        "type": "enter", "tranche_index": 0,
                        "limit": price_i * (1 - self.params.improve_pct),
                        "filled": False}
                    log.info("%s ARMED entry (limit %.4f)", name,
                             price_i * (1 - self.params.improve_pct))
            elif pos and trend and pos["tranches"] < len(self.params.tranches):
                add, _ = strat.should_add(daily, ma_f, i, pos["last_add_price"])
                if add:
                    self.state["intents"][name] = {
                        "type": "add", "tranche_index": pos["tranches"],
                        "limit": price_i * (1 - self.params.improve_pct),
                        "filled": False}
                    log.info("%s ARMED add (tranche %d)", name, pos["tranches"] + 1)
            self.save_state()

        # 3. Real-time limit-touch fill of the active intent.
        intent = self.state["intents"].get(name)
        if intent and not intent.get("filled") and latest <= intent["limit"]:
            self._buy_tranche(name, inst, latest, intent["tranche_index"],
                              stop_dist, "limit dip")
            self.state["intents"].pop(name, None)
            self.save_state()

    def step(self):
        for name in self.symbols:
            try:
                self._process(name)
            except Exception as e:  # noqa: BLE001
                log.exception("%s processing error (continuing): %s", name, e)
        self.save_state()

    def run(self):
        log.info("Strategy 4 live runner started. Symbols: %s", ", ".join(self.symbols))
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


# --------------------------------------------------------------------------- #
# Entry point + safety gates
# --------------------------------------------------------------------------- #
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
        log.error("ALPACA_BASE_URL points at the LIVE (real-money) endpoint. "
                  "Re-run with --live if you really mean it. Refusing for safety.")
        return 1
    if is_live_endpoint:
        log.warning("RUNNING AGAINST REAL MONEY (live endpoint).")

    params = PullbackParams(use_ema=args.ema)
    pf = Portfolio()
    trader = PullbackLiveTrader(pf, args.symbols, params)
    trader.reconcile()
    trader.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
