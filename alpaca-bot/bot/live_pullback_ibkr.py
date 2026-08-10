"""
bot/live_pullback_ibkr.py
=========================
Live runner for Strategy 4 on Interactive Brokers, EUR-only.

Standard IBKR ports are recognized automatically. A custom port is intentionally
ambiguous and requires IBKR_MODE=paper or IBKR_MODE=live so a custom live Gateway
cannot accidentally bypass the real-money confirmation gate.
"""

import argparse
import logging
import os
import sys

import config
from bot import ibkr_universe
from bot.portfolio_ibkr import IBKRPortfolio
from bot.strategies.trend_pullback import PullbackParams
import bot.live_pullback as lp

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("live_pullback_ibkr")

_LIVE_PORTS = {4001, 7496}
_PAPER_PORTS = {4002, 7497}


def _mode_for_port(port: int):
    if port in _LIVE_PORTS:
        return "live"
    if port in _PAPER_PORTS:
        return "paper"
    explicit = os.getenv("IBKR_MODE", "").strip().lower()
    return explicit if explicit in {"paper", "live"} else None


def main():
    ap = argparse.ArgumentParser(description="Live runner for Strategy 4 on IBKR (EUR-only).")
    ap.add_argument("--symbols", nargs="+", default=ibkr_universe.DEFAULT_SYMBOLS,
                    help="Names from ibkr_universe.IBKR_EUR_UNIVERSE.")
    ap.add_argument("--ema", action="store_true", help="Use EMAs instead of SMAs.")
    ap.add_argument("--live", action="store_true",
                    help="Required confirmation when IBKR_MODE/port is live.")
    args = ap.parse_args()

    ibkr_universe.register()
    port = int(os.getenv("IBKR_PORT", "4002"))
    mode = _mode_for_port(port)
    if mode is None:
        log.error("IBKR_PORT %s is non-standard. Set IBKR_MODE=paper or IBKR_MODE=live "
                  "explicitly; refusing to guess account mode.", port)
        return 1
    if mode == "live" and not args.live:
        log.error("IBKR runtime is LIVE. Re-run with --live to confirm real-money trading.")
        return 1
    if mode == "paper" and args.live:
        log.error("--live was supplied but IBKR runtime is explicitly paper; refusing ambiguity.")
        return 1

    suffix = "_live" if mode == "live" else ""
    lp.PULLBACK_TRADES_CSV = f"pullback_ibkr{suffix}_trades.csv"
    lp.PULLBACK_DAILY_PNL_CSV = f"pullback_ibkr{suffix}_daily_pnl.csv"
    state_file = f"pullback_ibkr{suffix}_state.json"
    if mode == "live":
        log.warning("RUNNING AGAINST REAL MONEY (IBKR port %s).", port)
    else:
        log.info("Using IBKR paper runtime on port %s.", port)

    unknown = [s for s in args.symbols if s not in config.PULLBACK_UNIVERSE]
    if unknown:
        log.error("Unknown symbol(s): %s. Valid: %s", ", ".join(unknown),
                  ", ".join(ibkr_universe.IBKR_EUR_UNIVERSE))
        return 1

    params = PullbackParams(use_ema=args.ema)
    try:
        pf = IBKRPortfolio()
    except Exception as e:  # noqa: BLE001
        log.error("Could not connect to IB Gateway/TWS: %s", e)
        return 1

    try:
        with lp.SingleInstanceLock(state_file + ".lock"):
            trader = lp.PullbackLiveTrader(pf, args.symbols, params, state_file=state_file)
            trader.reconcile()
            trader.run()
    finally:
        pf.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
