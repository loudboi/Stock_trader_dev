"""
bot/live_pullback_ibkr.py
=========================
Live runner for Strategy 4 (trend-pullback) on Interactive Brokers, EUR-only.

It reuses the *existing* PullbackLiveTrader (strategy logic, pyramiding, exits)
unchanged — the only differences are the broker adapter (IBKRPortfolio), the EUR
universe, and separate output files so an IBKR run never collides with an Alpaca
run. No Alpaca file is modified.

Run (paper Gateway on port 4002 by default):
    python -m bot.live_pullback_ibkr
    python -m bot.live_pullback_ibkr --symbols STOXX600 DAX SAP
Live (real money) requires both a live IBKR_PORT (4001/7496) and --live.

Prerequisites:
    • IB Gateway or TWS running and logged in, API enabled, this client's IP
      trusted, and the right market-data subscriptions for the EUR exchanges.
    • pip install -r requirements-ibkr.txt
Env: IBKR_HOST (127.0.0.1), IBKR_PORT (4002 paper Gateway), IBKR_CLIENT_ID (17).

Outputs: pullback_ibkr_trades.csv / pullback_ibkr_daily_pnl.csv /
         pullback_ibkr_state.json   (kept separate from the Alpaca pullback run).
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("live_pullback_ibkr")

_LIVE_PORTS = {4001, 7496}        # IB Gateway live, TWS live
_STATE_FILE = "pullback_ibkr_state.json"


def main():
    ap = argparse.ArgumentParser(
        description="Live runner for Strategy 4 on IBKR (EUR-only).")
    ap.add_argument("--symbols", nargs="+", default=ibkr_universe.DEFAULT_SYMBOLS,
                    help="Names from ibkr_universe.IBKR_EUR_UNIVERSE.")
    ap.add_argument("--ema", action="store_true", help="Use EMAs instead of SMAs.")
    ap.add_argument("--live", action="store_true",
                    help="Required to run against a LIVE (real-money) IBKR port.")
    args = ap.parse_args()

    # Make the existing trader resolve EUR names + write to separate files.
    ibkr_universe.register()
    lp.PULLBACK_TRADES_CSV = "pullback_ibkr_trades.csv"
    lp.PULLBACK_DAILY_PNL_CSV = "pullback_ibkr_daily_pnl.csv"

    port = int(os.getenv("IBKR_PORT", "4002"))
    if port in _LIVE_PORTS and not args.live:
        log.error("IBKR_PORT %s is a LIVE (real-money) port. Re-run with --live "
                  "if you really mean it. Refusing for safety.", port)
        return 1
    if port in _LIVE_PORTS:
        log.warning("RUNNING AGAINST REAL MONEY (live IBKR port %s).", port)
    else:
        log.info("Using IBKR port %s (paper).", port)

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
        log.error("Is the Gateway running, logged in, with API enabled on port %s?",
                  port)
        return 1

    trader = lp.PullbackLiveTrader(pf, args.symbols, params, state_file=_STATE_FILE)
    trader.reconcile()
    try:
        trader.run()
    finally:
        pf.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
