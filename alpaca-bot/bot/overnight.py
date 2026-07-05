"""
bot/overnight.py
================
Tests the well-documented "overnight effect": historically, a disproportionate
share of equity market gains has come from the CLOSE-to-OPEN (overnight) return,
while OPEN-to-CLOSE (intraday) has been flat or negative. This is a genuinely
different return decomposition from everything else in this project — it doesn't
predict direction, it asks "when during the day does the return actually happen?"

Two mechanical books, each equal-weighted across the universe:
  overnight_only(daily)  Buy at today's close, sell at tomorrow's open — captures
                         ONLY the close-to-open return.
  intraday_only(daily)   Buy at today's open, sell at today's close — captures
                         ONLY the open-to-close return.
Both require a full round-trip trade EVERY SINGLE DAY (unlike every other
strategy in this project, which only trades on a signal change), so realistic
slippage is charged twice a day, every day — the honest cost that determines
whether this famous anomaly survives being actually traded.

    python -m bot.overnight --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
"""

import argparse
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from bot.backtest_pullback import (compute_metrics, buy_hold_combined, fetch_all,
                                   plot_equity, _parse_date, INITIAL_EQUITY, SLIPPAGE)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("overnight")

RESULTS_PNG = "overnight_results.png"
# A full round-trip (enter + exit) happens every day for both books, so the daily
# cost is 2x whatever per-side cost is assumed — this default matches the project-
# standard "occasional trend trade" slippage convention used elsewhere, but that
# convention was calibrated for a handful of trades a year, not a daily round-trip.
# --cost-per-side lets you explore lower, more realistic costs for a hyper-liquid
# ETF (see the README finding: this strategy's viability is EXTREMELY sensitive to
# the cost assumption, unlike anything else in this project).
_DAILY_ROUNDTRIP_COST = 2 * SLIPPAGE


def overnight_return(daily: pd.DataFrame, cost_per_side: float = SLIPPAGE) -> pd.Series:
    """Close[t-1] -> Open[t]: holding ONLY overnight, flat during the session."""
    prev_close = daily["close"].shift(1)
    return (daily["open"] / prev_close - 1.0).fillna(0.0) - 2 * cost_per_side


def intraday_return(daily: pd.DataFrame, cost_per_side: float = SLIPPAGE) -> pd.Series:
    """Open[t] -> Close[t]: holding ONLY during the session, flat overnight."""
    return (daily["close"] / daily["open"] - 1.0).fillna(0.0) - 2 * cost_per_side


def portfolio_book(daily_data: dict, kind: str, cost_per_side: float = SLIPPAGE) -> pd.Series:
    """Equal-weight combination across the universe. kind: 'overnight'|'intraday'."""
    fn = overnight_return if kind == "overnight" else intraday_return
    per_symbol = {name: fn(daily, cost_per_side) for name, daily in daily_data.items()}
    return pd.concat(per_symbol, axis=1).mean(axis=1, skipna=True)


def run(daily_data, begin_ts, end_ts=None, cost_per_side=SLIPPAGE):
    overnight = portfolio_book(daily_data, "overnight", cost_per_side)
    intraday = portfolio_book(daily_data, "intraday", cost_per_side)
    bh = buy_hold_combined(daily_data, begin_ts)

    def m(r):
        eq = INITIAL_EQUITY * (1 + r[r.index >= begin_ts]).cumprod()
        return eq, compute_metrics([], eq)
    eq_on, m_on = m(overnight)
    eq_id, m_id = m(intraday)
    m_bh = compute_metrics([], bh)

    print(f"\nOVERNIGHT vs INTRADAY  (daily round-trip cost {2*cost_per_side:.3%} charged "
          f"to BOTH books, every day: {cost_per_side:.3%}/side)")
    print("=" * 62)
    print(f"{'Book':22}{'Return%':>10}{'MaxDD%':>10}{'Sharpe':>9}")
    print("-" * 62)
    print(f"{'buy_and_hold':22}{m_bh['total_return']*100:>10.1f}{m_bh['max_drawdown']*100:>10.1f}"
          f"{m_bh['sharpe']:>9.2f}")
    print(f"{'overnight_only':22}{m_on['total_return']*100:>10.1f}{m_on['max_drawdown']*100:>10.1f}"
          f"{m_on['sharpe']:>9.2f}")
    print(f"{'intraday_only':22}{m_id['total_return']*100:>10.1f}{m_id['max_drawdown']*100:>10.1f}"
          f"{m_id['sharpe']:>9.2f}")
    print("=" * 62)
    if m_on["sharpe"] > m_bh["sharpe"]:
        print(f"overnight_only BEATS buy-and-hold on Sharpe ({m_on['sharpe']:.2f} vs {m_bh['sharpe']:.2f}).")
    else:
        print(f"overnight_only trails buy-and-hold on Sharpe ({m_on['sharpe']:.2f} vs {m_bh['sharpe']:.2f}).")
    print("Note: the classic academic anomaly is well-documented BEFORE costs; this "
          "charges realistic daily round-trip slippage, which is the honest test of "
          "whether it's actually tradeable, not just a market-structure curiosity.")
    plot_equity({"overnight_only": eq_on, "intraday_only": eq_id}, eq_on, RESULTS_PNG, bh)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Overnight vs intraday return decomposition.")
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "GLD", "TLT"])
    ap.add_argument("--cost-per-side", type=float, default=SLIPPAGE,
                    help="Per-side transaction cost (default matches the project-wide "
                         "0.05%% slippage convention, calibrated for infrequent trades -- "
                         "this strategy trades daily, so try much lower e.g. 0.0001 (1bp) "
                         "for a hyper-liquid ETF to see the raw signal's cost sensitivity.")
    ap.add_argument("--months", type=int, default=240)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca")
    args = ap.parse_args()

    end_dt = _parse_date(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc))
    start_dt = (_parse_date(args.start) if args.start
                else end_dt - pd.Timedelta(days=int(args.months * 31)))
    log.info("Overnight/intraday window: %s -> %s", start_dt.date(), end_dt.date())

    daily_data, _, _ = fetch_all(args.symbols, "none", start_dt, end_dt, source=args.data_source)
    if not daily_data:
        log.error("No data fetched.")
        return 1
    return run(daily_data, begin_ts=start_dt, cost_per_side=args.cost_per_side)


if __name__ == "__main__":
    raise SystemExit(main())
