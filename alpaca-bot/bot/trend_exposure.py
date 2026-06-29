"""
bot/trend_exposure.py
=====================
Phase B — a trend-FILTERED EXPOSURE strategy, aimed squarely at beating
buy-and-hold (which the pullback strategy can't, because it sits in cash most of
the time even during uptrends).

The idea (classic 200-day-MA timing, à la Faber's tactical allocation):
  - While price is above its long moving average, be fully invested (optionally
    leveraged) in the asset.
  - When price drops below the MA, move to cash.
So you capture most of the uptrend but sidestep the big bear-market drawdowns
(2000-02, 2008, 2022) that buy-and-hold eats in full. With modest leverage on the
"risk-on" periods, that can beat buy-and-hold on return *and* Sharpe.

A `buffer` adds hysteresis: exit below the MA, but only re-enter once price is
`buffer` above it, which cuts whipsaw in choppy markets.

This is a daily ALLOCATION model, so it's backtested vectorized (no event loop):
the signal at each close is acted on the NEXT day's return (no lookahead), and a
slippage cost is charged whenever exposure flips.

Run from the project root:

    python -m bot.trend_exposure --symbols SPY QQQ GLD --start 2005-01-01 \
        --data-source yahoo --leverage 1.5 --buffer 0.01
"""

import argparse
import logging

import numpy as np
import pandas as pd

import config
from bot.backtest_pullback import (compute_metrics, buy_hold_equity, buy_hold_combined,
                                   fetch_all, plot_equity, print_vs_benchmark,
                                   _parse_date, INITIAL_EQUITY, SLIPPAGE)
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("trend_exposure")

RESULTS_PNG = "trend_exposure_results.png"


# --------------------------------------------------------------------------- #
# Signal + per-symbol backtest (pure, vectorized, testable offline)
# --------------------------------------------------------------------------- #
def exposure_series(close: pd.Series, ma_period: int, buffer: float = 0.0) -> pd.Series:
    """Target exposure (0 = cash, 1 = invested) from a long-MA trend filter.

    Hysteresis: go invested when close > MA*(1+buffer); go to cash when close < MA.
    Uses only data up to each bar (the MA includes the current close; the caller
    shifts by one day before applying it, so there's no lookahead)."""
    ma = close.rolling(ma_period, min_periods=ma_period).mean()
    upper = ma * (1 + buffer)
    state, out = 0.0, []
    for c, u, m in zip(close.values, upper.values, ma.values):
        if m != m:                       # NaN during warmup -> stay flat
            out.append(0.0)
            continue
        if state == 0.0 and c > u:
            state = 1.0
        elif state == 1.0 and c < m:
            state = 0.0
        out.append(state)
    return pd.Series(out, index=close.index)


def strategy_returns(daily: pd.DataFrame, ma_period: int, buffer: float,
                     leverage: float) -> pd.Series:
    """Daily strategy returns: yesterday's exposure × leverage × today's move,
    minus a slippage cost each time exposure flips."""
    close = daily["close"]
    target = exposure_series(close, ma_period, buffer)
    held = target.shift(1).fillna(0.0)               # act on the next bar (no lookahead)
    asset_ret = close.pct_change().fillna(0.0)
    gross = held * leverage * asset_ret
    flips = held.diff().abs().fillna(0.0)
    cost = flips * leverage * SLIPPAGE
    return gross - cost


def equity_from_returns(rets: pd.Series, begin_ts, initial=INITIAL_EQUITY) -> pd.Series:
    r = rets[rets.index >= begin_ts] if begin_ts is not None else rets
    if not len(r):
        return pd.Series(dtype=float)
    return initial * (1 + r).cumprod()


def _trade_count(daily, ma_period, buffer) -> int:
    """Number of times we go from cash to invested (round-trip entries)."""
    held = exposure_series(daily["close"], ma_period, buffer).shift(1).fillna(0.0)
    return int(((held > 0) & (held.shift(1).fillna(0.0) == 0)).sum())


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(daily_data, ma_period, buffer, leverage, begin_ts):
    n = len(daily_data)
    per_strat, per_bh, per_series, ret_frame = {}, {}, {}, {}
    for name, daily in daily_data.items():
        rets = strategy_returns(daily, ma_period, buffer, leverage)
        eq = equity_from_returns(rets, begin_ts)
        m = compute_metrics([], eq)
        m["trades"] = _trade_count(daily, ma_period, buffer)
        per_strat[name] = m
        per_series[name] = eq
        per_bh[name] = compute_metrics([], buy_hold_equity(daily, begin_ts))
        ret_frame[name] = rets
        log.info("%s: %d switches, return %.1f%%, MaxDD %.1f%%, Sharpe %.2f", name,
                 m["trades"], m["total_return"] * 100, m["max_drawdown"] * 100, m["sharpe"])

    # Combined: equal-weight across symbols, rebalanced daily (mean of returns).
    combined_ret = pd.concat(ret_frame, axis=1).mean(axis=1, skipna=True)
    combined_eq = equity_from_returns(combined_ret, begin_ts)
    combined = compute_metrics([], combined_eq)
    combined["trades"] = sum(m["trades"] for m in per_strat.values())
    combined_bh = compute_metrics([], buy_hold_combined(daily_data, begin_ts))

    print(f"\nTREND-EXPOSURE  ma={ma_period}  buffer={buffer:.0%}  leverage={leverage:g}"
          f"  (hold above MA, cash below; {SLIPPAGE:.2%} slippage per switch)")
    print_vs_benchmark(per_strat, per_bh, combined, combined_bh)
    plot_equity(per_series, combined_eq, RESULTS_PNG, buy_hold_combined(daily_data, begin_ts))
    return 0


def main():
    ap = argparse.ArgumentParser(description="Trend-filtered exposure backtest (beats-B&H attempt).")
    ap.add_argument("--symbols", nargs="+", default=config.PULLBACK_SYMBOLS)
    ap.add_argument("--ma-period", type=int, default=200, help="Long MA length (days).")
    ap.add_argument("--buffer", type=float, default=0.0,
                    help="Re-entry buffer above the MA (e.g. 0.01 = 1%%) to cut whipsaw.")
    ap.add_argument("--leverage", type=float, default=1.0,
                    help="Exposure while invested (1.0 = unlevered; 2.0 = 2x).")
    ap.add_argument("--months", type=int, default=240)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca",
                    help="yahoo = decades of free daily history for full-cycle tests.")
    args = ap.parse_args()

    end_dt = _parse_date(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc))
    start_dt = (_parse_date(args.start) if args.start
                else end_dt - pd.Timedelta(days=int(args.months * 31)))
    log.info("Trend-exposure window: %s -> %s", start_dt.date(), end_dt.date())

    daily_data, _, _ = fetch_all(args.symbols, "none", start_dt, end_dt, source=args.data_source)
    if not daily_data:
        log.error("No data fetched; aborting.")
        return 1
    return run(daily_data, args.ma_period, args.buffer, args.leverage, begin_ts=start_dt)


if __name__ == "__main__":
    raise SystemExit(main())
