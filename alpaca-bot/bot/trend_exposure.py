"""
bot/trend_exposure.py
=====================
Long/cash trend-exposure research model.

A close above a long moving average (optionally with a re-entry buffer) sets the
target exposure for the NEXT bar. Results are historical simulations, not evidence
that future buy-and-hold returns will be beaten.

Multi-symbol results use a fixed equal capital split. Missing returns for one sleeve
mean that sleeve is idle for that date; its capital is not silently reassigned to
other symbols.
"""

import argparse
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
from bot.backtest_pullback import (INITIAL_EQUITY, SLIPPAGE, _parse_date,
                                   buy_hold_combined, buy_hold_equity,
                                   compute_metrics, fetch_all, plot_equity,
                                   print_vs_benchmark)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("trend_exposure")
RESULTS_PNG = "trend_exposure_results.png"


def exposure_series(close: pd.Series, ma_period: int, buffer: float = 0.0) -> pd.Series:
    if ma_period <= 1 or buffer < 0:
        raise ValueError("ma_period must be >1 and buffer non-negative")
    ma = close.rolling(ma_period, min_periods=ma_period).mean()
    upper = ma * (1 + buffer)
    state, out = 0.0, []
    for c, u, m in zip(close.values, upper.values, ma.values):
        if pd.isna(m):
            out.append(0.0)
            continue
        if state == 0.0 and c > u:
            state = 1.0
        elif state == 1.0 and c < m:
            state = 0.0
        out.append(state)
    return pd.Series(out, index=close.index, dtype=float)


def strategy_returns(daily: pd.DataFrame, ma_period: int, buffer: float,
                     leverage: float, borrow_rate: float = 0.0,
                     periods_per_year: int = 252) -> pd.Series:
    if leverage < 0 or borrow_rate < 0 or periods_per_year <= 0:
        raise ValueError("invalid leverage/borrow_rate/periods_per_year")
    close = daily["close"].astype(float)
    target = exposure_series(close, ma_period, buffer)
    held = target.shift(1).fillna(0.0)
    asset_ret = close.pct_change(fill_method=None).fillna(0.0)
    gross = held * leverage * asset_ret
    flips = held.diff().abs().fillna(held.abs())
    cost = flips * leverage * SLIPPAGE
    financing = held * max(leverage - 1.0, 0.0) * (borrow_rate / periods_per_year)
    return gross - cost - financing


def equity_from_returns(rets: pd.Series, begin_ts, initial=INITIAL_EQUITY) -> pd.Series:
    r = rets[rets.index >= begin_ts] if begin_ts is not None else rets
    if not len(r):
        return pd.Series(dtype=float)
    out = initial * (1 + r).cumprod()
    out.attrs["initial_equity"] = initial
    return out


def _trade_count(daily, ma_period, buffer) -> int:
    held = exposure_series(daily["close"], ma_period, buffer).shift(1).fillna(0.0)
    return int(((held > 0) & (held.shift(1).fillna(0.0) == 0)).sum())


def _fixed_capital_mean(series_by_name: dict) -> pd.Series:
    if not series_by_name:
        return pd.Series(dtype=float)
    aligned = pd.concat(series_by_name, axis=1).fillna(0.0)
    return aligned.sum(axis=1) / len(series_by_name)


def run(daily_data, ma_period, buffer, leverage, begin_ts, borrow_rate=0.0):
    if not daily_data:
        raise ValueError("daily_data cannot be empty")
    per_strat, per_bh, per_series, ret_frame = {}, {}, {}, {}
    for name, daily in daily_data.items():
        rets = strategy_returns(daily, ma_period, buffer, leverage, borrow_rate)
        eq = equity_from_returns(rets, begin_ts)
        m = compute_metrics([], eq)
        m["trades"] = _trade_count(daily, ma_period, buffer)
        per_strat[name] = m
        per_series[name] = eq
        per_bh[name] = compute_metrics([], buy_hold_equity(daily, begin_ts))
        ret_frame[name] = rets

    combined_ret = _fixed_capital_mean(ret_frame)
    combined_eq = equity_from_returns(combined_ret, begin_ts)
    combined = compute_metrics([], combined_eq)
    combined["trades"] = sum(m["trades"] for m in per_strat.values())
    bh_series = buy_hold_combined(daily_data, begin_ts)
    combined_bh = compute_metrics([], bh_series)

    print(f"\nTREND-EXPOSURE HISTORICAL BACKTEST ma={ma_period} buffer={buffer:.1%} "
          f"leverage={leverage:g} borrow={borrow_rate:.1%}/yr")
    print_vs_benchmark(per_strat, per_bh, combined, combined_bh)
    plot_equity(per_series, combined_eq, RESULTS_PNG, bh_series)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Long/cash trend-exposure historical backtest.")
    ap.add_argument("--symbols", nargs="+", default=config.PULLBACK_SYMBOLS)
    ap.add_argument("--ma-period", type=int, default=200)
    ap.add_argument("--buffer", type=float, default=0.0)
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--borrow-rate", type=float, default=0.06)
    ap.add_argument("--months", type=int, default=240)
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca")
    args = ap.parse_args()
    if (args.ma_period <= 1 or args.buffer < 0 or args.leverage < 0 or
            args.borrow_rate < 0 or args.months <= 0):
        ap.error("invalid MA/buffer/leverage/borrow/month values")
    end_dt = _parse_date(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc))
    start_dt = (_parse_date(args.start) if args.start
                else end_dt - pd.Timedelta(days=args.months * 31))
    if start_dt >= end_dt:
        ap.error("start must precede end")
    daily_data, _, _ = fetch_all(args.symbols, "none", start_dt, end_dt,
                                 source=args.data_source)
    if len(daily_data) != len(args.symbols):
        log.error("Missing requested symbol data; refusing a silently changed universe.")
        return 1
    return run(daily_data, args.ma_period, args.buffer, args.leverage,
               begin_ts=start_dt, borrow_rate=args.borrow_rate)


if __name__ == "__main__":
    raise SystemExit(main())
