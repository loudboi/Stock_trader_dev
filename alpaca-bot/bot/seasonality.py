"""
bot/seasonality.py
==================
Calendar-effect research. The turn-of-month position is held on exactly the last
`days_before` trading days and first `days_after` trading days named by the rule.
Trading calendars are known in advance, so the calendar signal itself does not
need the price-signal one-day delay used by technical strategies.
"""

import argparse
import logging
from datetime import datetime, timezone

import pandas as pd

from bot.momentum_rotation import build_panel
from bot.backtest_pullback import (compute_metrics, buy_hold_combined, fetch_all,
                                   plot_equity, _parse_date, INITIAL_EQUITY, SLIPPAGE)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("seasonality")
RESULTS_PNG = "seasonality_results.png"
DAYS_BEFORE = 1
DAYS_AFTER = 3
_WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def turn_of_month_signal(index: pd.DatetimeIndex, days_before=DAYS_BEFORE,
                         days_after=DAYS_AFTER) -> pd.Series:
    if days_before < 0 or days_after < 0 or days_before + days_after <= 0:
        raise ValueError("turn-of-month window must contain at least one day")
    naive = index.tz_localize(None) if index.tz is not None else index
    month = pd.Series(naive, index=index).dt.to_period("M")
    from_end = month.groupby(month).cumcount(ascending=False)
    from_start = month.groupby(month).cumcount()
    return ((from_end < days_before) | (from_start < days_after)).astype(float)


def turn_of_month_returns(panel, days_before=DAYS_BEFORE, days_after=DAYS_AFTER):
    rets = panel.pct_change().fillna(0.0)
    w = turn_of_month_signal(panel.index, days_before, days_after)
    # Calendar membership is knowable before the session. Do not shift it into a
    # different set of days, which was the old off-by-one implementation.
    turnover = w.diff().abs().fillna(w.abs()) * SLIPPAGE
    return w * rets.mean(axis=1) - turnover


def weekday_stats(panel, begin_ts=None):
    rets = panel.pct_change().fillna(0.0).mean(axis=1)
    if begin_ts is not None:
        rets = rets[rets.index >= begin_ts]
    return {name: rets[rets.index.dayofweek == i] for i, name in enumerate(_WEEKDAY_NAMES)}


def print_weekday_stats(daily_data, begin_ts):
    by_day = weekday_stats(build_panel(daily_data), begin_ts)
    overall = pd.concat(by_day.values())
    print("\nDAY-OF-WEEK EFFECT (descriptive; not a trading recommendation)")
    for name, r in by_day.items():
        print(f"{name}: N={len(r)} mean={r.mean():+.4%} annualized-arithmetic={r.mean()*252:+.1%}")
    print(f"All: N={len(overall)} mean={overall.mean():+.4%}")


def run(daily_data, begin_ts, days_before=DAYS_BEFORE, days_after=DAYS_AFTER):
    panel = build_panel(daily_data)
    tom = turn_of_month_returns(panel, days_before, days_after)
    r = tom[tom.index >= begin_ts]
    eq = INITIAL_EQUITY * (1 + r).cumprod(); eq.attrs["initial_equity"] = INITIAL_EQUITY
    bh = buy_hold_combined(daily_data, begin_ts)
    m_tom, m_bh = compute_metrics([], eq), compute_metrics([], bh)
    pct = float(turn_of_month_signal(panel.index, days_before, days_after).mean())
    print(f"\nTURN-OF-MONTH: last {days_before} + first {days_after} trading days ({pct:.0%} invested)")
    print(f"TOM return={m_tom['total_return']:.1%} MaxDD={m_tom['max_drawdown']:.1%} Sharpe={m_tom['sharpe']:.2f}")
    print(f"B&H return={m_bh['total_return']:.1%} MaxDD={m_bh['max_drawdown']:.1%} Sharpe={m_bh['sharpe']:.2f}")
    plot_equity({"turn_of_month": eq}, eq, RESULTS_PNG, bh)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Calendar seasonality historical research.")
    ap.add_argument("--mode", choices=["tom", "weekday"], default="tom")
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "GLD", "TLT"])
    ap.add_argument("--days-before", type=int, default=DAYS_BEFORE)
    ap.add_argument("--days-after", type=int, default=DAYS_AFTER)
    ap.add_argument("--months", type=int, default=240)
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca")
    args = ap.parse_args()
    if args.months <= 0 or args.days_before < 0 or args.days_after < 0 or args.days_before + args.days_after <= 0:
        ap.error("invalid months/turn-of-month window")
    end_dt = _parse_date(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc))
    start_dt = _parse_date(args.start) if args.start else end_dt - pd.Timedelta(days=args.months * 31)
    if start_dt >= end_dt:
        ap.error("start must precede end")
    daily_data, _, _ = fetch_all(args.symbols, "none", start_dt, end_dt, source=args.data_source)
    if len(daily_data) != len(args.symbols):
        log.error("Missing data for requested symbols; refusing silently changed universe."); return 1
    if args.mode == "weekday":
        print_weekday_stats(daily_data, start_dt); return 0
    return run(daily_data, start_dt, args.days_before, args.days_after)


if __name__ == "__main__":
    raise SystemExit(main())
