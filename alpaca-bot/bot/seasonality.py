"""
bot/seasonality.py
==================
Tests the classic "turn-of-month" (TOM) effect (Ariel 1987, Lakonishok & Smidt
1988): equity returns are disproportionately concentrated around month
boundaries, attributed to systematic flows (payroll/401k contributions, pension
rebalancing). A PURE CALENDAR signal — no price data drives the decision, unlike
everything else in this project.

Fixed, literature-standard window (chosen before any backtest): invested on the
LAST 1 trading day of a month through the FIRST 3 trading days of the next month;
cash otherwise. Turnover-costed like every strategy in bot/lab.py (entry/exit
only on a signal change, NOT a daily round-trip like bot/overnight.py).

Also includes a day-of-week ("Monday effect") descriptive check (--mode weekday)
-- a direct statistical comparison of mean returns by weekday, not a backtested
strategy (holding only specific weekdays would face the same daily-round-trip
cost problem documented in bot/overnight.py).

    python -m bot.seasonality --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
    python -m bot.seasonality --mode weekday --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
"""

import argparse
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from bot.momentum_rotation import build_panel
from bot.backtest_pullback import (compute_metrics, buy_hold_combined, fetch_all,
                                   plot_equity, _parse_date, INITIAL_EQUITY, SLIPPAGE)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("seasonality")

RESULTS_PNG = "seasonality_results.png"
DAYS_BEFORE = 1     # last N trading days of the month
DAYS_AFTER = 3       # first N trading days of the next month


def turn_of_month_signal(index: pd.DatetimeIndex, days_before=DAYS_BEFORE,
                         days_after=DAYS_AFTER) -> pd.Series:
    """1.0 on turn-of-month days (last `days_before` of a month, first `days_after`
    of the next), 0.0 otherwise. Pure calendar function of the index."""
    naive = index.tz_localize(None) if index.tz is not None else index
    month = pd.Series(naive, index=index).dt.to_period("M")
    from_end = month.groupby(month).cumcount(ascending=False)
    from_start = month.groupby(month).cumcount()
    signal = ((from_end < days_before) | (from_start < days_after)).astype(float)
    return pd.Series(signal.values, index=index)


def turn_of_month_returns(panel: pd.DataFrame, days_before=DAYS_BEFORE,
                          days_after=DAYS_AFTER) -> pd.Series:
    """Equal-weight book, invested only on turn-of-month days (signal decided
    from the calendar alone, applied the NEXT day like every other strategy in
    this project — the position for day t is set from t-1's calendar signal, so
    there's no lookahead)."""
    rets = panel.pct_change().fillna(0.0)
    signal = turn_of_month_signal(panel.index, days_before, days_after)
    w = signal.shift(1).fillna(0.0)
    turnover = w.diff().abs().fillna(0.0) * SLIPPAGE
    return w * rets.mean(axis=1) - turnover


_WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def weekday_stats(panel: pd.DataFrame, begin_ts=None) -> dict:
    """Mean equal-weight daily return grouped by weekday (the classic 'Monday
    effect' / weekend-effect check) — a direct statistical comparison, not a
    tradeable strategy (see bot/overnight.py for why daily-frequency round-trips
    are brutal on costs; this is the same caution that applies to 'only trade
    Mondays')."""
    rets = panel.pct_change().fillna(0.0).mean(axis=1)
    if begin_ts is not None:
        rets = rets[rets.index >= begin_ts]
    return {name: rets[rets.index.dayofweek == i] for i, name in enumerate(_WEEKDAY_NAMES)}


def print_weekday_stats(daily_data, begin_ts):
    panel = build_panel(daily_data)
    by_day = weekday_stats(panel, begin_ts)
    overall = pd.concat(by_day.values())

    print(f"\nDAY-OF-WEEK EFFECT  (mean daily return by weekday, direct statistical check)")
    print("=" * 56)
    print(f"{'Day':6}{'N':>8}{'Mean %':>10}{'Annualized%':>13}")
    print("-" * 56)
    for name, r in by_day.items():
        print(f"{name:6}{len(r):>8}{r.mean()*100:>10.4f}{r.mean()*252*100:>13.1f}")
    print("-" * 56)
    print(f"{'All':6}{len(overall):>8}{overall.mean()*100:>10.4f}{overall.mean()*252*100:>13.1f}")
    print("=" * 56)
    best = max(by_day, key=lambda d: by_day[d].mean())
    worst = min(by_day, key=lambda d: by_day[d].mean())
    print(f"Best day: {best} ({by_day[best].mean()*100:.4f}%/day). "
          f"Worst day: {worst} ({by_day[worst].mean()*100:.4f}%/day).")
    print("This is a descriptive comparison, not a backtest of a tradeable strategy "
          "-- see the README/turn-of-month findings on how small, noisy differences "
          "in daily means rarely survive real transaction costs or significance testing.")


def run(daily_data, begin_ts, days_before=DAYS_BEFORE, days_after=DAYS_AFTER):
    panel = build_panel(daily_data)
    tom = turn_of_month_returns(panel, days_before, days_after)
    bh = buy_hold_combined(daily_data, begin_ts)

    eq_tom = INITIAL_EQUITY * (1 + tom[tom.index >= begin_ts]).cumprod()
    m_tom = compute_metrics([], eq_tom)
    m_bh = compute_metrics([], bh)
    pct_invested = float(turn_of_month_signal(panel.index, days_before, days_after).mean())

    print(f"\nTURN-OF-MONTH  invested last {days_before}d of month + first {days_after}d of "
          f"next ({pct_invested:.0%} of trading days)")
    print("=" * 62)
    print(f"{'Book':22}{'Return%':>10}{'MaxDD%':>10}{'Sharpe':>9}")
    print("-" * 62)
    print(f"{'buy_and_hold':22}{m_bh['total_return']*100:>10.1f}{m_bh['max_drawdown']*100:>10.1f}"
          f"{m_bh['sharpe']:>9.2f}")
    print(f"{'turn_of_month':22}{m_tom['total_return']*100:>10.1f}{m_tom['max_drawdown']*100:>10.1f}"
          f"{m_tom['sharpe']:>9.2f}")
    print("=" * 62)
    verdict = "BEATS" if m_tom["sharpe"] > m_bh["sharpe"] else "trails"
    print(f"turn_of_month {verdict} buy-and-hold on Sharpe ({m_tom['sharpe']:.2f} vs "
          f"{m_bh['sharpe']:.2f}) while invested only {pct_invested:.0%} of the time.")
    plot_equity({"turn_of_month": eq_tom}, eq_tom, RESULTS_PNG, bh)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Calendar seasonality effects.")
    ap.add_argument("--mode", choices=["tom", "weekday"], default="tom",
                    help="tom = turn-of-month strategy backtest (default); "
                         "weekday = day-of-week descriptive statistics.")
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "GLD", "TLT"])
    ap.add_argument("--days-before", type=int, default=DAYS_BEFORE)
    ap.add_argument("--days-after", type=int, default=DAYS_AFTER)
    ap.add_argument("--months", type=int, default=240)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca")
    args = ap.parse_args()

    end_dt = _parse_date(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc))
    start_dt = (_parse_date(args.start) if args.start
                else end_dt - pd.Timedelta(days=int(args.months * 31)))
    log.info("Seasonality window: %s -> %s (%s mode)", start_dt.date(), end_dt.date(), args.mode)

    daily_data, _, _ = fetch_all(args.symbols, "none", start_dt, end_dt, source=args.data_source)
    if not daily_data:
        log.error("No data fetched.")
        return 1
    if args.mode == "weekday":
        print_weekday_stats(daily_data, begin_ts=start_dt)
        return 0
    return run(daily_data, begin_ts=start_dt, days_before=args.days_before,
              days_after=args.days_after)


if __name__ == "__main__":
    raise SystemExit(main())
