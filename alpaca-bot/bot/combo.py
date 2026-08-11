"""
bot/combo.py
============
Research harness for blending a risk-based portfolio with a long/cash trend-
exposure sleeve. Historical results are evidence about a particular sample, not
proof or validation of a future edge.

The blend uses fixed capital allocations. If one symbol has no return on a date,
its capital stays idle instead of being implicitly reassigned to symbols that did
trade. ``--mode consistency`` (legacy alias ``walk``) reports chronological,
non-overlapping folds with fixed parameters; it is not fit/test walk-forward
optimization.
"""

import argparse
import logging
from datetime import datetime, timezone

import pandas as pd

import bot.trend_exposure as te
from bot.data import clean_daily_data
from bot.lab import (build_panel, combine_books, compute_metrics, erc, fold_bounds,
                     inverse_vol, min_var, slice_equity)
from bot.backtest_pullback import fetch_all, buy_hold_combined, _parse_date

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("combo")

RP_LOOKBACK = 20
COV_LOOKBACK = 60
TE_MA_PERIOD = 200
TE_BUFFER = 0.01
_ANNUAL = 252
_RP_STRATEGIES = {"inverse_vol": inverse_vol, "min_var": min_var, "erc": erc}


def leverage_returns(returns: pd.Series, leverage: float,
                     borrow_rate: float = 0.06) -> pd.Series:
    if leverage < 0:
        raise ValueError("leverage must be non-negative")
    if borrow_rate < 0:
        raise ValueError("borrow_rate must be non-negative")
    if leverage <= 1.0:
        return returns * leverage
    financing = (leverage - 1.0) * (borrow_rate / _ANNUAL)
    return returns * leverage - financing


def _fixed_capital_mean(series_by_name: dict) -> pd.Series:
    if not series_by_name:
        return pd.Series(dtype=float)
    aligned = pd.concat(series_by_name, axis=1).fillna(0.0)
    return aligned.sum(axis=1) / len(series_by_name)


def compute_books(daily_data: dict, rp_strategy="min_var", rp_lookback=RP_LOOKBACK,
                  cov_lookback=COV_LOOKBACK, te_ma_period=TE_MA_PERIOD,
                  te_buffer=TE_BUFFER) -> dict:
    if len(daily_data) < 2:
        raise ValueError("combo requires at least two symbols")
    if rp_strategy not in _RP_STRATEGIES:
        raise ValueError(f"unknown risk-based strategy {rp_strategy}")
    if rp_lookback <= 1 or cov_lookback <= 1 or te_ma_period <= 1:
        raise ValueError("lookback/MA periods must be > 1")
    if te_buffer < 0:
        raise ValueError("te_buffer cannot be negative")

    panel = build_panel(daily_data)
    if rp_strategy == "min_var":
        rp = min_var(panel, cov_lookback=cov_lookback)
    elif rp_strategy == "erc":
        rp = erc(panel, cov_lookback=cov_lookback)
    else:
        rp = inverse_vol(panel, lookback=rp_lookback)

    per_symbol = {
        name: te.strategy_returns(daily, ma_period=te_ma_period, buffer=te_buffer,
                                  leverage=1.0, borrow_rate=0.0)
        for name, daily in daily_data.items()
    }
    trend = _fixed_capital_mean(per_symbol)
    return {"rp": rp, "te": trend}


def print_score(daily_data, books, rp_weight, begin_ts, end_ts, leverage=1.0,
                borrow_rate=0.06, rp_label="min_var"):
    if not 0 <= rp_weight <= 1:
        raise ValueError("rp_weight must be in [0, 1]")
    blend = combine_books(books, weights={"rp": rp_weight, "te": 1.0 - rp_weight})
    levered = leverage_returns(blend, leverage, borrow_rate)
    bh = buy_hold_combined(daily_data, begin_ts).pct_change().fillna(0.0)

    def m(r):
        return compute_metrics([], slice_equity(r, begin_ts, end_ts))

    rows = [("buy_and_hold", m(bh)), (f"pure {rp_label}", m(books["rp"])),
            ("pure trend_exposure", m(books["te"])),
            (f"blend ({rp_weight:.0%} {rp_label} / {1-rp_weight:.0%} TE)", m(blend))]
    if leverage != 1.0:
        rows.append((f"blend @ {leverage:g}x", m(levered)))

    print(f"\n{rp_label.upper()} + TREND-EXPOSURE — historical comparison")
    print(f"{'Book':34}{'Return%':>10}{'MaxDD%':>10}{'Sharpe':>9}")
    for name, mm in rows:
        print(f"{name:34}{mm['total_return']*100:>10.1f}{mm['max_drawdown']*100:>10.1f}"
              f"{mm['sharpe']:>9.2f}")
    return rows


def print_walk(daily_data, books, rp_weight, start_dt, end_dt, folds,
               leverage=1.0, borrow_rate=0.06, rp_label="min_var"):
    """Legacy function name; performs chronological consistency folds only."""
    if folds <= 0:
        raise ValueError("folds must be positive")
    blend = combine_books(books, weights={"rp": rp_weight, "te": 1.0 - rp_weight})
    active = leverage_returns(blend, leverage, borrow_rate) if leverage != 1.0 else blend
    bh = buy_hold_combined(daily_data, start_dt).pct_change().fillna(0.0)
    bounds = fold_bounds(start_dt, end_dt, folds)

    print(f"\nCHRONOLOGICAL CONSISTENCY — legacy --mode walk alias; no fitting ({folds} folds)")
    wins = {"rp": 0, "te": 0, "bh": 0}
    for k, (fs, fe) in enumerate(bounds):
        final = k == len(bounds) - 1
        def m(r):
            return compute_metrics([], slice_equity(r, fs, fe, end_inclusive=final))
        m_bh, m_rp, m_te, m_bl = m(bh), m(books["rp"]), m(books["te"]), m(active)
        wins["rp"] += int(m_bl["sharpe"] > m_rp["sharpe"])
        wins["te"] += int(m_bl["sharpe"] > m_te["sharpe"])
        wins["bh"] += int(m_bl["sharpe"] > m_bh["sharpe"])
        print(f"{k+1}: {fs.date()}->{fe.date()} B&H={m_bh['sharpe']:.2f} "
              f"{rp_label}={m_rp['sharpe']:.2f} TE={m_te['sharpe']:.2f} "
              f"blend={m_bl['sharpe']:.2f}")
    print("SUMMARY: "
          f"blend beat {rp_label} {wins['rp']}/{folds}, TE {wins['te']}/{folds}, "
          f"B&H {wins['bh']}/{folds} folds.")
    return wins


def main():
    ap = argparse.ArgumentParser(description="Blend risk-based + trend-exposure research books.")
    ap.add_argument("--mode", choices=["score", "consistency", "walk"], default="score",
                    help="consistency (legacy alias walk) = fixed-parameter chronological folds")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rp-strategy", choices=list(_RP_STRATEGIES), default="min_var")
    ap.add_argument("--rp-weight", type=float, default=0.5)
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--borrow-rate", type=float, default=0.06)
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "GLD", "TLT"])
    ap.add_argument("--months", type=int, default=240)
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca")
    ap.add_argument("--clean-outliers", action="store_true")
    ap.add_argument("--max-abs-return", type=float, default=0.15)
    args = ap.parse_args()

    if (args.folds <= 0 or args.months <= 0 or not 0 <= args.rp_weight <= 1 or
            args.leverage < 0 or args.borrow_rate < 0 or args.max_abs_return <= 0):
        ap.error("invalid folds/months/weights/leverage/rates/outlier threshold")
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
    if args.clean_outliers:
        daily_data = clean_daily_data(daily_data, args.max_abs_return)

    books = compute_books(daily_data, rp_strategy=args.rp_strategy)
    if args.mode in {"consistency", "walk"}:
        print_walk(daily_data, books, args.rp_weight, start_dt, end_dt, args.folds,
                   args.leverage, args.borrow_rate, rp_label=args.rp_strategy)
    else:
        print_score(daily_data, books, args.rp_weight, start_dt, end_dt,
                    args.leverage, args.borrow_rate, rp_label=args.rp_strategy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
