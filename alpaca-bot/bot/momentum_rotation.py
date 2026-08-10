"""
bot/momentum_rotation.py
========================
Cross-sectional momentum with an absolute (positive-momentum) filter.

A multi-asset panel starts only once every requested asset has a price. Thereafter
prices are forward-filled across market-specific holidays so an unavailable market
contributes a 0 return rather than causing its capital to be reallocated implicitly.
"""

import argparse
import logging
from datetime import datetime, timezone

import pandas as pd

from bot.backtest_pullback import (compute_metrics, buy_hold_combined, fetch_all,
                                   plot_equity, _parse_date, INITIAL_EQUITY, SLIPPAGE)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("momentum_rotation")
RESULTS_PNG = "momentum_rotation_results.png"
_DAYS_PER_MONTH = 21


def build_panel(daily_data: dict) -> pd.DataFrame:
    if not daily_data:
        return pd.DataFrame()
    panel = pd.DataFrame({n: d["close"] for n, d in daily_data.items()}).sort_index()
    first = []
    for col in panel:
        valid = panel[col].first_valid_index()
        if valid is None:
            raise ValueError(f"No price observations for {col}")
        first.append(valid)
    panel = panel.loc[max(first):].ffill()
    if panel.isna().any().any():
        raise ValueError("Price panel still contains missing values after common inception")
    return panel


def select_weights(scores: pd.Series, top_k: int) -> pd.Series:
    if top_k <= 0 or top_k > len(scores):
        raise ValueError("top_k must be between 1 and the universe size")
    w = pd.Series(0.0, index=scores.index)
    pos = scores[scores > 0].dropna().sort_values(ascending=False)
    chosen = pos.index[:top_k]
    if len(chosen):
        w[chosen] = 1.0 / top_k
    return w


def momentum(panel: pd.DataFrame, base_i: int, lookback: int, skip: int):
    if lookback <= 0 or skip < 0:
        raise ValueError("lookback must be positive and skip non-negative")
    end = base_i - skip
    start = end - lookback
    if start < 0 or end < 0:
        return None
    return panel.iloc[end] / panel.iloc[start] - 1.0


def weight_panel(panel, lookback_months, skip_months, top_k):
    if panel.empty:
        return pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
    if lookback_months <= 0 or skip_months < 0:
        raise ValueError("lookback_months must be positive and skip_months non-negative")
    if top_k <= 0 or top_k > len(panel.columns):
        raise ValueError("top_k must be between 1 and the universe size")
    lookback, skip = lookback_months * _DAYS_PER_MONTH, skip_months * _DAYS_PER_MONTH
    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    cur = pd.Series(0.0, index=panel.columns); last = None
    for i, ts in enumerate(panel.index):
        period = (ts.year, ts.month)
        if last is not None and period != last:
            scores = momentum(panel, i - 1, lookback, skip)
            if scores is not None:
                cur = select_weights(scores, top_k)
        weights.iloc[i] = cur.values; last = period
    return weights


def portfolio_returns(panel, weights):
    if not panel.index.equals(weights.index) or list(panel.columns) != list(weights.columns):
        raise ValueError("weights must align exactly with the price panel")
    rets = panel.pct_change().fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    return (weights * rets).sum(axis=1) - turnover * SLIPPAGE


def run(daily_data, lookback_months, skip_months, top_k, begin_ts):
    panel = build_panel(daily_data)
    weights = weight_panel(panel, lookback_months, skip_months, top_k)
    r = portfolio_returns(panel, weights)
    r = r[r.index >= max(begin_ts, panel.index[0])] if begin_ts is not None else r
    equity = INITIAL_EQUITY * (1 + r).cumprod(); equity.attrs["initial_equity"] = INITIAL_EQUITY
    strat = compute_metrics([], equity)
    strat["trades"] = int((weights.diff().abs().sum(axis=1) > 1e-9).sum())
    bh_series = buy_hold_combined(daily_data, begin_ts)
    bh = compute_metrics([], bh_series)
    print(f"\nMOMENTUM ROTATION  lookback={lookback_months}m skip={skip_months}m top_k={top_k}")
    print(f"strategy: return {strat['total_return']:.1%}, maxDD {strat['max_drawdown']:.1%}, Sharpe {strat['sharpe']:.2f}")
    print(f"B&H:      return {bh['total_return']:.1%}, maxDD {bh['max_drawdown']:.1%}, Sharpe {bh['sharpe']:.2f}")
    plot_equity({"rotation": equity}, equity, RESULTS_PNG, bh_series)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Dual-momentum rotation historical simulation.")
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "GLD", "TLT", "EFA", "EEM", "IWM"])
    ap.add_argument("--lookback-months", type=int, default=12)
    ap.add_argument("--skip-months", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--months", type=int, default=240)
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca")
    args = ap.parse_args()
    if args.months <= 0 or args.lookback_months <= 0 or args.skip_months < 0 or args.top_k <= 0:
        ap.error("months/lookback/top-k must be positive; skip-months must be non-negative")
    if args.top_k > len(args.symbols):
        ap.error("--top-k cannot exceed the number of symbols")
    end_dt = _parse_date(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc))
    start_dt = _parse_date(args.start) if args.start else end_dt - pd.Timedelta(days=args.months * 31)
    if start_dt >= end_dt:
        ap.error("start must precede end")
    daily_data, _, _ = fetch_all(args.symbols, "none", start_dt, end_dt, source=args.data_source)
    if len(daily_data) != len(args.symbols):
        log.error("Expected data for %d symbols, received %d; refusing a silently changed universe.",
                  len(args.symbols), len(daily_data)); return 1
    return run(daily_data, args.lookback_months, args.skip_months, args.top_k, start_dt)


if __name__ == "__main__":
    raise SystemExit(main())
