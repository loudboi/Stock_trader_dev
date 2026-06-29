"""
bot/momentum_rotation.py
========================
Phase C — cross-sectional (relative-strength) momentum with an absolute filter,
i.e. "dual momentum" (Antonacci). The other research-backed path to beating
buy-and-hold.

Each month:
  - rank the universe by trailing momentum (return over the last `lookback`
    months, optionally skipping the most recent `skip` month — the classic 12-1),
  - hold the top `top_k` equal-weight,
  - BUT only the names whose momentum is positive (the absolute filter). If fewer
    than `top_k` qualify, the rest of the book sits in cash.
So in a broad downturn the whole universe goes negative and you rotate to cash,
which is how momentum rotation dodges bears while still chasing the leaders.

Give it a DIVERSIFIED universe (equities + bonds + gold + international) so there's
usually *something* trending — that's where rotation earns its keep.

Backtested vectorized on monthly rebalances with turnover costs; weights are set
from data through the prior close and applied the next day (no lookahead).

Run from the project root:

    python -m bot.momentum_rotation --symbols SPY QQQ GLD TLT EFA EEM IWM \
        --start 2005-01-01 --data-source yahoo --lookback-months 12 --top-k 2
"""

import argparse
import logging
from datetime import datetime, timezone

import pandas as pd

import config
from bot.backtest_pullback import (compute_metrics, buy_hold_combined, fetch_all,
                                   plot_equity, _parse_date, INITIAL_EQUITY, SLIPPAGE)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("momentum_rotation")

RESULTS_PNG = "momentum_rotation_results.png"
_DAYS_PER_MONTH = 21


# --------------------------------------------------------------------------- #
# Pure pieces (testable offline)
# --------------------------------------------------------------------------- #
def build_panel(daily_data: dict) -> pd.DataFrame:
    """{name: daily df} -> a close-price panel (date × symbol), union-aligned, ffilled."""
    panel = pd.DataFrame({n: d["close"] for n, d in daily_data.items()})
    return panel.sort_index().ffill()


def select_weights(scores: pd.Series, top_k: int) -> pd.Series:
    """Equal-weight the top `top_k` names with POSITIVE momentum; cash for the rest.

    Dividing by top_k (not the number that qualify) means a partly-negative
    universe leaves part of the book in cash — the absolute/dual-momentum filter."""
    w = pd.Series(0.0, index=scores.index)
    pos = scores[scores > 0].sort_values(ascending=False)
    chosen = pos.index[:top_k]
    if len(chosen):
        w[chosen] = 1.0 / top_k
    return w


def momentum(panel: pd.DataFrame, base_i: int, lookback: int, skip: int):
    """Trailing return per symbol as of integer row `base_i`, or None if too early."""
    end = base_i - skip
    start = base_i - skip - lookback
    if start < 0:
        return None
    return panel.iloc[end] / panel.iloc[start] - 1.0


def weight_panel(panel, lookback_months, skip_months, top_k) -> pd.DataFrame:
    """Daily target weights. Rebalanced on the first trading day of each month from
    momentum computed through the PRIOR close (so applying them to the same day's
    return is lookahead-free)."""
    lookback = lookback_months * _DAYS_PER_MONTH
    skip = skip_months * _DAYS_PER_MONTH
    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    cur = pd.Series(0.0, index=panel.columns)
    last_period = None
    for i, ts in enumerate(panel.index):
        period = (ts.year, ts.month)
        if last_period is not None and period != last_period:
            scores = momentum(panel, i - 1, lookback, skip)   # through prior close
            if scores is not None:
                cur = select_weights(scores, top_k)
        weights.iloc[i] = cur.values
        last_period = period
    return weights


def portfolio_returns(panel, weights) -> pd.Series:
    """Daily portfolio return: yesterday's-info weights × today's asset returns,
    minus turnover cost on rebalance days."""
    rets = panel.pct_change().fillna(0.0)
    gross = (weights * rets).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    return gross - turnover * SLIPPAGE


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(daily_data, lookback_months, skip_months, top_k, begin_ts):
    panel = build_panel(daily_data)
    weights = weight_panel(panel, lookback_months, skip_months, top_k)
    port_ret = portfolio_returns(panel, weights)

    r = port_ret[port_ret.index >= begin_ts] if begin_ts is not None else port_ret
    equity = INITIAL_EQUITY * (1 + r).cumprod()
    strat = compute_metrics([], equity)
    # rebalances that actually changed the book
    rebal = int((weights.diff().abs().sum(axis=1) > 1e-9).sum())
    strat["trades"] = rebal

    bh_series = buy_hold_combined(daily_data, begin_ts)
    bh = compute_metrics([], bh_series)

    beat = "YES" if strat["total_return"] > bh["total_return"] else "no"
    print(f"\nMOMENTUM ROTATION  lookback={lookback_months}m  skip={skip_months}m  "
          f"top_k={top_k}  universe={len(daily_data)}  ({SLIPPAGE:.2%} slippage/turnover)")
    print("=" * 70)
    print(f"{'':12}{'Return%':>10}{'MaxDD%':>10}{'Sharpe':>9}{'Rebalances':>12}")
    print("-" * 70)
    print(f"{'Rotation':12}{strat['total_return']*100:>10.1f}{strat['max_drawdown']*100:>10.1f}"
          f"{strat['sharpe']:>9.2f}{strat['trades']:>12}")
    print(f"{'Buy & hold':12}{bh['total_return']*100:>10.1f}{bh['max_drawdown']*100:>10.1f}"
          f"{bh['sharpe']:>9.2f}{'-':>12}")
    print("=" * 70)
    if beat == "YES":
        print(f"Rotation BEAT buy-and-hold on return ({strat['total_return']*100:.1f}% "
              f"vs {bh['total_return']*100:.1f}%), Sharpe {strat['sharpe']:.2f} vs {bh['sharpe']:.2f}.")
    else:
        print(f"Rotation TRAILED buy-and-hold on return ({strat['total_return']*100:.1f}% "
              f"vs {bh['total_return']*100:.1f}%); Sharpe {strat['sharpe']:.2f} vs {bh['sharpe']:.2f}, "
              f"MaxDD {strat['max_drawdown']*100:.1f}% vs {bh['max_drawdown']*100:.1f}%.")

    plot_equity({"rotation": equity}, equity, RESULTS_PNG, bh_series)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Dual-momentum rotation backtest (beats-B&H attempt).")
    ap.add_argument("--symbols", nargs="+",
                    default=["SPY", "QQQ", "GLD", "TLT", "EFA", "EEM", "IWM"],
                    help="Universe to rotate across — diversify it (stocks/bonds/gold/intl).")
    ap.add_argument("--lookback-months", type=int, default=12)
    ap.add_argument("--skip-months", type=int, default=1, help="Skip most-recent month (12-1).")
    ap.add_argument("--top-k", type=int, default=2, help="How many leaders to hold.")
    ap.add_argument("--months", type=int, default=240)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca")
    args = ap.parse_args()

    end_dt = _parse_date(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc))
    start_dt = (_parse_date(args.start) if args.start
                else end_dt - pd.Timedelta(days=int(args.months * 31)))
    log.info("Rotation window: %s -> %s", start_dt.date(), end_dt.date())

    daily_data, _, _ = fetch_all(args.symbols, "none", start_dt, end_dt, source=args.data_source)
    if len(daily_data) < 2:
        log.error("Need at least 2 symbols with data to rotate; got %d.", len(daily_data))
        return 1
    return run(daily_data, args.lookback_months, args.skip_months, args.top_k,
               begin_ts=start_dt)


if __name__ == "__main__":
    raise SystemExit(main())
