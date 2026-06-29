"""
bot/lab.py
==========
A "strategy lab" — several research-backed approaches to beating buy-and-hold,
all in one comparable harness so you can rank them head-to-head against B&H over a
full cycle (use --data-source yahoo). Each strategy is a pure function that returns
a daily portfolio-return series; the runner builds equity, computes metrics, and
prints one scoreboard.

Strategies:
  vol_target      Scale an equal-weight book to a constant target volatility
                  (lever calm markets, cut risk in turbulent ones).
  inverse_vol     Risk parity: weight assets by 1/volatility (always invested).
  managed_futures Diversified time-series trend (hold above MA), inverse-vol
                  weighted — "crisis alpha".
  mean_reversion  Buy short-term-oversold (RSI) dips while in an uptrend.
  trend_vol       Vol-targeted trend: hold above MA, then target portfolio vol.
  ensemble        Equal blend of the daily returns of the above (diversify across
                  strategies — the one real free lunch).

Costs modelled: 0.05% slippage on turnover, and a borrow rate on leverage >1×.
All signals act on the NEXT day (shift by one), so there is no lookahead.

    python -m bot.lab --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
    python -m bot.lab --strategies trend_vol ensemble --target-vol 0.12
"""

import argparse
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
from bot.momentum_rotation import build_panel
from bot.backtest_pullback import (compute_metrics, buy_hold_combined, fetch_all,
                                   plot_equity, _parse_date, INITIAL_EQUITY, SLIPPAGE)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("lab")

RESULTS_PNG = "lab_results.png"
_ANNUAL = 252


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def daily_returns(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pct_change().fillna(0.0)


def realized_vol(rets, lookback: int) -> pd.Series:
    """Annualized trailing volatility (works on a Series or per-column DataFrame)."""
    return rets.rolling(lookback).std() * np.sqrt(_ANNUAL)


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _turn_cost(weights: pd.DataFrame) -> pd.Series:
    return weights.diff().abs().sum(axis=1).fillna(0.0) * SLIPPAGE


# --------------------------------------------------------------------------- #
# Strategies — each returns a daily portfolio-return Series over the full panel
# --------------------------------------------------------------------------- #
def vol_target(panel, target_vol=0.15, lookback=20, max_leverage=2.0,
               borrow_rate=0.06) -> pd.Series:
    rets = daily_returns(panel)
    base = rets.mean(axis=1)                                  # equal-weight book
    scale = (target_vol / realized_vol(base, lookback)).clip(upper=max_leverage)
    scale = scale.shift(1).fillna(0.0)                        # no lookahead
    financing = (scale - 1.0).clip(lower=0) * (borrow_rate / _ANNUAL)
    cost = scale.diff().abs().fillna(0.0) * SLIPPAGE
    return scale * base - financing - cost


def inverse_vol(panel, lookback=20) -> pd.Series:
    rets = daily_returns(panel)
    inv = (1.0 / realized_vol(rets, lookback)).replace([np.inf, -np.inf], np.nan)
    w = inv.div(inv.sum(axis=1), axis=0).fillna(0.0)         # risk-parity weights
    w = w.shift(1).fillna(0.0)
    return (w * rets).sum(axis=1) - _turn_cost(w)


def managed_futures(panel, ma_period=200, vol_lookback=20) -> pd.Series:
    rets = daily_returns(panel)
    ma = panel.rolling(ma_period, min_periods=ma_period).mean()
    trend = (panel > ma).astype(float)                       # long/flat per asset
    raw = (trend / realized_vol(rets, vol_lookback)).replace([np.inf, -np.inf], np.nan)
    raw = raw.fillna(0.0)
    w = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)  # cash if none trend
    w = w.shift(1).fillna(0.0)
    return (w * rets).sum(axis=1) - _turn_cost(w)


def mean_reversion(panel, ma_trend=200, rsi_period=2, buy=10.0, exit_=70.0) -> pd.Series:
    rets = daily_returns(panel)
    ma = panel.rolling(ma_trend, min_periods=ma_trend).mean()
    uptrend = panel > ma
    pos = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    for col in panel.columns:
        r = rsi(panel[col], rsi_period)
        state, out = 0.0, []
        for up, rv in zip(uptrend[col].values, r.values):
            if rv != rv:                                     # RSI warmup
                out.append(0.0)
                continue
            if state == 0.0 and up and rv < buy:
                state = 1.0
            elif state == 1.0 and (rv > exit_ or not up):
                state = 0.0
            out.append(state)
        pos[col] = out
    w = pos.div(pos.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)  # equal-weight active
    w = w.shift(1).fillna(0.0)
    return (w * rets).sum(axis=1) - _turn_cost(w)


def trend_vol(panel, ma_period=200, target_vol=0.15, vol_lookback=20,
              max_leverage=2.0, borrow_rate=0.06) -> pd.Series:
    rets = daily_returns(panel)
    ma = panel.rolling(ma_period, min_periods=ma_period).mean()
    trend = (panel > ma).astype(float)
    w = trend.div(trend.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    w = w.shift(1).fillna(0.0)
    base = (w * rets).sum(axis=1)
    scale = (target_vol / realized_vol(base, vol_lookback)).clip(upper=max_leverage)
    scale = scale.shift(1).fillna(0.0)
    financing = (scale - 1.0).clip(lower=0) * (borrow_rate / _ANNUAL)
    return scale * base - financing - _turn_cost(w) - scale.diff().abs().fillna(0.0) * SLIPPAGE


def ensemble(components: dict) -> pd.Series:
    """Equal-weight blend of several strategies' daily return series."""
    return pd.concat(components.values(), axis=1).mean(axis=1)


_STRATEGIES = {
    "vol_target": vol_target,
    "inverse_vol": inverse_vol,
    "managed_futures": managed_futures,
    "mean_reversion": mean_reversion,
    "trend_vol": trend_vol,
}


# --------------------------------------------------------------------------- #
# Runner / reporting
# --------------------------------------------------------------------------- #
def run(daily_data, names, begin_ts, params):
    panel = build_panel(daily_data)
    comp_returns = {}
    for name in names:
        fn = _STRATEGIES[name]
        kw = {k: v for k, v in params.items()
              if k in fn.__code__.co_varnames}
        comp_returns[name] = fn(panel, **kw)
    if "ensemble" in (names if isinstance(names, list) else []) or params.get("with_ensemble"):
        comp_returns["ensemble"] = ensemble(
            {k: v for k, v in comp_returns.items() if k in _STRATEGIES})

    def metrics_for(rets):
        r = rets[rets.index >= begin_ts] if begin_ts is not None else rets
        eq = INITIAL_EQUITY * (1 + r).cumprod()
        return eq, compute_metrics([], eq)

    results, series = {}, {}
    for name, rets in comp_returns.items():
        eq, m = metrics_for(rets)
        results[name] = m
        series[name] = eq

    bh_series = buy_hold_combined(daily_data, begin_ts)
    bh = compute_metrics([], bh_series)

    # Scoreboard, ranked by Sharpe.
    order = sorted(results, key=lambda n: results[n]["sharpe"], reverse=True)
    print(f"\nSTRATEGY LAB  universe={len(daily_data)}  vs equal-weight buy-and-hold")
    print("=" * 74)
    print(f"{'Strategy':16}{'Return%':>11}{'MaxDD%':>10}{'Sharpe':>9}{'Beat B&H (Shp)':>18}")
    print("-" * 74)
    print(f"{'buy_and_hold':16}{bh['total_return']*100:>11.1f}{bh['max_drawdown']*100:>10.1f}"
          f"{bh['sharpe']:>9.2f}{'(benchmark)':>18}")
    print("-" * 74)
    for name in order:
        m = results[name]
        verdict = "YES" if m["sharpe"] > bh["sharpe"] else "no"
        print(f"{name:16}{m['total_return']*100:>11.1f}{m['max_drawdown']*100:>10.1f}"
              f"{m['sharpe']:>9.2f}{verdict:>18}")
    print("=" * 74)
    winners = [n for n in order if results[n]["sharpe"] > bh["sharpe"]]
    print(f"Beat B&H on Sharpe: {', '.join(winners) if winners else 'none'}.")
    best_ret = max(results, key=lambda n: results[n]["total_return"])
    if results[best_ret]["total_return"] > bh["total_return"]:
        print(f"Beat B&H on raw return: {best_ret} "
              f"({results[best_ret]['total_return']*100:.0f}% vs {bh['total_return']*100:.0f}%).")
    else:
        print(f"Beat B&H on raw return: none (best was {best_ret} at "
              f"{results[best_ret]['total_return']*100:.0f}% vs {bh['total_return']*100:.0f}%).")

    plot_equity(series, series[order[0]], RESULTS_PNG, bh_series)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Strategy lab: beat-buy-and-hold approaches.")
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "GLD", "TLT"])
    ap.add_argument("--strategies", nargs="+",
                    default=list(_STRATEGIES) + ["ensemble"],
                    choices=list(_STRATEGIES) + ["ensemble"])
    ap.add_argument("--ma-period", type=int, default=200)
    ap.add_argument("--target-vol", type=float, default=0.15)
    ap.add_argument("--vol-lookback", type=int, default=20)
    ap.add_argument("--max-leverage", type=float, default=2.0)
    ap.add_argument("--borrow-rate", type=float, default=0.06)
    ap.add_argument("--months", type=int, default=240)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca")
    args = ap.parse_args()

    end_dt = _parse_date(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc))
    start_dt = (_parse_date(args.start) if args.start
                else end_dt - pd.Timedelta(days=int(args.months * 31)))
    log.info("Lab window: %s -> %s", start_dt.date(), end_dt.date())

    daily_data, _, _ = fetch_all(args.symbols, "none", start_dt, end_dt, source=args.data_source)
    if len(daily_data) < 2:
        log.error("Need >= 2 symbols with data; got %d.", len(daily_data))
        return 1

    names = [s for s in args.strategies if s in _STRATEGIES]
    params = dict(ma_period=args.ma_period, target_vol=args.target_vol,
                  vol_lookback=args.vol_lookback, max_leverage=args.max_leverage,
                  borrow_rate=args.borrow_rate,
                  with_ensemble="ensemble" in args.strategies)
    return run(daily_data, names, begin_ts=start_dt, params=params)


if __name__ == "__main__":
    raise SystemExit(main())
