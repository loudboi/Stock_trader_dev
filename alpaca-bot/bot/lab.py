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
  inverse_vol     Risk parity (naive): weight assets by 1/volatility.
  erc             Equal-risk-contribution risk parity (uses the full covariance
                  matrix, not just each asset's own vol) — the textbook risk-parity
                  construction. Monthly rebalance, 60-day trailing covariance.
  min_var         Long-only global minimum-variance portfolio (analytic solution
                  on trailing covariance, negative weights clipped to 0).
  rp_voltarget    Inverse-vol weights, then the whole book scaled to a constant
                  target volatility — the standard institutional combination.
  managed_futures Diversified time-series trend (hold above MA), inverse-vol
                  weighted — "crisis alpha".
  mean_reversion  Buy short-term-oversold (RSI) dips while in an uptrend.
  trend_vol       Vol-targeted trend: hold above MA, then target portfolio vol.
  ensemble        Equal blend of the daily returns of the above (diversify across
                  strategies — the one real free lunch).

All new strategies (erc, min_var, rp_voltarget) use FIXED, standard textbook
parameters (60-day covariance, monthly rebalance, 10-15% vol target) — chosen
before looking at results, not swept for the best backtest number. Costs modelled:
0.05% slippage on turnover, and a borrow rate on leverage >1×. All signals act on
the NEXT day (shift by one), so there is no lookahead.

--mode walk splits the full window into sequential folds and checks whether each
strategy's edge over buy-and-hold is CONSISTENT across time (not just a good
average) — with no per-fold parameter fitting, since there's nothing to fit.

    python -m bot.lab --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
    python -m bot.lab --strategies trend_vol ensemble --target-vol 0.12
    python -m bot.lab --mode walk --folds 5 --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
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


def rp_voltarget(panel, target_vol=0.10, lookback=20, max_leverage=1.5,
                 borrow_rate=0.06) -> pd.Series:
    """Vol-targeted risk parity: inverse-vol weights, then scale the whole book to
    a constant target volatility (the standard institutional construction)."""
    rets = daily_returns(panel)
    inv = (1.0 / realized_vol(rets, lookback)).replace([np.inf, -np.inf], np.nan)
    w = inv.div(inv.sum(axis=1), axis=0).fillna(0.0).shift(1).fillna(0.0)
    base = (w * rets).sum(axis=1)
    scale = (target_vol / realized_vol(base, lookback)).clip(upper=max_leverage)
    scale = scale.shift(1).fillna(0.0)
    financing = (scale - 1.0).clip(lower=0) * (borrow_rate / _ANNUAL)
    return scale * base - financing - _turn_cost(w) - scale.diff().abs().fillna(0.0) * SLIPPAGE


# --- covariance-based portfolio construction (monthly rebalance) --------------- #
def _erc_weights(cov: np.ndarray) -> np.ndarray:
    """Equal-risk-contribution weights (each asset contributes equal portfolio risk)."""
    n = cov.shape[0]
    w = 1.0 / np.sqrt(np.diag(cov).clip(1e-12))
    w /= w.sum()
    for _ in range(500):
        rc = w * (cov @ w)                    # risk contributions
        if rc.sum() <= 0:
            break
        w = w * np.sqrt(rc.mean() / np.maximum(rc, 1e-12))
        w = np.clip(w, 1e-9, None)
        w /= w.sum()
    return w


def _minvar_weights(cov: np.ndarray) -> np.ndarray:
    """Long-only global minimum-variance weights (analytic, negatives clipped)."""
    n = cov.shape[0]
    try:
        inv = np.linalg.pinv(cov)
        ones = np.ones(n)
        w = inv @ ones / (ones @ inv @ ones)
    except np.linalg.LinAlgError:
        return np.ones(n) / n
    w = np.clip(w, 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.ones(n) / n


def _monthly_cov_strategy(panel, lookback, weight_fn) -> pd.Series:
    """Rebalance monthly to weight_fn(trailing covariance); hold in between."""
    rets = daily_returns(panel)
    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    cur = np.ones(len(panel.columns)) / len(panel.columns)
    last = None
    for i, ts in enumerate(panel.index):
        per = (ts.year, ts.month)
        if last is not None and per != last and i >= lookback:
            cov = rets.iloc[i - lookback:i].cov().values
            if np.all(np.isfinite(cov)):
                cur = weight_fn(cov)
        weights.iloc[i] = cur
        last = per
    weights = weights.shift(1).fillna(0.0)
    return (weights * rets).sum(axis=1) - _turn_cost(weights)


def erc(panel, cov_lookback=60) -> pd.Series:
    return _monthly_cov_strategy(panel, cov_lookback, _erc_weights)


def min_var(panel, cov_lookback=60) -> pd.Series:
    return _monthly_cov_strategy(panel, cov_lookback, _minvar_weights)


def ensemble(components: dict) -> pd.Series:
    """Equal-weight blend of several strategies' daily return series."""
    return pd.concat(components.values(), axis=1).mean(axis=1)


_STRATEGIES = {
    "vol_target": vol_target,
    "inverse_vol": inverse_vol,
    "erc": erc,
    "min_var": min_var,
    "rp_voltarget": rp_voltarget,
    "managed_futures": managed_futures,
    "mean_reversion": mean_reversion,
    "trend_vol": trend_vol,
}


# --------------------------------------------------------------------------- #
# Runner / reporting
# --------------------------------------------------------------------------- #
def compute_strategy_returns(daily_data, names, params) -> dict:
    """Full-history daily return series per strategy (unsliced). Callers slice
    by whatever period they need (a single window, or per-fold for walk-forward) —
    the return series itself is computed ONCE, with no per-period refitting, so
    slicing it later can't introduce any lookahead or parameter-selection bias."""
    panel = build_panel(daily_data)
    comp_returns = {}
    for name in names:
        fn = _STRATEGIES[name]
        kw = {k: v for k, v in params.items() if k in fn.__code__.co_varnames}
        comp_returns[name] = fn(panel, **kw)
    if params.get("with_ensemble"):
        comp_returns["ensemble"] = ensemble(
            {k: v for k, v in comp_returns.items() if k in _STRATEGIES})
    return comp_returns


def slice_equity(rets: pd.Series, start_ts=None, end_ts=None,
                 initial=INITIAL_EQUITY) -> pd.Series:
    """Equity curve of a return series restricted to [start_ts, end_ts], rebased
    to `initial` at the first bar in range (so each slice is self-contained)."""
    r = rets
    if start_ts is not None:
        r = r[r.index >= start_ts]
    if end_ts is not None:
        r = r[r.index <= end_ts]
    if not len(r):
        return pd.Series(dtype=float)
    return initial * (1 + r).cumprod()


def run(daily_data, names, begin_ts, params):
    comp_returns = compute_strategy_returns(daily_data, names, params)

    results, series = {}, {}
    for name, rets in comp_returns.items():
        eq = slice_equity(rets, begin_ts)
        results[name] = compute_metrics([], eq)
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


# --------------------------------------------------------------------------- #
# Walk-forward: consistency across time, not parameter fitting
# --------------------------------------------------------------------------- #
# These strategies deliberately have no tunable parameters to search (that was
# the point — avoid overfitting). So "walk-forward" here isn't fit-in-sample /
# test-out-of-sample; it's simpler and stronger: each strategy's return series is
# computed ONCE over the full history with its fixed parameters, then sliced into
# sequential folds and checked against buy-and-hold on that SAME fold. No fitting
# happens per fold, so this only answers one question: is the full-window result
# an average that holds up regime by regime, or is it a few great years carrying
# a mediocre track record?
def fold_bounds(start_ts, end_ts, folds: int) -> list:
    """Split [start_ts, end_ts] into `folds` equal, sequential, non-overlapping
    (start, end) periods."""
    total = (end_ts - start_ts) / folds
    return [(start_ts + total * k, start_ts + total * (k + 1)) for k in range(folds)]


def run_walk_forward(daily_data, names, params, start_ts, end_ts, folds=5) -> list:
    comp_returns = compute_strategy_returns(daily_data, names, params)
    out = []
    for fs, fe in fold_bounds(start_ts, end_ts, folds):
        bh_eq = buy_hold_combined(daily_data, begin_ts=fs)
        bh_eq = bh_eq[bh_eq.index <= fe]
        row = {"start": fs, "end": fe, "bh": compute_metrics([], bh_eq), "strategies": {}}
        for name, rets in comp_returns.items():
            row["strategies"][name] = compute_metrics([], slice_equity(rets, fs, fe))
        out.append(row)
    return out


def print_walk_forward(per_fold):
    names = list(per_fold[0]["strategies"]) if per_fold else []
    print("\n" + "=" * 88)
    print(f"WALK-FORWARD  {len(per_fold)} sequential folds, each vs buy-and-hold on that SAME fold")
    print("(fixed parameters throughout — no per-fold fitting; this checks CONSISTENCY,")
    print(" not the best backtest number)")
    print("=" * 88)
    print(f"{'Fold':>4}  {'Window':>23}  {'B&H Shp':>8}  Strategy Sharpe ('*' = beat B&H)")
    print("-" * 88)
    beat = {n: 0 for n in names}
    sharpes = {n: [] for n in names}
    for k, row in enumerate(per_fold):
        win = f"{row['start'].date()}->{row['end'].date()}"
        bits = []
        for n in names:
            shp = row["strategies"][n]["sharpe"]
            sharpes[n].append(shp)
            won = shp > row["bh"]["sharpe"]
            beat[n] += int(won)
            bits.append(f"{n}={shp:.2f}{'*' if won else ' '}")
        print(f"{k+1:>4}  {win:>23}  {row['bh']['sharpe']:>8.2f}  " + "  ".join(bits))
    print("-" * 88)
    print("SUMMARY (folds beaten / total, mean Sharpe across folds):")
    bh_mean = sum(r["bh"]["sharpe"] for r in per_fold) / len(per_fold)
    print(f"  buy_and_hold{'':<4}mean Sharpe {bh_mean:.2f}  (benchmark)")
    for n in sorted(names, key=lambda x: -beat[x]):
        mean_s = sum(sharpes[n]) / len(sharpes[n])
        robust = "ROBUST" if beat[n] == len(per_fold) else (
            "inconsistent" if 0 < beat[n] < len(per_fold) else "never beat B&H")
        print(f"  {n:<16}{beat[n]}/{len(per_fold)} folds beat B&H, "
              f"mean Sharpe {mean_s:.2f}  [{robust}]")
    print("=" * 88)
    print("A strategy that only wins on the FULL-window average but not most folds is")
    print("regime-dependent luck, not a durable edge.")


def main():
    ap = argparse.ArgumentParser(description="Strategy lab: beat-buy-and-hold approaches.")
    ap.add_argument("--mode", choices=["score", "walk"], default="score",
                    help="score = single full-window scoreboard (default); "
                         "walk = split into folds and check consistency across time.")
    ap.add_argument("--folds", type=int, default=5, help="Walk-forward fold count.")
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
    log.info("Lab window: %s -> %s (%s mode)", start_dt.date(), end_dt.date(), args.mode)

    daily_data, _, _ = fetch_all(args.symbols, "none", start_dt, end_dt, source=args.data_source)
    if len(daily_data) < 2:
        log.error("Need >= 2 symbols with data; got %d.", len(daily_data))
        return 1

    names = [s for s in args.strategies if s in _STRATEGIES]
    params = dict(ma_period=args.ma_period, target_vol=args.target_vol,
                  vol_lookback=args.vol_lookback, max_leverage=args.max_leverage,
                  borrow_rate=args.borrow_rate,
                  with_ensemble="ensemble" in args.strategies)

    if args.mode == "walk":
        per_fold = run_walk_forward(daily_data, names, params, start_dt, end_dt, args.folds)
        print_walk_forward(per_fold)
        return 0
    return run(daily_data, names, begin_ts=start_dt, params=params)


if __name__ == "__main__":
    raise SystemExit(main())
