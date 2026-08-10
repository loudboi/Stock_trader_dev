"""
bot/lab.py
==========
Comparable research harness for fixed-rule portfolio strategies.

`--mode consistency` (legacy alias: `walk`) is a chronological consistency check,
not fit/test walk-forward optimization: strategy rules are computed once and then
measured in non-overlapping sequential folds. True parameter-selection walk-forward
lives in bot/sweep.py.

All strategy weights act one bar after the information used to calculate them.
Turnover is charged on the actual dollar-weight vector, including leverage/vol
scaling. Missing sub-book returns are treated as idle capital (0 return), not as a
signal to reallocate their capital to another sleeve.
"""

import argparse
import inspect
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from bot import regime as rg
from bot.data import clean_daily_data
from bot.momentum_rotation import build_panel, momentum, _DAYS_PER_MONTH
from bot.backtest_pullback import (compute_metrics, buy_hold_combined, fetch_all,
                                   plot_equity, _parse_date, INITIAL_EQUITY, SLIPPAGE)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("lab")
RESULTS_PNG = "lab_results.png"
_ANNUAL = 252


def daily_returns(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pct_change().fillna(0.0)


def realized_vol(rets, lookback: int):
    if lookback <= 1:
        raise ValueError("lookback must be > 1")
    return rets.rolling(lookback).std() * np.sqrt(_ANNUAL)


def rsi(series: pd.Series, period: int) -> pd.Series:
    if period <= 0:
        raise ValueError("RSI period must be positive")
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _turn_cost(weights: pd.DataFrame) -> pd.Series:
    return weights.diff().abs().sum(axis=1).fillna(0.0) * SLIPPAGE


def _validate_positive(**values):
    bad = [k for k, v in values.items() if v <= 0]
    if bad:
        raise ValueError("Must be positive: " + ", ".join(bad))


def vol_target(panel, target_vol=0.15, lookback=20, max_leverage=2.0,
               borrow_rate=0.06) -> pd.Series:
    _validate_positive(target_vol=target_vol, lookback=lookback, max_leverage=max_leverage)
    if borrow_rate < 0:
        raise ValueError("borrow_rate must be non-negative")
    rets = daily_returns(panel)
    base_w = pd.DataFrame(1.0 / len(panel.columns), index=panel.index, columns=panel.columns)
    base = (base_w * rets).sum(axis=1)
    scale = (target_vol / realized_vol(base, lookback)).clip(upper=max_leverage).shift(1).fillna(0.0)
    w = base_w.mul(scale, axis=0)
    financing = (scale - 1.0).clip(lower=0) * (borrow_rate / _ANNUAL)
    return (w * rets).sum(axis=1) - financing - _turn_cost(w)


def inverse_vol(panel, lookback=20) -> pd.Series:
    _validate_positive(lookback=lookback)
    rets = daily_returns(panel)
    inv = (1.0 / realized_vol(rets, lookback)).replace([np.inf, -np.inf], np.nan)
    w = inv.div(inv.sum(axis=1), axis=0).fillna(0.0).shift(1).fillna(0.0)
    return (w * rets).sum(axis=1) - _turn_cost(w)


def managed_futures(panel, ma_period=200, vol_lookback=20) -> pd.Series:
    _validate_positive(ma_period=ma_period, vol_lookback=vol_lookback)
    rets = daily_returns(panel)
    ma = panel.rolling(ma_period, min_periods=ma_period).mean()
    trend = (panel > ma).astype(float)
    raw = (trend / realized_vol(rets, vol_lookback)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    w = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).shift(1).fillna(0.0)
    return (w * rets).sum(axis=1) - _turn_cost(w)


def mean_reversion(panel, ma_trend=200, rsi_period=2, buy=10.0, exit_=70.0) -> pd.Series:
    _validate_positive(ma_trend=ma_trend, rsi_period=rsi_period)
    if not 0 <= buy < exit_ <= 100:
        raise ValueError("require 0 <= buy < exit <= 100")
    rets = daily_returns(panel)
    ma = panel.rolling(ma_trend, min_periods=ma_trend).mean()
    uptrend = panel > ma
    pos = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    for col in panel.columns:
        rv = rsi(panel[col], rsi_period)
        state, out = 0.0, []
        for up, value in zip(uptrend[col].values, rv.values):
            if value != value:
                out.append(0.0); continue
            if state == 0.0 and up and value < buy:
                state = 1.0
            elif state == 1.0 and (value > exit_ or not up):
                state = 0.0
            out.append(state)
        pos[col] = out
    w = pos.div(pos.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).shift(1).fillna(0.0)
    return (w * rets).sum(axis=1) - _turn_cost(w)


def trend_vol(panel, ma_period=200, target_vol=0.15, vol_lookback=20,
              max_leverage=2.0, borrow_rate=0.06) -> pd.Series:
    _validate_positive(ma_period=ma_period, target_vol=target_vol,
                       vol_lookback=vol_lookback, max_leverage=max_leverage)
    if borrow_rate < 0:
        raise ValueError("borrow_rate must be non-negative")
    rets = daily_returns(panel)
    ma = panel.rolling(ma_period, min_periods=ma_period).mean()
    target = (panel > ma).astype(float)
    target = target.div(target.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    base_w = target.shift(1).fillna(0.0)
    base = (base_w * rets).sum(axis=1)
    scale = (target_vol / realized_vol(base, vol_lookback)).clip(upper=max_leverage).shift(1).fillna(0.0)
    w = base_w.mul(scale, axis=0)
    financing = (scale - 1.0).clip(lower=0) * (borrow_rate / _ANNUAL)
    return (w * rets).sum(axis=1) - financing - _turn_cost(w)


def rp_voltarget(panel, target_vol=0.10, lookback=20, max_leverage=1.5,
                 borrow_rate=0.06) -> pd.Series:
    _validate_positive(target_vol=target_vol, lookback=lookback, max_leverage=max_leverage)
    if borrow_rate < 0:
        raise ValueError("borrow_rate must be non-negative")
    rets = daily_returns(panel)
    inv = (1.0 / realized_vol(rets, lookback)).replace([np.inf, -np.inf], np.nan)
    base_w = inv.div(inv.sum(axis=1), axis=0).fillna(0.0).shift(1).fillna(0.0)
    base = (base_w * rets).sum(axis=1)
    scale = (target_vol / realized_vol(base, lookback)).clip(upper=max_leverage).shift(1).fillna(0.0)
    w = base_w.mul(scale, axis=0)
    financing = (scale - 1.0).clip(lower=0) * (borrow_rate / _ANNUAL)
    return (w * rets).sum(axis=1) - financing - _turn_cost(w)


def _erc_weights(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if cov.shape != (n, n) or n == 0:
        raise ValueError("covariance matrix must be non-empty and square")
    w = 1.0 / np.sqrt(np.diag(cov).clip(1e-12)); w /= w.sum()
    for _ in range(500):
        rc = w * (cov @ w)
        if rc.sum() <= 0:
            break
        new = w * np.sqrt(rc.mean() / np.maximum(rc, 1e-12))
        new = np.clip(new, 1e-12, None); new /= new.sum()
        if np.max(np.abs(new - w)) < 1e-10:
            w = new; break
        w = new
    return w


def _minvar_weights(cov: np.ndarray) -> np.ndarray:
    """True constrained long-only global minimum-variance solution."""
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if n == 0 or cov.shape != (n, n) or not np.all(np.isfinite(cov)):
        raise ValueError("covariance matrix must be finite, square and non-empty")
    cov = (cov + cov.T) / 2.0 + np.eye(n) * 1e-12
    x0 = np.ones(n) / n
    result = minimize(lambda w: float(w @ cov @ w), x0,
                      jac=lambda w: 2.0 * cov @ w,
                      bounds=[(0.0, 1.0)] * n,
                      constraints=[{"type": "eq", "fun": lambda w: float(w.sum() - 1.0),
                                    "jac": lambda w: np.ones_like(w)}],
                      method="SLSQP", options={"ftol": 1e-12, "maxiter": 1000})
    if not result.success or not np.all(np.isfinite(result.x)):
        log.warning("Long-only min-var optimizer failed (%s); using equal weights.", result.message)
        return x0
    w = np.clip(result.x, 0.0, 1.0)
    return w / w.sum()


def _monthly_cov_strategy(panel, lookback, weight_fn) -> pd.Series:
    _validate_positive(lookback=lookback)
    rets = daily_returns(panel)
    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    cur = np.ones(len(panel.columns)) / len(panel.columns)
    last = None
    for i, ts in enumerate(panel.index):
        period = (ts.year, ts.month)
        if last is not None and period != last and i >= lookback:
            cov = rets.iloc[i - lookback:i].cov().values
            if np.all(np.isfinite(cov)):
                cur = weight_fn(cov)
        weights.iloc[i] = cur
        last = period
    weights = weights.shift(1).fillna(0.0)
    return (weights * rets).sum(axis=1) - _turn_cost(weights)


def erc(panel, cov_lookback=60):
    return _monthly_cov_strategy(panel, cov_lookback, _erc_weights)


def min_var(panel, cov_lookback=60):
    return _monthly_cov_strategy(panel, cov_lookback, _minvar_weights)


def trend_ls(panel, ma_period=200, short_borrow=0.01) -> pd.Series:
    _validate_positive(ma_period=ma_period)
    if short_borrow < 0:
        raise ValueError("short_borrow must be non-negative")
    rets = daily_returns(panel); ma = panel.rolling(ma_period, min_periods=ma_period).mean()
    signal = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    signal[panel > ma] = 1.0; signal[panel < ma] = -1.0
    w = signal.div(signal.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).shift(1).fillna(0.0)
    return ((w * rets).sum(axis=1)
            - w.clip(upper=0).abs().sum(axis=1) * (short_borrow / _ANNUAL)
            - _turn_cost(w))


def managed_futures_ls(panel, ma_period=200, vol_lookback=20, short_borrow=0.01):
    _validate_positive(ma_period=ma_period, vol_lookback=vol_lookback)
    if short_borrow < 0:
        raise ValueError("short_borrow must be non-negative")
    rets = daily_returns(panel); ma = panel.rolling(ma_period, min_periods=ma_period).mean()
    signal = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    signal[panel > ma] = 1.0; signal[panel < ma] = -1.0
    inv = (1.0 / realized_vol(rets, vol_lookback)).replace([np.inf, -np.inf], np.nan)
    raw = (signal * inv).fillna(0.0)
    w = raw.div(raw.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).shift(1).fillna(0.0)
    return ((w * rets).sum(axis=1)
            - w.clip(upper=0).abs().sum(axis=1) * (short_borrow / _ANNUAL)
            - _turn_cost(w))


def _inverse_vol_target(panel, lookback=20):
    rets = daily_returns(panel)
    inv = (1.0 / realized_vol(rets, lookback)).replace([np.inf, -np.inf], np.nan)
    return rets, inv.div(inv.sum(axis=1), axis=0).fillna(0.0)


def regime_gated_rp(panel, regime, deleverage_to=0.0, vol_lookback=20):
    if not 0 <= deleverage_to <= 1:
        raise ValueError("deleverage_to must be in [0,1]")
    rets, base = _inverse_vol_target(panel, vol_lookback)
    aligned = rg.align_to_panel(regime, panel.index)
    exposure = pd.Series(1.0, index=panel.index); exposure[aligned == -1] = deleverage_to
    w = base.mul(exposure, axis=0).shift(1).fillna(0.0)
    return (w * rets).sum(axis=1) - _turn_cost(w)


def regime_leverage_rp(panel, regime, calm_leverage=1.5, normal_leverage=1.0,
                       elevated_leverage=0.3, vol_lookback=20, borrow_rate=0.06):
    for value in (calm_leverage, normal_leverage, elevated_leverage):
        if value < 0:
            raise ValueError("regime leverage values must be non-negative")
    if borrow_rate < 0:
        raise ValueError("borrow_rate must be non-negative")
    rets, base = _inverse_vol_target(panel, vol_lookback)
    aligned = rg.align_to_panel(regime, panel.index)
    exposure = pd.Series(normal_leverage, index=panel.index)
    exposure[aligned == 1] = calm_leverage; exposure[aligned == -1] = elevated_leverage
    w = base.mul(exposure, axis=0).shift(1).fillna(0.0)
    shifted = exposure.shift(1).fillna(normal_leverage)
    financing = (shifted - 1.0).clip(lower=0) * (borrow_rate / _ANNUAL)
    return (w * rets).sum(axis=1) - financing - _turn_cost(w)


def regime_gated_trend_ls(panel, regime, ma_period=200, short_borrow=0.01):
    if short_borrow < 0:
        raise ValueError("short_borrow must be non-negative")
    rets = daily_returns(panel); ma = panel.rolling(ma_period, min_periods=ma_period).mean()
    signal = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    signal[panel > ma] = 1.0; signal[panel < ma] = -1.0
    stress = rg.align_to_panel(regime, panel.index).eq(-1)
    allowed = pd.DataFrame({c: stress for c in signal.columns})
    gated = signal.where((signal >= 0) | allowed, 0.0)
    w = gated.div(gated.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).shift(1).fillna(0.0)
    return ((w * rets).sum(axis=1)
            - w.clip(upper=0).abs().sum(axis=1) * (short_borrow / _ANNUAL)
            - _turn_cost(w))


def xsmom_weights(panel, lookback_months=12, skip_months=1, top_frac=0.3):
    if len(panel.columns) < 2:
        raise ValueError("xsmom requires at least two assets")
    if lookback_months <= 0 or skip_months < 0:
        raise ValueError("lookback_months must be positive and skip_months non-negative")
    if not 0 < top_frac <= 0.5:
        raise ValueError("top_frac must be in (0, 0.5]")
    lookback, skip = lookback_months * _DAYS_PER_MONTH, skip_months * _DAYS_PER_MONTH
    n = len(panel.columns); k = min(n // 2, max(1, int(round(n * top_frac))))
    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    cur = pd.Series(0.0, index=panel.columns); last = None
    for i, ts in enumerate(panel.index):
        period = (ts.year, ts.month)
        if last is not None and period != last:
            scores = momentum(panel, i - 1, lookback, skip)
            if scores is not None and scores.notna().sum() >= 2 * k:
                ranked = scores.dropna().sort_values(ascending=False)
                w = pd.Series(0.0, index=panel.columns)
                longs, shorts = ranked.index[:k], ranked.index[-k:]
                if not set(longs) & set(shorts):
                    w[longs], w[shorts] = 0.5 / k, -0.5 / k
                    cur = w
        weights.iloc[i] = cur.values; last = period
    return weights


def xsmom_ls(panel, lookback_months=12, skip_months=1, top_frac=0.3, short_borrow=0.01):
    if short_borrow < 0:
        raise ValueError("short_borrow must be non-negative")
    rets = daily_returns(panel)
    w = xsmom_weights(panel, lookback_months, skip_months, top_frac).shift(1).fillna(0.0)
    return ((w * rets).sum(axis=1)
            - w.clip(upper=0).abs().sum(axis=1) * (short_borrow / _ANNUAL)
            - _turn_cost(w))


def ensemble(components: dict) -> pd.Series:
    if not components:
        raise ValueError("ensemble requires at least one component strategy")
    return pd.concat(components, axis=1).fillna(0.0).mean(axis=1)


def combine_books(book_returns: dict, weights: dict = None) -> pd.Series:
    if not book_returns:
        raise ValueError("book_returns cannot be empty")
    if weights is None:
        weights = {k: 1.0 / len(book_returns) for k in book_returns}
    if set(weights) != set(book_returns):
        raise ValueError("weights must provide exactly one weight per book")
    if any(v < 0 for v in weights.values()) or abs(sum(weights.values()) - 1.0) > 1e-8:
        raise ValueError("book weights must be non-negative and sum to 1")
    aligned = pd.concat(book_returns, axis=1).fillna(0.0)
    return (aligned * pd.Series(weights)).sum(axis=1)


_STRATEGIES = {
    "vol_target": vol_target, "inverse_vol": inverse_vol, "erc": erc,
    "min_var": min_var, "rp_voltarget": rp_voltarget,
    "managed_futures": managed_futures, "mean_reversion": mean_reversion,
    "trend_vol": trend_vol, "trend_ls": trend_ls,
    "xsmom_ls": xsmom_ls, "managed_futures_ls": managed_futures_ls,
}
_REGIME_STRATEGIES = {
    "regime_gated_rp": regime_gated_rp, "regime_leverage_rp": regime_leverage_rp,
    "regime_gated_trend_ls": regime_gated_trend_ls,
}


def compute_strategy_returns(daily_data, names, params):
    all_strategies = {**_STRATEGIES, **_REGIME_STRATEGIES}
    panel = build_panel(daily_data)
    requested = list(names)
    if not requested and params.get("with_ensemble"):
        requested = list(_STRATEGIES)
    comp = {}
    for name in requested:
        if name not in all_strategies:
            raise ValueError(f"Unknown strategy {name}")
        fn = all_strategies[name]
        accepted = inspect.signature(fn).parameters
        kw = {k: v for k, v in params.items() if k in accepted}
        comp[name] = fn(panel, **kw)
    if params.get("with_ensemble"):
        components = {k: v for k, v in comp.items() if k in all_strategies}
        comp["ensemble"] = ensemble(components)
    return comp


def slice_equity(rets, start_ts=None, end_ts=None, initial=INITIAL_EQUITY,
                 end_inclusive=True):
    r = rets
    if start_ts is not None:
        r = r[r.index >= start_ts]
    if end_ts is not None:
        r = r[r.index <= end_ts] if end_inclusive else r[r.index < end_ts]
    if not len(r):
        return pd.Series(dtype=float)
    out = initial * (1 + r).cumprod(); out.attrs["initial_equity"] = initial
    return out


def run(daily_data, names, begin_ts, params):
    comp = compute_strategy_returns(daily_data, names, params)
    if not comp:
        log.error("No strategy returns to score."); return 1
    results = {n: compute_metrics([], slice_equity(r, begin_ts)) for n, r in comp.items()}
    series = {n: slice_equity(r, begin_ts) for n, r in comp.items()}
    bh_series = buy_hold_combined(daily_data, begin_ts); bh = compute_metrics([], bh_series)
    order = sorted(results, key=lambda n: results[n]["sharpe"], reverse=True)
    print("\nSTRATEGY LAB (historical simulation; not validation of future edge)")
    print(f"{'Strategy':24}{'Return%':>11}{'MaxDD%':>10}{'Sharpe':>9}")
    print(f"{'buy_and_hold':24}{bh['total_return']*100:>11.1f}{bh['max_drawdown']*100:>10.1f}{bh['sharpe']:>9.2f}")
    for name in order:
        m = results[name]
        print(f"{name:24}{m['total_return']*100:>11.1f}{m['max_drawdown']*100:>10.1f}{m['sharpe']:>9.2f}")
    plot_equity(series, series[order[0]], RESULTS_PNG, bh_series)
    return 0


def fold_bounds(start_ts, end_ts, folds):
    if folds <= 0 or start_ts >= end_ts:
        raise ValueError("folds must be positive and start must precede end")
    total = (end_ts - start_ts) / folds
    return [(start_ts + total * k, start_ts + total * (k + 1)) for k in range(folds)]


def run_walk_forward(daily_data, names, params, start_ts, end_ts, folds=5):
    """Legacy API: chronological consistency folds, not parameter walk-forward."""
    comp = compute_strategy_returns(daily_data, names, params); out = []
    bounds = fold_bounds(start_ts, end_ts, folds)
    for k, (fs, fe) in enumerate(bounds):
        final = k == len(bounds) - 1
        bh = buy_hold_combined(daily_data, begin_ts=fs)
        bh = bh[bh.index <= fe] if final else bh[bh.index < fe]
        row = {"start": fs, "end": fe, "bh": compute_metrics([], bh), "strategies": {}}
        for name, rets in comp.items():
            row["strategies"][name] = compute_metrics([], slice_equity(
                rets, fs, fe, end_inclusive=final))
        out.append(row)
    return out


def print_walk_forward(per_fold):
    names = list(per_fold[0]["strategies"]) if per_fold else []
    print(f"\nCHRONOLOGICAL CONSISTENCY — {len(per_fold)} non-overlapping folds")
    beat = {n: 0 for n in names}
    for k, row in enumerate(per_fold, 1):
        bits = []
        for n in names:
            shp = row["strategies"][n]["sharpe"]
            won = shp > row["bh"]["sharpe"]; beat[n] += int(won)
            bits.append(f"{n}={shp:.2f}{'*' if won else ''}")
        print(f"{k}: {row['start'].date()}->{row['end'].date()} B&H={row['bh']['sharpe']:.2f} " + " ".join(bits))
    for n in names:
        print(f"{n}: beat B&H in {beat[n]}/{len(per_fold)} folds")


def _validate_args(ap, args):
    if args.folds <= 0 or args.months <= 0:
        ap.error("--folds and --months must be positive")
    if not (0 < args.target_vol <= 2) or args.vol_lookback <= 1 or args.max_leverage <= 0:
        ap.error("target vol/lookback/leverage values are invalid")
    if args.borrow_rate < 0 or args.short_borrow < 0:
        ap.error("borrow rates cannot be negative")
    if not 0 < args.top_frac <= 0.5:
        ap.error("--top-frac must be in (0, 0.5]")


def main():
    ap = argparse.ArgumentParser(description="Research strategy comparison.")
    ap.add_argument("--mode", choices=["score", "consistency", "walk"], default="score",
                    help="consistency (legacy alias walk) slices fixed rules into sequential folds; no fitting occurs.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "GLD", "TLT"])
    ap.add_argument("--strategies", nargs="+", default=list(_STRATEGIES) + ["ensemble"],
                    choices=list(_STRATEGIES) + list(_REGIME_STRATEGIES) + ["ensemble"])
    ap.add_argument("--regime-indicator", choices=["none", "vix", "curve", "credit"], default="none")
    ap.add_argument("--ma-period", type=int, default=200)
    ap.add_argument("--target-vol", type=float, default=0.15)
    ap.add_argument("--vol-lookback", type=int, default=20)
    ap.add_argument("--max-leverage", type=float, default=2.0)
    ap.add_argument("--borrow-rate", type=float, default=0.06)
    ap.add_argument("--short-borrow", type=float, default=0.01)
    ap.add_argument("--top-frac", type=float, default=0.3)
    ap.add_argument("--months", type=int, default=240)
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca")
    ap.add_argument("--clean-outliers", action="store_true")
    ap.add_argument("--max-abs-return", type=float, default=0.15)
    args = ap.parse_args(); _validate_args(ap, args)

    end_dt = _parse_date(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc))
    start_dt = _parse_date(args.start) if args.start else end_dt - pd.Timedelta(days=args.months * 31)
    if start_dt >= end_dt:
        ap.error("start must precede end")
    daily_data, _, _ = fetch_all(args.symbols, "none", start_dt, end_dt, source=args.data_source)
    if len(daily_data) < 2:
        log.error("Need at least two symbols with data; got %d", len(daily_data)); return 1
    if args.clean_outliers:
        daily_data = clean_daily_data(daily_data, args.max_abs_return)

    names = [s for s in args.strategies if s != "ensemble"]
    needs_regime = any(n in _REGIME_STRATEGIES for n in names)
    if needs_regime and args.regime_indicator == "none":
        ap.error("regime-conditioned strategies require --regime-indicator")
    params = dict(ma_period=args.ma_period, target_vol=args.target_vol,
                  vol_lookback=args.vol_lookback, max_leverage=args.max_leverage,
                  borrow_rate=args.borrow_rate, short_borrow=args.short_borrow,
                  top_frac=args.top_frac, with_ensemble="ensemble" in args.strategies)
    if needs_regime:
        macro_start = start_dt - pd.Timedelta(days=120)
        fetchers = {"vix": lambda: rg.vix_regime(rg.fetch_vix(macro_start, end_dt)),
                    "curve": lambda: rg.curve_regime(rg.fetch_curve(macro_start, end_dt)),
                    "credit": lambda: rg.credit_regime(rg.fetch_credit(macro_start, end_dt))}
        params["regime"] = fetchers[args.regime_indicator]()
    if args.mode in {"consistency", "walk"}:
        per_fold = run_walk_forward(daily_data, names, params, start_dt, end_dt, args.folds)
        print_walk_forward(per_fold); return 0
    return run(daily_data, names, start_dt, params)


if __name__ == "__main__":
    raise SystemExit(main())
