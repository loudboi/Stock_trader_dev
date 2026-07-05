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

  trend_ls        Time-series trend, LONG AND SHORT: long an asset above its
                  long MA, short it below (instead of going to cash) — the
                  classic managed-futures/CTA construction.
  xsmom_ls        Cross-sectional momentum, LONG AND SHORT: each month, long the
                  strongest-momentum assets and short the weakest, equal dollar
                  amounts (market-neutral) — the classic academic momentum factor
                  (12-1 lookback, Jegadeesh & Titman-style).
  managed_futures_ls  Same long/short trend signal as trend_ls, but INVERSE-VOL
                  weighted across active signals (like managed_futures, but
                  allowed to short) instead of equal-weighted — so a low-vol FX
                  pair and a high-vol commodity don't get equal dollar weight.

Macro-regime-conditioned (need --regime-indicator {vix,curve,credit}; see
bot/regime.py for the classification — VIX 15/25, yield-curve sign, credit-vs-
Treasury sign, all fixed conventional thresholds):
  regime_gated_rp      Inverse-vol risk parity, de-risked to `deleverage_to`
                  exposure whenever the regime signals stress.
  regime_leverage_rp   Inverse-vol risk parity, LEVERED UP in a calm regime and
                  delevered in a stressed one (an external, often forward-
                  looking signal timing leverage instead of the book's own
                  trailing realized vol).
  regime_gated_trend_ls  Same long/short trend signal as trend_ls, but SHORTS
                  are only allowed when the regime signals stress — testing
                  whether gating shorts to a genuine macro-stress window fixes
                  trend_ls's core problem (shorting an asset with persistent
                  structural drift, which fights that drift most of the time).

All new strategies (erc, min_var, rp_voltarget, trend_ls, xsmom_ls) use FIXED,
standard textbook parameters (60-day covariance, monthly rebalance, 10-15% vol
target, 12-1 momentum) — chosen before looking at results, not swept for the best
backtest number. Costs modelled: 0.05% slippage on turnover, a borrow rate on
leverage >1×, and a separate --short-borrow rate charged on short notional (default
1%/yr, a liquid-ETF assumption — hard-to-borrow single names can cost far more or
be unborrowable; see the module note on trend_ls/xsmom_ls). All signals act on the
NEXT day (shift by one), so there is no lookahead.

--mode walk splits the full window into sequential folds and checks whether each
strategy's edge over buy-and-hold is CONSISTENT across time (not just a good
average) — with no per-fold parameter fitting, since there's nothing to fit.

    python -m bot.lab --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
    python -m bot.lab --strategies trend_vol ensemble --target-vol 0.12
    python -m bot.lab --mode walk --folds 5 --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
    python -m bot.lab --strategies trend_ls xsmom_ls --symbols SPY QQQ GLD TLT IWM EFA --data-source yahoo --start 2005-01-01
    python -m bot.lab --strategies regime_leverage_rp --regime-indicator vix --symbols SPY QQQ GLD TLT --data-source yahoo --start 2005-01-01
"""

import argparse
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
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


# --- long/short strategies ----------------------------------------------------- #
# HONEST NOTE on short_borrow: this charges an annual cost on short notional
# (like the leverage borrow_rate, but economically distinct — it's a securities-
# lending fee, not margin interest). The 1%/yr default assumes cheap, liquid,
# easy-to-borrow names (broad index/sector ETFs). Real single stocks — especially
# heavily-shorted or small-cap ones — can cost far more (double digits) or simply
# not be borrowable, and a broker can recall shares/force-cover at the worst time.
# Treat any result here as an upper bound on a real short book's performance.
def trend_ls(panel, ma_period=200, short_borrow=0.01) -> pd.Series:
    """Time-series trend, long AND short: long an asset above its long MA, short
    it below (instead of sitting in cash) — the classic managed-futures/CTA
    construction. Equal-weighted across every asset with an active signal."""
    rets = daily_returns(panel)
    ma = panel.rolling(ma_period, min_periods=ma_period).mean()
    signal = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    signal[panel > ma] = 1.0
    signal[panel < ma] = -1.0
    active = signal.abs().sum(axis=1).replace(0, np.nan)
    w = signal.div(active, axis=0).fillna(0.0).shift(1).fillna(0.0)
    gross = (w * rets).sum(axis=1)
    short_notional = w.clip(upper=0).abs().sum(axis=1)
    financing = short_notional * (short_borrow / _ANNUAL)
    return gross - financing - _turn_cost(w)


def managed_futures_ls(panel, ma_period=200, vol_lookback=20, short_borrow=0.01) -> pd.Series:
    """Time-series trend, long AND short, INVERSE-VOL weighted (not equal-weighted
    across active signals like trend_ls) — the standard managed-futures
    construction, so a low-vol instrument (e.g. an FX pair) and a high-vol one
    (e.g. a commodity) don't get the same dollar weight. Same long-above/short-
    below-the-MA signal as trend_ls; this is the long-only managed_futures'
    natural long/short sibling."""
    rets = daily_returns(panel)
    ma = panel.rolling(ma_period, min_periods=ma_period).mean()
    signal = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    signal[panel > ma] = 1.0
    signal[panel < ma] = -1.0
    inv_vol = (1.0 / realized_vol(rets, vol_lookback)).replace([np.inf, -np.inf], np.nan)
    raw = (signal * inv_vol).fillna(0.0)
    w = raw.div(raw.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    w = w.shift(1).fillna(0.0)
    gross = (w * rets).sum(axis=1)
    short_notional = w.clip(upper=0).abs().sum(axis=1)
    financing = short_notional * (short_borrow / _ANNUAL)
    return gross - financing - _turn_cost(w)


# --- macro-regime-conditioned strategies --------------------------------------- #
# `regime` is a pre-classified macro series (+1 calm/risk-on, -1 elevated/stress,
# 0 neutral — see bot/regime.py) with ITS OWN (macro) calendar; each function
# forward-fills it onto the panel's trading calendar via rg.align_to_panel. Which
# macro indicator (VIX / yield curve / credit spread) is fed in as `regime` is a
# CLI choice (--regime-indicator), not something these functions hardcode — the
# same construction is tested against three independent macro signals.
def regime_gated_rp(panel, regime, deleverage_to=0.0, vol_lookback=20) -> pd.Series:
    """Inverse-vol risk parity, scaled down to `deleverage_to` exposure whenever
    the macro regime signals stress (-1) — an early-warning de-risk overlay on
    top of the one construction (inverse_vol) already shown to beat B&H
    risk-adjusted. Tests whether a macro signal improves on that further, or just
    adds whipsaw."""
    rets = daily_returns(panel)
    inv = (1.0 / realized_vol(rets, vol_lookback)).replace([np.inf, -np.inf], np.nan)
    base_w = inv.div(inv.sum(axis=1), axis=0).fillna(0.0)
    aligned = rg.align_to_panel(regime, panel.index).fillna(0)
    exposure = pd.Series(1.0, index=panel.index)
    exposure[aligned == -1] = deleverage_to
    w = base_w.mul(exposure, axis=0).shift(1).fillna(0.0)
    return (w * rets).sum(axis=1) - _turn_cost(w)


def regime_leverage_rp(panel, regime, calm_leverage=1.5, normal_leverage=1.0,
                       elevated_leverage=0.3, vol_lookback=20, borrow_rate=0.06) -> pd.Series:
    """Inverse-vol risk parity, LEVERED UP in a calm macro regime and delevered in
    a stressed one. Tests whether an external, often forward-looking signal (e.g.
    VIX, which is option-implied) times leverage better than the book's own
    trailing realized vol — vol_target/rp_voltarget already tried the latter and
    financing costs ate the edge; this is a genuinely different mechanism."""
    rets = daily_returns(panel)
    inv = (1.0 / realized_vol(rets, vol_lookback)).replace([np.inf, -np.inf], np.nan)
    base_w = inv.div(inv.sum(axis=1), axis=0).fillna(0.0)
    aligned = rg.align_to_panel(regime, panel.index).fillna(0)
    exposure = pd.Series(normal_leverage, index=panel.index)
    exposure[aligned == 1] = calm_leverage
    exposure[aligned == -1] = elevated_leverage
    w = base_w.mul(exposure, axis=0).shift(1).fillna(0.0)
    exposure_shifted = exposure.shift(1).fillna(normal_leverage)
    financing = (exposure_shifted - 1.0).clip(lower=0) * (borrow_rate / _ANNUAL)
    return (w * rets).sum(axis=1) - financing - _turn_cost(w)


def regime_gated_trend_ls(panel, regime, ma_period=200, short_borrow=0.01) -> pd.Series:
    """Same long/short trend signal as trend_ls (long above the MA, short below),
    but SHORTS are only allowed when the macro regime signals stress (-1) — e.g.
    an inverted yield curve or credit-spread widening. Long signals are always
    allowed. Tests whether gating shorts to a genuine macro-stress window fixes
    trend_ls's core problem (found in round 1/2): shorting an asset with
    persistent structural drift fights that drift most of the time."""
    rets = daily_returns(panel)
    ma = panel.rolling(ma_period, min_periods=ma_period).mean()
    signal = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    signal[panel > ma] = 1.0
    signal[panel < ma] = -1.0
    aligned = rg.align_to_panel(regime, panel.index).fillna(1)   # unknown -> no shorts
    stress_day = (aligned == -1)
    allow_short = pd.DataFrame({c: stress_day for c in signal.columns})
    gated = signal.where((signal >= 0) | allow_short, 0.0)
    active = gated.abs().sum(axis=1).replace(0, np.nan)
    w = gated.div(active, axis=0).fillna(0.0).shift(1).fillna(0.0)
    gross = (w * rets).sum(axis=1)
    short_notional = w.clip(upper=0).abs().sum(axis=1)
    financing = short_notional * (short_borrow / _ANNUAL)
    return gross - financing - _turn_cost(w)


def xsmom_weights(panel, lookback_months=12, skip_months=1, top_frac=0.3) -> pd.DataFrame:
    """Pre-shift target weights for the cross-sectional momentum long/short book:
    monthly rebalance, long the top `top_frac` and short the bottom `top_frac` of
    the universe by trailing (skip-adjusted) momentum, 0.5 notional a side (net
    zero -> dollar-neutral). Exposed separately from xsmom_ls so the weight
    construction (neutrality, ranking) is directly testable."""
    lookback = lookback_months * _DAYS_PER_MONTH
    skip = skip_months * _DAYS_PER_MONTH
    n = len(panel.columns)
    k = max(1, int(round(n * top_frac)))
    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    cur = pd.Series(0.0, index=panel.columns)
    last_period = None
    for i, ts in enumerate(panel.index):
        period = (ts.year, ts.month)
        if last_period is not None and period != last_period:
            scores = momentum(panel, i - 1, lookback, skip)
            if scores is not None:
                ranked = scores.sort_values(ascending=False)
                w = pd.Series(0.0, index=panel.columns)
                w[ranked.index[:k]] = 0.5 / k
                w[ranked.index[-k:]] = -0.5 / k
                cur = w
        weights.iloc[i] = cur.values
        last_period = period
    return weights


def xsmom_ls(panel, lookback_months=12, skip_months=1, top_frac=0.3,
            short_borrow=0.01) -> pd.Series:
    """Classic cross-sectional momentum factor (12-1, Jegadeesh & Titman-style):
    each month, rank assets by trailing (skip-adjusted) momentum, go long the top
    `top_frac` and short the bottom `top_frac`, each equal-weighted and matched to
    0.5 notional a side — a market-neutral (dollar-neutral) long/short book."""
    rets = daily_returns(panel)
    weights = xsmom_weights(panel, lookback_months, skip_months, top_frac).shift(1).fillna(0.0)
    gross = (weights * rets).sum(axis=1)
    short_notional = weights.clip(upper=0).abs().sum(axis=1)
    financing = short_notional * (short_borrow / _ANNUAL)
    return gross - financing - _turn_cost(weights)


def ensemble(components: dict) -> pd.Series:
    """Equal-weight blend of several strategies' daily return series."""
    return pd.concat(components.values(), axis=1).mean(axis=1)


def combine_books(book_returns: dict, weights: dict = None) -> pd.Series:
    """Blend several already-computed strategy return series into ONE combined-
    capital book, each possibly from a DIFFERENT universe/trading calendar (e.g.
    an equity risk-parity book + an FX/commodity trend book). `weights` are fixed
    CAPITAL ALLOCATIONS (must sum to ~1); default is equal-weight.

    Unlike ensemble() (which averages and silently reweights on any day a series
    is missing, via pandas' default skip-NaN mean), a day one sub-book has no data
    (e.g. a holiday on that market) contributes its allocated weight x 0 return —
    the correct behavior for a real multi-market capital split, since that
    capital is sitting idle that day, not reallocated to the other book."""
    if weights is None:
        n = len(book_returns)
        weights = {k: 1.0 / n for k in book_returns}
    total_w = sum(weights.values())
    if abs(total_w - 1.0) > 1e-6:
        log.warning("combine_books weights sum to %.4f, not 1.0.", total_w)
    aligned = pd.concat(book_returns, axis=1).fillna(0.0)
    w = pd.Series(weights)
    return (aligned * w).sum(axis=1)


_STRATEGIES = {
    "vol_target": vol_target,
    "inverse_vol": inverse_vol,
    "erc": erc,
    "min_var": min_var,
    "rp_voltarget": rp_voltarget,
    "managed_futures": managed_futures,
    "mean_reversion": mean_reversion,
    "trend_vol": trend_vol,
    "trend_ls": trend_ls,
    "xsmom_ls": xsmom_ls,
    "managed_futures_ls": managed_futures_ls,
}

# Regime-conditioned strategies: require a --regime-indicator (they take a
# `regime` kwarg with no default). Kept in a separate dict so plain `_STRATEGIES`
# usage (walk-forward, existing CLIs) is unaffected; main() merges them in only
# when the user opts into a regime indicator.
_REGIME_STRATEGIES = {
    "regime_gated_rp": regime_gated_rp,
    "regime_leverage_rp": regime_leverage_rp,
    "regime_gated_trend_ls": regime_gated_trend_ls,
}


# --------------------------------------------------------------------------- #
# Runner / reporting
# --------------------------------------------------------------------------- #
def compute_strategy_returns(daily_data, names, params) -> dict:
    """Full-history daily return series per strategy (unsliced). Callers slice
    by whatever period they need (a single window, or per-fold for walk-forward) —
    the return series itself is computed ONCE, with no per-period refitting, so
    slicing it later can't introduce any lookahead or parameter-selection bias."""
    all_strategies = {**_STRATEGIES, **_REGIME_STRATEGIES}
    panel = build_panel(daily_data)
    comp_returns = {}
    for name in names:
        fn = all_strategies[name]
        kw = {k: v for k, v in params.items() if k in fn.__code__.co_varnames}
        comp_returns[name] = fn(panel, **kw)
    if params.get("with_ensemble"):
        comp_returns["ensemble"] = ensemble(
            {k: v for k, v in comp_returns.items() if k in all_strategies})
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
                    choices=list(_STRATEGIES) + list(_REGIME_STRATEGIES) + ["ensemble"])
    ap.add_argument("--regime-indicator", choices=["none", "vix", "curve", "credit"],
                    default="none",
                    help="Macro regime signal for regime_gated_rp/regime_leverage_rp/"
                         "regime_gated_trend_ls (bot/regime.py). Fixed, conventional "
                         "thresholds (VIX 15/25, curve sign, credit-vs-Treasury sign) "
                         "-- chosen before backtesting, not tuned per run. 'credit' "
                         "needs HYG, which only exists from 2007-04-11.")
    ap.add_argument("--ma-period", type=int, default=200)
    ap.add_argument("--target-vol", type=float, default=0.15)
    ap.add_argument("--vol-lookback", type=int, default=20)
    ap.add_argument("--max-leverage", type=float, default=2.0)
    ap.add_argument("--borrow-rate", type=float, default=0.06)
    ap.add_argument("--short-borrow", type=float, default=0.01,
                    help="Annual securities-lending cost on short notional (trend_ls, "
                         "xsmom_ls). Default 1%%/yr assumes cheap, liquid, easy-to-borrow "
                         "names — real single stocks can cost far more or be unborrowable.")
    ap.add_argument("--top-frac", type=float, default=0.3,
                    help="xsmom_ls: fraction of the universe held long / short each side.")
    ap.add_argument("--months", type=int, default=240)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca")
    ap.add_argument("--clean-outliers", action="store_true",
                    help="Patch implausible single-day price moves (bad ticks, or a "
                         "real zero-crossing event like WTI's 2020-04-20 negative "
                         "print) before backtesting. Off by default so existing "
                         "results are unaffected; recommended for FX (=X) / futures "
                         "(=F) tickers, which have real, documented data-quality "
                         "issues in free feeds. See bot/data.py clean_price_series.")
    ap.add_argument("--max-abs-return", type=float, default=0.15,
                    help="Cap used by --clean-outliers (fixed, not a tuning knob).")
    args = ap.parse_args()

    end_dt = _parse_date(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc))
    start_dt = (_parse_date(args.start) if args.start
                else end_dt - pd.Timedelta(days=int(args.months * 31)))
    log.info("Lab window: %s -> %s (%s mode)", start_dt.date(), end_dt.date(), args.mode)

    daily_data, _, _ = fetch_all(args.symbols, "none", start_dt, end_dt, source=args.data_source)
    if len(daily_data) < 2:
        log.error("Need >= 2 symbols with data; got %d.", len(daily_data))
        return 1
    if args.clean_outliers:
        daily_data = clean_daily_data(daily_data, args.max_abs_return)
        log.info("Cleaned implausible single-day moves (cap %.0f%%).", args.max_abs_return * 100)

    all_strategies = {**_STRATEGIES, **_REGIME_STRATEGIES}
    names = [s for s in args.strategies if s in all_strategies]
    needs_regime = any(n in _REGIME_STRATEGIES for n in names)
    if needs_regime and args.regime_indicator == "none":
        log.error("Requested a regime-conditioned strategy but --regime-indicator "
                  "is 'none'. Pass --regime-indicator {vix,curve,credit}.")
        return 1

    params = dict(ma_period=args.ma_period, target_vol=args.target_vol,
                  vol_lookback=args.vol_lookback, max_leverage=args.max_leverage,
                  borrow_rate=args.borrow_rate, short_borrow=args.short_borrow,
                  top_frac=args.top_frac,
                  with_ensemble="ensemble" in args.strategies)

    if needs_regime:
        macro_start = start_dt - pd.Timedelta(days=120)   # lead for e.g. credit's 60d lookback
        if args.regime_indicator == "vix":
            params["regime"] = rg.vix_regime(rg.fetch_vix(macro_start, end_dt))
        elif args.regime_indicator == "curve":
            params["regime"] = rg.curve_regime(rg.fetch_curve(macro_start, end_dt))
        elif args.regime_indicator == "credit":
            params["regime"] = rg.credit_regime(rg.fetch_credit(macro_start, end_dt))
        n_stress = int((params["regime"] == -1).sum())
        log.info("Regime indicator: %s (%d/%d days classified 'stress').",
                 args.regime_indicator, n_stress, len(params["regime"]))

    if args.mode == "walk":
        per_fold = run_walk_forward(daily_data, names, params, start_dt, end_dt, args.folds)
        print_walk_forward(per_fold)
        return 0
    return run(daily_data, names, begin_ts=start_dt, params=params)


if __name__ == "__main__":
    raise SystemExit(main())
