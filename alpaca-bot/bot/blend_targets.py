"""Causal target weights for the fixed profitability-research blend.

This module intentionally contains no broker/order code.  It translates the fixed
research sleeves into long-only target weights that can later be consumed by a
paper-only rebalancer after independent parity and execution testing.

Default blend (fixed from 2006-2024 development data, not the 2025-2026 holdout):

* 25% equal-capital 150-day trend exposure, 0% re-entry buffer;
* 25% equal-capital 200-day trend exposure, 1% re-entry buffer;
* 25% 9-month / skip-1-month absolute momentum, top 3;
* 25% 60-day long-only minimum variance, monthly update.

A sleeve may hold cash.  Combined risky-asset weights therefore sum to <= 1.
Signals use closes available at or before the decision close; returned close targets
are intended for the following return interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot import lab
from bot import momentum_rotation as mr
from bot import trend_exposure as te


DEFAULT_SYMBOLS = ("SPY", "QQQ", "GLD", "TLT", "EFA", "EEM", "IWM")


@dataclass(frozen=True)
class BlendSpec:
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    trend_fast_ma: int = 150
    trend_fast_buffer: float = 0.0
    trend_slow_ma: int = 200
    trend_slow_buffer: float = 0.01
    momentum_lookback_months: int = 9
    momentum_skip_months: int = 1
    momentum_top_k: int = 3
    minvar_lookback: int = 60
    trend_fast_sleeve: float = 0.25
    trend_slow_sleeve: float = 0.25
    momentum_sleeve: float = 0.25
    minvar_sleeve: float = 0.25

    def validate(self) -> None:
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must be a non-empty unique tuple")
        if self.trend_fast_ma <= 1 or self.trend_slow_ma <= 1:
            raise ValueError("trend moving averages must be > 1")
        if self.trend_fast_buffer < 0 or self.trend_slow_buffer < 0:
            raise ValueError("trend buffers must be non-negative")
        if self.momentum_lookback_months <= 0 or self.momentum_skip_months < 0:
            raise ValueError("invalid momentum lookback/skip")
        if not 1 <= self.momentum_top_k <= len(self.symbols):
            raise ValueError("momentum_top_k must be within the universe")
        if self.minvar_lookback <= 1:
            raise ValueError("minvar_lookback must be > 1")
        sleeves = (self.trend_fast_sleeve, self.trend_slow_sleeve,
                   self.momentum_sleeve, self.minvar_sleeve)
        if any(x < 0 for x in sleeves) or sum(sleeves) > 1 + 1e-12:
            raise ValueError("sleeve weights must be non-negative and sum to <= 1")


DEFAULT_SPEC = BlendSpec()


def build_panel(daily_data: dict[str, pd.DataFrame], spec: BlendSpec = DEFAULT_SPEC) -> pd.DataFrame:
    """Build the common-inception close panel in the spec's canonical column order."""
    spec.validate()
    missing = [s for s in spec.symbols if s not in daily_data]
    extra = [s for s in daily_data if s not in spec.symbols]
    if missing or extra:
        raise ValueError(f"daily_data universe mismatch; missing={missing}, extra={extra}")
    ordered = {s: daily_data[s] for s in spec.symbols}
    panel = mr.build_panel(ordered)
    if panel.empty:
        raise ValueError("price panel cannot be empty")
    return panel.loc[:, list(spec.symbols)].astype(float)


def trend_close_weights(panel: pd.DataFrame, ma_period: int, buffer: float) -> pd.DataFrame:
    """Equal-capital trend sleeve target decided at each close for the next interval."""
    n = len(panel.columns)
    if n == 0:
        raise ValueError("panel must contain assets")
    out = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    for symbol in panel.columns:
        out[symbol] = te.exposure_series(panel[symbol], ma_period, buffer) / n
    return out


def minvar_close_weights(panel: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Reproduce the research min-var *pre-shift* monthly target weights.

    The research return function shifts these close targets by one row before they
    earn returns.  Covariance at a month transition uses only rows strictly before
    that transition date.
    """
    if lookback <= 1:
        raise ValueError("lookback must be > 1")
    rets = lab.daily_returns(panel)
    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    cur = np.ones(len(panel.columns), dtype=float) / len(panel.columns)
    last_period = None
    for i, ts in enumerate(panel.index):
        period = (ts.year, ts.month)
        if last_period is not None and period != last_period and i >= lookback:
            cov = rets.iloc[i - lookback:i].cov().values
            if np.all(np.isfinite(cov)):
                cur = lab._minvar_weights(cov)
        weights.iloc[i] = cur
        last_period = period
    return weights


def momentum_execution_weights(panel: pd.DataFrame, spec: BlendSpec = DEFAULT_SPEC) -> pd.DataFrame:
    """Weights held over each row's close-to-close return in the research model."""
    spec.validate()
    return mr.weight_panel(panel, spec.momentum_lookback_months,
                           spec.momentum_skip_months, spec.momentum_top_k)


def momentum_close_weights(panel: pd.DataFrame, spec: BlendSpec = DEFAULT_SPEC) -> pd.DataFrame:
    """Convert research execution weights into prior-close targets.

    `momentum_rotation.weight_panel` changes the held weights on the first trading
    date of a new month using information through the previous close.  Therefore
    the close target that causes row t's holding is stored at row t-1 here.
    The final row is carried forward; callers needing the true next-session target
    should use :func:`latest_target` with an explicit next-session timestamp.
    """
    held = momentum_execution_weights(panel, spec)
    target = held.shift(-1)
    if len(target):
        target.iloc[-1] = held.iloc[-1]
    return target.fillna(0.0)


def component_close_weights(panel: pd.DataFrame,
                            spec: BlendSpec = DEFAULT_SPEC) -> dict[str, pd.DataFrame]:
    """Return close-decision weights for each 100%-notional component sleeve."""
    spec.validate()
    return {
        "trend150": trend_close_weights(panel, spec.trend_fast_ma,
                                         spec.trend_fast_buffer),
        "trend200": trend_close_weights(panel, spec.trend_slow_ma,
                                         spec.trend_slow_buffer),
        "momentum9": momentum_close_weights(panel, spec),
        "minvar60": minvar_close_weights(panel, spec.minvar_lookback),
    }


def combined_close_weights(panel: pd.DataFrame,
                           spec: BlendSpec = DEFAULT_SPEC) -> pd.DataFrame:
    """Aggregate the fixed sleeves into executable long-only risky-asset targets."""
    c = component_close_weights(panel, spec)
    out = (c["trend150"] * spec.trend_fast_sleeve
           + c["trend200"] * spec.trend_slow_sleeve
           + c["momentum9"] * spec.momentum_sleeve
           + c["minvar60"] * spec.minvar_sleeve)
    if (out < -1e-12).any().any():
        raise RuntimeError("blend generated a short target")
    gross = out.sum(axis=1)
    if (gross > 1 + 1e-10).any():
        raise RuntimeError("blend generated gross exposure above 100%")
    return out.clip(lower=0.0)


def _momentum_target_next_session(panel: pd.DataFrame, next_session: pd.Timestamp,
                                  spec: BlendSpec) -> pd.Series:
    held = momentum_execution_weights(panel, spec)
    current = held.iloc[-1].copy()
    last_ts = pd.Timestamp(panel.index[-1])
    nxt = pd.Timestamp(next_session)
    if nxt.tzinfo is None and last_ts.tzinfo is not None:
        nxt = nxt.tz_localize(last_ts.tz)
    elif nxt.tzinfo is not None and last_ts.tzinfo is not None:
        nxt = nxt.tz_convert(last_ts.tz)
    if nxt <= last_ts:
        raise ValueError("next_session must be after the latest close")
    if (nxt.year, nxt.month) == (last_ts.year, last_ts.month):
        return current

    lookback = spec.momentum_lookback_months * mr._DAYS_PER_MONTH
    skip = spec.momentum_skip_months * mr._DAYS_PER_MONTH
    scores = mr.momentum(panel, len(panel) - 1, lookback, skip)
    if scores is None:
        return pd.Series(0.0, index=panel.columns)
    return mr.select_weights(scores, spec.momentum_top_k)


def latest_target(daily_data: dict[str, pd.DataFrame], next_session,
                  spec: BlendSpec = DEFAULT_SPEC) -> pd.Series:
    """Target risky-asset weights decided at the latest close for `next_session`.

    The returned Series has a `cash_weight` attr.  No execution assumptions are
    made here; this is intentionally a pure decision function.
    """
    panel = build_panel(daily_data, spec)
    fast = trend_close_weights(panel, spec.trend_fast_ma,
                               spec.trend_fast_buffer).iloc[-1]
    slow = trend_close_weights(panel, spec.trend_slow_ma,
                               spec.trend_slow_buffer).iloc[-1]
    momentum = _momentum_target_next_session(panel, pd.Timestamp(next_session), spec)
    minvar = minvar_close_weights(panel, spec.minvar_lookback).iloc[-1]
    target = (fast * spec.trend_fast_sleeve
              + slow * spec.trend_slow_sleeve
              + momentum * spec.momentum_sleeve
              + minvar * spec.minvar_sleeve).clip(lower=0.0)
    gross = float(target.sum())
    if gross > 1 + 1e-10:
        raise RuntimeError(f"target gross exposure {gross:.6f} exceeds 100%")
    target.attrs["cash_weight"] = max(0.0, 1.0 - gross)
    target.attrs["decision_close"] = pd.Timestamp(panel.index[-1]).isoformat()
    target.attrs["next_session"] = pd.Timestamp(next_session).isoformat()
    return target


def gross_returns_from_close_targets(panel: pd.DataFrame,
                                     close_targets: pd.DataFrame) -> pd.Series:
    """Return gross (pre-cost) returns for targets executed for the next interval."""
    if not panel.index.equals(close_targets.index) or list(panel.columns) != list(close_targets.columns):
        raise ValueError("panel and targets must align exactly")
    held = close_targets.shift(1).fillna(0.0)
    return (held * panel.pct_change(fill_method=None).fillna(0.0)).sum(axis=1)
