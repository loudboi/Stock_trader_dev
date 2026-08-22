"""Deterministic targets for the frozen volatility-managed QQQ strategy.

The research winner is an *active exposure-sizing* rule, not a static allocation:

* underlying: QQQ;
* decision cadence: monthly (the completed close immediately before the first
  trading session of a new month);
* volatility estimate: annualized sample standard deviation of the prior 20 daily
  total-return-adjusted QQQ returns;
* signal lag: one completed session inside the decision close, matching the frozen
  backtest exactly and preventing same-close information from entering the signal;
* target exposure: 25% annualized volatility divided by realized volatility;
* target clamp: 0.50x to 1.75x gross QQQ exposure;
* execution no-trade band: 0.20 exposure points.

This module deliberately contains no broker, borrowing, tax, or order-submission
code.  It only computes the research target and whether a monthly rebalance is large
enough to act on.  The input close series must already use the same total-return
corporate-action treatment as the research signal (Yahoo adjusted close in research;
an explicitly adjusted-all broker feed is required for paper parity work).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolManagedSpec:
    symbol: str = "QQQ"
    lookback: int = 20
    annualization: int = 252
    target_vol: float = 0.25
    min_exposure: float = 0.50
    max_exposure: float = 1.75
    no_trade_band: float = 0.20
    signal_lag_sessions: int = 1

    def validate(self) -> None:
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValueError("symbol must be a non-empty string")
        if self.lookback < 2:
            raise ValueError("lookback must be >= 2")
        if self.annualization <= 0:
            raise ValueError("annualization must be positive")
        if not np.isfinite(self.target_vol) or self.target_vol <= 0:
            raise ValueError("target_vol must be finite and positive")
        if not np.isfinite(self.min_exposure) or self.min_exposure < 0:
            raise ValueError("min_exposure must be finite and non-negative")
        if not np.isfinite(self.max_exposure) or self.max_exposure < self.min_exposure:
            raise ValueError("max_exposure must be finite and >= min_exposure")
        if self.max_exposure > 2.0:
            raise ValueError("max_exposure above 2.0x is outside the validated strategy")
        if not np.isfinite(self.no_trade_band) or not 0 <= self.no_trade_band <= 1:
            raise ValueError("no_trade_band must be within [0, 1]")
        if self.signal_lag_sessions < 1:
            raise ValueError("signal_lag_sessions must be >= 1 to preserve causality")


DEFAULT_SPEC = VolManagedSpec()


@dataclass(frozen=True)
class VolManagedTarget:
    symbol: str
    target_exposure: float
    realized_vol: float
    decision_close: pd.Timestamp
    next_session: pd.Timestamp
    monthly_rebalance_due: bool


def _validated_close(adjusted_close: pd.Series) -> pd.Series:
    if adjusted_close is None:
        raise ValueError("adjusted_close is required")
    close = pd.Series(adjusted_close, copy=True).astype(float)
    if close.empty:
        raise ValueError("adjusted_close cannot be empty")
    if close.index.has_duplicates:
        raise ValueError("adjusted_close index must be unique")
    if not close.index.is_monotonic_increasing:
        raise ValueError("adjusted_close index must be increasing")
    if (~np.isfinite(close)).any() or (close <= 0).any():
        raise ValueError("adjusted_close must contain finite positive prices")
    return close


def lagged_realized_volatility(
    adjusted_close: pd.Series,
    spec: VolManagedSpec = DEFAULT_SPEC,
) -> pd.Series:
    """Annualized realized volatility with the frozen research signal lag.

    At a decision close ``t``, the default one-session signal lag means the value at
    ``t`` uses returns ending at ``t-1``.  A target executed on the following session
    therefore cannot use the decision close's own return.  This matches the frozen
    research implementation rather than silently improving its timing after seeing
    the holdout.
    """
    spec.validate()
    close = _validated_close(adjusted_close)
    returns = close.pct_change(fill_method=None)
    raw = returns.rolling(spec.lookback, min_periods=spec.lookback).std(ddof=1)
    return raw.mul(sqrt(spec.annualization)).shift(spec.signal_lag_sessions)


def target_exposure_from_vol(
    realized_vol: float,
    spec: VolManagedSpec = DEFAULT_SPEC,
) -> float:
    """Map a positive realized-volatility estimate to the frozen gross exposure."""
    spec.validate()
    vol = float(realized_vol)
    if not np.isfinite(vol) or vol <= 0:
        raise ValueError("realized_vol must be finite and positive")
    raw = spec.target_vol / vol
    return float(np.clip(raw, spec.min_exposure, spec.max_exposure))


def _aligned_next_session(decision_close: pd.Timestamp, next_session) -> pd.Timestamp:
    decision = pd.Timestamp(decision_close)
    nxt = pd.Timestamp(next_session)
    if decision.tzinfo is not None and nxt.tzinfo is None:
        nxt = nxt.tz_localize(decision.tzinfo)
    elif decision.tzinfo is not None and nxt.tzinfo is not None:
        nxt = nxt.tz_convert(decision.tzinfo)
    elif decision.tzinfo is None and nxt.tzinfo is not None:
        raise ValueError("next_session timezone must match the timezone-naive close index")
    if nxt <= decision:
        raise ValueError("next_session must be after the latest completed close")
    return nxt


def latest_target(
    adjusted_close: pd.Series,
    next_session,
    spec: VolManagedSpec = DEFAULT_SPEC,
) -> VolManagedTarget:
    """Compute the frozen target from the latest completed close history.

    ``monthly_rebalance_due`` is true only when ``next_session`` is in a different
    calendar month from the latest completed close.  Callers must not manufacture a
    business-day approximation for ``next_session``; a broker/exchange calendar is
    required at the execution layer.
    """
    spec.validate()
    close = _validated_close(adjusted_close)
    decision = pd.Timestamp(close.index[-1])
    nxt = _aligned_next_session(decision, next_session)
    vol_series = lagged_realized_volatility(close, spec)
    realized = float(vol_series.iloc[-1])
    if not np.isfinite(realized) or realized <= 0:
        need = spec.lookback + spec.signal_lag_sessions + 1
        raise ValueError(
            f"insufficient valid history for realized volatility; need at least {need} closes"
        )
    target = target_exposure_from_vol(realized, spec)
    monthly_due = (nxt.year, nxt.month) != (decision.year, decision.month)
    return VolManagedTarget(
        symbol=spec.symbol,
        target_exposure=target,
        realized_vol=realized,
        decision_close=decision,
        next_session=nxt,
        monthly_rebalance_due=monthly_due,
    )


def should_rebalance(
    current_exposure: float,
    target: VolManagedTarget,
    spec: VolManagedSpec = DEFAULT_SPEC,
) -> bool:
    """Apply the frozen monthly cadence and 20-point no-trade band.

    The max exposure is a *target* cap.  As in the research simulation, normal price
    drift can temporarily leave actual exposure slightly outside the target range;
    this function does not introduce an untested intramonth hard-rebalance rule.
    Margin and broker safety controls belong in a separate execution/risk layer.
    """
    spec.validate()
    current = float(current_exposure)
    if not np.isfinite(current) or current < 0:
        raise ValueError("current_exposure must be finite and non-negative")
    if not isinstance(target, VolManagedTarget):
        raise TypeError("target must be a VolManagedTarget")
    if target.symbol != spec.symbol:
        raise ValueError("target symbol does not match spec")
    if not target.monthly_rebalance_due:
        return False
    return abs(current - target.target_exposure) >= spec.no_trade_band - 1e-12
