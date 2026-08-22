"""Regression tests for the frozen volatility-managed QQQ target engine."""

from math import sqrt

import numpy as np
import pandas as pd
import pytest

from bot.vol_managed_targets import (
    DEFAULT_SPEC,
    VolManagedSpec,
    VolManagedTarget,
    lagged_realized_volatility,
    latest_target,
    should_rebalance,
    target_exposure_from_vol,
)


def _close(periods=80, start="2025-01-02"):
    idx = pd.bdate_range(start, periods=periods, tz="UTC")
    # Deterministic heterogeneous return sequence with non-zero realized volatility.
    r = 0.0006 + 0.012 * np.sin(np.arange(periods - 1) / 2.7) + 0.004 * np.cos(
        np.arange(periods - 1) / 5.1
    )
    values = 100.0 * np.cumprod(np.r_[1.0, 1.0 + r])
    return pd.Series(values, index=idx, name="QQQ")


def test_exposure_mapping_has_fixed_floor_cap_and_center():
    assert target_exposure_from_vol(0.25) == pytest.approx(1.0)
    assert target_exposure_from_vol(0.20) == pytest.approx(1.25)
    assert target_exposure_from_vol(0.05) == pytest.approx(DEFAULT_SPEC.max_exposure)
    assert target_exposure_from_vol(0.80) == pytest.approx(DEFAULT_SPEC.min_exposure)


def test_latest_volatility_matches_frozen_one_session_signal_lag():
    close = _close(50)
    actual = lagged_realized_volatility(close).iloc[-1]
    returns = close.pct_change(fill_method=None)
    # At decision close t, the frozen model excludes return[t] and uses the prior
    # 20 completed returns.  This locks live/paper parity to the research timing.
    expected = returns.iloc[-21:-1].std(ddof=1) * sqrt(DEFAULT_SPEC.annualization)
    assert actual == pytest.approx(expected, rel=0, abs=1e-14)


def test_future_prices_cannot_change_a_past_realized_volatility_signal():
    close = _close(90)
    cutoff = close.index[55]
    base = lagged_realized_volatility(close).loc[cutoff]
    altered = close.copy()
    # Change only observations after the signal being checked.
    altered.loc[altered.index > cutoff] *= np.linspace(
        0.45, 2.2, int((altered.index > cutoff).sum())
    )
    changed = lagged_realized_volatility(altered).loc[cutoff]
    assert changed == pytest.approx(base, rel=0, abs=1e-14)


def test_latest_target_requires_explicit_future_session_and_reports_monthly_due():
    close = _close(70)
    decision = close.index[-1]
    with pytest.raises(ValueError, match="next_session"):
        latest_target(close, decision)

    same_month = decision + pd.Timedelta(days=1)
    if same_month.month != decision.month:
        same_month = decision - pd.Timedelta(days=1)
        # Need a future timestamp while staying in the same month.
        same_month = decision + pd.Timedelta(hours=1)
    same = latest_target(close, same_month)
    assert same.monthly_rebalance_due is False

    next_month = decision + pd.offsets.MonthBegin(1)
    monthly = latest_target(close, next_month)
    assert monthly.monthly_rebalance_due is True
    assert monthly.symbol == "QQQ"
    assert DEFAULT_SPEC.min_exposure <= monthly.target_exposure <= DEFAULT_SPEC.max_exposure
    assert monthly.decision_close == decision


def test_no_trade_band_and_monthly_cadence_match_research_rule():
    decision = pd.Timestamp("2026-07-31", tz="UTC")
    nxt = pd.Timestamp("2026-08-03", tz="UTC")
    target = VolManagedTarget(
        symbol="QQQ",
        target_exposure=1.75,
        realized_vol=0.12,
        decision_close=decision,
        next_session=nxt,
        monthly_rebalance_due=True,
    )
    assert should_rebalance(1.60, target) is False  # 15 percentage points
    assert should_rebalance(1.55, target) is True   # exactly 20 points
    assert should_rebalance(1.45, target) is True

    intra_month = VolManagedTarget(
        symbol="QQQ",
        target_exposure=1.75,
        realized_vol=0.12,
        decision_close=pd.Timestamp("2026-08-10", tz="UTC"),
        next_session=pd.Timestamp("2026-08-11", tz="UTC"),
        monthly_rebalance_due=False,
    )
    assert should_rebalance(0.50, intra_month) is False


def test_insufficient_or_invalid_data_fail_closed():
    with pytest.raises(ValueError, match="insufficient valid history"):
        latest_target(_close(DEFAULT_SPEC.lookback + 1), pd.Timestamp("2025-03-03", tz="UTC"))

    bad = _close(40)
    bad.iloc[-3] = np.nan
    with pytest.raises(ValueError, match="finite positive"):
        latest_target(bad, bad.index[-1] + pd.Timedelta(days=1))

    with pytest.raises(ValueError, match="causality"):
        VolManagedSpec(signal_lag_sessions=0).validate()


def test_current_exposure_validation_and_symbol_mismatch_fail_closed():
    target = VolManagedTarget(
        symbol="QQQ",
        target_exposure=1.0,
        realized_vol=0.25,
        decision_close=pd.Timestamp("2026-07-31", tz="UTC"),
        next_session=pd.Timestamp("2026-08-03", tz="UTC"),
        monthly_rebalance_due=True,
    )
    with pytest.raises(ValueError, match="current_exposure"):
        should_rebalance(-0.1, target)
    with pytest.raises(ValueError, match="symbol"):
        should_rebalance(1.0, target, VolManagedSpec(symbol="SPY"))
