"""Regression tests for the fixed profitability-blend target-weight engine."""

import numpy as np
import pandas as pd
import pytest

from alpaca.data.enums import Adjustment, DataFeed

import bot.lab as lab
import bot.momentum_rotation as mr
import bot.trend_exposure as te
from bot.blend_market_data import adjusted_daily_request
from bot.blend_targets import (
    DEFAULT_SPEC,
    build_panel,
    combined_close_weights,
    gross_returns_from_close_targets,
    latest_target,
    minvar_close_weights,
)


def _daily_data(periods=420):
    idx = pd.bdate_range("2023-01-03", periods=periods, tz="UTC")
    t = np.arange(periods, dtype=float)
    out = {}
    for j, symbol in enumerate(DEFAULT_SPEC.symbols):
        # Deterministic but heterogeneous paths: positive drift, cycles and several
        # different volatility scales make momentum/min-var choices non-trivial.
        close = (80 + 7 * j) * np.exp(
            (0.00015 + j * 0.000025) * t
            + (0.018 + j * 0.002) * np.sin(t / (13 + j))
            + 0.006 * np.cos(t / (5 + 2 * j))
        )
        out[symbol] = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.002,
                "low": close * 0.998,
                "close": close,
                "volume": np.full(periods, 1_000_000.0 + j),
            },
            index=idx,
        )
    return out


def test_combined_targets_are_long_only_and_never_exceed_one():
    panel = build_panel(_daily_data())
    w = combined_close_weights(panel)
    assert (w >= -1e-12).all().all()
    assert (w.sum(axis=1) <= 1.0 + 1e-10).all()
    assert list(w.columns) == list(DEFAULT_SPEC.symbols)


def test_latest_target_reports_cash_and_requires_future_session():
    data = _daily_data()
    panel = build_panel(data)
    next_session = panel.index[-1] + pd.offsets.BDay(1)
    target = latest_target(data, next_session)
    assert 0.0 <= target.attrs["cash_weight"] <= 1.0
    assert abs(float(target.sum()) + target.attrs["cash_weight"] - 1.0) < 1e-10
    with pytest.raises(ValueError, match="next_session"):
        latest_target(data, panel.index[-1])


def test_minvar_prior_target_cannot_change_when_future_prices_change():
    data = _daily_data()
    panel = build_panel(data)
    base = minvar_close_weights(panel, DEFAULT_SPEC.minvar_lookback)
    cutoff = 300
    altered = panel.copy()
    # Modify only observations strictly after the comparison row.
    altered.iloc[cutoff + 1:] *= np.linspace(0.5, 1.8, len(altered) - cutoff - 1)[:, None]
    changed = minvar_close_weights(altered, DEFAULT_SPEC.minvar_lookback)
    pd.testing.assert_series_equal(base.iloc[cutoff], changed.iloc[cutoff])


def test_blend_gross_returns_match_fixed_research_sleeves_without_costs():
    data = _daily_data()
    panel = build_panel(data)
    actual = gross_returns_from_close_targets(panel, combined_close_weights(panel))

    old_te, old_mr, old_lab = te.SLIPPAGE, mr.SLIPPAGE, lab.SLIPPAGE
    try:
        te.SLIPPAGE = mr.SLIPPAGE = lab.SLIPPAGE = 0.0
        fast = pd.concat(
            [te.strategy_returns(data[s], DEFAULT_SPEC.trend_fast_ma,
                                 DEFAULT_SPEC.trend_fast_buffer, 1.0, 0.0)
             for s in DEFAULT_SPEC.symbols], axis=1).fillna(0.0).mean(axis=1)
        slow = pd.concat(
            [te.strategy_returns(data[s], DEFAULT_SPEC.trend_slow_ma,
                                 DEFAULT_SPEC.trend_slow_buffer, 1.0, 0.0)
             for s in DEFAULT_SPEC.symbols], axis=1).fillna(0.0).mean(axis=1)
        momentum_w = mr.weight_panel(panel, DEFAULT_SPEC.momentum_lookback_months,
                                     DEFAULT_SPEC.momentum_skip_months,
                                     DEFAULT_SPEC.momentum_top_k)
        momentum = mr.portfolio_returns(panel, momentum_w)
        minimum_variance = lab.min_var(panel, DEFAULT_SPEC.minvar_lookback)
        expected = (fast * DEFAULT_SPEC.trend_fast_sleeve
                    + slow * DEFAULT_SPEC.trend_slow_sleeve
                    + momentum * DEFAULT_SPEC.momentum_sleeve
                    + minimum_variance * DEFAULT_SPEC.minvar_sleeve)
    finally:
        te.SLIPPAGE, mr.SLIPPAGE, lab.SLIPPAGE = old_te, old_mr, old_lab

    expected = expected.reindex(actual.index).fillna(0.0)
    assert np.max(np.abs(actual.values - expected.values)) < 1e-12


def test_alpaca_blend_history_request_is_explicitly_adjusted_all():
    req = adjusted_daily_request(
        "SPY", pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2026-01-01", tz="UTC"), feed=DataFeed.IEX)
    assert req.adjustment == Adjustment.ALL
    assert req.feed == DataFeed.IEX
