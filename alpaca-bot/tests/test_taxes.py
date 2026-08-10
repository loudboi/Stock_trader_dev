"""Offline tests for the Slovenian tax-scenario helpers."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.taxes as tx


def _returns(start, end, growth):
    idx = pd.date_range(start, end, freq="D")
    daily = growth ** (1 / len(idx)) - 1
    return pd.Series(daily, index=idx)


def test_current_crypto_constant_is_not_proposed_tax_rate():
    assert tx.CRYPTO_TAX_RATE == 0.0
    assert tx.CRYPTO_PROPOSED_TAX_RATE == 0.25


def test_completed_years_uses_calendar_anniversary():
    assert tx.completed_years("2020-02-29", "2025-02-28") == 4
    assert tx.completed_years("2020-02-29", "2025-03-01") == 5
    assert tx.completed_years("2010-06-15", "2025-06-14") == 14
    assert tx.completed_years("2010-06-15", "2025-06-15") == 15


def test_date_based_security_rates():
    a = pd.Timestamp("2020-01-01")
    assert tx.slovenia_rate_for_dates(a, "2024-12-31") == 0.25
    assert tx.slovenia_rate_for_dates(a, "2025-01-01") == 0.20
    assert tx.slovenia_rate_for_dates(a, "2030-01-01") == 0.15
    assert tx.slovenia_rate_for_dates(a, "2035-01-01") == 0.0


def test_duration_compatibility_helper_is_not_exact_365_cliff():
    assert tx.slovenia_rate_for_holding(1) == 0.25
    assert tx.slovenia_rate_for_holding(6 * 365) == 0.20
    assert tx.slovenia_rate_for_holding(11 * 365) == 0.15
    assert tx.slovenia_rate_for_holding(16 * 365) == 0.0
    with pytest.raises(ValueError):
        tx.slovenia_rate_for_holding(-1)


def test_after_tax_active_taxes_completed_positive_year():
    r = _returns("2020-01-01", "2020-12-31", 1.10)
    eq = tx.after_tax_active(r, tax_rate=0.25)
    assert abs(eq.iloc[-1] - 107_500.0) < 2.0


def test_after_tax_active_does_not_carry_ordinary_loss_to_next_year():
    r1 = _returns("2020-01-01", "2020-12-31", 0.90)
    r2 = _returns("2021-01-01", "2021-12-31", 1.20)
    eq = tx.after_tax_active(pd.concat([r1, r2]), tax_rate=0.25)
    expected = 90_000 * 1.20 - 0.25 * (90_000 * 0.20)
    assert abs(eq.iloc[-1] - expected) < 2.0


def test_after_tax_active_incomplete_final_year_is_not_forced_realized():
    r = _returns("2024-01-01", "2024-06-30", 1.10)
    eq = tx.after_tax_active(r, tax_rate=0.25)
    assert abs(eq.iloc[-1] - 110_000.0) < 2.0
    eq_liq = tx.after_tax_active(r, tax_rate=0.25, realize_final=True)
    assert abs(eq_liq.iloc[-1] - 107_500.0) < 2.0


def test_buy_hold_uses_calendar_holding_period_and_final_sale_only():
    r = _returns("2020-02-29", "2025-02-28", 1.50)
    eq = tx.after_tax_buy_hold(r)
    assert abs(eq.iloc[-1] - (150_000 - 0.25 * 50_000)) < 2.0
    r2 = _returns("2020-02-29", "2025-03-01", 1.50)
    eq2 = tx.after_tax_buy_hold(r2)
    assert abs(eq2.iloc[-1] - (150_000 - 0.20 * 50_000)) < 2.0


def test_buy_hold_can_leave_final_position_unrealized():
    r = _returns("2020-01-01", "2024-01-01", 1.50)
    eq = tx.after_tax_buy_hold(r, realize_final=False)
    assert abs(eq.iloc[-1] - 150_000.0) < 2.0


def test_buy_hold_exempt_after_fifteen_completed_years():
    r = _returns("2005-01-01", "2020-01-01", 2.0)
    eq = tx.after_tax_buy_hold(r)
    assert abs(eq.iloc[-1] - 200_000.0) < 3.0


def test_exposure_based_taxes_only_on_actual_exit():
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    close = pd.Series(np.linspace(100, 150, 20), index=idx)
    held = pd.Series([0.0] * 2 + [1.0] * 10 + [0.0] * 8, index=idx)
    eq = tx.after_tax_exposure_based(close, held)
    assert eq.iloc[11] > eq.iloc[12]
    assert eq.iloc[-1] == eq.iloc[12]


def test_exposure_based_open_final_trade_remains_unrealized():
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    close = pd.Series(100 * 1.001 ** np.arange(100), index=idx)
    held = pd.Series(1.0, index=idx)
    eq = tx.after_tax_exposure_based(close, held)
    expected = 100_000 * close / close.iloc[0]
    assert np.allclose(eq.values, expected.values)


def test_core_satellite_extremes_match_component_models():
    idx = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    core = pd.Series(0.0001, index=idx)
    sat = pd.Series(0.0002, index=idx)
    all_core = tx.after_tax_core_satellite(core, sat, 1.0)
    all_sat = tx.after_tax_core_satellite(core, sat, 0.0)
    assert np.allclose(all_core.values, tx.after_tax_buy_hold(core).values)
    assert np.allclose(all_sat.values, tx.after_tax_active(sat).values)


def test_cross_offset_is_conservative_noop_without_tax_lots(caplog):
    idx = pd.date_range("2020-01-01", "2021-12-31", freq="D")
    core = pd.Series(0.0001, index=idx)
    sat = pd.Series(-0.0001, index=idx)
    a = tx.after_tax_core_satellite(core, sat, 0.7, cross_offset_losses=False)
    b = tx.after_tax_core_satellite(core, sat, 0.7, cross_offset_losses=True)
    assert np.allclose(a.values, b.values)
    assert "ignored" in caplog.text


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        tx.after_tax_active(pd.Series([0.1]), tax_rate=1.5)
    with pytest.raises(ValueError):
        tx.after_tax_core_satellite(pd.Series([0.0]), pd.Series([0.0]), 1.2)
