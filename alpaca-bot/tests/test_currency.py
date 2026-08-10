"""Offline tests for EUR/USD conversion helpers."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.currency as cur


def _series(vals, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(vals), freq="D")
    return pd.Series(vals, index=idx)


def test_unhedged_matches_usd_when_fx_is_flat():
    usd_ret = _series([0.01, 0.02, -0.01, 0.005])
    fx = _series([1.10] * 4)
    eur = cur.unhedged_eur_equity(usd_ret, fx)
    assert np.allclose(eur.values, 100_000 * (1 + usd_ret).cumprod().values)


def test_full_usd_exposure_uses_exact_fx_ratio():
    usd_ret = _series([0.0, 0.10, 0.0])
    fx = _series([1.20, 1.10, 1.00])
    eur = cur.unhedged_eur_equity(usd_ret, fx, invested_frac=1.0)
    expected_factors = pd.Series([
        1.0,
        1.10 / (1.10 / 1.20),
        1.0 / (1.00 / 1.10),
    ], index=usd_ret.index)
    assert np.allclose(eur.values, 100_000 * expected_factors.cumprod().values)


def test_partial_exposure_is_exact_mixed_eur_cash_usd_asset_wealth():
    usd_ret = _series([0.0, 0.05, 0.0])
    fx = _series([1.20, 1.10, 1.10])
    frac = _series([0.5, 0.5, 0.5])
    eur = cur.unhedged_eur_equity(usd_ret, fx, invested_frac=frac)
    day1_factor = 0.5 + (0.5 + 0.05) / (1.10 / 1.20)
    assert abs(eur.iloc[1] - 100_000 * day1_factor) < 1e-6


def test_zero_exposure_has_no_fx_move_without_usd_return():
    usd_ret = _series([0.0, 0.0, 0.0])
    fx = _series([1.0, 1.1, 1.2])
    eq = cur.unhedged_eur_equity(usd_ret, fx, invested_frac=0.0)
    assert np.allclose(eq.values, 100_000)


def test_zero_exposure_preserves_usd_transaction_cost_contribution():
    # Example exit row: no remaining market exposure, but a 10bp USD transaction
    # cost must still reduce EUR wealth instead of being discarded or rejected.
    usd_ret = _series([0.0, -0.001, 0.0])
    fx = _series([1.0, 1.1, 1.1])
    eq = cur.unhedged_eur_equity(usd_ret, fx, invested_frac=0.0)
    expected_day1_factor = 1.0 + (-0.001) / 1.1
    assert abs(eq.iloc[1] - 100_000 * expected_day1_factor) < 1e-6


def test_fx_is_forward_filled_but_never_backfilled_from_future():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    r = pd.Series(0.0, index=idx)
    fx_late = pd.Series([1.1, 1.2], index=idx[2:])
    with pytest.raises(ValueError, match="refusing to backfill"):
        cur.unhedged_eur_equity(r, fx_late)
    fx_sparse = pd.Series([1.0, 1.2], index=[idx[0], idx[3]])
    aligned = cur.align_fx_causally(fx_sparse, idx)
    assert aligned.tolist() == [1.0, 1.0, 1.0, 1.2]


def test_hedge_cost_is_charged_only_on_exposed_fraction():
    r = _series([0.0] * 252)
    full = cur.hedged_eur_equity(r, annual_hedge_cost=0.02, invested_frac=1.0)
    half = cur.hedged_eur_equity(r, annual_hedge_cost=0.02, invested_frac=0.5)
    cash = cur.hedged_eur_equity(r, annual_hedge_cost=0.02, invested_frac=0.0)
    assert full.iloc[-1] < half.iloc[-1] < cash.iloc[-1]
    assert np.allclose(cash.values, 100_000)


def test_invalid_fraction_cost_and_initial_rejected():
    r = _series([0.0, 0.0])
    fx = _series([1.1, 1.1])
    with pytest.raises(ValueError):
        cur.unhedged_eur_equity(r, fx, invested_frac=1.2)
    with pytest.raises(ValueError):
        cur.hedged_eur_equity(r, annual_hedge_cost=-0.01)
    with pytest.raises(ValueError):
        cur.unhedged_eur_equity(r, fx, initial=0)


def test_empty_series():
    assert cur.unhedged_eur_equity(pd.Series(dtype=float), pd.Series(dtype=float)).empty
    assert cur.hedged_eur_equity(pd.Series(dtype=float)).empty


def test_te_invested_fraction_is_fixed_capital_mean():
    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    up = pd.Series(100 * 1.001 ** np.arange(300), index=idx)
    down = pd.Series(100 * 0.999 ** np.arange(300), index=idx)
    data = {"UP": pd.DataFrame({"close": up}), "DOWN": pd.DataFrame({"close": down})}
    frac = cur.te_invested_fraction(data, ma_period=50, buffer=0.0)
    assert len(frac) == 300
    assert ((frac >= 0) & (frac <= 1)).all()
    assert frac.iloc[150:].mean() > 0
