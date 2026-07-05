"""
Offline tests for the EUR/USD currency-exposure model (bot/currency.py). No
network.

Run:  pytest tests/test_currency.py    (or: python tests/test_currency.py)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.currency as cur


def _series(vals, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(vals), freq="D")
    return pd.Series(vals, index=idx)


def test_unhedged_matches_usd_when_fx_is_flat():
    usd_ret = _series([0.01, 0.02, -0.01, 0.005])
    fx_flat = _series([1.10, 1.10, 1.10, 1.10])
    eur_eq = cur.unhedged_eur_equity(usd_ret, fx_flat, invested_frac=1.0, initial=100_000.0)
    usd_eq = 100_000.0 * (1 + usd_ret).cumprod()
    assert np.allclose(eur_eq.values, usd_eq.values)


def test_unhedged_eur_gains_when_usd_strengthens():
    # USD strengthening (EURUSD falling, i.e. fewer USD per EUR) means a fixed
    # USD amount converts to MORE EUR.
    usd_ret = _series([0.0, 0.0, 0.0])
    fx_falling = _series([1.20, 1.10, 1.00])   # USD strengthens vs EUR
    eur_eq = cur.unhedged_eur_equity(usd_ret, fx_falling, invested_frac=1.0, initial=100_000.0)
    assert eur_eq.iloc[-1] > eur_eq.iloc[0]


def test_unhedged_eur_loses_when_usd_weakens():
    usd_ret = _series([0.0, 0.0, 0.0])
    fx_rising = _series([1.00, 1.10, 1.20])    # USD weakens vs EUR
    eur_eq = cur.unhedged_eur_equity(usd_ret, fx_rising, invested_frac=1.0, initial=100_000.0)
    assert eur_eq.iloc[-1] < eur_eq.iloc[0]


def test_invested_frac_zero_means_no_fx_exposure():
    usd_ret = _series([0.0, 0.0, 0.0])
    fx_rising = _series([1.00, 1.10, 1.20])
    eur_eq = cur.unhedged_eur_equity(usd_ret, fx_rising, invested_frac=0.0, initial=100_000.0)
    assert np.allclose(eur_eq.values, 100_000.0)   # flat -- no USD return, no FX exposure either


def test_invested_frac_series_only_applies_fx_move_on_invested_days():
    usd_ret = _series([0.0, 0.0, 0.0, 0.0])
    fx = _series([1.00, 1.10, 1.10, 1.20])     # move happens on day index 1 and 3
    frac = _series([1.0, 1.0, 0.0, 0.0])       # invested days 0-1, flat/cash days 2-3
    eur_eq = cur.unhedged_eur_equity(usd_ret, fx, invested_frac=frac, initial=100_000.0)
    # Day 1's FX move (1.00->1.10, USD weakens) should hit while invested=1.0;
    # day 3's FX move (1.10->1.20) should NOT apply since invested=0.0 there.
    expected_after_day1 = 100_000.0 / (1.10 / 1.00)
    assert abs(eur_eq.iloc[1] - expected_after_day1) < 1e-6
    assert abs(eur_eq.iloc[-1] - eur_eq.iloc[1]) < 1e-6   # unchanged after that, flat/uninvested


def test_unhedged_empty_series():
    assert cur.unhedged_eur_equity(pd.Series(dtype=float), pd.Series(dtype=float)).empty


def test_hedged_equals_usd_equity_when_cost_is_zero():
    usd_ret = _series([0.01, -0.02, 0.03])
    hedged_eq = cur.hedged_eur_equity(usd_ret, annual_hedge_cost=0.0, initial=100_000.0)
    usd_eq = 100_000.0 * (1 + usd_ret).cumprod()
    assert np.allclose(hedged_eq.values, usd_eq.values)


def test_hedged_costs_less_than_unhedged_pure_usd_return_when_cost_is_positive():
    usd_ret = _series([0.0] * 252)     # flat local return for a year
    hedged_eq = cur.hedged_eur_equity(usd_ret, annual_hedge_cost=0.015, initial=100_000.0)
    assert hedged_eq.iloc[-1] < 100_000.0
    assert abs(hedged_eq.iloc[-1] - 100_000.0 * (1 - 0.015)) < 50.0   # ~1.5% annual drag


def test_hedged_empty_series():
    assert cur.hedged_eur_equity(pd.Series(dtype=float)).empty


def test_te_invested_fraction_is_mean_of_per_symbol_exposure():
    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    # Symbol A: rising -> above its MA -> invested most of the time.
    close_a = pd.Series(100 * 1.001 ** np.arange(300), index=idx)
    # Symbol B: falling -> below its MA -> flat/cash most of the time.
    close_b = pd.Series(100 * 0.999 ** np.arange(300), index=idx)
    daily_data = {"A": pd.DataFrame({"close": close_a}), "B": pd.DataFrame({"close": close_b})}
    frac = cur.te_invested_fraction(daily_data, ma_period=50, buffer=0.0)
    assert len(frac) == 300
    assert (frac >= 0.0).all() and (frac <= 1.0).all()
    # A trends up the whole time (mostly invested) -> mean should skew > 0
    # somewhere in the back half once the 50-day MA has enough history.
    assert frac.iloc[150:].mean() > 0.0


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        fn()
        print(f"{name} OK")
    print(f"\nALL {len(fns)} CURRENCY TESTS PASSED")
