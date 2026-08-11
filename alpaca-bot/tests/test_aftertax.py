"""Offline integration tests for after-tax research scenarios."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.aftertax as aftertax
import bot.combo as combo
import bot.currency as cur


def _daily_data(n=1800, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-01", periods=n, freq="B", tz="UTC")
    out = {}
    for name in ("A", "B", "C", "D"):
        close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
        out[name] = pd.DataFrame({"open": close, "high": close, "low": close,
                                  "close": close, "volume": np.ones(n)}, index=idx)
    return out


def test_apply_currency_usd_is_noop():
    r = pd.Series([0.01, -0.02, 0.03])
    assert aftertax._apply_currency(r, "blend", "usd") is r


def test_apply_currency_smart_and_hedged_use_exposure_fraction():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    frac = pd.Series(np.linspace(0.2, 1.0, 10), index=idx)
    r = pd.Series(0.001 * frac, index=idx)
    fx = pd.Series(np.linspace(1.20, 1.10, 10), index=idx)
    smart = aftertax._apply_currency(r, "te", "eur_smart", fx_close=fx, te_frac=frac)
    expected = cur.unhedged_eur_equity(r, fx, invested_frac=frac).pct_change().fillna(0.0)
    assert np.allclose(smart, expected)
    hedged = aftertax._apply_currency(r, "te", "eur_hedged", te_frac=frac,
                                      hedge_cost=0.02)
    expected_h = cur.hedged_eur_equity(r, 0.02, invested_frac=frac).pct_change().fillna(0.0)
    assert np.allclose(hedged, expected_h)


def test_te_after_tax_uses_exit_timed_binary_model():
    data = _daily_data(n=800, seed=2)
    idx = next(iter(data.values())).index
    eq = aftertax._te_after_tax_equity(data, idx[250], idx[-1])
    assert len(eq) > 0
    assert eq.attrs["initial_equity"] == aftertax.INITIAL_EQUITY


def test_print_score_runs_and_labels_scenario_limits(capsys):
    data = _daily_data(seed=3)
    books = combo.compute_books(data)
    idx = next(iter(data.values())).index
    aftertax.print_score(data, books, idx[250], idx[-1])
    out = capsys.readouterr().out
    assert "AFTER-TAX SCENARIO COMPARISON" in out
    assert "not tax-lot accounting" in out
    assert "trend exposure" in out
    assert "CORE-SATELLITE STRESS GRID" in out


def test_print_score_eur_smart_runs_with_causal_fx(capsys):
    data = _daily_data(seed=4)
    books = combo.compute_books(data)
    idx = next(iter(data.values())).index
    fx = pd.Series(np.linspace(1.25, 1.05, len(idx)), index=idx)
    frac = cur.te_invested_fraction(data, combo.TE_MA_PERIOD, combo.TE_BUFFER)
    aftertax.print_score(data, books, idx[250], idx[-1], currency="eur_smart",
                         fx_close=fx, te_frac=frac)
    out = capsys.readouterr().out
    assert "currency=eur_smart" in out


def test_print_checkpoints_uses_calendar_offsets_and_no_cross_offset_claim(capsys):
    data = _daily_data(n=5000, seed=5)
    books = combo.compute_books(data)
    idx = next(iter(data.values())).index
    wins, checks = aftertax.print_checkpoints(data, books, idx[0], 0.7,
                                              checkpoint_years=(5, 10, 15))
    out = capsys.readouterr().out
    assert "CORE-SATELLITE CHECKPOINTS" in out and "SUMMARY" in out
    assert "cross" not in out.lower()
    assert 0 <= wins <= checks


def test_invalid_score_tax_rate_rejected():
    data = _daily_data(n=500)
    books = combo.compute_books(data)
    idx = next(iter(data.values())).index
    with pytest.raises(ValueError):
        aftertax.print_score(data, books, idx[250], idx[-1], tax_rate=2.0)



def test_te_after_tax_staggered_inception_does_not_create_partial_capital(monkeypatch):
    idx_a = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    idx_b = pd.date_range("2020-01-03", periods=3, freq="D", tz="UTC")
    data = {
        "A": pd.DataFrame({"close": 100.0}, index=idx_a),
        "B": pd.DataFrame({"close": 100.0}, index=idx_b),
    }
    monkeypatch.setattr(
        aftertax, "_te_held_by_symbol",
        lambda d: {name: pd.Series(1.0, index=df.index) for name, df in d.items()})
    monkeypatch.setattr(
        aftertax.te, "strategy_returns",
        lambda daily, *args: pd.Series(0.0, index=daily.index))
    monkeypatch.setattr(
        aftertax.tx, "after_tax_binary_strategy",
        lambda returns, held, initial=100_000.0, **kwargs:
            pd.Series(float(initial), index=returns.index))

    eq = aftertax._te_after_tax_equity(data, idx_a[0], idx_a[-1], initial=100_000.0)
    assert eq.index[0] == idx_b[0]
    assert np.allclose(eq.values, 100_000.0)
    assert eq.attrs["initial_equity"] == 100_000.0
