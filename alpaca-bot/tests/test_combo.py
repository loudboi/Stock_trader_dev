"""Offline tests for the risk+trend combo research tool."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.combo as combo


def _daily_data(n=400, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-01", periods=n, freq="B", tz="UTC")
    out = {}
    for name in ("A", "B", "C"):
        close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
        out[name] = pd.DataFrame({"open": close, "high": close, "low": close,
                                  "close": close, "volume": np.ones(n)}, index=idx)
    return out


def test_compute_books_returns_rp_and_te():
    books = combo.compute_books(_daily_data())
    assert set(books) == {"rp", "te"}
    assert len(books["rp"]) == len(books["te"]) == 400
    assert books["rp"].iloc[0] == 0 and books["te"].iloc[0] == 0


def test_risk_strategy_selection_changes_only_risk_leg():
    data = _daily_data(seed=2)
    mv = combo.compute_books(data, "min_var")
    iv = combo.compute_books(data, "inverse_vol")
    erc = combo.compute_books(data, "erc")
    assert not np.allclose(mv["rp"], iv["rp"])
    assert not np.allclose(mv["rp"], erc["rp"])
    assert np.allclose(mv["te"], iv["te"])


def test_fixed_capital_mean_does_not_reallocate_missing_sleeve():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    a = pd.Series([0.1, 0.1, 0.1], index=idx)
    b = pd.Series([0.2, np.nan, 0.2], index=idx)
    out = combo._fixed_capital_mean({"a": a, "b": b})
    assert np.allclose(out.values, [0.15, 0.05, 0.15])


def test_blend_weight_extremes_match_pure_components():
    books = combo.compute_books(_daily_data(seed=3))
    all_rp = combo.combine_books(books, {"rp": 1.0, "te": 0.0})
    all_te = combo.combine_books(books, {"rp": 0.0, "te": 1.0})
    assert np.allclose(all_rp, books["rp"])
    assert np.allclose(all_te, books["te"])


def test_leverage_returns_scaling_and_financing():
    r = pd.Series([0.01, -0.02, 0.005])
    assert np.allclose(combo.leverage_returns(r, 1.0), r)
    lev = combo.leverage_returns(r, 2.0, 0.10)
    assert np.allclose(lev, r * 2 - 0.10 / combo._ANNUAL)
    with pytest.raises(ValueError):
        combo.leverage_returns(r, -1)


def test_print_score_runs_without_overclaim(capsys):
    data = _daily_data(seed=4)
    books = combo.compute_books(data)
    idx = next(iter(data.values())).index
    combo.print_score(data, books, 0.5, idx[100], idx[-1])
    out = capsys.readouterr().out
    assert "historical comparison" in out
    assert "pure trend_exposure" in out
    assert "proven" not in out.lower()


def test_consistency_output_is_not_called_true_walk_forward(capsys):
    data = _daily_data(n=650, seed=5)
    books = combo.compute_books(data)
    idx = next(iter(data.values())).index
    combo.print_walk(data, books, 0.5, idx[100], idx[-1], 3)
    out = capsys.readouterr().out
    assert "CHRONOLOGICAL CONSISTENCY" in out
    assert "legacy --mode walk alias; no fitting" in out
    assert "SUMMARY" in out


def test_invalid_compute_inputs_rejected():
    with pytest.raises(ValueError):
        combo.compute_books({"A": _daily_data()["A"]})
    with pytest.raises(ValueError):
        combo.compute_books(_daily_data(), rp_strategy="bad")
