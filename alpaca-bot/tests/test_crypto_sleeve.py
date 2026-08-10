"""Offline tests for the crypto-sleeve research tool."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.combo as combo
import bot.crypto_sleeve as cs
import bot.taxes as tx


def _daily_data(n=900, seed=1, names=("A", "B", "C", "D")):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    out = {}
    for name in names:
        close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
        out[name] = pd.DataFrame({"open": close, "high": close, "low": close,
                                  "close": close, "volume": np.ones(n)}, index=idx)
    return out


def _btc_daily(n=900, seed=9, start="2015-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC")
    close = 100 * np.cumprod(1 + rng.normal(0.0008, 0.04, n))
    return pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": np.ones(n)}, index=idx)


def test_compute_btc_sleeve_is_lagged_return_series():
    btc = _btc_daily()
    sleeve = cs.compute_btc_sleeve(btc)
    assert len(sleeve) == len(btc) and sleeve.iloc[0] == 0.0
    with pytest.raises(ValueError):
        cs.compute_btc_sleeve(btc, ma_period=1)


def test_score_uses_common_overlap_and_reports_historical_language(capsys):
    data = _daily_data(seed=2)
    books = combo.compute_books(data)
    btc = _btc_daily(n=500, seed=3, start="2016-01-01")
    idx = next(iter(data.values())).index
    cs.print_score(data, books, btc, idx[0], idx[-1])
    out = capsys.readouterr().out
    assert "historical comparison" in out
    assert "Correlation(blend, trend-filtered BTC)" in out
    assert str(btc.index[0].date()) in out
    assert "blend + 10% BTC" in out


def test_consistency_output_is_not_presented_as_true_walk_forward(capsys):
    data = _daily_data(n=1200, seed=4)
    books = combo.compute_books(data)
    btc = _btc_daily(n=1200, seed=5)
    idx = next(iter(data.values())).index
    cs.print_walk(data, books, btc, idx[250], idx[-1], 3, 0.10)
    out = capsys.readouterr().out
    assert "CHRONOLOGICAL CONSISTENCY" in out
    assert "legacy walk alias" in out
    assert "SUMMARY" in out


def test_current_and_proposal_crypto_rates_are_distinct():
    assert tx.CRYPTO_TAX_RATE == 0.0
    assert tx.CRYPTO_PROPOSED_TAX_RATE == 0.25


def test_aftertax_output_explicitly_marks_proposal_not_law(capsys):
    data = _daily_data(n=1200, seed=6)
    books = combo.compute_books(data)
    btc = _btc_daily(n=1200, seed=7)
    idx = next(iter(data.values())).index
    cs.print_aftertax(data, books, btc, idx[250], idx[-1])
    out = capsys.readouterr().out
    assert "CRYPTO SLEEVE TAX SCENARIOS" in out
    assert "PROPOSAL/HYPOTHETICAL" in out
    assert "not treated as enacted law" in out
    assert "grandfather" not in out.lower()
    assert "blend + 10% BTC" in out


def test_proposal_tax_scenario_cannot_improve_crypto_only_wealth():
    btc = cs.compute_btc_sleeve(_btc_daily(n=1200, seed=8)).iloc[250:]
    current = cs._crypto_after_tax(btc, 1.0, tx.CRYPTO_TAX_RATE)
    proposed = cs._crypto_after_tax(btc, 1.0, tx.CRYPTO_PROPOSED_TAX_RATE)
    assert proposed.iloc[-1] <= current.iloc[-1] + 1e-9


def test_invalid_consistency_inputs_rejected():
    data = _daily_data(n=600)
    books = combo.compute_books(data)
    btc = _btc_daily(n=600)
    idx = next(iter(data.values())).index
    with pytest.raises(ValueError):
        cs.print_walk(data, books, btc, idx[250], idx[-1], folds=0)
    with pytest.raises(ValueError):
        cs.print_walk(data, books, btc, idx[250], idx[-1], folds=3, btc_weight=2.0)
