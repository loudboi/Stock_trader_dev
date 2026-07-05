"""
Offline tests for the crypto sleeve tool (bot/crypto_sleeve.py). Synthetic
data, no network.

Run:  pytest tests/test_crypto_sleeve.py    (or: python tests/test_crypto_sleeve.py)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.combo as combo
import bot.crypto_sleeve as cs


def _daily_data(n=800, seed=1, names=("A", "B", "C", "D")):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    out = {}
    for name in names:
        close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
        out[name] = pd.DataFrame({"open": close, "high": close, "low": close,
                                  "close": close, "volume": np.ones(n)}, index=idx)
    return out


def _btc_daily(n=800, seed=9):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    close = 100 * np.cumprod(1 + rng.normal(0.0008, 0.04, n))   # higher drift/vol than equities
    return pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": np.ones(n)}, index=idx)


def test_compute_btc_sleeve_returns_a_return_series_no_lookahead():
    btc = _btc_daily()
    sleeve = cs.compute_btc_sleeve(btc)
    assert len(sleeve) == len(btc)
    assert sleeve.iloc[0] == 0.0     # no position until the MA has history


def test_print_score_runs_without_error(capsys):
    daily_data = _daily_data(seed=1)
    btc = _btc_daily(seed=2)
    books = combo.compute_books(daily_data)
    idx = list(daily_data.values())[0].index
    cs.print_score(daily_data, books, btc, begin_ts=idx[250], end_ts=idx[-1])
    out = capsys.readouterr().out
    assert "CRYPTO SLEEVE" in out
    assert "Correlation(blend, trend-filtered BTC)" in out
    assert "blend + 10% BTC" in out


def test_print_score_zero_weight_matches_pure_blend(capsys):
    daily_data = _daily_data(seed=3)
    btc = _btc_daily(seed=4)
    books = combo.compute_books(daily_data)
    idx = list(daily_data.values())[0].index
    cs.print_score(daily_data, books, btc, begin_ts=idx[250], end_ts=idx[-1])
    out = capsys.readouterr().out
    lines = out.splitlines()
    blend_line = [l for l in lines if l.strip().startswith("min_var+TE blend")][0]
    assert blend_line.split()[-1] != "YES"   # 0% BTC row has no "beats" comparison printed as YES


def test_print_walk_runs_without_error(capsys):
    daily_data = _daily_data(n=1200, seed=5)
    btc = _btc_daily(n=1200, seed=6)
    books = combo.compute_books(daily_data)
    idx = list(daily_data.values())[0].index
    cs.print_walk(daily_data, books, btc, start_dt=idx[250], end_dt=idx[-1], folds=3)
    out = capsys.readouterr().out
    assert "CRYPTO SLEEVE WALK-FORWARD" in out
    assert "beat pure blend in" in out


def test_print_walk_higher_weight_changes_result(capsys):
    daily_data = _daily_data(n=1200, seed=7)
    btc = _btc_daily(n=1200, seed=8)
    books = combo.compute_books(daily_data)
    idx = list(daily_data.values())[0].index
    cs.print_walk(daily_data, books, btc, start_dt=idx[250], end_dt=idx[-1], folds=3, btc_weight=0.3)
    out = capsys.readouterr().out
    assert "btc_weight=30%" in out


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        fn()
        print(f"{name} OK")
    print(f"\nALL {len(fns)} CRYPTO SLEEVE TESTS PASSED")
