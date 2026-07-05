"""
Offline tests for the RP+TE combo tool (bot/combo.py). Synthetic data, no
network.

Run:  pytest tests/test_combo.py    (or: python tests/test_combo.py)
"""
import os
import sys

import numpy as np
import pandas as pd

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
    daily_data = _daily_data()
    books = combo.compute_books(daily_data)
    assert set(books) == {"rp", "te"}
    assert len(books["rp"]) == len(books["te"]) == 400
    assert books["rp"].iloc[0] == 0.0 and books["te"].iloc[0] == 0.0   # no lookahead


def test_blend_weight_extremes_match_pure_components():
    daily_data = _daily_data(seed=2)
    books = combo.compute_books(daily_data)
    all_rp = combo.combine_books(books, weights={"rp": 1.0, "te": 0.0})
    all_te = combo.combine_books(books, weights={"rp": 0.0, "te": 1.0})
    assert np.allclose(all_rp.values, books["rp"].values)
    assert np.allclose(all_te.values, books["te"].values)


def test_print_score_runs_without_error(capsys):
    daily_data = _daily_data(seed=3)
    books = combo.compute_books(daily_data)
    begin = list(daily_data.values())[0].index[100]
    end = list(daily_data.values())[0].index[-1]
    combo.print_score(daily_data, books, rp_weight=0.5, begin_ts=begin, end_ts=end)
    out = capsys.readouterr().out
    assert "RP+TE COMBO" in out and "pure risk_parity" in out and "pure trend_exposure" in out


def test_print_walk_runs_without_error(capsys):
    daily_data = _daily_data(n=600, seed=4)
    books = combo.compute_books(daily_data)
    idx = list(daily_data.values())[0].index
    combo.print_walk(daily_data, books, rp_weight=0.5, start_dt=idx[100], end_dt=idx[-1], folds=3)
    out = capsys.readouterr().out
    assert "WALK-FORWARD" in out and "Blend beat pure RP" in out


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        fn()
        print(f"{name} OK")
    print(f"\nALL {len(fns)} COMBO TESTS PASSED")
