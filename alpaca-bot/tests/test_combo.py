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
    assert "MIN_VAR+TE COMBO" in out and "pure min_var" in out and "pure trend_exposure" in out


def test_compute_books_rp_strategy_selection():
    daily_data = _daily_data(seed=6)
    mv_books = combo.compute_books(daily_data, rp_strategy="min_var")
    iv_books = combo.compute_books(daily_data, rp_strategy="inverse_vol")
    # Different constructions -> different (non-identical) return series.
    assert not np.allclose(mv_books["rp"].values, iv_books["rp"].values)
    # trend-exposure leg is unaffected by the rp_strategy choice.
    assert np.allclose(mv_books["te"].values, iv_books["te"].values)


def test_print_score_inverse_vol_label(capsys):
    daily_data = _daily_data(seed=7)
    books = combo.compute_books(daily_data, rp_strategy="inverse_vol")
    begin = list(daily_data.values())[0].index[100]
    end = list(daily_data.values())[0].index[-1]
    combo.print_score(daily_data, books, rp_weight=0.5, begin_ts=begin, end_ts=end,
                      rp_label="inverse_vol")
    out = capsys.readouterr().out
    assert "INVERSE_VOL+TE COMBO" in out and "pure inverse_vol" in out


def test_leverage_returns_unlevered_is_unchanged():
    r = pd.Series([0.01, -0.02, 0.005])
    assert np.allclose(combo.leverage_returns(r, 1.0, 0.06).values, r.values)


def test_leverage_returns_scales_and_charges_financing():
    r = pd.Series([0.01, -0.02, 0.005])
    levered = combo.leverage_returns(r, 2.0, borrow_rate=0.10)
    expected_financing = 1.0 * (0.10 / combo._ANNUAL)      # (leverage-1) * rate/252
    assert np.allclose(levered.values, r.values * 2.0 - expected_financing)


def test_leverage_returns_higher_borrow_costs_more():
    r = pd.Series(np.full(50, 0.001))
    cheap = combo.leverage_returns(r, 1.5, borrow_rate=0.02)
    expensive = combo.leverage_returns(r, 1.5, borrow_rate=0.20)
    assert (1 + expensive).prod() < (1 + cheap).prod()


def test_print_score_with_leverage_runs_without_error(capsys):
    daily_data = _daily_data(seed=5)
    books = combo.compute_books(daily_data)
    begin = list(daily_data.values())[0].index[100]
    end = list(daily_data.values())[0].index[-1]
    combo.print_score(daily_data, books, rp_weight=0.5, begin_ts=begin, end_ts=end,
                      leverage=1.5, borrow_rate=0.06)
    out = capsys.readouterr().out
    assert "Levered blend vs buy&hold" in out


def test_print_walk_runs_without_error(capsys):
    daily_data = _daily_data(n=600, seed=4)
    books = combo.compute_books(daily_data)
    idx = list(daily_data.values())[0].index
    combo.print_walk(daily_data, books, rp_weight=0.5, start_dt=idx[100], end_dt=idx[-1], folds=3)
    out = capsys.readouterr().out
    assert "WALK-FORWARD" in out and "beat pure min_var" in out


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        fn()
        print(f"{name} OK")
    print(f"\nALL {len(fns)} COMBO TESTS PASSED")
