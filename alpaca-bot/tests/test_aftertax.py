"""
Offline tests for the after-tax comparison tool (bot/aftertax.py). Synthetic
data, no network.

Run:  pytest tests/test_aftertax.py    (or: python tests/test_aftertax.py)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.combo as combo
import bot.aftertax as aftertax


def _daily_data(n=6000, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-01", periods=n, freq="B", tz="UTC")
    out = {}
    for name in ("A", "B", "C", "D"):
        close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
        out[name] = pd.DataFrame({"open": close, "high": close, "low": close,
                                  "close": close, "volume": np.ones(n)}, index=idx)
    return out


def test_print_score_runs_without_error(capsys):
    daily_data = _daily_data(seed=1)
    books = combo.compute_books(daily_data)
    idx = list(daily_data.values())[0].index
    aftertax.print_score(daily_data, books, begin_ts=idx[100], end_ts=idx[-1])
    out = capsys.readouterr().out
    assert "AFTER-TAX COMPARISON" in out
    assert "buy_and_hold" in out
    assert "CORE-SATELLITE" in out


def test_print_score_buy_hold_is_tax_exempt_over_long_window(capsys):
    # A window well past the 15y+1d exemption threshold: buy-and-hold's
    # pre-tax and post-tax Sharpe must be identical (0% tax owed).
    daily_data = _daily_data(n=6000, seed=2)     # ~23 years of business days
    books = combo.compute_books(daily_data)
    idx = list(daily_data.values())[0].index
    aftertax.print_score(daily_data, books, begin_ts=idx[0], end_ts=idx[-1])
    out = capsys.readouterr().out
    bh_line = [l for l in out.splitlines() if l.strip().startswith("buy_and_hold")][0]
    fields = bh_line.split()
    pretax_sharpe, posttax_sharpe = float(fields[1]), float(fields[2])
    assert np.isclose(pretax_sharpe, posttax_sharpe, atol=0.001)


def test_print_score_all_core_weight_row_matches_pure_buy_hold(capsys):
    daily_data = _daily_data(seed=3)
    books = combo.compute_books(daily_data)
    idx = list(daily_data.values())[0].index
    aftertax.print_score(daily_data, books, begin_ts=idx[100], end_ts=idx[-1])
    out = capsys.readouterr().out
    lines = out.splitlines()
    bh_sharpe = float([l for l in lines if l.strip().startswith("buy_and_hold")][0].split()[2])
    core_1_0_line = [l for l in lines if l.strip().startswith("1.0/0.0")][0]
    core_1_0_sharpe = float(core_1_0_line.split()[1])
    assert np.isclose(core_1_0_sharpe, bh_sharpe, atol=0.001)
    assert core_1_0_line.strip().endswith("no")     # 100% core == pure B&H, doesn't "beat" itself


def test_print_checkpoints_runs_without_error(capsys):
    daily_data = _daily_data(n=6000, seed=4)
    books = combo.compute_books(daily_data)
    idx = list(daily_data.values())[0].index
    aftertax.print_checkpoints(daily_data, books, begin_ts=idx[0], core_weight=0.7)
    out = capsys.readouterr().out
    assert "CORE-SATELLITE CHECKPOINTS" in out
    assert "beat pure buy-and-hold in" in out


def test_print_checkpoints_skips_dates_beyond_available_data(capsys):
    # Only ~4 years of data -- none of the default (10..21y) checkpoints fit.
    daily_data = _daily_data(n=1000, seed=5)
    books = combo.compute_books(daily_data)
    idx = list(daily_data.values())[0].index
    aftertax.print_checkpoints(daily_data, books, begin_ts=idx[0], core_weight=0.7)
    out = capsys.readouterr().out
    assert "beat pure buy-and-hold in 0/0 checkpoints." in out


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        fn()
        print(f"{name} OK")
    print(f"\nALL {len(fns)} AFTERTAX TESTS PASSED")
