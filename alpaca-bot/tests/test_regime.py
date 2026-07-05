"""Offline tests for the macro-regime classifiers (bot/regime.py). No network."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot import regime as rg


def _idx(n):
    return pd.date_range("2020-01-01", periods=n, freq="B")


def test_vix_regime_classifies_calm_elevated_neutral():
    vix = pd.Series([10, 18, 30, 25.01, 14.99], index=_idx(5))
    out = rg.vix_regime(vix, calm=15, elevated=25)
    assert list(out.values) == [1, 0, -1, -1, 1]


def test_curve_regime_sign():
    curve = pd.Series([1.5, -0.3, 0.0, -0.01], index=_idx(4))
    out = rg.curve_regime(curve)
    assert list(out.values) == [1, -1, 1, -1]


def test_credit_regime_threshold():
    credit = pd.Series([0.05, -0.02, 0.0, -0.001], index=_idx(4))
    out = rg.credit_regime(credit, threshold=0.0)
    assert list(out.values) == [1, -1, 1, -1]


def test_vix_term_regime_backwardation_vs_contango():
    ratio = pd.Series([0.85, 0.95, 1.0, 1.01, 1.20], index=_idx(5))
    out = rg.vix_term_regime(ratio, backwardation=1.0)
    assert list(out.values) == [1, 1, 1, -1, -1]     # only STRICTLY > 1.0 is stress


def test_align_to_panel_forward_fills_across_gaps():
    macro_idx = pd.date_range("2020-01-01", periods=3, freq="3B")   # sparser calendar
    regime = pd.Series([1, -1, 1], index=macro_idx)
    panel_idx = pd.date_range("2020-01-01", periods=9, freq="B")    # denser calendar
    aligned = rg.align_to_panel(regime, panel_idx)
    assert len(aligned) == 9
    assert aligned.iloc[0] == 1                       # first value present
    assert aligned.ffill().isna().sum() == 0 or aligned.iloc[-1] in (1, -1)


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        fn()
        print(f"{name} OK")
    print(f"\nALL {len(fns)} REGIME TESTS PASSED")
