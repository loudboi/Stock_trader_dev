"""
Offline tests for the dynamic RP/TE allocation mechanism (bot/dynamic_combo.py).
Synthetic data, no network.

Run:  pytest tests/test_dynamic_combo.py  (or: python tests/test_dynamic_combo.py)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.dynamic_combo as dc


def _panel(cols: dict, start="2005-01-01"):
    n = len(next(iter(cols.values())))
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC")
    return pd.DataFrame({k: np.asarray(v, float) for k, v in cols.items()}, index=idx)


# --------------------------------------------------------------------------- #
# breadth_signal
# --------------------------------------------------------------------------- #
def test_breadth_signal_is_one_when_all_above_ma():
    up = np.linspace(100, 200, 250)          # steady uptrend -> ends well above its MA
    panel = _panel({"A": up, "B": up * 1.1, "C": up * 0.9})
    b = dc.breadth_signal(panel, ma_period=20)
    assert b.iloc[-1] == 1.0


def test_breadth_signal_is_zero_when_all_below_ma():
    down = np.linspace(200, 100, 250)
    panel = _panel({"A": down, "B": down * 1.1, "C": down * 0.9})
    b = dc.breadth_signal(panel, ma_period=20)
    assert b.iloc[-1] == 0.0


def test_breadth_signal_mixed_participation():
    up = np.linspace(100, 200, 250)
    down = np.linspace(200, 100, 250)
    panel = _panel({"UP1": up, "UP2": up * 1.05, "DOWN": down})
    b = dc.breadth_signal(panel, ma_period=20)
    assert abs(b.iloc[-1] - 2 / 3) < 1e-9


# --------------------------------------------------------------------------- #
# dynamic_rp_weight
# --------------------------------------------------------------------------- #
def test_dynamic_rp_weight_bands_and_shift():
    idx = pd.date_range("2020-01-01", periods=5, freq="B", tz="UTC")
    signal = pd.Series([0.2, 0.5, 0.8, 0.2, 0.8], index=idx)   # low, mid, high, low, high
    w = dc.dynamic_rp_weight(signal, idx, low_thresh=0.4, high_thresh=0.6,
                             low_rp=0.3, mid_rp=0.5, high_rp=0.7)
    # Shifted by 1: day i's weight reflects day i-1's signal.
    assert w.iloc[0] == 0.5                     # no prior day -> mid default
    assert w.iloc[1] == 0.7                      # yesterday's signal was 0.2 (low) -> high_rp
    assert w.iloc[2] == 0.5                      # yesterday's signal was 0.5 (mid) -> mid_rp
    assert w.iloc[3] == 0.3                      # yesterday's signal was 0.8 (high) -> low_rp
    assert w.iloc[4] == 0.7                      # yesterday's signal was 0.2 (low) -> high_rp


def test_dynamic_rp_weight_forward_fills_sparser_signal_calendar():
    panel_idx = pd.date_range("2020-01-01", periods=10, freq="B", tz="UTC")
    sparse_idx = panel_idx[::3]                  # a sparser "macro calendar"
    signal = pd.Series([0.8, 0.2, 0.8, 0.2], index=sparse_idx[:4])
    w = dc.dynamic_rp_weight(signal, panel_idx, low_thresh=0.4, high_thresh=0.6)
    assert len(w) == len(panel_idx)
    assert not w.isna().any()


# --------------------------------------------------------------------------- #
# dynamic_blend_returns
# --------------------------------------------------------------------------- #
def test_dynamic_blend_matches_manual_weighted_sum():
    idx = pd.date_range("2020-01-01", periods=5, freq="B", tz="UTC")
    rp = pd.Series([0.01, 0.02, -0.01, 0.0, 0.005], index=idx)
    te = pd.Series([-0.01, 0.0, 0.02, 0.01, -0.005], index=idx)
    weight = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5], index=idx)   # constant -> no turnover cost
    blended = dc.dynamic_blend_returns({"rp": rp, "te": te}, weight)
    expected = rp * 0.5 + te * 0.5
    assert np.allclose(blended.values, expected.values)


def test_dynamic_blend_charges_turnover_on_weight_changes():
    idx = pd.date_range("2020-01-01", periods=4, freq="B", tz="UTC")
    rp = pd.Series(np.zeros(4), index=idx)
    te = pd.Series(np.zeros(4), index=idx)
    weight = pd.Series([0.5, 0.5, 0.8, 0.8], index=idx)   # jumps once
    blended = dc.dynamic_blend_returns({"rp": rp, "te": te}, weight)
    assert blended.iloc[0] == 0.0 and blended.iloc[1] == 0.0    # no weight change -> no cost
    assert blended.iloc[2] < 0.0                                 # weight jumped -> turnover cost


def test_dynamic_blend_extremes_match_pure_books():
    idx = pd.date_range("2020-01-01", periods=5, freq="B", tz="UTC")
    rp = pd.Series([0.01, 0.02, -0.01, 0.0, 0.005], index=idx)
    te = pd.Series([-0.01, 0.0, 0.02, 0.01, -0.005], index=idx)
    all_rp = dc.dynamic_blend_returns({"rp": rp, "te": te}, pd.Series(1.0, index=idx))
    all_te = dc.dynamic_blend_returns({"rp": rp, "te": te}, pd.Series(0.0, index=idx))
    assert np.allclose(all_rp.values, rp.values)
    assert np.allclose(all_te.values, te.values)


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        fn()
        print(f"{name} OK")
    print(f"\nALL {len(fns)} DYNAMIC-COMBO TESTS PASSED")
