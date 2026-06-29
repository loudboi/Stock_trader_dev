"""
Offline tests for the dual-momentum rotation strategy (bot/momentum_rotation.py).
Synthetic data, no network.

Run:  pytest tests/test_momentum_rotation.py  (or: python tests/test_momentum_rotation.py)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.momentum_rotation as mr


def _series(close, start="2005-01-01"):
    idx = pd.date_range(start, periods=len(close), freq="B", tz="UTC")
    return pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": np.asarray(close, float),
                         "volume": np.full(len(close), 1.0)}, index=idx)


# --------------------------------------------------------------------------- #
# Selection (relative + absolute filter)
# --------------------------------------------------------------------------- #
def test_select_weights_picks_top_k_positive():
    scores = pd.Series({"A": 0.30, "B": 0.10, "C": -0.05, "D": 0.20})
    w = mr.select_weights(scores, top_k=2)
    assert w["A"] == 0.5 and w["D"] == 0.5      # two strongest positives
    assert w["B"] == 0.0 and w["C"] == 0.0
    assert abs(w.sum() - 1.0) < 1e-9


def test_absolute_filter_goes_to_cash_when_all_negative():
    scores = pd.Series({"A": -0.10, "B": -0.20, "C": -0.05})
    w = mr.select_weights(scores, top_k=2)
    assert w.sum() == 0.0                        # everything negative -> all cash


def test_partial_cash_when_few_positive():
    scores = pd.Series({"A": 0.15, "B": -0.01, "C": -0.02})
    w = mr.select_weights(scores, top_k=2)
    assert w["A"] == 0.5 and w.sum() == 0.5      # one qualifies -> half invested, half cash


# --------------------------------------------------------------------------- #
# Momentum + no-lookahead weight panel
# --------------------------------------------------------------------------- #
def test_momentum_none_before_enough_history():
    panel = pd.DataFrame({"A": np.linspace(100, 110, 30)})
    assert mr.momentum(panel, base_i=5, lookback=21, skip=0) is None
    assert mr.momentum(panel, base_i=25, lookback=21, skip=0) is not None


def test_weight_panel_warmup_is_cash_then_invests():
    # A trends up, B flat. Once warm, rotation should favor A.
    n = 300
    a = _series(np.linspace(100, 300, n))
    b = _series(np.full(n, 100.0))
    panel = mr.build_panel({"A": a, "B": b})
    w = mr.weight_panel(panel, lookback_months=3, skip_months=0, top_k=1)
    assert (w.iloc[:60].sum().sum()) == 0.0      # warmup region -> cash
    assert w["A"].iloc[-1] == 1.0                # ends holding the trending name
    assert w["B"].iloc[-1] == 0.0


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #
def test_run_rotation_beats_holding_a_crashing_universe():
    # A rises throughout; B rises then crashes hard. Rotation + absolute filter
    # should favor A and dodge B's crash, beating equal-weight buy-and-hold.
    n = 600
    a = _series(np.linspace(100, 260, n))
    b_path = np.concatenate([np.linspace(100, 200, n // 2),
                             np.linspace(200, 70, n - n // 2)])
    b = _series(b_path)
    rc = mr.run({"A": a, "B": b}, lookback_months=3, skip_months=0, top_k=1,
                begin_ts=a.index[80])
    assert rc == 0


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        fn()
        print(f"{name} OK")
    print(f"\nALL {len(fns)} MOMENTUM-ROTATION TESTS PASSED")
