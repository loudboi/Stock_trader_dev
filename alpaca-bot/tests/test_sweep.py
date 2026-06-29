"""
Offline tests for the parameter-sweep / walk-forward mechanics (bot/sweep.py).

The backtester it sits on is covered elsewhere; here we check the grid building,
parameter application, ranking, fold splitting, and that walk-forward returns a
chosen-params + in/out-of-sample structure. Synthetic data, no network.

Run:  pytest tests/test_sweep.py    (or: python tests/test_sweep.py)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from bot.strategies.trend_pullback import PullbackParams
import bot.sweep as sweep


def _register(name="TEST"):
    config.PULLBACK_UNIVERSE[name] = config.Instrument(
        name=name, api_symbol=name, asset_class="equity",
        strategy="trend_pullback", timeframe="1Day",
        can_short=False, qty_decimals=0)


def _daily(close, start="2023-01-01"):
    n = len(close)
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"open": close - 0.2, "high": close + 1.0,
                         "low": close - 1.0, "close": close,
                         "volume": np.full(n, 1_000_000.0)}, index=idx)


def _synthetic(n=600):
    np.random.seed(1)
    close = np.linspace(100, 260, n) + np.random.normal(0, 1.2, n)
    for k in range(240, n, 70):              # periodic light-volume dips
        close[k:k + 4] -= 6
    d = _daily(close)
    for k in range(240, n, 70):
        d.iloc[k:k + 4, d.columns.get_loc("volume")] *= 0.5
    return {"TEST": d}


# --------------------------------------------------------------------------- #
# Grid building / param application
# --------------------------------------------------------------------------- #
def test_build_grid_cartesian_product():
    grid = sweep.build_grid({"add_step": [0.02, 0.03], "atr_mult": [1.5, 2.0]})
    # touch_band / vol_contraction fall back to their single defaults -> 2*2*1*1.
    assert len(grid) == 4
    assert all(set(c) == {"add_step", "atr_mult", "touch_band", "vol_contraction"}
               for c in grid)


def test_params_for_overrides_only_named_axes():
    base = PullbackParams()
    p = sweep.params_for(base, {"add_step": 0.05, "atr_mult": 3.0,
                                "touch_band": 0.02, "vol_contraction": 0.9})
    assert p.add_step == 0.05 and p.atr_mult == 3.0
    assert p.ma_fast == base.ma_fast and p.improve_pct == base.improve_pct


def test_fold_bounds_partition_and_order():
    s = pd.Timestamp("2023-01-01", tz="UTC")
    e = pd.Timestamp("2025-01-01", tz="UTC")
    folds = sweep.fold_bounds(s, e, 4)
    assert len(folds) == 4
    assert folds[0][0] == s and folds[-1][2] == e
    for fs, split, fe in folds:               # in-sample precedes out-of-sample
        assert fs < split < fe


# --------------------------------------------------------------------------- #
# Grid + walk-forward over synthetic data
# --------------------------------------------------------------------------- #
def test_run_grid_ranks_by_sharpe():
    _register()
    data = _synthetic()
    grid = sweep.build_grid({"add_step": [0.02, 0.04], "atr_mult": [1.5, 2.5]})
    results = sweep.run_grid(data, {}, PullbackParams(), grid,
                             exec_is_intraday=False)
    assert len(results) == len(grid)
    sharpes = [m["sharpe"] for _, m in results]
    assert sharpes == sorted(sharpes, reverse=True)     # ranked desc
    assert all("trades" in m for _, m in results)


def test_walk_forward_returns_is_and_oos_per_fold():
    _register()
    data = _synthetic()
    grid = sweep.build_grid({"add_step": [0.02, 0.04], "atr_mult": [1.5, 2.5]})
    s = data["TEST"].index[0]
    e = data["TEST"].index[-1]
    folds = sweep.run_walk_forward(data, {}, PullbackParams(), grid,
                                   exec_is_intraday=False, start_ts=s, end_ts=e,
                                   folds=2)
    assert len(folds) == 2
    for f in folds:
        assert set(f["best"]) == {"add_step", "atr_mult", "touch_band", "vol_contraction"}
        assert "sharpe" in f["is"] and "sharpe" in f["oos"]
        assert f["is_window"][0] < f["is_window"][1] <= f["oos_window"][0] < f["oos_window"][1]


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        fn()
        print(f"{name} OK")
    print(f"\nALL {len(fns)} SWEEP TESTS PASSED")
