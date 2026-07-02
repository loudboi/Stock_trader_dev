"""Offline tests for the GEX backtest engine + screen logic. No network."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gex_lab import backtest as bt
from gex_lab import screen as sc


def _df(close, high=None, ptrans=100, ntrans=90, pos_gex=110, cotmp=95):
    n = len(close)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    close = np.asarray(close, float)
    high = close if high is None else np.asarray(high, float)
    df = pd.DataFrame({"open": close, "high": high, "low": close, "close": close}, index=idx)
    for name, v in (("ptrans", ptrans), ("ntrans", ntrans),
                    ("pos_gex", pos_gex), ("cotmp", cotmp)):
        df[name] = v
    return df


# --------------------------------------------------------------------------- #
# Entries / exits
# --------------------------------------------------------------------------- #
def test_winning_trade_hits_t1():
    df = _df([99, 101, 104, 107, 111, 112])          # cross >100 at bar1, high>=110 at bar4
    trades = bt.simulate(df)
    assert len(trades) == 1
    assert trades[0]["reason"] == "T1 +GEX"
    assert trades[0]["exit"] == 110 and trades[0]["return_pct"] > 0


def test_stop1_close_below_ntrans():
    df = _df([99, 101, 100, 89, 88])                 # enter bar1, close 89 < nTrans 90
    trades = bt.simulate(df)
    assert len(trades) == 1 and trades[0]["reason"].startswith("stop1")
    assert trades[0]["return_pct"] < 0


def test_low_rr_setup_is_not_entered():
    df = _df([99, 101, 102], pos_gex=102)            # R:R = (102-101)/(101-100)=1 < 2
    assert bt.simulate(df) == []


def test_cushion_filter_blocks_when_too_close_to_put_mass():
    # spot 101, cotmp 100.5 -> cushion ~0.5% < 2% -> no entry despite good R:R.
    df = _df([99, 101, 104], cotmp=100.5)
    assert bt.simulate(df) == []


def test_time_stop_fires_when_stalled_past_day7():
    # Enters at bar1 (~101), then drifts flat well under 50% of the way to 110.
    close = [99, 101] + [101.5] * 9
    df = _df(close)
    trades = bt.simulate(df)
    assert len(trades) == 1
    assert trades[0]["reason"].startswith("stop3") or trades[0]["reason"].startswith("stop4")


def test_metrics_basic():
    trades = [{"return_pct": 0.2}, {"return_pct": -0.1}, {"return_pct": 0.3}]
    m = bt.metrics(trades)
    assert m["trades"] == 3
    assert abs(m["win_rate"] - 2 / 3) < 1e-9
    assert abs(m["profit_factor"] - 0.5 / 0.1) < 1e-9


def test_demo_panel_runs_end_to_end():
    m = bt.run(bt.demo_panel(seed=1, n_names=4, days=150))
    assert m["trades"] >= 0        # engine executes; may or may not trade


# --------------------------------------------------------------------------- #
# Screen status logic
# --------------------------------------------------------------------------- #
def test_screen_confirmed():
    lv = {"spot": 101, "ptrans": 100, "pos_gex": 110, "ntrans": 90,
          "cotmp": 95, "net_gex": 1e9}
    assert sc.screen_row("X", lv)["status"] == "CONFIRMED"


def test_screen_pending_just_below_ptrans():
    lv = {"spot": 99.7, "ptrans": 100, "pos_gex": 110, "ntrans": 90,
          "cotmp": 95, "net_gex": 1e9}
    assert sc.screen_row("X", lv)["status"] == "PENDING"


def test_screen_blocked_low_rr():
    lv = {"spot": 101, "ptrans": 100, "pos_gex": 102, "ntrans": 90,
          "cotmp": 95, "net_gex": 1e9}
    assert sc.screen_row("X", lv)["status"] == "BLOCKED"


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        fn()
        print(f"{name} OK")
    print(f"\nALL {len(fns)} BACKTEST/SCREEN TESTS PASSED")
