import pandas as pd

from gex_lab import backtest as bt


def test_pending_cross_is_cancelled_if_next_open_loses_locked_ptrans():
    idx = pd.date_range("2026-01-05", periods=4, freq="B", tz="UTC")
    # Day 1 closes through pTrans=100, but day 2 gaps back below it. The thesis
    # is no longer confirmed at executable time, so no position may be opened.
    df = pd.DataFrame({
        "open": [99.0, 100.0, 99.0, 102.0],
        "high": [99.5, 101.5, 105.0, 104.0],
        "low": [98.5, 99.5, 98.0, 101.0],
        "close": [99.0, 101.0, 99.0, 103.0],
        "ptrans": [100.0] * 4,
        "ntrans": [90.0] * 4,
        "pos_gex": [110.0] * 4,
        "cotmp": [95.0] * 4,
    }, index=idx)
    assert bt.simulate(df) == []
