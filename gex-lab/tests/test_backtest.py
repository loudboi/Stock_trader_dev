"""Offline tests for point-in-time GEX backtest and screen logic."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gex_lab import backtest as bt
from gex_lab import screen as sc


def _df(close, opens=None, high=None, low=None,
        ptrans=100, ntrans=90, pos_gex=110, cotmp=95):
    n = len(close)
    idx = pd.date_range("2025-01-01", periods=n, freq="B", tz="UTC")
    close = np.asarray(close, float)
    opens = close if opens is None else np.asarray(opens, float)
    high = np.maximum(opens, close) if high is None else np.asarray(high, float)
    low = np.minimum(opens, close) if low is None else np.asarray(low, float)
    df = pd.DataFrame({"open": opens, "high": high, "low": low, "close": close}, index=idx)
    for name, value in (("ptrans", ptrans), ("ntrans", ntrans),
                        ("pos_gex", pos_gex), ("cotmp", cotmp)):
        df[name] = value
    return df


def test_close_signal_enters_next_session_open_not_same_close():
    df = _df([99, 101, 104, 111], opens=[99, 100, 102, 108], high=[99, 101, 104, 111])
    trades = bt.simulate(df)
    assert len(trades) == 1
    assert trades[0]["signal_date"] == df.index[1]
    assert trades[0]["entry_date"] == df.index[2]
    assert trades[0]["entry"] == 102
    assert trades[0]["reason"] == "T1 +GEX"


def test_gap_above_locked_target_cancels_pending_entry():
    df = _df([99, 101, 112], opens=[99, 100, 112], high=[99, 101, 113])
    assert bt.simulate(df) == []


def test_stop1_close_below_ntrans():
    df = _df([99, 101, 100, 89, 88], opens=[99, 100, 100, 95, 88])
    trades = bt.simulate(df)
    assert len(trades) == 1
    assert "stop1" in trades[0]["reason"]
    assert trades[0]["return_pct"] < 0


def test_same_bar_target_and_price_stop_uses_adverse_stop():
    df = _df([99, 101, 100, 89], opens=[99, 100, 100, 100],
             high=[99, 101, 111, 111], low=[99, 100, 85, 85])
    trades = bt.simulate(df)
    assert len(trades) == 1
    assert trades[0]["reason"].startswith("ambiguous target/stop")
    assert trades[0]["exit"] == 89


def test_low_rr_and_cushion_filters_block_signal():
    assert bt.simulate(_df([99, 101, 102], pos_gex=102)) == []
    assert bt.simulate(_df([99, 101, 104], cotmp=100.5)) == []


def test_metrics_and_demo_run():
    m = bt.metrics([{"return_pct": 0.2}, {"return_pct": -0.1}, {"return_pct": 0.3}])
    assert m["trades"] == 3 and abs(m["profit_factor"] - 5.0) < 1e-9
    assert bt.run(bt.demo_panel(seed=1, n_names=3, days=120))["trades"] >= 0


def test_screen_status_and_assumption_fields():
    lv = {"spot": 101, "ptrans": 100, "pos_gex": 110, "ntrans": 90,
          "cotmp": 95, "net_gex": 1e9, "risk_free": 0.03,
          "dividend_yield": 0.01,
          "dealer_sign_model": "calls_positive_puts_negative"}
    row = sc.screen_row("X", lv)
    assert row["status"] == "CONFIRMED"
    assert row["risk_free"] == 0.03 and row["dividend_yield"] == 0.01
    lv["spot"] = 99.7
    assert sc.screen_row("X", lv)["status"] == "PENDING"
    lv.update(spot=101, pos_gex=102)
    assert sc.screen_row("X", lv)["status"] == "BLOCKED"


def test_snapshot_filename_never_overwrites_same_date(tmp_path):
    df = pd.DataFrame([{"ticker": "X", "spot": 100, "ptrans": 99,
                        "ntrans": 90, "pos_gex": 110, "cotmp": 95}])
    asof1 = pd.Timestamp("2026-08-10T20:00:00.000001Z")
    asof2 = pd.Timestamp("2026-08-10T20:00:00.000002Z")
    p1 = sc.save_snapshot(df, str(tmp_path), asof=asof1)
    p2 = sc.save_snapshot(df, str(tmp_path), asof=asof2)
    assert p1 != p2 and os.path.exists(p1) and os.path.exists(p2)
    with pytest.raises(FileExistsError):
        sc.save_snapshot(df, str(tmp_path), asof=asof1)


def _write_snapshot(path, date, ptrans):
    pd.DataFrame([{"date": date, "ticker": "X", "ptrans": ptrans,
                   "ntrans": 90, "pos_gex": 110, "cotmp": 95}]).to_csv(path, index=False)


def test_load_real_levels_are_timezone_safe_and_never_used_same_snapshot_date(tmp_path):
    _write_snapshot(tmp_path / "gex_2026-01-05.csv", "2026-01-05", 100)
    _write_snapshot(tmp_path / "gex_2026-01-06.csv", "2026-01-06", 101)

    idx = pd.date_range("2026-01-05", periods=5, freq="B", tz="UTC")
    px = pd.DataFrame({"open": [99, 100, 102, 103, 104],
                       "high": [100, 101, 103, 104, 105],
                       "low": [98, 99, 101, 102, 103],
                       "close": [99, 101, 102, 103, 104]}, index=idx)
    loader = lambda ticker, start, end: px
    panel = bt.load_real(str(tmp_path), price_loader=loader, max_level_age_days=4)
    df = panel["X"]
    # Jan 5 snapshot must NOT be usable on Jan 5; it starts Jan 6.
    assert pd.isna(df.loc[pd.Timestamp("2026-01-05", tz="UTC"), "ptrans"])
    assert df.loc[pd.Timestamp("2026-01-06", tz="UTC"), "ptrans"] == 100
    assert df.loc[pd.Timestamp("2026-01-07", tz="UTC"), "ptrans"] == 101
    assert str(df.index.tz) == "UTC"


def test_load_real_rejects_unsupported_price_source(tmp_path):
    with pytest.raises(ValueError):
        bt.load_real(str(tmp_path), price_source="alpaca")
