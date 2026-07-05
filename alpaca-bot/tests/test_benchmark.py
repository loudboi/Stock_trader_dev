"""
Offline tests for the buy-and-hold benchmark and the Yahoo data normalizer
(bot/backtest_pullback.py buy_hold_*, bot/data.py normalize_ohlcv). No network.

Run:  pytest tests/test_benchmark.py    (or: python tests/test_benchmark.py)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.backtest_pullback as bp
from bot import data


def _daily(close, start="2020-01-01"):
    n = len(close)
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": np.full(n, 1.0)}, index=idx)


# --------------------------------------------------------------------------- #
# Buy-and-hold
# --------------------------------------------------------------------------- #
def test_buy_hold_equity_scales_with_price():
    daily = _daily([100, 110, 90, 120])
    eq = bp.buy_hold_equity(daily, initial=100_000)
    assert eq.iloc[0] == 100_000                       # starts at initial
    assert abs(eq.iloc[1] - 110_000) < 1e-6            # +10%
    assert abs(eq.iloc[-1] - 120_000) < 1e-6           # +20% at the end


def test_buy_hold_equity_respects_begin_ts():
    daily = _daily([100, 110, 90, 120])
    begin = daily.index[1]                             # start at the 110 bar
    eq = bp.buy_hold_equity(daily, begin_ts=begin, initial=100_000)
    assert eq.index[0] == begin and eq.iloc[0] == 100_000
    assert abs(eq.iloc[-1] - 100_000 * 120 / 110) < 1e-6


def test_buy_hold_combined_equal_weight():
    a = _daily([100, 200])        # +100%
    b = _daily([100, 100])        # flat
    combined = bp.buy_hold_combined({"A": a, "B": b}, initial=100_000)
    # 50k each: A doubles to 100k, B stays 50k -> 150k total.
    assert abs(combined.iloc[0] - 100_000) < 1e-6
    assert abs(combined.iloc[-1] - 150_000) < 1e-6


def test_buy_hold_metrics_plug_into_compute_metrics():
    daily = _daily(list(np.linspace(100, 130, 50)))
    eq = bp.buy_hold_equity(daily)
    m = bp.compute_metrics([], eq)                     # no trades, just the curve
    assert m["trades"] == 0
    assert m["total_return"] > 0 and m["max_drawdown"] <= 0


# --------------------------------------------------------------------------- #
# Yahoo normalizer (fed yfinance-shaped frames, no network)
# --------------------------------------------------------------------------- #
def test_normalize_flat_columns_and_tz():
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-03"])   # tz-naive, like Yahoo
    raw = pd.DataFrame({"Open": [1, 2], "High": [2, 3], "Low": [0.5, 1.5],
                        "Close": [1.5, 2.5], "Volume": [100, 200]}, index=idx)
    df = data.normalize_ohlcv(raw)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index.tz) == "UTC"
    assert df["close"].tolist() == [1.5, 2.5]


def test_normalize_multiindex_columns_and_adj_close():
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-03"])
    cols = pd.MultiIndex.from_tuples(
        [("Open", "SPY"), ("High", "SPY"), ("Low", "SPY"),
         ("Close", "SPY"), ("Volume", "SPY")])
    raw = pd.DataFrame([[1, 2, 0.5, 1.4, 100], [2, 3, 1.5, 2.4, 200]],
                       index=idx, columns=cols)
    df = data.normalize_ohlcv(raw)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2


def test_normalize_empty():
    assert data.normalize_ohlcv(pd.DataFrame()).empty
    assert data.normalize_ohlcv(None).empty


# --------------------------------------------------------------------------- #
# Outlier/bad-tick cleaning (discovered need: real free-data corruption in
# FX/futures tickers — see bot/data.py clean_price_series docstring)
# --------------------------------------------------------------------------- #
def test_clean_price_series_patches_a_spurious_spike():
    px = pd.Series([100.0, 101.0, 250.0, 102.0, 103.0])   # a bad tick at index 2
    cleaned = data.clean_price_series(px, max_abs_return=0.15)
    assert cleaned.iloc[2] == cleaned.iloc[1]              # carried forward, not the spike
    assert cleaned.iloc[3] == 102.0                        # unaffected downstream values untouched


def test_clean_price_series_handles_a_negative_price_crossing():
    # The real April 2020 WTI event: price crosses through/below zero, which makes
    # a percentage return mathematically undefined, not just large.
    px = pd.Series([18.27, -37.63, 10.01, 13.78])
    cleaned = data.clean_price_series(px, max_abs_return=0.15)
    assert (cleaned > 0).all()                              # no non-positive price survives
    assert cleaned.iloc[1] == cleaned.iloc[0]                # the crossing day carried forward


def test_clean_price_series_leaves_normal_moves_alone():
    px = pd.Series([100.0, 102.0, 99.0, 101.5, 98.0])       # all well under 15%/day
    cleaned = data.clean_price_series(px, max_abs_return=0.15)
    assert (cleaned == px).all()


def test_clean_daily_data_only_touches_close():
    df = pd.DataFrame({"open": [1, 1], "high": [1, 1], "low": [1, 1],
                       "close": [100.0, 500.0], "volume": [10, 10]})
    cleaned = data.clean_daily_data({"X": df}, max_abs_return=0.15)["X"]
    assert cleaned["close"].iloc[1] == 100.0                # patched
    assert (cleaned["open"] == df["open"]).all()             # other columns untouched
    assert data.clean_daily_data({"EMPTY": pd.DataFrame()})["EMPTY"].empty


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        fn()
        print(f"{name} OK")
    print(f"\nALL {len(fns)} BENCHMARK/DATA TESTS PASSED")
