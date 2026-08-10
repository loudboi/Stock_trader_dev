"""Offline tests for buy-and-hold benchmarks and research data normalization."""
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


def test_buy_hold_equity_scales_with_price():
    daily = _daily([100, 110, 90, 120])
    eq = bp.buy_hold_equity(daily, initial=100_000)
    assert eq.iloc[0] == 100_000
    assert abs(eq.iloc[1] - 110_000) < 1e-6
    assert abs(eq.iloc[-1] - 120_000) < 1e-6


def test_buy_hold_equity_respects_begin_ts():
    daily = _daily([100, 110, 90, 120])
    begin = daily.index[1]
    eq = bp.buy_hold_equity(daily, begin_ts=begin, initial=100_000)
    assert eq.index[0] == begin and eq.iloc[0] == 100_000
    assert abs(eq.iloc[-1] - 100_000 * 120 / 110) < 1e-6


def test_buy_hold_combined_equal_weight():
    a = _daily([100, 200])
    b = _daily([100, 100])
    combined = bp.buy_hold_combined({"A": a, "B": b}, initial=100_000)
    assert abs(combined.iloc[0] - 100_000) < 1e-6
    assert abs(combined.iloc[-1] - 150_000) < 1e-6


def test_buy_hold_combined_does_not_create_late_inception_capital():
    a = _daily([100, 110, 120, 130], start="2020-01-01")
    b = _daily([100, 110], start="2020-01-03")
    combined = bp.buy_hold_combined({"A": a, "B": b}, initial=100_000)
    # Both sleeves are funded at the common investable start; there must not be a
    # sudden +50k jump simply because B did not exist on earlier dates.
    assert combined.index[0] >= b.index[0]
    assert abs(combined.iloc[0] - 100_000) < 1e-6


def test_buy_hold_metrics_plug_into_compute_metrics():
    eq = bp.buy_hold_equity(_daily(list(np.linspace(100, 130, 50))))
    m = bp.compute_metrics([], eq)
    assert m["trades"] == 0
    assert m["total_return"] > 0 and m["max_drawdown"] <= 0


def test_normalize_flat_columns_and_tz():
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-03"])
    raw = pd.DataFrame({"Open": [1, 2], "High": [2, 3], "Low": [0.5, 1.5],
                        "Close": [1.5, 2.5], "Volume": [100, 200]}, index=idx)
    df = data.normalize_ohlcv(raw)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index.tz) == "UTC"
    assert df["close"].tolist() == [1.5, 2.5]


def test_normalize_multiindex_columns():
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-03"])
    cols = pd.MultiIndex.from_tuples([
        ("Open", "SPY"), ("High", "SPY"), ("Low", "SPY"),
        ("Close", "SPY"), ("Volume", "SPY")])
    raw = pd.DataFrame([[1, 2, 0.5, 1.4, 100], [2, 3, 1.5, 2.4, 200]],
                       index=idx, columns=cols)
    df = data.normalize_ohlcv(raw)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2


def test_normalize_empty():
    assert data.normalize_ohlcv(pd.DataFrame()).empty
    assert data.normalize_ohlcv(None).empty


def test_isolated_spike_is_quarantined_without_cascading():
    px = pd.Series([100.0, 101.0, 250.0, 102.0, 103.0])
    mask = data.quarantine_price_mask(px, max_abs_return=0.15)
    assert mask.tolist() == [False, False, True, False, False]
    cleaned = data.clean_price_series(px, max_abs_return=0.15)
    assert cleaned.tolist() == [100.0, 101.0, 102.0, 103.0]


def test_negative_price_is_quarantined_but_following_positive_bar_survives():
    px = pd.Series([18.27, -37.63, 10.01, 13.78])
    mask = data.quarantine_price_mask(px, max_abs_return=0.15)
    assert mask.tolist() == [False, True, False, False]
    cleaned = data.clean_price_series(px, max_abs_return=0.15)
    assert cleaned.tolist() == [18.27, 10.01, 13.78]


def test_large_one_way_market_move_is_flagged_not_erased():
    px = pd.Series([100.0, 125.0, 126.0, 127.0])
    suspect = data.suspect_price_mask(px, max_abs_return=0.15)
    quarantine = data.quarantine_price_mask(px, max_abs_return=0.15)
    assert suspect.iloc[1]
    assert not quarantine.any()
    assert data.clean_price_series(px, max_abs_return=0.15).equals(px)


def test_clean_daily_data_drops_whole_bad_bar_and_records_provenance():
    idx = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({
        "open": [100, 101, 250, 102, 103],
        "high": [101, 102, 251, 103, 104],
        "low": [99, 100, 249, 101, 102],
        "close": [100, 101, 250, 102, 103],
        "volume": [10, 10, 10, 10, 10]}, index=idx)
    cleaned = data.clean_daily_data({"X": df}, max_abs_return=0.15)["X"]
    assert idx[2] not in cleaned.index
    assert idx[3] in cleaned.index
    assert cleaned.attrs["quarantined_timestamps"] == [idx[2].isoformat()]
    assert idx[2].isoformat() in cleaned.attrs["suspect_timestamps"]


def test_clean_daily_data_does_not_guess_about_terminal_large_move():
    idx = pd.date_range("2020-01-01", periods=2, freq="D", tz="UTC")
    df = pd.DataFrame({"open": [100, 500], "high": [101, 501], "low": [99, 499],
                       "close": [100.0, 500.0], "volume": [10, 10]}, index=idx)
    cleaned = data.clean_daily_data({"X": df}, max_abs_return=0.15)["X"]
    assert cleaned.index.equals(df.index)
    assert cleaned.attrs["quarantined_timestamps"] == []
    assert cleaned.attrs["suspect_timestamps"] == [idx[1].isoformat()]
    assert data.clean_daily_data({"EMPTY": pd.DataFrame()})["EMPTY"].empty
