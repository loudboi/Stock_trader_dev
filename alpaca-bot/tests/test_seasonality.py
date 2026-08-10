"""Offline tests for calendar seasonality research."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.seasonality as sea


def test_turn_of_month_signal_flags_exact_last_and_first_days():
    idx = pd.date_range("2024-01-01", "2024-02-10", freq="B")
    signal = sea.turn_of_month_signal(idx, days_before=1, days_after=3)
    jan = idx[idx.to_period("M") == "2024-01"]
    feb = idx[idx.to_period("M") == "2024-02"]
    assert signal.loc[jan[-1]] == 1.0
    assert signal.loc[jan[-2]] == 0.0
    assert (signal.loc[feb[:3]] == 1.0).all()
    assert signal.loc[feb[3]] == 0.0


def test_turn_of_month_signal_fraction_matches_window_size():
    idx = pd.date_range("2020-01-01", "2025-12-31", freq="B")
    signal = sea.turn_of_month_signal(idx, days_before=1, days_after=3)
    assert 0.15 < signal.mean() < 0.25


def test_turn_of_month_returns_use_named_calendar_days_not_shifted_days():
    idx = pd.date_range("2024-01-22", "2024-02-09", freq="B")
    # Engineer a +10% close-to-close return only on Feb's first trading day.
    close = pd.Series(100.0, index=idx)
    feb = idx[idx.to_period("M") == "2024-02"]
    first_feb = feb[0]
    close.loc[first_feb:] = 110.0
    panel = pd.DataFrame({"A": close}, index=idx)
    r = sea.turn_of_month_returns(panel, days_before=1, days_after=3)
    assert r.loc[first_feb] > 0.09
    # The fourth February day is not part of first-3 TOM exposure.
    fourth_feb = feb[3]
    assert abs(r.loc[fourth_feb] + sea.SLIPPAGE) < 1e-12  # exit turnover only


def test_inactive_days_have_zero_return_except_entry_exit_turnover():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    panel = pd.DataFrame({"A": 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))}, index=idx)
    r = sea.turn_of_month_returns(panel, days_before=1, days_after=3)
    signal = sea.turn_of_month_signal(idx, 1, 3)
    turnover = signal.diff().abs().fillna(signal.abs())
    inactive_no_turn = idx[(signal == 0) & (turnover == 0)]
    assert len(inactive_no_turn) > 0
    assert (r.reindex(inactive_no_turn).abs() < 1e-12).all()


def test_turn_of_month_calendar_signal_is_causal_without_price_shift():
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    panel = pd.DataFrame({"A": np.linspace(100, 120, 100)}, index=idx)
    signal = sea.turn_of_month_signal(idx)
    r = sea.turn_of_month_returns(panel)
    # A date's membership depends only on the known trading calendar. If the first
    # row is in-window there can be an entry cost but no manufactured price return.
    expected_first = -sea.SLIPPAGE if signal.iloc[0] else 0.0
    assert abs(r.iloc[0] - expected_first) < 1e-12


def test_invalid_turn_of_month_window_rejected():
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    with pytest.raises(ValueError):
        sea.turn_of_month_signal(idx, 0, 0)


def test_weekday_stats_groups_by_day_of_week():
    idx = pd.date_range("2024-01-01", periods=50, freq="B")
    close = 100 * np.cumprod(1 + np.linspace(0.001, -0.001, 50))
    panel = pd.DataFrame({"A": close}, index=idx)
    by_day = sea.weekday_stats(panel)
    assert set(by_day) == {"Mon", "Tue", "Wed", "Thu", "Fri"}
    for name, r in by_day.items():
        assert (r.index.dayofweek == sea._WEEKDAY_NAMES.index(name)).all()
    assert sum(len(r) for r in by_day.values()) == len(panel)


def test_weekday_stats_respects_begin_ts():
    idx = pd.date_range("2024-01-01", periods=50, freq="B")
    panel = pd.DataFrame({"A": 100 * np.cumprod(1 + np.full(50, 0.001))}, index=idx)
    begin = idx[20]
    for r in sea.weekday_stats(panel, begin_ts=begin).values():
        assert (r.index >= begin).all()


def test_print_weekday_stats_runs_without_error(capsys):
    rng = np.random.default_rng(4)
    idx = pd.date_range("2020-01-01", periods=300, freq="B", tz="UTC")
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 300))
    daily_data = {"A": pd.DataFrame({"open": close, "high": close, "low": close,
                                     "close": close, "volume": np.ones(300)}, index=idx)}
    sea.print_weekday_stats(daily_data, begin_ts=idx[50])
    out = capsys.readouterr().out
    assert "DAY-OF-WEEK EFFECT" in out and "Best day" in out and "Worst day" in out


def test_run_smoke_end_to_end():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2020-01-01", periods=500, freq="B", tz="UTC")
    daily_data = {}
    for name in ("A", "B"):
        close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 500))
        daily_data[name] = pd.DataFrame({"open": close, "high": close, "low": close,
                                         "close": close, "volume": np.ones(500)}, index=idx)
    assert sea.run(daily_data, begin_ts=idx[50]) == 0
