"""
Offline tests for the turn-of-month effect (bot/seasonality.py). Synthetic data,
no network.

Run:  pytest tests/test_seasonality.py    (or: python tests/test_seasonality.py)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.seasonality as sea


def test_turn_of_month_signal_flags_last_and_first_days():
    # January 2024 has 23 business days; February starts right after.
    idx = pd.date_range("2024-01-01", "2024-02-10", freq="B")
    signal = sea.turn_of_month_signal(idx, days_before=1, days_after=3)
    jan = idx[idx.to_period("M") == "2024-01"]
    feb = idx[idx.to_period("M") == "2024-02"]
    # Last day of January flagged.
    assert signal.loc[jan[-1]] == 1.0
    # Second-to-last day of January NOT flagged (outside the 1-day window).
    assert signal.loc[jan[-2]] == 0.0
    # First three days of February flagged; the fourth is not.
    assert (signal.loc[feb[:3]] == 1.0).all()
    assert signal.loc[feb[3]] == 0.0
    # A mid-month day is not flagged.
    mid_jan = jan[len(jan) // 2]
    assert signal.loc[mid_jan] == 0.0


def test_turn_of_month_signal_fraction_matches_window_size():
    idx = pd.date_range("2020-01-01", "2025-12-31", freq="B")
    signal = sea.turn_of_month_signal(idx, days_before=1, days_after=3)
    # Roughly 4 invested days out of ~21 trading days/month -> ~19%.
    assert 0.15 < signal.mean() < 0.25


def test_turn_of_month_returns_only_active_on_signal_days():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    panel = pd.DataFrame({"A": 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))}, index=idx)
    r = sea.turn_of_month_returns(panel, days_before=1, days_after=3)
    signal = sea.turn_of_month_signal(idx, 1, 3)
    # Shifted by one day (act on yesterday's signal) -> non-TOM days (excluding
    # the day right after a TOM window, which still carries turnover cost/no
    # position) should have zero raw exposure. Check a definitely-inactive day.
    inactive_days = idx[(signal.shift(2).fillna(0) == 0) & (signal.shift(1).fillna(0) == 0)]
    assert len(inactive_days) > 0
    assert (r.reindex(inactive_days[5:-5]).abs() < 1e-9).all()


def test_turn_of_month_no_lookahead():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    panel = pd.DataFrame({"A": 100 * np.cumprod(1 + rng.normal(0, 0.01, 100))}, index=idx)
    r = sea.turn_of_month_returns(panel)
    assert r.iloc[0] == 0.0          # shift(1) -> first bar always flat


def test_weekday_stats_groups_by_day_of_week():
    idx = pd.date_range("2024-01-01", periods=50, freq="B")     # Mon-Fri business days
    close = 100 * np.cumprod(1 + np.linspace(0.001, -0.001, 50))
    panel = pd.DataFrame({"A": close}, index=idx)
    by_day = sea.weekday_stats(panel)
    assert set(by_day) == {"Mon", "Tue", "Wed", "Thu", "Fri"}
    for name, r in by_day.items():
        assert (r.index.dayofweek == sea._WEEKDAY_NAMES.index(name)).all()
    total_n = sum(len(r) for r in by_day.values())
    assert total_n == len(panel)     # fillna(0.0) keeps every row, including day 1


def test_weekday_stats_respects_begin_ts():
    idx = pd.date_range("2024-01-01", periods=50, freq="B")
    close = 100 * np.cumprod(1 + np.full(50, 0.001))
    panel = pd.DataFrame({"A": close}, index=idx)
    begin = idx[20]
    by_day = sea.weekday_stats(panel, begin_ts=begin)
    for r in by_day.values():
        assert (r.index >= begin).all()


def test_print_weekday_stats_runs_without_error(capsys):
    rng = np.random.default_rng(4)
    idx = pd.date_range("2020-01-01", periods=300, freq="B", tz="UTC")
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 300))
    daily_data = {"A": pd.DataFrame({"open": close, "high": close, "low": close,
                                    "close": close, "volume": np.ones(300)}, index=idx)}
    sea.print_weekday_stats(daily_data, begin_ts=idx[50])
    out = capsys.readouterr().out
    assert "DAY-OF-WEEK EFFECT" in out and "Best day" in out


def test_run_smoke_end_to_end():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2020-01-01", periods=500, freq="B", tz="UTC")
    daily_data = {}
    for name in ("A", "B"):
        close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 500))
        daily_data[name] = pd.DataFrame({"open": close, "high": close, "low": close,
                                         "close": close, "volume": np.ones(500)}, index=idx)
    rc = sea.run(daily_data, begin_ts=idx[50])
    assert rc == 0


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        fn()
        print(f"{name} OK")
    print(f"\nALL {len(fns)} SEASONALITY TESTS PASSED")
