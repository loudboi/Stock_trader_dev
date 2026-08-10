"""Small regressions for calendar-sensitive tax scenario boundaries."""

import pandas as pd

import bot.taxes as tx


def test_year_end_event_does_not_assume_every_december_28_is_year_end():
    idx = pd.DatetimeIndex([pd.Timestamp("2026-12-28", tz="UTC")])
    assert tx._year_end_event(idx, 0, realize_final=False) is False


def test_year_end_event_recognizes_final_business_day_without_next_year_row():
    # 2026-12-31 is a Thursday; the next generic business day is in 2027.
    idx = pd.DatetimeIndex([pd.Timestamp("2026-12-31", tz="UTC")])
    assert tx._year_end_event(idx, 0, realize_final=False) is True


def test_explicit_final_realization_still_overrides_calendar_position():
    idx = pd.DatetimeIndex([pd.Timestamp("2026-08-10", tz="UTC")])
    assert tx._year_end_event(idx, 0, realize_final=True) is True
