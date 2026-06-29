"""
Offline unit tests for the Strategy 4 logic (bot/strategies/trend_pullback.py).

Pure and index-based, so each branch is exercised with hand-built frames and
hand-built moving-average series (no network, no indicators warmup needed).

Run:  pytest tests/test_trend_pullback.py     (or: python tests/test_trend_pullback.py)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.strategies.trend_pullback import TrendPullbackStrategy, PullbackParams


def _frame(n=40, close=100.0, vol=1_000_000.0):
    """A flat OHLCV frame we then poke specific bars into."""
    idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    c = np.full(n, float(close))
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1,
                         "close": c, "volume": np.full(n, float(vol))}, index=idx)


def _const(series_like, value):
    return pd.Series(np.full(len(series_like), float(value)), index=series_like.index)


def _strat(**params):
    return TrendPullbackStrategy(None, PullbackParams(**params))


# --------------------------------------------------------------------------- #
# Principle 1 — trend gate
# --------------------------------------------------------------------------- #
def test_trend_ok_passes_in_aligned_uptrend():
    s = _strat()
    df = _frame()
    df.loc[df.index[-1], "close"] = 120
    ma_f = _const(df, 110)   # close > 50MA
    ma_s = _const(df, 100)   # close > 200MA and 50MA > 200MA
    assert s.trend_ok(df, ma_f, ma_s, len(df) - 1)


def test_trend_ok_fails_when_below_fast_ma():
    s = _strat()
    df = _frame()
    df.loc[df.index[-1], "close"] = 105
    ma_f = _const(df, 110)   # close < 50MA -> gate fails
    ma_s = _const(df, 100)
    assert not s.trend_ok(df, ma_f, ma_s, len(df) - 1)


def test_trend_ok_fails_when_fast_below_slow():
    s = _strat()
    df = _frame()
    df.loc[df.index[-1], "close"] = 120
    ma_f = _const(df, 110)
    ma_s = _const(df, 115)   # 50MA < 200MA (not a golden-cross alignment)
    assert not s.trend_ok(df, ma_f, ma_s, len(df) - 1)


# --------------------------------------------------------------------------- #
# Principle 2 — pullback entry
# --------------------------------------------------------------------------- #
def test_pullback_entry_fires_on_light_volume_rebound():
    s = _strat()
    df = _frame(n=40, close=120, vol=1_000_000)
    i = len(df) - 1
    ma_f = _const(df, 100)               # 50MA at 100

    # A dip toward the MA over the lookback, on contracted volume.
    for j in range(i - 4, i):
        df.loc[df.index[j], "low"] = 101      # within touch_band (2%) of the MA
        df.loc[df.index[j], "volume"] = 400_000  # well under the 0.90x baseline
    df.loc[df.index[i], "volume"] = 400_000
    # Current bar rebounds: closes up vs prior and back above the MA, low >= MA*0.98.
    df.loc[df.index[i - 1], "close"] = 104
    df.loc[df.index[i], "close"] = 108
    df.loc[df.index[i], "low"] = 105

    ok, reason = s.entry_signal(df, ma_f, i)
    assert ok and "pullback" in reason


def test_pullback_entry_blocked_when_volume_not_contracted():
    s = _strat()
    df = _frame(n=40, close=120, vol=1_000_000)
    i = len(df) - 1
    ma_f = _const(df, 100)
    for j in range(i - 4, i + 1):
        df.loc[df.index[j], "low"] = 101
        df.loc[df.index[j], "volume"] = 2_000_000   # volume EXPANDED, not contracted
    df.loc[df.index[i - 1], "close"] = 104
    df.loc[df.index[i], "close"] = 108
    df.loc[df.index[i], "low"] = 105
    ok, _ = s._pullback_entry(df, ma_f, i)
    assert ok is False


# --------------------------------------------------------------------------- #
# Principle 3 — breakout entry
# --------------------------------------------------------------------------- #
def test_breakout_entry_fires_on_tight_range_then_volume_break():
    s = _strat()
    df = _frame(n=30, close=100, vol=1_000_000)
    i = len(df) - 1
    c = s.p.consolidation_bars
    # Prior `c` bars form a tight range around 100 (well within 6%).
    for j in range(i - c, i):
        df.loc[df.index[j], "high"] = 101
        df.loc[df.index[j], "low"] = 99
        df.loc[df.index[j], "close"] = 100
        df.loc[df.index[j], "volume"] = 1_000_000
    # Current bar closes above the range high on >1.5x volume.
    df.loc[df.index[i], "close"] = 103
    df.loc[df.index[i], "volume"] = 2_000_000
    ok, reason = s._breakout_entry(df, i)
    assert ok and "breakout" in reason


def test_breakout_entry_blocked_without_volume():
    s = _strat()
    df = _frame(n=30, close=100, vol=1_000_000)
    i = len(df) - 1
    c = s.p.consolidation_bars
    for j in range(i - c, i):
        df.loc[df.index[j], "high"] = 101
        df.loc[df.index[j], "low"] = 99
        df.loc[df.index[j], "close"] = 100
    df.loc[df.index[i], "close"] = 103
    df.loc[df.index[i], "volume"] = 1_000_000   # no volume expansion
    ok, _ = s._breakout_entry(df, i)
    assert ok is False


# --------------------------------------------------------------------------- #
# Principle 4 — phased adds
# --------------------------------------------------------------------------- #
def test_should_add_when_price_extends_above_last_add():
    s = _strat(add_step=0.03)
    df = _frame()
    i = len(df) - 1
    ma_f = _const(df, 100)
    df.loc[df.index[i], "close"] = 110         # > MA and > +3% above last add (105)
    add, _ = s.should_add(df, ma_f, i, last_add_price=105)
    assert add is True


def test_should_not_add_below_threshold_or_below_ma():
    s = _strat(add_step=0.03)
    df = _frame()
    i = len(df) - 1
    ma_f = _const(df, 100)
    df.loc[df.index[i], "close"] = 106         # only +1% over last add (105)
    assert s.should_add(df, ma_f, i, 105)[0] is False
    df.loc[df.index[i], "close"] = 99          # below the MA entirely
    assert s.should_add(df, ma_f, i, 90)[0] is False


# --------------------------------------------------------------------------- #
# Volatility stop distance
# --------------------------------------------------------------------------- #
def test_stop_distance_floor_and_atr_scaling():
    s = _strat(atr_mult=2.0, min_stop=0.05)
    # Low ATR -> floored at 5%.
    assert s.stop_distance(atr=0.5, price=100) == 0.05
    # High ATR -> 2 * ATR / price.
    assert abs(s.stop_distance(atr=5.0, price=100) - 0.10) < 1e-12
    # Degenerate inputs fall back to the floor.
    assert s.stop_distance(atr=float("nan"), price=100) == 0.05
    assert s.stop_distance(atr=2.0, price=0) == 0.05


# --------------------------------------------------------------------------- #
# Principle 5 — trend-break exit
# --------------------------------------------------------------------------- #
def test_trend_exit_on_close_below_fast_ma():
    s = _strat()
    df = _frame()
    i = len(df) - 1
    ma_f = _const(df, 100)
    df.loc[df.index[i], "close"] = 98          # below the 50MA
    res = s.trend_exit(df, ma_f, i)
    assert res is not None and "50MA" in res[1]


def test_trend_exit_on_close_below_structural_low():
    s = _strat(structural_low_lookback=10)
    df = _frame()
    i = len(df) - 1
    ma_f = _const(df, 90)                       # above MA, so not the MA branch
    for j in range(i - 10, i):
        df.loc[df.index[j], "low"] = 95
    df.loc[df.index[i], "close"] = 94           # below the 10-bar swing low (95)
    res = s.trend_exit(df, ma_f, i)
    assert res is not None and "structural" in res[1]


def test_no_trend_exit_while_holding_the_trend():
    s = _strat()
    df = _frame()
    i = len(df) - 1
    ma_f = _const(df, 100)
    for j in range(i - 10, i):
        df.loc[df.index[j], "low"] = 95
    df.loc[df.index[i], "close"] = 110          # above MA and above the swing low
    assert s.trend_exit(df, ma_f, i) is None


def test_warmup_covers_slow_ma():
    assert _strat(ma_slow=200).warmup() == 205


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"{fn.__name__} OK")
    print(f"\nALL {len(fns)} TREND-PULLBACK TESTS PASSED")
