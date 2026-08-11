"""Offline tests for the overnight/intraday return decomposition."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.overnight as on


def _daily(opens, closes, start="2020-01-01"):
    n = len(opens)
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC")
    o, c = np.asarray(opens, float), np.asarray(closes, float)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c),
                         "close": c, "volume": np.ones(n)}, index=idx)


def test_overnight_return_uses_prior_close_to_todays_open():
    closes = [100, 101, 102.01, 103.0301]
    opens = closes[:]
    r = on.overnight_return(_daily(opens, closes))
    # There is no prior close on the first row, therefore no overnight trade and no cost.
    assert r.iloc[0] == 0.0
    expected = 101 / 100 - 1.0 - on._DAILY_ROUNDTRIP_COST
    assert abs(r.iloc[1] - expected) < 1e-9


def test_intraday_return_uses_todays_open_to_close():
    daily = _daily([100, 100, 100, 100], [100, 102, 99, 100])
    r = on.intraday_return(daily)
    assert abs(r.iloc[1] - (0.02 - on._DAILY_ROUNDTRIP_COST)) < 1e-9
    assert abs(r.iloc[2] - (-0.01 - on._DAILY_ROUNDTRIP_COST)) < 1e-9


def test_daily_roundtrip_cost_is_double_the_slippage_convention():
    from bot.backtest_pullback import SLIPPAGE
    assert abs(on._DAILY_ROUNDTRIP_COST - 2 * SLIPPAGE) < 1e-12


def test_cost_per_side_is_configurable_and_scales_only_traded_rows():
    daily = _daily([100, 101, 102.01, 103.0301], [100, 101, 102.01, 103.0301])
    r_default = on.overnight_return(daily)
    r_zero = on.overnight_return(daily, cost_per_side=0.0)
    r_high = on.overnight_return(daily, cost_per_side=0.01)
    assert r_zero.iloc[0] == r_default.iloc[0] == r_high.iloc[0] == 0.0
    assert (r_zero.iloc[1:] > r_default.iloc[1:]).all()
    assert (r_default.iloc[1:] > r_high.iloc[1:]).all()
    assert abs((r_zero.iloc[1] - r_high.iloc[1]) - 0.02) < 1e-9


def test_portfolio_book_equal_weights_across_symbols():
    a = _daily([100, 100], [100, 110])
    b = _daily([100, 100], [100, 90])
    book = on.portfolio_book({"A": a, "B": b}, "intraday")
    expected = ((0.10 - on._DAILY_ROUNDTRIP_COST) +
                (-0.10 - on._DAILY_ROUNDTRIP_COST)) / 2
    assert abs(book.iloc[1] - expected) < 1e-9


def test_missing_market_keeps_its_capital_idle():
    a = _daily([100, 100, 100], [100, 110, 100])
    b = _daily([100, 100], [100, 100], start="2020-01-02")
    book = on.portfolio_book({"A": a, "B": b}, "intraday", cost_per_side=0.0)
    # On A's first date B has no observation; A still receives only its fixed 50% capital.
    assert abs(book.iloc[0]) < 1e-12


def test_overnight_and_intraday_compound_to_close_to_close_before_costs():
    daily = _daily([100, 102, 100, 101], [100, 105, 98, 103])
    on_ret = on.overnight_return(daily, cost_per_side=0.0)
    id_ret = on.intraday_return(daily, cost_per_side=0.0)
    full = daily["close"].pct_change().fillna(0.0)
    combined = (1 + on_ret) * (1 + id_ret) - 1
    assert np.allclose(combined.iloc[1:].values, full.iloc[1:].values, atol=1e-9)


def test_negative_cost_rejected():
    with pytest.raises(ValueError):
        on.overnight_return(_daily([100, 101], [100, 101]), cost_per_side=-0.01)


def test_run_smoke_end_to_end():
    rng = np.random.default_rng(1)
    n = 300
    daily_data = {}
    for name in ("A", "B"):
        closes = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
        opens = closes * (1 + rng.normal(0, 0.002, n))
        daily_data[name] = _daily(opens, closes)
    assert on.run(daily_data, begin_ts=daily_data["A"].index[50]) == 0
