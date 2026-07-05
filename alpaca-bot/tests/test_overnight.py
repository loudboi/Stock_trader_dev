"""
Offline tests for the overnight/intraday return decomposition (bot/overnight.py).
Synthetic OHLC data, no network.

Run:  pytest tests/test_overnight.py    (or: python tests/test_overnight.py)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.overnight as on


def _daily(opens, closes, start="2020-01-01"):
    n = len(opens)
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC")
    o, c = np.asarray(opens, float), np.asarray(closes, float)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c),
                         "close": c, "volume": np.ones(n)}, index=idx)


def test_overnight_return_uses_prior_close_to_todays_open():
    # Flat sessions (open == close), but each day gaps up 1% overnight.
    closes = [100, 101, 102.01, 103.0301]
    opens = [100, 101, 102.01, 103.0301]        # session itself is flat
    daily = _daily(opens, closes)
    r = on.overnight_return(daily)
    assert r.iloc[0] == -on._DAILY_ROUNDTRIP_COST     # first bar has no prior close -> 0% + cost
    expected = 101 / 100 - 1.0 - on._DAILY_ROUNDTRIP_COST
    assert abs(r.iloc[1] - expected) < 1e-9


def test_intraday_return_uses_todays_open_to_close():
    opens = [100, 100, 100, 100]
    closes = [100, 102, 99, 100]                 # sessions move, no overnight gaps assumed
    daily = _daily(opens, closes)
    r = on.intraday_return(daily)
    assert abs(r.iloc[1] - (102 / 100 - 1.0 - on._DAILY_ROUNDTRIP_COST)) < 1e-9
    assert abs(r.iloc[2] - (99 / 100 - 1.0 - on._DAILY_ROUNDTRIP_COST)) < 1e-9


def test_daily_roundtrip_cost_is_double_the_slippage_convention():
    from bot.backtest_pullback import SLIPPAGE
    assert abs(on._DAILY_ROUNDTRIP_COST - 2 * SLIPPAGE) < 1e-12


def test_cost_per_side_is_configurable_and_scales_the_drag():
    closes = [100, 101, 102.01, 103.0301]
    opens = closes[:]
    daily = _daily(opens, closes)
    r_default = on.overnight_return(daily)                       # SLIPPAGE default
    r_zero = on.overnight_return(daily, cost_per_side=0.0)
    r_high = on.overnight_return(daily, cost_per_side=0.01)
    assert (r_zero.iloc[1:] > r_default.iloc[1:]).all()          # no cost -> higher return
    assert (r_default.iloc[1:] > r_high.iloc[1:]).all()          # higher cost -> lower return
    assert abs((r_zero.iloc[1] - r_high.iloc[1]) - 2 * 0.01) < 1e-9


def test_portfolio_book_equal_weights_across_symbols():
    a = _daily([100, 100], [100, 110])     # +10% intraday
    b = _daily([100, 100], [100, 90])      # -10% intraday
    book = on.portfolio_book({"A": a, "B": b}, "intraday")
    expected = ((0.10 - on._DAILY_ROUNDTRIP_COST) + (-0.10 - on._DAILY_ROUNDTRIP_COST)) / 2
    assert abs(book.iloc[1] - expected) < 1e-9


def test_overnight_and_intraday_sum_to_close_to_close_return_before_costs():
    # Without the round-trip cost, overnight + intraday returns should reconstruct
    # the full close-to-close return (basic decomposition identity).
    closes = [100, 105, 98, 103]
    opens = [100, 102, 100, 101]
    daily = _daily(opens, closes)
    on_ret = on.overnight_return(daily) + on._DAILY_ROUNDTRIP_COST     # strip the cost
    id_ret = on.intraday_return(daily) + on._DAILY_ROUNDTRIP_COST
    full = daily["close"].pct_change().fillna(0.0)
    combined = (1 + on_ret) * (1 + id_ret) - 1
    assert np.allclose(combined.iloc[1:].values, full.iloc[1:].values, atol=1e-9)


def test_run_smoke_end_to_end():
    rng = np.random.default_rng(1)
    n = 300
    daily_data = {}
    for name in ("A", "B"):
        closes = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
        opens = closes * (1 + rng.normal(0, 0.002, n))
        daily_data[name] = _daily(opens, closes)
    rc = on.run(daily_data, begin_ts=daily_data["A"].index[50])
    assert rc == 0


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        fn()
        print(f"{name} OK")
    print(f"\nALL {len(fns)} OVERNIGHT TESTS PASSED")
