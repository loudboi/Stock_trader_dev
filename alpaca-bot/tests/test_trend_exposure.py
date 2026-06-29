"""
Offline tests for the trend-filtered exposure strategy (bot/trend_exposure.py).
Synthetic data, no network.

Run:  pytest tests/test_trend_exposure.py  (or: python tests/test_trend_exposure.py)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.trend_exposure as te


def _daily(close):
    n = len(close)
    idx = pd.date_range("2000-01-01", periods=n, freq="D", tz="UTC")
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": np.full(n, 1.0)}, index=idx)


# --------------------------------------------------------------------------- #
# Exposure signal
# --------------------------------------------------------------------------- #
def test_exposure_invested_above_ma_cash_below():
    # Rise above the MA, then fall below it.
    close = pd.Series(list(np.linspace(100, 140, 30)) + list(np.linspace(140, 80, 30)))
    exp = te.exposure_series(close, ma_period=10, buffer=0.0)
    assert exp.iloc[:9].sum() == 0           # warmup (MA undefined) -> flat
    assert exp.iloc[25] == 1.0               # solidly in the uptrend
    assert exp.iloc[-1] == 0.0               # moved to cash in the downtrend


def test_exposure_buffer_creates_hysteresis():
    # Sit just above the MA: a buffer should keep us from flip-flopping in.
    close = _daily(list(np.linspace(100, 110, 40)))["close"]
    no_buf = te.exposure_series(close, ma_period=10, buffer=0.0)
    big_buf = te.exposure_series(close, ma_period=10, buffer=0.10)
    # A 10% re-entry buffer should delay/reduce time invested vs no buffer.
    assert big_buf.sum() <= no_buf.sum()


def test_signal_has_no_lookahead():
    # Exposure is decided from data up to each bar; strategy_returns shifts it by
    # one day, so today's return can't depend on knowing today's close direction.
    close = _daily(list(np.linspace(100, 150, 60)))
    rets = te.strategy_returns(close, ma_period=10, buffer=0.0, leverage=1.0)
    # First tradeable return is only after warmup + the one-day shift (MA valid at
    # index 9 -> exposure shifts to index 10), so indices 0..9 are flat.
    assert rets.iloc[0] == 0.0
    assert (rets.iloc[:10] == 0.0).all()


# --------------------------------------------------------------------------- #
# Returns / leverage
# --------------------------------------------------------------------------- #
def test_cash_in_downtrend_avoids_the_loss():
    # Uptrend then crash. Strategy should exit and dodge most of the crash.
    up = np.linspace(100, 200, 60)
    crash = np.linspace(200, 100, 30)
    daily = _daily(np.concatenate([up, crash]))
    strat_eq = te.equity_from_returns(
        te.strategy_returns(daily, ma_period=20, buffer=0.0, leverage=1.0), begin_ts=None)
    bh_ret = daily["close"].iloc[-1] / daily["close"].iloc[0] - 1   # = 0 (round trip)
    strat_ret = strat_eq.iloc[-1] / strat_eq.iloc[0] - 1
    assert strat_ret > bh_ret                # captured up-leg, sat out the crash


def test_leverage_amplifies_invested_returns():
    daily = _daily(np.linspace(100, 200, 120))   # clean uptrend
    eq1 = te.equity_from_returns(
        te.strategy_returns(daily, 20, 0.0, 1.0), None)
    eq2 = te.equity_from_returns(
        te.strategy_returns(daily, 20, 0.0, 2.0), None)
    assert eq2.iloc[-1] > eq1.iloc[-1]       # 2x makes more in a pure uptrend


def test_borrow_rate_only_charges_the_levered_portion():
    daily = _daily(np.linspace(100, 200, 200))   # clean uptrend, always invested late
    base = te.strategy_returns(daily, 20, 0.0, leverage=1.0, borrow_rate=0.10)
    no_fee = te.strategy_returns(daily, 20, 0.0, leverage=2.0, borrow_rate=0.0)
    fee = te.strategy_returns(daily, 20, 0.0, leverage=2.0, borrow_rate=0.10)
    # Unlevered: borrow rate must not matter (nothing borrowed).
    assert np.allclose(base.values,
                       te.strategy_returns(daily, 20, 0.0, 1.0, 0.0).values)
    # Levered: financing drags the return below the no-fee version.
    assert te.equity_from_returns(fee, None).iloc[-1] < \
           te.equity_from_returns(no_fee, None).iloc[-1]


def test_run_smoke_two_symbols():
    a = _daily(np.linspace(100, 180, 300))
    b = _daily(np.linspace(100, 140, 300))
    rc = te.run({"A": a, "B": b}, ma_period=50, buffer=0.0, leverage=1.0,
                begin_ts=a.index[60])
    assert rc == 0


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        fn()
        print(f"{name} OK")
    print(f"\nALL {len(fns)} TREND-EXPOSURE TESTS PASSED")
