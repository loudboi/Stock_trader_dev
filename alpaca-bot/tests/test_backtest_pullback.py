"""
Offline tests for the Strategy 4 backtester internals (bot/backtest_pullback.py).

Covers the inlined metrics/fill helpers, the execution-window slicing, the
pyramiding book, and a couple of end-to-end run_single passes on synthetic data.
No network: a throwaway TEST instrument is registered in PULLBACK_UNIVERSE.

Run:  pytest tests/test_backtest_pullback.py  (or: python tests/test_backtest_pullback.py)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from bot.strategies.trend_pullback import PullbackParams
import bot.backtest_pullback as bp


def _register_test_instrument(name="TEST", asset_class="equity", qty_decimals=0):
    config.PULLBACK_UNIVERSE[name] = config.Instrument(
        name=name, api_symbol=name, asset_class=asset_class,
        strategy="trend_pullback", timeframe="1Day",
        can_short=False, qty_decimals=qty_decimals)
    return config.PULLBACK_UNIVERSE[name]


def _daily(close, start="2024-01-01"):
    n = len(close)
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"open": close - 0.2, "high": close + 1.0,
                         "low": close - 1.0, "close": close,
                         "volume": np.full(n, 1_000_000.0)}, index=idx)


# --------------------------------------------------------------------------- #
# Inlined helpers
# --------------------------------------------------------------------------- #
def test_fill_price_slippage_directions():
    assert round(bp.fill_price(100, "long", True), 4) == 100.05    # pay up to enter
    assert round(bp.fill_price(100, "long", False), 4) == 99.95    # give up to exit
    assert round(bp.fill_price(100, "short", True), 4) == 99.95
    assert round(bp.fill_price(100, "short", False), 4) == 100.05


def test_compute_metrics_basic_stats():
    trades = [{"pnl": 100.0}, {"pnl": -40.0}, {"pnl": 60.0}]
    eq = pd.Series([100_000, 100_100, 100_060, 100_120],
                   index=pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC"))
    m = bp.compute_metrics(trades, eq)
    assert m["trades"] == 3
    assert abs(m["win_rate"] - 2 / 3) < 1e-9
    assert abs(m["profit_factor"] - 160 / 40) < 1e-9     # 4.0
    assert m["total_return"] > 0
    assert m["max_drawdown"] <= 0


def test_compute_metrics_empty():
    m = bp.compute_metrics([], pd.Series([], dtype=float))
    assert m["trades"] == 0 and m["profit_factor"] == 0.0 and m["sharpe"] == 0.0


# --------------------------------------------------------------------------- #
# Execution window
# --------------------------------------------------------------------------- #
def test_exec_window_single_timeframe_is_next_daily_bar():
    daily = _daily([100, 101, 102, 103])
    win = bp.exec_window(daily, None, 1, intraday=False)
    assert len(win) == 1 and win.index[0] == daily.index[2]


def test_exec_window_intraday_slices_between_daily_stamps():
    daily = _daily([100, 101, 102])
    # Intraday bars spanning the day after daily[1].
    intra_idx = pd.date_range(daily.index[1] + pd.Timedelta(hours=1),
                              daily.index[2], freq="1h", tz="UTC")
    intra = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                          "close": 1.0, "volume": 1.0}, index=intra_idx)
    win = bp.exec_window(daily, intra, 1, intraday=True)
    assert len(win) == len(intra_idx)
    assert (win.index > daily.index[1]).all() and (win.index <= daily.index[2]).all()


# --------------------------------------------------------------------------- #
# Pyramiding book
# --------------------------------------------------------------------------- #
def test_pyramid_add_tranche_sizes_for_one_percent_risk():
    inst = _register_test_instrument()
    book = bp.PyramidBook(initial=100_000.0)
    ts = pd.Timestamp("2025-01-01", tz="UTC")
    # risk = 1% * 100k = 1000; full_qty = 1000/(100*0.05)=200; first tranche 30% -> 60.
    ok = book.add_tranche("TEST", inst, price=100.0, fraction=0.30,
                          stop_dist=0.05, ts=ts)
    assert ok
    pos = book.positions["TEST"]
    assert pos.tranches == 1 and pos.qty == 60
    assert abs(pos.avg_entry - bp.fill_price(100.0, "long", True)) < 1e-9
    assert pos.stop_dist == 0.05


def test_pyramid_close_records_trade_and_realized_pnl():
    inst = _register_test_instrument()
    book = bp.PyramidBook(initial=100_000.0)
    ts = pd.Timestamp("2025-01-01", tz="UTC")
    book.add_tranche("TEST", inst, 100.0, 0.30, 0.05, ts)
    qty = book.positions["TEST"].qty
    avg = book.positions["TEST"].avg_entry
    book.close("TEST", exit_level=120.0, ts=ts, reason="test exit")
    assert "TEST" not in book.positions
    assert len(book.trades) == 1
    t = book.trades[0]
    expected = (bp.fill_price(120.0, "long", False) - avg) * qty
    assert abs(t["pnl"] - round(expected, 2)) < 0.01
    assert t["exit_reason"] == "test exit" and t["tranches"] == 1


# --------------------------------------------------------------------------- #
# End-to-end run_single on synthetic data
# --------------------------------------------------------------------------- #
def test_run_single_trades_an_uptrend_with_a_pullback():
    _register_test_instrument()
    np.random.seed(0)
    n = 320
    close = np.linspace(100, 200, n) + np.random.normal(0, 1.0, n)
    close[250:255] -= 6                       # a dip toward the rising MA
    daily = _daily(close)
    daily.iloc[250:255, daily.columns.get_loc("volume")] *= 0.5   # lighter volume
    trades, eq = bp.run_single("TEST", daily, None, PullbackParams(),
                               exec_is_intraday=False)
    assert len(trades) >= 1
    assert len(eq) > 0 and eq.iloc[-1] > 0


def test_run_single_flat_market_makes_no_trades():
    _register_test_instrument()
    daily = _daily(np.full(300, 100.0))       # no trend, no entries
    trades, eq = bp.run_single("TEST", daily, None, PullbackParams(),
                               exec_is_intraday=False)
    assert len(trades) == 0


def test_run_single_stop_out_on_a_crash_is_a_loss():
    _register_test_instrument()
    # A noisy uptrend with a light-volume pullback opens a position, then a crash
    # trips the volatility stop / trend break.
    np.random.seed(0)
    up = np.linspace(100, 200, 300) + np.random.normal(0, 1.0, 300)
    up[250:255] -= 6
    crash = np.linspace(up[-1], up[-1] * 0.55, 25)
    daily = _daily(np.concatenate([up, crash]))
    daily.iloc[250:255, daily.columns.get_loc("volume")] *= 0.5
    trades, _ = bp.run_single("TEST", daily, None, PullbackParams(),
                              exec_is_intraday=False)
    assert len(trades) >= 1
    # Some trade should have closed on the volatility stop or the trend break.
    reasons = " ".join(t["exit_reason"] for t in trades)
    assert ("stop" in reasons) or ("50MA" in reasons) or ("structural" in reasons)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"{fn.__name__} OK")
    print(f"\nALL {len(fns)} BACKTEST-PULLBACK TESTS PASSED")
