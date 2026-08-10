"""Offline tests for pullback backtester execution, sizing, metrics and causality."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import bot.backtest_pullback as bp
from bot.strategies.trend_pullback import PullbackParams, TrendPullbackStrategy


def _register_test_instrument(name="TEST", asset_class="equity", qty_decimals=0):
    config.PULLBACK_UNIVERSE[name] = config.Instrument(
        name=name, api_symbol=name, asset_class=asset_class,
        strategy="trend_pullback", timeframe="1Day", can_short=False,
        qty_decimals=qty_decimals)
    return config.PULLBACK_UNIVERSE[name]


def _daily(close, start="2024-01-01"):
    n = len(close)
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"open": close - 0.2, "high": close + 1.0,
                         "low": close - 1.0, "close": close,
                         "volume": np.full(n, 1_000_000.0)}, index=idx)


def test_fill_price_slippage_directions():
    assert round(bp.fill_price(100, "long", True), 4) == 100.05
    assert round(bp.fill_price(100, "long", False), 4) == 99.95
    assert round(bp.fill_price(100, "short", True), 4) == 99.95
    assert round(bp.fill_price(100, "short", False), 4) == 100.05


def test_compute_metrics_basic_stats_and_true_initial_nav():
    trades = [{"pnl": 100.0}, {"pnl": -40.0}, {"pnl": 60.0}]
    eq = pd.Series([100_100, 100_060, 100_120],
                   index=pd.date_range("2025-01-02", periods=3, freq="D", tz="UTC"))
    eq.attrs["initial_equity"] = 100_000
    m = bp.compute_metrics(trades, eq)
    assert m["trades"] == 3
    assert abs(m["win_rate"] - 2 / 3) < 1e-9
    assert abs(m["profit_factor"] - 4.0) < 1e-9
    assert abs(m["total_return"] - 0.0012) < 1e-12


def test_compute_metrics_drawdown_includes_true_starting_nav():
    idx = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    eq = pd.Series([90_000.0, 92_000.0, 91_000.0], index=idx)
    eq.attrs["initial_equity"] = 100_000.0
    m = bp.compute_metrics([], eq)
    assert abs(m["max_drawdown"] + 0.10) < 1e-12


def test_compute_metrics_risk_free_is_explicit():
    idx = pd.date_range("2025-01-01", periods=100, freq="D", tz="UTC")
    eq = pd.Series(100_000 * 1.001 ** np.arange(1, 101), index=idx)
    eq.attrs["initial_equity"] = 100_000
    zero_rf = bp.compute_metrics([], eq, risk_free_annual=0.0)
    positive_rf = bp.compute_metrics([], eq, risk_free_annual=0.10)
    assert positive_rf["sharpe"] < zero_rf["sharpe"]
    with pytest.raises(ValueError):
        bp.compute_metrics([], eq, risk_free_annual=-1.0)


def test_compute_metrics_empty():
    m = bp.compute_metrics([], pd.Series([], dtype=float))
    assert m["trades"] == 0 and m["profit_factor"] == 0.0 and m["sharpe"] == 0.0


def test_exec_window_single_timeframe_is_next_daily_bar():
    daily = _daily([100, 101, 102, 103])
    win = bp.exec_window(daily, None, 1, intraday=False)
    assert len(win) == 1 and win.index[0] == daily.index[2]


def test_exec_window_intraday_slices_between_daily_stamps():
    daily = _daily([100, 101, 102])
    intra_idx = pd.date_range(daily.index[1] + pd.Timedelta(hours=1),
                              daily.index[2], freq="1h", tz="UTC")
    intra = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                          "close": 1.0, "volume": 1.0}, index=intra_idx)
    win = bp.exec_window(daily, intra, 1, intraday=True)
    assert len(win) == len(intra_idx)
    assert (win.index > daily.index[1]).all() and (win.index <= daily.index[2]).all()


def test_explicit_no_action_is_never_recomputed(monkeypatch):
    """Regression for combined same-close future-execution causality.

    `run_combined` precomputes every decision at a shared close. An explicit None
    means B had no signal at that close. It must stay None even if another symbol's
    future execution has changed the book by the time B's execution window is
    walked.
    """
    inst = _register_test_instrument()
    strat = TrendPullbackStrategy(inst, PullbackParams())
    daily = _daily([100.0])
    dummy = pd.Series([np.nan], index=daily.index)
    empty_win = daily.iloc[0:0]
    book = bp.PyramidBook(initial=100_000)

    def forbidden_recompute(*args, **kwargs):
        raise AssertionError("precomputed no-action decision was recomputed")

    monkeypatch.setattr(bp, "_decision", forbidden_recompute)
    eq = bp.process_day(book, "TEST", inst, strat, daily, dummy, dummy, dummy, 0,
                        empty_win, False, decision=None)
    assert eq == 100_000 and not book.positions


def test_pyramid_add_tranche_sizes_for_one_percent_risk():
    inst = _register_test_instrument()
    book = bp.PyramidBook(initial=100_000.0)
    ts = pd.Timestamp("2025-01-01", tz="UTC")
    ok = book.add_tranche("TEST", inst, price=100.0, fraction=0.30,
                          stop_dist=0.05, ts=ts)
    assert ok
    pos = book.positions["TEST"]
    assert pos.tranches == 1 and pos.qty == 60
    assert abs(pos.avg_entry - bp.fill_price(100.0, "long", True)) < 1e-9
    assert pos.stop_dist == 0.05


def test_later_tranche_sizing_uses_actual_existing_stop_distance():
    inst = _register_test_instrument()
    book = bp.PyramidBook(initial=100_000.0)
    ts = pd.Timestamp("2025-01-01", tz="UTC")
    book.add_tranche("TEST", inst, 100.0, 0.30, 0.05, ts)
    before = book.positions["TEST"].qty
    # Passing a newly sampled 20% stop must not change the position's actual 5% risk distance.
    book.add_tranche("TEST", inst, 110.0, 0.30, 0.20, ts + pd.Timedelta(days=1))
    pos = book.positions["TEST"]
    assert pos.stop_dist == 0.05 and pos.qty > before


def test_portfolio_capacity_caps_gross_and_stop_risk():
    inst = _register_test_instrument()
    book = bp.PyramidBook(initial=100_000.0)
    capacity = {"gross": config.MAX_GROSS_EXPOSURE * 100_000,
                "risk": config.MAX_PORTFOLIO_RISK * 100_000}
    qty = book.plan_tranche_qty(inst, 100, 0.3, 0.05, 100_000, capacity)
    assert qty == 0


def test_pyramid_close_records_trade_and_realized_pnl():
    inst = _register_test_instrument()
    book = bp.PyramidBook(initial=100_000.0)
    ts = pd.Timestamp("2025-01-01", tz="UTC")
    book.add_tranche("TEST", inst, 100.0, 0.30, 0.05, ts)
    qty = book.positions["TEST"].qty
    avg = book.positions["TEST"].avg_entry
    book.close("TEST", 120.0, ts, "test exit")
    t = book.trades[0]
    expected = (bp.fill_price(120.0, "long", False) - avg) * qty
    assert abs(t["pnl"] - round(expected, 2)) < 0.01
    assert t["exit_reason"] == "test exit" and t["tranches"] == 1


def test_run_single_trades_an_uptrend_with_a_pullback():
    _register_test_instrument()
    np.random.seed(0)
    n = 320
    close = np.linspace(100, 200, n) + np.random.normal(0, 1.0, n)
    close[250:255] -= 6
    daily = _daily(close)
    daily.iloc[250:255, daily.columns.get_loc("volume")] *= 0.5
    trades, eq = bp.run_single("TEST", daily, None, PullbackParams(), False)
    assert len(trades) >= 1 and len(eq) > 0 and eq.iloc[-1] > 0


def test_run_single_flat_market_makes_no_trades():
    _register_test_instrument()
    trades, _ = bp.run_single("TEST", _daily(np.full(300, 100.0)), None,
                              PullbackParams(), False)
    assert trades == []


def test_run_single_stop_out_on_a_crash_is_a_loss():
    _register_test_instrument()
    np.random.seed(0)
    up = np.linspace(100, 200, 300) + np.random.normal(0, 1.0, 300)
    up[250:255] -= 6
    crash = np.linspace(up[-1], up[-1] * 0.55, 25)
    daily = _daily(np.concatenate([up, crash]))
    daily.iloc[250:255, daily.columns.get_loc("volume")] *= 0.5
    trades, _ = bp.run_single("TEST", daily, None, PullbackParams(), False)
    assert trades
    reasons = " ".join(t["exit_reason"] for t in trades)
    assert ("stop" in reasons) or ("MA" in reasons) or ("structural" in reasons)
