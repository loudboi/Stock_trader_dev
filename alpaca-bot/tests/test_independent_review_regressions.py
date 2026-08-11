"""Regressions for independent live-safety review findings."""
import os
import sys
import types

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.live_pullback as lp
from bot import portfolio_ibkr as pix
from bot.strategies.trend_pullback import PullbackParams


class CaptureNotifier:
    def __init__(self):
        self.messages = []

    def notify(self, message):
        self.messages.append(message)


class SafetyPortfolio:
    def __init__(self):
        self.position = None
        self.statuses = {}
        self.cancel_results = {}
        self.cancel_calls = []
        self.stop_result = "stop-new"
        self.stop_calls = []
        self.market_result = "buy-1"
        self.market_calls = 0
        self.close_result = "close-1"
        self.close_calls = 0
        self.fill_price = None
        self.latest = 100.0
        self.bar_calls = 0
        self.bar_df = pd.DataFrame()

    def get_position_raw(self, inst):
        return dict(self.position) if self.position else None

    def order_status(self, order_id):
        value = self.statuses.get(str(order_id))
        return dict(value) if value is not None else None

    def cancel_order(self, order_id):
        oid = str(order_id)
        self.cancel_calls.append(oid)
        return bool(self.cancel_results.get(oid, False))

    def submit_stop_order(self, inst, qty, stop_price):
        self.stop_calls.append((inst.name, qty, stop_price))
        return self.stop_result

    def submit_market_order(self, inst, qty, side):
        self.market_calls += 1
        return self.market_result

    def close_position_raw(self, inst):
        self.close_calls += 1
        return self.close_result

    def recent_fill_price(self, inst, side, order_id=None, since=None):
        return self.fill_price

    def get_equity(self):
        return 100_000.0

    def latest_price(self, inst):
        return self.latest

    def is_tradable_now(self, inst):
        return True

    def get_historical_bars(self, inst, tf, start, end):
        self.bar_calls += 1
        return self.bar_df


def _position(qty=60.0, avg=100.0, stop_id=None):
    return {
        "qty": qty, "avg_entry": avg, "stop_dist": 0.05,
        "tranches": 1, "last_add_price": avg, "last_price": avg,
        "stop_order_id": stop_id,
        "stop_level": (95.0 if stop_id else None),
        "stop_qty": (qty if stop_id else 0.0),
        "entry_time": "2026-08-01T12:00:00+00:00", "entry_orders": [],
    }


def _trader(tmp_path, pf, symbol="SPY", adopt=False):
    n = CaptureNotifier()
    t = lp.PullbackLiveTrader(
        pf, [symbol], PullbackParams(),
        state_file=str(tmp_path / "state.json"),
        notifier=n, adopt_existing=adopt)
    return t, n


def test_external_flat_does_not_forget_unconfirmed_protective_stop(tmp_path):
    pf = SafetyPortfolio()
    t, n = _trader(tmp_path, pf)
    t.state["positions"]["SPY"] = _position(stop_id="stop-A")
    assert t._finalize_external_close(
        "SPY", t.instruments["SPY"], t.state["positions"]["SPY"]) is False
    assert "SPY" in t.state["positions"]
    assert t.state["positions"]["SPY"]["stop_order_id"] == "stop-A"
    assert pf.cancel_calls == ["stop-A"]
    assert any("ORPHAN-STOP" in m for m in n.messages)


def test_buy_flat_race_keeps_old_stop_known_until_terminal(tmp_path):
    pf = SafetyPortfolio()
    t, n = _trader(tmp_path, pf)
    t.state["positions"]["SPY"] = _position(qty=50.0, stop_id="stop-A")
    pending = {
        "type": "buy", "order_id": "buy-1", "before_qty": 50.0,
        "before_avg_entry": 100.0, "tranche_index": 1,
        "stop_dist": 0.05, "signal_price": 101.0,
        "requested_qty": 20.0, "how": "test",
        "submitted_at": "2026-08-10T12:00:00+00:00",
    }
    t.state["pending"]["SPY"] = pending
    assert t._settle_terminal_buy(
        "SPY", t.instruments["SPY"], pending, None, 20.0) is False
    assert t.state["positions"]["SPY"]["stop_order_id"] == "stop-A"
    assert "SPY" in t.state["pending"]
    assert any("ORPHAN-STOP" in m for m in n.messages)


def test_ambiguous_buy_submission_blocks_without_resubmitting(tmp_path):
    pf = SafetyPortfolio()
    pf.market_result = None
    t, n = _trader(tmp_path, pf)
    assert t._buy_tranche(
        "SPY", t.instruments["SPY"], 100.0, 0, 0.05, "test") is False
    assert t.state["pending"]["SPY"]["type"] == "buy_submit"
    assert pf.market_calls == 1
    assert t._buy_tranche(
        "SPY", t.instruments["SPY"], 100.0, 0, 0.05, "test") is False
    assert pf.market_calls == 1
    assert any("CRITICAL ORDER RECONCILIATION" in m for m in n.messages)


def test_pending_buy_reserves_portfolio_capacity(tmp_path):
    pf = SafetyPortfolio()
    t, _ = _trader(tmp_path, pf)
    t.state["pending"]["SPY"] = {
        "type": "buy_submit", "submission_id": "s1",
        "before_qty": 0.0, "before_avg_entry": 0.0,
        "tranche_index": 0, "stop_dist": 0.05,
        "signal_price": 100.0, "requested_qty": 100.0,
        "how": "test", "submitted_at": "2026-08-10T12:00:00+00:00",
    }
    room = t._managed_capacity(100_000.0)
    assert room["gross_room"] == 90_000.0
    assert room["risk_room"] == 4_500.0


def test_adoption_reports_missing_equity_stop(tmp_path):
    pf = SafetyPortfolio()
    pf.stop_result = None
    t, n = _trader(tmp_path, pf, adopt=True)
    broker = {"side": "long", "qty": 10.0, "avg_entry": 100.0}
    t._handle_untracked_broker_long(
        "SPY", t.instruments["SPY"], broker)
    assert t.state["positions"]["SPY"]["stop_order_id"] is None
    assert any("protective stop unavailable" in m for m in n.messages)
    assert any("CRITICAL" in m for m in n.messages)


def test_unrecoverable_old_pending_with_broker_long_alerts(tmp_path):
    pf = SafetyPortfolio()
    pf.position = {"side": "long", "qty": 60.0, "avg_entry": 100.0}
    t, n = _trader(tmp_path, pf)
    t.state["pending"]["SPY"] = {
        "type": "buy", "order_id": "old-buy", "before_qty": 0.0,
        "before_avg_entry": 0.0, "tranche_index": 0,
        "stop_dist": 0.05, "signal_price": 100.0,
        "requested_qty": 60.0, "how": "test",
        "submitted_at": "2026-08-01T12:00:00+00:00",
    }
    assert t._confirm_pending("SPY", t.instruments["SPY"]) is False
    assert any("CRITICAL ORDER RECONCILIATION" in m for m in n.messages)


def test_stale_intent_is_not_fresh_for_fallback(tmp_path):
    pf = SafetyPortfolio()
    t, _ = _trader(tmp_path, pf)
    daily = pd.DataFrame(
        index=pd.date_range("2026-08-01", periods=3, freq="D", tz="UTC"))
    fresh = {"signal_ts": str(daily.index[1])}
    stale = {"signal_ts": str(daily.index[0])}
    assert t._intent_is_fresh_for_fallback(fresh, daily, 2)
    assert not t._intent_is_fresh_for_fallback(stale, daily, 2)


def test_ibkr_inactive_is_terminal_nonworking_state():
    trade = types.SimpleNamespace(
        order=types.SimpleNamespace(orderId=7),
        orderStatus=types.SimpleNamespace(
            status="Inactive", filled=0, remaining=10))
    _, status = pix.IBKRPortfolio._normalized_trade_status(trade)
    assert status["terminal"] is True
    assert status["filled_qty"] == 0.0


def test_ibkr_already_flat_sentinel_normalizes_terminally():
    pf = pix.IBKRPortfolio.__new__(pix.IBKRPortfolio)
    pf._ensure = lambda: (_ for _ in ()).throw(
        AssertionError("sentinel should not query IBKR"))
    status = pf.order_status("already-flat")
    assert status == {
        "status": "already-flat", "filled_qty": 0.0,
        "qty": 0.0, "terminal": True}
