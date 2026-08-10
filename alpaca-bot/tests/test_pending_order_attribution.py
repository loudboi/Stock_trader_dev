"""Pending strategy orders must be reconciled to their own broker order state."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.live_pullback as lp
from bot.strategies.trend_pullback import PullbackParams


class QuietNotifier:
    def __init__(self):
        self.messages = []

    def notify(self, message):
        self.messages.append(message)


class AttributionPortfolio:
    def __init__(self):
        self.position = None
        self.status = None
        self.cancel_result = False
        self.cancel_calls = []
        self.fill_price = None
        self.stop_orders = []

    def get_position_raw(self, inst):
        return dict(self.position) if self.position else None

    def order_status(self, order_id):
        return dict(self.status) if self.status is not None else None

    def cancel_order(self, order_id):
        self.cancel_calls.append(str(order_id))
        return self.cancel_result

    def recent_fill_price(self, inst, side, order_id=None, since=None):
        return self.fill_price

    def submit_stop_order(self, inst, qty, stop_price):
        self.stop_orders.append((qty, stop_price))
        return f"stop-{len(self.stop_orders)}"

    def get_historical_bars(self, inst, tf, start, end):
        import pandas as pd
        return pd.DataFrame()


def _trader(tmp_path, pf):
    return lp.PullbackLiveTrader(
        pf, ["SPY"], PullbackParams(), state_file=str(tmp_path / "state.json"),
        notifier=QuietNotifier())


def _pending_buy():
    return {
        "type": "buy", "order_id": "buy-1", "before_qty": 0.0,
        "tranche_index": 0, "stop_dist": 0.05, "signal_price": 100.0,
        "requested_qty": 60.0, "how": "test",
        "submitted_at": "2026-08-10T12:00:00+00:00",
    }


def _position(qty=60.0):
    return {
        "qty": qty, "avg_entry": 100.0, "stop_dist": 0.05,
        "tranches": 1, "last_add_price": 100.0, "last_price": 100.0,
        "stop_order_id": None, "stop_level": None, "stop_qty": 0.0,
        "entry_time": "2026-08-01T12:00:00+00:00", "entry_orders": [],
    }


def _pending_close(pos):
    return {
        "type": "close", "order_id": "sell-1", "reason": "test close",
        "requested_exit_price": 94.0, "position": dict(pos),
        "submitted_at": "2026-08-10T12:00:00+00:00",
    }


def test_external_buy_does_not_complete_nonterminal_bot_buy(tmp_path):
    pf = AttributionPortfolio()
    pf.position = {"side": "long", "qty": 60.0, "avg_entry": 101.0}
    pf.status = {"status": "new", "filled_qty": 0.0, "qty": 60.0, "terminal": False}
    t = _trader(tmp_path, pf)
    t.state["pending"]["SPY"] = _pending_buy()

    assert t._confirm_pending("SPY", t.instruments["SPY"]) is False
    assert "SPY" in t.state["pending"]
    assert "SPY" not in t.state["positions"]
    assert pf.cancel_calls == ["buy-1"]


def test_external_flat_does_not_complete_nonterminal_bot_close(tmp_path):
    pf = AttributionPortfolio()
    pf.position = None  # human/external close happened, but bot sell is still working
    pf.status = {"status": "new", "filled_qty": 0.0, "qty": 60.0, "terminal": False}
    t = _trader(tmp_path, pf)
    pos = _position()
    t.state["positions"]["SPY"] = pos
    t.state["pending"]["SPY"] = _pending_close(pos)

    assert t._confirm_pending("SPY", t.instruments["SPY"]) is False
    assert "SPY" in t.state["pending"]
    assert "SPY" in t.state["positions"]
    assert pf.cancel_calls == ["sell-1"]


def test_terminal_cancelled_buy_with_zero_bot_fill_does_not_adopt_external_long(tmp_path):
    pf = AttributionPortfolio()
    pf.position = {"side": "long", "qty": 60.0, "avg_entry": 101.0}
    pf.status = {"status": "canceled", "filled_qty": 0.0, "qty": 60.0, "terminal": True}
    t = _trader(tmp_path, pf)
    t.state["pending"]["SPY"] = _pending_buy()

    assert t._confirm_pending("SPY", t.instruments["SPY"]) is True
    assert "SPY" not in t.state["pending"]
    assert "SPY" not in t.state["positions"]


def test_terminal_filled_buy_uses_exact_order_fill_state(tmp_path):
    pf = AttributionPortfolio()
    pf.position = {"side": "long", "qty": 60.0, "avg_entry": 100.5}
    pf.status = {"status": "filled", "filled_qty": 60.0, "qty": 60.0, "terminal": True}
    pf.fill_price = 100.5
    t = _trader(tmp_path, pf)
    t.state["pending"]["SPY"] = _pending_buy()

    assert t._confirm_pending("SPY", t.instruments["SPY"]) is True
    assert t.state["positions"]["SPY"]["qty"] == 60.0
    assert t.state["positions"]["SPY"]["entry_orders"][0]["filled_qty"] == 60.0


def test_full_bot_close_then_external_rebuy_is_not_managed_as_partial_remainder(tmp_path, monkeypatch):
    pf = AttributionPortfolio()
    # Bot sold all 60, then a separate external action bought 10 before reconciliation.
    pf.position = {"side": "long", "qty": 10.0, "avg_entry": 97.0}
    pf.status = {"status": "filled", "filled_qty": 60.0, "qty": 60.0, "terminal": True}
    pf.fill_price = 94.0
    t = _trader(tmp_path, pf)
    pos = _position()
    t.state["positions"]["SPY"] = pos
    t.state["pending"]["SPY"] = _pending_close(pos)
    monkeypatch.setattr(lp, "PULLBACK_TRADES_CSV", str(tmp_path / "trades.csv"))

    assert t._confirm_pending("SPY", t.instruments["SPY"]) is True
    assert "SPY" not in t.state["pending"]
    assert "SPY" not in t.state["positions"]
