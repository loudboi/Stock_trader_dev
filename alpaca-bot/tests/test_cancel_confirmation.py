"""Broker-adapter cancellation confirmation tests; no network or broker required."""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot import portfolio as alp
from bot import portfolio_ibkr as ibp


class FakeAlpacaTrading:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.cancel_calls = []
        self.last = self.statuses[0] if self.statuses else "new"

    def get_order_by_id(self, order_id):
        if self.statuses:
            self.last = self.statuses.pop(0)
        return types.SimpleNamespace(status=self.last, filled_qty="0", qty="1")

    def cancel_order_by_id(self, order_id):
        self.cancel_calls.append(str(order_id))


def _alpaca_adapter(statuses):
    pf = alp.Portfolio.__new__(alp.Portfolio)
    pf.trading = FakeAlpacaTrading(statuses)
    return pf


def test_alpaca_cancel_waits_through_pending_cancel(monkeypatch):
    monkeypatch.setattr(alp, "_CANCEL_CONFIRM_POLLS", 3)
    monkeypatch.setattr(alp, "_CANCEL_POLL_SECONDS", 0.0)
    pf = _alpaca_adapter(["new", "pending_cancel", "canceled"])
    assert pf.cancel_order("42") is True
    assert pf.trading.cancel_calls == ["42"]


def test_alpaca_cancel_fails_if_order_fills_first(monkeypatch):
    monkeypatch.setattr(alp, "_CANCEL_CONFIRM_POLLS", 2)
    monkeypatch.setattr(alp, "_CANCEL_POLL_SECONDS", 0.0)
    pf = _alpaca_adapter(["new", "filled"])
    assert pf.cancel_order("42") is False


def test_alpaca_cancel_does_not_treat_request_acceptance_as_confirmation(monkeypatch):
    monkeypatch.setattr(alp, "_CANCEL_CONFIRM_POLLS", 2)
    monkeypatch.setattr(alp, "_CANCEL_POLL_SECONDS", 0.0)
    pf = _alpaca_adapter(["new", "pending_cancel", "pending_cancel"])
    assert pf.cancel_order("42") is False


def test_alpaca_nonfinal_special_states_are_not_terminal():
    for state in ("pending_cancel", "done_for_day", "stopped", "suspended", "calculated", "replaced"):
        pf = _alpaca_adapter([state])
        assert pf.order_status("42")["terminal"] is False


class FakeIB:
    def __init__(self):
        self.cancel_calls = []
        self.sleep_calls = []

    def cancelOrder(self, order):
        self.cancel_calls.append(order)

    def sleep(self, seconds):
        self.sleep_calls.append(seconds)


def _ibkr_adapter(statuses):
    pf = ibp.IBKRPortfolio.__new__(ibp.IBKRPortfolio)
    pf.ib = FakeIB()
    pf._ensure = lambda: None
    order = object()
    pf._strategy_orders = {"7": order}
    states = list(statuses)
    last = states[0] if states else {"status": "submitted", "terminal": False}

    def status(_order_id):
        nonlocal last
        if states:
            last = states.pop(0)
        return dict(last)

    pf.order_status = status
    return pf, order


def test_ibkr_cancel_waits_for_cancelled(monkeypatch):
    monkeypatch.setattr(ibp, "_CANCEL_CONFIRM_POLLS", 3)
    monkeypatch.setattr(ibp, "_CANCEL_POLL_SECONDS", 0.0)
    pf, order = _ibkr_adapter([
        {"status": "submitted", "terminal": False},
        {"status": "pendingcancel", "terminal": False},
        {"status": "cancelled", "terminal": True},
    ])
    assert pf.cancel_order("7") is True
    assert pf.ib.cancel_calls == [order]
    assert "7" not in pf._strategy_orders


def test_ibkr_cancel_fails_if_order_fills_first(monkeypatch):
    monkeypatch.setattr(ibp, "_CANCEL_CONFIRM_POLLS", 2)
    monkeypatch.setattr(ibp, "_CANCEL_POLL_SECONDS", 0.0)
    pf, _ = _ibkr_adapter([
        {"status": "submitted", "terminal": False},
        {"status": "filled", "terminal": True},
    ])
    assert pf.cancel_order("7") is False
    assert "7" not in pf._strategy_orders


def test_ibkr_cancel_timeout_keeps_order_for_later_reconciliation(monkeypatch):
    monkeypatch.setattr(ibp, "_CANCEL_CONFIRM_POLLS", 2)
    monkeypatch.setattr(ibp, "_CANCEL_POLL_SECONDS", 0.0)
    pf, _ = _ibkr_adapter([
        {"status": "submitted", "terminal": False},
        {"status": "pendingcancel", "terminal": False},
        {"status": "pendingcancel", "terminal": False},
    ])
    assert pf.cancel_order("7") is False
    assert "7" in pf._strategy_orders
