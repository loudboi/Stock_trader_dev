"""IBKR persisted pending orders must be recoverable after a session restart."""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot import portfolio_ibkr as pix
from bot import ibkr_universe as uni


def _trade(order_id, status, filled, remaining):
    return types.SimpleNamespace(
        order=types.SimpleNamespace(orderId=order_id),
        orderStatus=types.SimpleNamespace(status=status, filled=filled, remaining=remaining))


def _fill(order_id, side, price, shares, when):
    return types.SimpleNamespace(
        contract=types.SimpleNamespace(conId=111),
        execution=types.SimpleNamespace(
            orderId=order_id, side=side, price=price, shares=shares, time=when))


class RestartIB:
    def __init__(self):
        self.completed = []
        self.execution_history = []
        self.completed_calls = 0
        self.execution_calls = 0

    def trades(self):
        return []  # new API session has no in-memory trade history

    def reqCompletedOrders(self, apiOnly):
        self.completed_calls += 1
        assert apiOnly is True
        return list(self.completed)

    def fills(self):
        return []  # likewise empty after reconnect

    def reqExecutions(self):
        self.execution_calls += 1
        return list(self.execution_history)


def _adapter(fake):
    pf = pix.IBKRPortfolio.__new__(pix.IBKRPortfolio)
    pf.ib = fake
    pf._ensure = lambda: None
    pf._contract = lambda inst: types.SimpleNamespace(conId=111)
    pf._strategy_orders = {}
    return pf


def test_order_status_falls_back_to_completed_orders_after_restart():
    fake = RestartIB()
    fake.completed = [_trade(77, "Filled", 12, 0)]
    pf = _adapter(fake)
    status = pf.order_status("77")
    assert status == {"status": "filled", "filled_qty": 12.0, "qty": 12.0, "terminal": True}
    assert fake.completed_calls == 1


def test_exact_fill_price_falls_back_to_execution_history_after_restart():
    fake = RestartIB()
    fake.execution_history = [
        _fill(77, "BOT", 100.0, 1, "2026-08-10T12:00:00Z"),
        _fill(77, "BOT", 102.0, 3, "2026-08-10T12:00:01Z"),
    ]
    pf = _adapter(fake)
    px = pf.recent_fill_price(uni.IBKR_EUR_UNIVERSE["SAP"], "buy", order_id="77")
    assert abs(px - 101.5) < 1e-12
    assert fake.execution_calls == 1
