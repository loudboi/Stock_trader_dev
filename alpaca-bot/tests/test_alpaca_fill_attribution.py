"""Alpaca generic external-close fill recovery should aggregate the relevant sells."""
import os
import sys
import types

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from bot import portfolio as alp


class OrderHistoryTrading:
    def __init__(self, orders):
        self.orders = orders
        self.filters = []

    def get_orders(self, filter):
        self.filters.append(filter)
        return list(self.orders)


def _order(price, qty, filled_at):
    return types.SimpleNamespace(
        filled_avg_price=str(price), filled_qty=str(qty),
        filled_at=pd.Timestamp(filled_at).to_pydatetime())


def _adapter(orders):
    pf = alp.Portfolio.__new__(alp.Portfolio)
    pf.trading = OrderHistoryTrading(orders)
    return pf


def test_external_fill_price_is_weighted_across_sells_since_reconciliation_boundary():
    pf = _adapter([
        _order(105.0, 100, "2026-08-10T11:00:00Z"),  # before boundary, exclude
        _order(95.0, 20, "2026-08-10T12:00:00Z"),
        _order(90.0, 10, "2026-08-10T12:05:00Z"),
    ])
    px = pf.recent_fill_price(
        config.resolve_instrument("SPY"), "sell",
        since="2026-08-10T11:59:00Z")
    assert abs(px - (95.0 * 20 + 90.0 * 10) / 30.0) < 1e-12


def test_unbounded_generic_fill_lookup_keeps_latest_order_semantics():
    pf = _adapter([
        _order(95.0, 20, "2026-08-10T12:00:00Z"),
        _order(90.0, 10, "2026-08-10T12:05:00Z"),
    ])
    assert pf.recent_fill_price(config.resolve_instrument("SPY"), "sell") == 90.0
