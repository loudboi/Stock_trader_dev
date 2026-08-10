"""IBKR fill recovery should return an order-level weighted average."""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot import portfolio_ibkr as pix
from bot import ibkr_universe as uni


class FillIB:
    def __init__(self, fills):
        self._fills = fills

    def fills(self):
        return list(self._fills)


def _fill(order_id, side, price, shares, when):
    return types.SimpleNamespace(
        contract=types.SimpleNamespace(conId=111),
        execution=types.SimpleNamespace(
            orderId=order_id, side=side, price=price, shares=shares, time=when))


def _adapter(fills):
    pf = pix.IBKRPortfolio.__new__(pix.IBKRPortfolio)
    pf.ib = FillIB(fills)
    pf._ensure = lambda: None
    pf._contract = lambda inst: types.SimpleNamespace(conId=111)
    return pf


def test_exact_order_fill_price_is_volume_weighted_average():
    pf = _adapter([
        _fill(7, "BOT", 100.0, 2, "2026-08-10T12:00:00Z"),
        _fill(7, "BOT", 103.0, 1, "2026-08-10T12:00:01Z"),
        _fill(8, "BOT", 120.0, 5, "2026-08-10T12:00:02Z"),
    ])
    px = pf.recent_fill_price(uni.IBKR_EUR_UNIVERSE["SAP"], "buy", order_id="7")
    assert abs(px - 101.0) < 1e-12


def test_unscoped_recent_fill_uses_vwap_of_latest_order_not_last_execution_only():
    pf = _adapter([
        _fill(7, "SLD", 100.0, 1, "2026-08-10T12:00:00Z"),
        _fill(8, "SLD", 90.0, 1, "2026-08-10T12:00:01Z"),
        _fill(8, "SLD", 96.0, 3, "2026-08-10T12:00:02Z"),
    ])
    px = pf.recent_fill_price(uni.IBKR_EUR_UNIVERSE["SAP"], "sell")
    assert abs(px - 94.5) < 1e-12
