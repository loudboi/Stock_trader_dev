"""Offline tests for the IBKR adapter — no Gateway/network required."""
import os
import sys
import types
from datetime import datetime

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot import ibkr_universe as uni
from bot import portfolio_ibkr as pix


class FakeContract:
    def __init__(self, conId, symbol="X"):
        self.conId = conId
        self.symbol = symbol
        self.localSymbol = symbol


class FakeRow:
    def __init__(self, tag, value, currency):
        self.tag, self.value, self.currency = tag, value, currency


class FakePosition:
    def __init__(self, contract, position, avgCost):
        self.contract, self.position, self.avgCost = contract, position, avgCost


class FakeBar:
    def __init__(self, date, o, h, l, c, v):
        self.date, self.open, self.high, self.low, self.close, self.volume = date, o, h, l, c, v


class FakeTicker:
    def __init__(self, last=None, close=None, mkt=None):
        self.last, self.close, self._mkt = last, close, mkt

    def marketPrice(self):
        return self._mkt


class FakeIB:
    def __init__(self):
        self._connected = True
        self.orders = []
        self._summary = []
        self._positions = []
        self._bars = []
        self._tickers = []
        self._next_order_id = 1
        self._open_trades = []

    def isConnected(self):
        return self._connected

    def accountSummary(self):
        return self._summary

    def positions(self):
        return self._positions

    def reqHistoricalData(self, *a, **k):
        return self._bars

    def reqTickers(self, *a, **k):
        return self._tickers

    def placeOrder(self, contract, order):
        order.orderId = self._next_order_id
        self._next_order_id += 1
        self.orders.append((order.action, order.totalQuantity))
        trade = types.SimpleNamespace(order=order)
        self._open_trades.append(trade)
        return trade

    def openTrades(self):
        return list(self._open_trades)

    def cancelOrder(self, order):
        self._open_trades = [t for t in self._open_trades if t.order is not order]

    def sleep(self, n):
        pass


def _adapter_with_fake(fake, contract=FakeContract(111)):
    """Build an IBKRPortfolio without connecting, wired to a FakeIB."""
    pf = pix.IBKRPortfolio.__new__(pix.IBKRPortfolio)
    pf.ib = fake
    pf.base_currency = "EUR"
    pf.cache_dir = "/tmp/ibkr_test_cache"
    os.makedirs(pf.cache_dir, exist_ok=True)
    pf._contracts = {}
    pf._hours = {}
    pf._bar_cache = {}
    pf._strategy_orders = {}
    pf._ensure = lambda: None
    pf._contract = lambda inst: contract
    return pf


def test_contract_kwargs():
    spec = uni.ibkr_spec(uni.IBKR_EUR_UNIVERSE["SAP"])
    assert pix.contract_kwargs(spec) == {
        "symbol": "SAP", "exchange": "SMART", "currency": "EUR", "primaryExchange": "IBIS"}


def test_map_position():
    assert pix.map_ib_position(0, 10) is None
    assert pix.map_ib_position(5, 12.5) == {"side": "long", "qty": 5.0, "avg_entry": 12.5}
    assert pix.map_ib_position(-3, 9.0) == {"side": "short", "qty": 3.0, "avg_entry": 9.0}


def test_equity_from_summary_requires_requested_currency():
    rows = [FakeRow("NetLiquidation", "1234.50", "USD"),
            FakeRow("NetLiquidation", "1000.00", "EUR"),
            FakeRow("BuyingPower", "5000", "EUR")]
    assert pix.equity_from_summary(rows, "EUR") == 1000.0
    assert pix.equity_from_summary([FakeRow("NetLiquidation", "777", "USD")], "EUR") is None
    assert pix.equity_from_summary([], "EUR") is None


def test_get_equity_fails_closed_on_wrong_currency():
    fake = FakeIB()
    fake._summary = [FakeRow("NetLiquidation", "777", "USD")]
    pf = _adapter_with_fake(fake)
    with pytest.raises(RuntimeError, match="EUR"):
        pf.get_equity()


def test_bars_and_merge_and_duration():
    bars = [FakeBar("2025-02-04", 1, 2, 0.5, 1.5, 100),
            FakeBar("2025-02-05", 1.5, 2.5, 1.0, 2.0, 120)]
    df = pix.bars_to_df(bars)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2 and str(df.index.tz) == "UTC"
    newer = pix.bars_to_df([FakeBar("2025-02-05", 9, 9, 9, 9, 9),
                            FakeBar("2025-02-06", 2, 3, 1, 2.5, 130)])
    merged = pix.merge_bar_cache(df, newer)
    assert len(merged) == 3
    assert merged.loc["2025-02-05 00:00:00+00:00", "close"] == 9
    assert pix.duration_str("2025-02-04", "2025-02-10").endswith(" D")
    assert pix.duration_str("2024-01-01", "2026-01-01").endswith(" Y")


def test_market_hours():
    now = datetime(2026, 6, 25, 12, 0)
    assert pix.market_open_from_hours("20260625:0900-20260625:1730", now) is True
    assert pix.market_open_from_hours("20260625:0900-1730", now) is True
    assert pix.market_open_from_hours("20260625:CLOSED", now) is False
    assert pix.market_open_from_hours("20260626:0900-1730", now) is None
    assert pix.market_open_from_hours("", now) is None
    assert pix.market_open_from_hours("20260625:0900-1730", datetime(2026, 6, 25, 7, 0)) is False


def test_exchange_now_resolves_exchange_timezone():
    from datetime import timezone as _tz
    now_utc = datetime.now(_tz.utc)
    now_est = pix.exchange_now("US/Eastern")
    assert now_est.tzinfo is not None
    assert abs((now_est - now_utc).total_seconds()) < 5


def test_exchange_now_fails_closed_on_bad_or_missing_timezone():
    for tz_id in ("", None, "Not/AZone"):
        assert pix.exchange_now(tz_id) is None


def test_get_equity():
    fake = FakeIB()
    fake._summary = [FakeRow("NetLiquidation", "25000", "EUR")]
    assert _adapter_with_fake(fake).get_equity() == 25000.0


def test_get_position_and_close_returns_order_id():
    c = FakeContract(conId=999)
    fake = FakeIB()
    fake._positions = [FakePosition(c, 7, 100.0)]
    pf = _adapter_with_fake(fake, contract=c)
    assert pf.get_position_raw(uni.IBKR_EUR_UNIVERSE["SAP"]) == {
        "side": "long", "qty": 7.0, "avg_entry": 100.0}
    oid = pf.close_position_raw(uni.IBKR_EUR_UNIVERSE["SAP"])
    assert oid == "1"
    assert fake.orders == [("SELL", 7.0)]


def test_close_when_flat_returns_explicit_token():
    fake = FakeIB()
    pf = _adapter_with_fake(fake)
    assert pf.close_position_raw(uni.IBKR_EUR_UNIVERSE["DAX"]) == "already-flat"
    assert fake.orders == []


def test_latest_price_does_not_use_previous_close_as_live_quote():
    fake = FakeIB()
    fake._tickers = [FakeTicker(last=float("nan"), close=42.0, mkt=float("nan"))]
    pf = _adapter_with_fake(fake)
    assert pf.latest_price(uni.IBKR_EUR_UNIVERSE["ASML"]) is None


def test_submit_order_returns_id_and_rejects_zero_qty():
    fake = FakeIB()
    pf = _adapter_with_fake(fake)
    assert pf.submit_market_order(uni.IBKR_EUR_UNIVERSE["SAP"], 3, "buy") == "1"
    assert fake.orders == [("BUY", 3)]
    assert pf.submit_market_order(uni.IBKR_EUR_UNIVERSE["SAP"], 0, "buy") is None


def test_historical_caching_and_slice():
    inst = uni.IBKR_EUR_UNIVERSE["STOXX600"]
    fake = FakeIB()
    fake._bars = [FakeBar("2025-02-04", 1, 1, 1, 1, 10),
                  FakeBar("2025-02-05", 2, 2, 2, 2, 10),
                  FakeBar("2025-02-06", 3, 3, 3, 3, 10)]
    pf = _adapter_with_fake(fake)
    pf.cache_dir = "/tmp/ibkr_test_cache2"
    os.makedirs(pf.cache_dir, exist_ok=True)
    for f in os.listdir(pf.cache_dir):
        os.remove(os.path.join(pf.cache_dir, f))
    df = pf.get_historical_bars(inst, "1Day", "2025-02-04", "2025-02-06")
    assert len(df) == 3
    fake._bars = [FakeBar("2025-02-06", 9, 9, 9, 9, 10),
                  FakeBar("2025-02-07", 4, 4, 4, 4, 10)]
    df2 = pf.get_historical_bars(inst, "1Day", "2025-02-04", "2025-02-07")
    assert len(df2) == 4
    assert df2.loc["2025-02-06 00:00:00+00:00", "close"] == 9
    df3 = pf.get_historical_bars(inst, "1Day", "2025-02-06", "2025-02-07")
    assert len(df3) == 2
