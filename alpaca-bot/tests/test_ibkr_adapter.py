"""
Offline tests for the IBKR adapter — no Gateway/network required.

Validates the pure translation/caching/hours logic and the adapter's method
behavior against a FakeIB that mimics the ib_async surface we use.
"""
import os
import sys
import types
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from bot import ibkr_universe as uni
from bot import portfolio_ibkr as pix


# --------------------------------------------------------------------------- #
# Fakes that mimic the bits of ib_async we touch
# --------------------------------------------------------------------------- #
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
        self.date, self.open, self.high, self.low, self.close, self.volume = \
            date, o, h, l, c, v


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
        self.orders.append((order.action, order.totalQuantity))
        return types.SimpleNamespace(order=order)

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
    pf._ensure = lambda: None
    pf._contract = lambda inst: contract
    return pf


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_contract_kwargs():
    spec = uni.ibkr_spec(uni.IBKR_EUR_UNIVERSE["SAP"])
    kw = pix.contract_kwargs(spec)
    assert kw == {"symbol": "SAP", "exchange": "SMART",
                  "currency": "EUR", "primaryExchange": "IBIS"}
    print("contract_kwargs OK")


def test_map_position():
    assert pix.map_ib_position(0, 10) is None
    assert pix.map_ib_position(5, 12.5) == {"side": "long", "qty": 5.0, "avg_entry": 12.5}
    assert pix.map_ib_position(-3, 9.0) == {"side": "short", "qty": 3.0, "avg_entry": 9.0}
    print("map_ib_position OK")


def test_equity_from_summary():
    rows = [FakeRow("NetLiquidation", "1234.50", "USD"),
            FakeRow("NetLiquidation", "1000.00", "EUR"),
            FakeRow("BuyingPower", "5000", "EUR")]
    assert pix.equity_from_summary(rows, "EUR") == 1000.0          # prefers EUR
    rows2 = [FakeRow("NetLiquidation", "777", "USD")]
    assert pix.equity_from_summary(rows2, "EUR") == 777.0          # fallback
    assert pix.equity_from_summary([], "EUR") is None
    print("equity_from_summary OK")


def test_bars_and_merge_and_duration():
    bars = [FakeBar("2025-02-04", 1, 2, 0.5, 1.5, 100),
            FakeBar("2025-02-05", 1.5, 2.5, 1.0, 2.0, 120)]
    df = pix.bars_to_df(bars)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2 and str(df.index.tz) == "UTC"
    newer = pix.bars_to_df([FakeBar("2025-02-05", 9, 9, 9, 9, 9),   # overwrites
                            FakeBar("2025-02-06", 2, 3, 1, 2.5, 130)])
    merged = pix.merge_bar_cache(df, newer)
    assert len(merged) == 3
    assert merged.loc["2025-02-05 00:00:00+00:00", "close"] == 9   # last write wins
    assert pix.duration_str("2025-02-04", "2025-02-10").endswith(" D")
    assert pix.duration_str("2024-01-01", "2026-01-01").endswith(" Y")
    print("bars_to_df / merge / duration OK")


def test_market_hours():
    now = datetime(2026, 6, 25, 12, 0)        # 1200
    assert pix.market_open_from_hours("20260625:0900-20260625:1730", now) is True
    assert pix.market_open_from_hours("20260625:0900-1730", now) is True
    assert pix.market_open_from_hours("20260625:CLOSED", now) is False
    assert pix.market_open_from_hours("20260626:0900-1730", now) is None  # no today
    assert pix.market_open_from_hours("", now) is None
    early = datetime(2026, 6, 25, 7, 0)       # 0700, before open
    assert pix.market_open_from_hours("20260625:0900-1730", early) is False
    print("market_open_from_hours OK")


# --------------------------------------------------------------------------- #
# Adapter methods (against FakeIB)
# --------------------------------------------------------------------------- #
def test_get_equity():
    fake = FakeIB()
    fake._summary = [FakeRow("NetLiquidation", "25000", "EUR")]
    pf = _adapter_with_fake(fake)
    assert pf.get_equity() == 25000.0
    print("get_equity OK")


def test_get_position_and_close():
    c = FakeContract(conId=999)
    fake = FakeIB()
    fake._positions = [FakePosition(c, 7, 100.0)]
    pf = _adapter_with_fake(fake, contract=c)
    r = pf.get_position_raw(uni.IBKR_EUR_UNIVERSE["SAP"])
    assert r == {"side": "long", "qty": 7.0, "avg_entry": 100.0}
    # closing a long places a SELL for the full qty
    ok = pf.close_position_raw(uni.IBKR_EUR_UNIVERSE["SAP"])
    assert ok and fake.orders == [("SELL", 7.0)]
    print("get_position_raw + close_position_raw OK")


def test_close_when_flat():
    fake = FakeIB()                      # no positions
    pf = _adapter_with_fake(fake)
    assert pf.close_position_raw(uni.IBKR_EUR_UNIVERSE["DAX"]) is True
    assert fake.orders == []             # nothing to do, no order sent
    print("close_position_raw when flat OK")


def test_latest_price_fallback():
    fake = FakeIB()
    fake._tickers = [FakeTicker(last=float("nan"), close=42.0, mkt=float("nan"))]
    pf = _adapter_with_fake(fake)
    assert pf.latest_price(uni.IBKR_EUR_UNIVERSE["ASML"]) == 42.0   # skips NaN→close
    print("latest_price fallback OK")


def test_submit_order():
    fake = FakeIB()
    pf = _adapter_with_fake(fake)
    assert pf.submit_market_order(uni.IBKR_EUR_UNIVERSE["SAP"], 3, "buy") is True
    assert fake.orders == [("BUY", 3)]
    assert pf.submit_market_order(uni.IBKR_EUR_UNIVERSE["SAP"], 0, "buy") is False  # qty 0
    print("submit_market_order OK")


def test_historical_caching_and_slice():
    inst = uni.IBKR_EUR_UNIVERSE["STOXX600"]
    fake = FakeIB()
    fake._bars = [FakeBar("2025-02-04", 1, 1, 1, 1, 10),
                  FakeBar("2025-02-05", 2, 2, 2, 2, 10),
                  FakeBar("2025-02-06", 3, 3, 3, 3, 10)]
    pf = _adapter_with_fake(fake)
    pf.cache_dir = "/tmp/ibkr_test_cache2"
    os.makedirs(pf.cache_dir, exist_ok=True)
    # clear any prior disk cache
    for f in os.listdir(pf.cache_dir):
        os.remove(os.path.join(pf.cache_dir, f))
    df = pf.get_historical_bars(inst, "1Day", "2025-02-04", "2025-02-06")
    assert len(df) == 3
    # second call returns a tail that overwrites + extends; cache merges
    fake._bars = [FakeBar("2025-02-06", 9, 9, 9, 9, 10),
                  FakeBar("2025-02-07", 4, 4, 4, 4, 10)]
    df2 = pf.get_historical_bars(inst, "1Day", "2025-02-04", "2025-02-07")
    assert len(df2) == 4
    assert df2.loc["2025-02-06 00:00:00+00:00", "close"] == 9   # merged overwrite
    # slice excludes out-of-range
    df3 = pf.get_historical_bars(inst, "1Day", "2025-02-06", "2025-02-07")
    assert len(df3) == 2
    print("historical caching + slice OK")


if __name__ == "__main__":
    test_contract_kwargs()
    test_map_position()
    test_equity_from_summary()
    test_bars_and_merge_and_duration()
    test_market_hours()
    test_get_equity()
    test_get_position_and_close()
    test_close_when_flat()
    test_latest_price_fallback()
    test_submit_order()
    test_historical_caching_and_slice()
    print("\nALL IBKR ADAPTER TESTS PASSED")
