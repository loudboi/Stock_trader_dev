"""
bot/portfolio.py
================
Alpaca broker adapter used by the live pullback runner and historical-data tools.

The important contract is that a broker/API failure is never represented as a
valid flat position. `get_position_raw()` returns None only when Alpaca explicitly
reports that the position does not exist; all other API failures are raised so the
caller can stop/retry instead of corrupting trading state.
"""

import logging
from datetime import datetime, timezone, timedelta

import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (MarketOrderRequest, StopOrderRequest,
                                     GetOrdersRequest)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import (StockBarsRequest, CryptoBarsRequest,
                                  StockLatestTradeRequest, CryptoLatestTradeRequest)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

import config

log = logging.getLogger("portfolio")

_TF_MAP = {
    "15Min": (TimeFrame(15, TimeFrameUnit.Minute), None),
    "1Hour": (TimeFrame(1, TimeFrameUnit.Hour), None),
    "4Hour": (TimeFrame(1, TimeFrameUnit.Hour), "4h"),
    "1Day": (TimeFrame.Day, None),
}


def _api_status(exc):
    """Best-effort HTTP status extraction across alpaca-py APIError versions."""
    for name in ("status_code", "status", "http_status"):
        value = getattr(exc, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return None


class Portfolio:
    def __init__(self):
        paper = "paper" in config.ALPACA_BASE_URL.lower()
        self.paper = paper
        self.trading = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY,
                                     paper=paper)
        self.stock_data = StockHistoricalDataClient(config.ALPACA_API_KEY,
                                                    config.ALPACA_SECRET_KEY)
        self.crypto_data = CryptoHistoricalDataClient(config.ALPACA_API_KEY,
                                                      config.ALPACA_SECRET_KEY)

    # ------------------------------------------------------------------ #
    # Account / positions
    # ------------------------------------------------------------------ #
    def get_equity(self) -> float:
        return float(self.trading.get_account().equity)

    @staticmethod
    def _side_str(side) -> str:
        return side.value if hasattr(side, "value") else str(side)

    def get_position_raw(self, instrument):
        """Return one broker position, or None only when Alpaca says it is flat.

        A transient/auth/rate-limit API error must propagate. Treating it as flat
        can make the live runner log a fake stop-out and drop protection/state.
        """
        try:
            p = self.trading.get_open_position(instrument.api_symbol)
        except APIError as e:
            if _api_status(e) == 404:
                return None
            raise
        return {
            "side": self._side_str(p.side),
            "qty": abs(float(p.qty)),
            "avg_entry": float(p.avg_entry_price),
        }

    # ------------------------------------------------------------------ #
    # Market hours
    # ------------------------------------------------------------------ #
    def is_tradable_now(self, instrument) -> bool:
        if instrument.asset_class == "crypto":
            return True
        try:
            return bool(self.trading.get_clock().is_open)
        except APIError as e:
            log.warning("Clock check failed: %s", e)
            return False

    # ------------------------------------------------------------------ #
    # Bars
    # ------------------------------------------------------------------ #
    def get_historical_bars(self, instrument, tf_key: str, start, end) -> pd.DataFrame:
        tf, resample = _TF_MAP[tf_key]
        try:
            if instrument.asset_class == "crypto":
                req = CryptoBarsRequest(symbol_or_symbols=instrument.api_symbol,
                                        timeframe=tf, start=start, end=end)
                bars = self.crypto_data.get_crypto_bars(req)
            else:
                req = StockBarsRequest(symbol_or_symbols=instrument.api_symbol,
                                       timeframe=tf, start=start, end=end,
                                       feed=DataFeed.IEX)
                bars = self.stock_data.get_stock_bars(req)
        except APIError as e:
            log.warning("Bar fetch failed for %s: %s", instrument.name, e)
            return pd.DataFrame()

        df = bars.df
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel(0)
        df = df.sort_index()
        cols = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in cols if c in df.columns]].copy()
        if resample:
            # Anchor stock 4-hour bars to the US regular-session open instead of
            # arbitrary UTC midnight boundaries. Crypto remains UTC-anchored.
            kwargs = {"label": "right", "closed": "right"}
            if instrument.asset_class != "crypto":
                kwargs.update(origin="start_day", offset="14h30min")
            df = (df.resample(resample, **kwargs)
                  .agg({"open": "first", "high": "max", "low": "min",
                        "close": "last", "volume": "sum"})
                  .dropna())
        return df

    # ------------------------------------------------------------------ #
    # Latest price
    # ------------------------------------------------------------------ #
    def latest_price(self, instrument):
        """Most recent trade price, or None when live market data is unavailable."""
        try:
            if instrument.asset_class == "crypto":
                req = CryptoLatestTradeRequest(symbol_or_symbols=instrument.api_symbol)
                res = self.crypto_data.get_crypto_latest_trade(req)
            else:
                req = StockLatestTradeRequest(symbol_or_symbols=instrument.api_symbol)
                res = self.stock_data.get_stock_latest_trade(req)
            px = float(res[instrument.api_symbol].price)
            return px if px > 0 else None
        except Exception as e:  # noqa: BLE001 - data outage is represented as no price
            log.debug("Latest price failed for %s: %s", instrument.name, e)
            return None

    # ------------------------------------------------------------------ #
    # Orders
    # ------------------------------------------------------------------ #
    def submit_market_order(self, instrument, qty: float, side: str):
        """Submit a market order and return its broker order id, or None on rejection.

        Submission/acceptance is deliberately not called a fill. The live runner
        confirms the resulting broker position before mutating local state.
        """
        tif = TimeInForce.GTC if instrument.asset_class == "crypto" else TimeInForce.DAY
        order = MarketOrderRequest(symbol=instrument.api_symbol, qty=qty,
                                   side=OrderSide(side), time_in_force=tif)
        try:
            submitted = self.trading.submit_order(order_data=order)
            return str(submitted.id)
        except APIError as e:
            log.error("Order failed (%s %s %s): %s", side, qty, instrument.name, e)
            return None

    def close_position_raw(self, instrument):
        """Request a market close and return its order id, or None on rejection."""
        try:
            submitted = self.trading.close_position(instrument.api_symbol)
            return str(submitted.id)
        except APIError as e:
            log.error("Close failed for %s: %s", instrument.name, e)
            return None

    def submit_stop_order(self, instrument, qty: float, stop_price: float):
        if instrument.asset_class == "crypto":
            log.info("%s: broker stop not placed (crypto); in-process stop active.",
                     instrument.name)
            return None
        if qty <= 0 or stop_price <= 0:
            return None
        req = StopOrderRequest(symbol=instrument.api_symbol, qty=qty,
                               side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
                               stop_price=round(stop_price, 2))
        try:
            order = self.trading.submit_order(order_data=req)
            return str(order.id)
        except APIError as e:
            log.warning("Stop order failed for %s: %s (in-process stop active).",
                        instrument.name, e)
            return None

    def cancel_order(self, order_id: str) -> bool:
        if not order_id:
            return True
        try:
            self.trading.cancel_order_by_id(order_id)
            return True
        except APIError as e:
            # Alpaca may report not-found/already-final after a race; callers still
            # need to know cancellation was not positively confirmed.
            log.warning("Cancel order %s was not confirmed: %s", order_id, e)
            return False

    def recent_fill_price(self, instrument, side: str, order_id=None):
        """Return a fill price, preferring the exact broker order id when known."""
        try:
            if order_id:
                o = self.trading.get_order_by_id(order_id)
                if getattr(o, "filled_avg_price", None):
                    return float(o.filled_avg_price)
                return None
            req = GetOrdersRequest(status=QueryOrderStatus.CLOSED,
                                   symbols=[instrument.api_symbol],
                                   side=OrderSide(side), limit=20, nested=False)
            for o in self.trading.get_orders(filter=req):
                if getattr(o, "filled_avg_price", None):
                    return float(o.filled_avg_price)
        except APIError as e:
            log.debug("recent_fill_price failed for %s: %s", instrument.name, e)
        return None
