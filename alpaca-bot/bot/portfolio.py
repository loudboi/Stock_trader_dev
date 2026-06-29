"""
bot/portfolio.py
================
The only module that talks to Alpaca. Everything else stays broker-agnostic.

Uses alpaca-py (the current, maintained SDK). All SDK specifics live here, so the
rest of the project calls clean Portfolio methods and never imports alpaca directly.

This is the surface Strategy 4's live runner and backtester use:
  - account equity + the current position for one instrument
  - historical candles (4h resampled from 1h, since Alpaca has no native 4h)
  - latest price, market order submission, position close
  - market-hours / clock checks

Trade logging and restart-safe state live in bot/live_pullback.py (the strategy
keeps its own pyramided-position state), so they are intentionally not here.
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

# Map our timeframe strings to (alpaca TimeFrame, resample_rule_or_None).
_TF_MAP = {
    "15Min": (TimeFrame(15, TimeFrameUnit.Minute), None),
    "1Hour": (TimeFrame(1, TimeFrameUnit.Hour), None),
    "4Hour": (TimeFrame(1, TimeFrameUnit.Hour), "4h"),   # resample 1h -> 4h
    "1Day": (TimeFrame.Day, None),
}


class Portfolio:
    def __init__(self):
        paper = "paper" in config.ALPACA_BASE_URL.lower()
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
        # PositionSide enum -> "long"/"short"
        return side.value if hasattr(side, "value") else str(side)

    def get_position_raw(self, instrument):
        """Return {'side','qty','avg_entry'} for one instrument, or None if flat."""
        try:
            p = self.trading.get_open_position(instrument.api_symbol)
        except APIError:
            return None
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
            return True  # 24/7
        try:
            return bool(self.trading.get_clock().is_open)
        except APIError as e:
            log.warning("Clock check failed: %s", e)
            return False

    # ------------------------------------------------------------------ #
    # Bars
    # ------------------------------------------------------------------ #
    def get_historical_bars(self, instrument, tf_key: str, start, end) -> pd.DataFrame:
        """Fetch OHLCV bars for an instrument over [start, end] at tf_key.

        Returns a single-symbol DataFrame indexed by timestamp with columns
        open/high/low/close/volume (4h resampled from 1h when tf_key == '4Hour').
        start/end are datetimes.
        """
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
        # alpaca-py returns a (symbol, timestamp) MultiIndex; flatten to timestamp.
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel(0)
        df = df.sort_index()

        cols = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in cols if c in df.columns]].copy()
        if resample:
            df = (df.resample(resample, label="right", closed="right")
                  .agg({"open": "first", "high": "max", "low": "min",
                        "close": "last", "volume": "sum"})
                  .dropna())
        return df

    # ------------------------------------------------------------------ #
    # Latest price
    # ------------------------------------------------------------------ #
    def latest_price(self, instrument):
        """Most recent trade price, or None if unavailable."""
        try:
            if instrument.asset_class == "crypto":
                req = CryptoLatestTradeRequest(symbol_or_symbols=instrument.api_symbol)
                res = self.crypto_data.get_crypto_latest_trade(req)
            else:
                req = StockLatestTradeRequest(symbol_or_symbols=instrument.api_symbol)
                res = self.stock_data.get_stock_latest_trade(req)
            return float(res[instrument.api_symbol].price)
        except (APIError, KeyError, Exception):  # noqa: BLE001
            return None

    # ------------------------------------------------------------------ #
    # Orders
    # ------------------------------------------------------------------ #
    def submit_market_order(self, instrument, qty: float, side: str) -> bool:
        """side is 'buy' or 'sell'. Returns True on success."""
        tif = TimeInForce.GTC if instrument.asset_class == "crypto" else TimeInForce.DAY
        order = MarketOrderRequest(symbol=instrument.api_symbol, qty=qty,
                                   side=OrderSide(side), time_in_force=tif)
        try:
            self.trading.submit_order(order_data=order)
            return True
        except APIError as e:
            log.error("Order failed (%s %s %s): %s", side, qty, instrument.name, e)
            return False

    def close_position_raw(self, instrument) -> bool:
        """Market close of the whole position (no logging)."""
        try:
            self.trading.close_position(instrument.api_symbol)
            return True
        except APIError as e:
            log.error("Close failed for %s: %s", instrument.name, e)
            return False

    def submit_stop_order(self, instrument, qty: float, stop_price: float):
        """Rest a protective SELL stop at the broker (long-only Strategy 4).

        Returns the order id (str) on success, or None if not placed. Alpaca
        crypto doesn't take a plain stop order, so for crypto we return None and
        let the in-process stop cover it. The equity stop is GTC so it survives
        restarts / downtime.
        """
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
            log.debug("Cancel order %s failed: %s", order_id, e)
            return False

    def recent_fill_price(self, instrument, side: str):
        """filled_avg_price of the most recent filled order on `side`, or None.

        Used to recover the true exit price when a resting stop fills while we're
        between loops (or down).
        """
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.CLOSED,
                                   symbols=[instrument.api_symbol],
                                   side=OrderSide(side), limit=20, nested=False)
            for o in self.trading.get_orders(filter=req):
                if getattr(o, "filled_avg_price", None):
                    return float(o.filled_avg_price)
        except APIError as e:
            log.debug("recent_fill_price failed for %s: %s", instrument.name, e)
        return None
