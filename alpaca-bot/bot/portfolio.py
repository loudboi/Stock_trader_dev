"""
bot/portfolio.py
================
Alpaca broker adapter used by the live pullback runner and historical-data tools.

A broker/API failure is never represented as a valid flat position. Position reads
return None only when Alpaca explicitly reports that the position does not exist.
Cancellation succeeds only after the broker reports a non-working cancellation
state; acceptance of a cancel request is not treated as confirmation. Generic
external-close fill recovery can aggregate all relevant closed sell orders after a
caller-supplied reconciliation boundary.
"""

import logging
import time
from datetime import datetime

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
_CANCEL_CONFIRM_POLLS = 30
_CANCEL_POLL_SECONDS = 0.1
_CANCEL_CONFIRMED = {"canceled", "cancelled", "expired", "rejected"}
_TERMINAL_ORDER_STATES = {"filled", *_CANCEL_CONFIRMED}


def _api_status(exc):
    for name in ("status_code", "status", "http_status"):
        value = getattr(exc, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return None


def _as_utc(value):
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


class Portfolio:
    def __init__(self):
        self.paper = "paper" in config.ALPACA_BASE_URL.lower()
        self.trading = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY,
                                     paper=self.paper)
        self.stock_data = StockHistoricalDataClient(config.ALPACA_API_KEY,
                                                    config.ALPACA_SECRET_KEY)
        self.crypto_data = CryptoHistoricalDataClient(config.ALPACA_API_KEY,
                                                      config.ALPACA_SECRET_KEY)

    def get_equity(self) -> float:
        value = float(self.trading.get_account().equity)
        if value <= 0:
            raise RuntimeError("Alpaca reported non-positive account equity")
        return value

    @staticmethod
    def _side_str(side) -> str:
        return side.value if hasattr(side, "value") else str(side)

    def get_position_raw(self, instrument):
        try:
            p = self.trading.get_open_position(instrument.api_symbol)
        except APIError as e:
            if _api_status(e) == 404:
                return None
            raise
        return {"side": self._side_str(p.side), "qty": abs(float(p.qty)),
                "avg_entry": float(p.avg_entry_price)}

    def is_tradable_now(self, instrument) -> bool:
        if instrument.asset_class == "crypto":
            return True
        try:
            return bool(self.trading.get_clock().is_open)
        except APIError as e:
            log.warning("Clock check failed: %s", e)
            return False

    def get_historical_bars(self, instrument, tf_key: str, start, end) -> pd.DataFrame:
        if tf_key not in _TF_MAP:
            raise ValueError(f"Unsupported timeframe: {tf_key}")
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
        idx = pd.DatetimeIndex(df.index)
        df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        df = df.sort_index()
        cols = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in cols if c in df.columns]].copy()
        if resample:
            if instrument.asset_class == "crypto":
                df = (df.resample(resample, origin="start_day", label="right", closed="right")
                      .agg({"open": "first", "high": "max", "low": "min",
                            "close": "last", "volume": "sum"}).dropna())
            else:
                local = df.tz_convert("America/New_York")
                local = (local.resample(resample, origin="start_day", offset="9h30min",
                                        label="right", closed="right")
                         .agg({"open": "first", "high": "max", "low": "min",
                               "close": "last", "volume": "sum"}).dropna())
                df = local.tz_convert("UTC")
        return df

    def latest_price(self, instrument):
        try:
            if instrument.asset_class == "crypto":
                req = CryptoLatestTradeRequest(symbol_or_symbols=instrument.api_symbol)
                res = self.crypto_data.get_crypto_latest_trade(req)
            else:
                req = StockLatestTradeRequest(symbol_or_symbols=instrument.api_symbol)
                res = self.stock_data.get_stock_latest_trade(req)
            px = float(res[instrument.api_symbol].price)
            return px if px > 0 else None
        except Exception as e:  # noqa: BLE001
            log.debug("Latest price failed for %s: %s", instrument.name, e)
            return None

    def submit_market_order(self, instrument, qty: float, side: str):
        if qty <= 0:
            return None
        tif = TimeInForce.GTC if instrument.asset_class == "crypto" else TimeInForce.DAY
        req = MarketOrderRequest(symbol=instrument.api_symbol, qty=qty,
                                 side=OrderSide(side), time_in_force=tif)
        try:
            return str(self.trading.submit_order(order_data=req).id)
        except APIError as e:
            log.error("Order failed (%s %s %s): %s", side, qty, instrument.name, e)
            return None

    def close_position_raw(self, instrument):
        try:
            return str(self.trading.close_position(instrument.api_symbol).id)
        except APIError as e:
            log.error("Close failed for %s: %s", instrument.name, e)
            return None

    def order_status(self, order_id):
        """Normalized order state used by live reconciliation.

        Only states documented as final/no-further-execution are marked terminal.
        States such as pending_cancel, stopped, suspended, calculated, done_for_day,
        and replaced are deliberately not collapsed into terminal success because
        they can still imply unresolved execution or a successor order.
        """
        try:
            o = self.trading.get_order_by_id(order_id)
            raw = o.status.value if hasattr(o.status, "value") else str(o.status)
            status = str(raw).lower()
            return {
                "status": status,
                "filled_qty": float(getattr(o, "filled_qty", 0) or 0),
                "qty": float(getattr(o, "qty", 0) or 0),
                "terminal": status in _TERMINAL_ORDER_STATES,
            }
        except APIError as e:
            log.warning("Could not read order status %s: %s", order_id, e)
            return None

    def submit_stop_order(self, instrument, qty: float, stop_price: float):
        if instrument.asset_class == "crypto" or qty <= 0 or stop_price <= 0:
            return None
        req = StopOrderRequest(symbol=instrument.api_symbol, qty=qty,
                               side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
                               stop_price=round(stop_price, 2))
        try:
            return str(self.trading.submit_order(order_data=req).id)
        except APIError as e:
            log.warning("Stop order failed for %s: %s", instrument.name, e)
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Request cancellation and return True only after broker confirmation."""
        if not order_id:
            return True

        current = self.order_status(order_id)
        if current:
            if current["status"] in _CANCEL_CONFIRMED:
                return True
            if current["status"] == "filled":
                return False
        try:
            self.trading.cancel_order_by_id(order_id)
        except APIError as e:
            latest = self.order_status(order_id)
            if latest and latest["status"] in _CANCEL_CONFIRMED:
                return True
            log.warning("Cancel request for order %s failed/unconfirmed: %s", order_id, e)
            return False

        for _ in range(_CANCEL_CONFIRM_POLLS):
            status = self.order_status(order_id)
            if status:
                if status["status"] in _CANCEL_CONFIRMED:
                    return True
                if status["status"] == "filled":
                    log.warning("Order %s filled before cancellation was confirmed.", order_id)
                    return False
            time.sleep(_CANCEL_POLL_SECONDS)
        log.warning("Cancel request for order %s was accepted but not terminally confirmed.", order_id)
        return False

    def recent_fill_price(self, instrument, side: str, order_id=None, since=None):
        """Return an exact order average, or a bounded external-close weighted average."""
        since_ts = _as_utc(since)
        try:
            if order_id:
                o = self.trading.get_order_by_id(order_id)
                px = getattr(o, "filled_avg_price", None)
                return float(px) if px else None
            req = GetOrdersRequest(status=QueryOrderStatus.CLOSED,
                                   symbols=[instrument.api_symbol],
                                   side=OrderSide(side), limit=500, nested=False)
            candidates = []
            for o in self.trading.get_orders(filter=req):
                px = getattr(o, "filled_avg_price", None)
                if not px:
                    continue
                filled_at = _as_utc(getattr(o, "filled_at", None))
                if since_ts is not None:
                    if filled_at is None or filled_at < since_ts:
                        continue
                try:
                    qty = float(getattr(o, "filled_qty", 0) or 0)
                except (TypeError, ValueError):
                    qty = 0.0
                candidates.append({"time": filled_at, "price": float(px),
                                   "qty": max(0.0, qty)})
            if not candidates:
                return None
            if since_ts is None:
                with_time = [c for c in candidates if c["time"] is not None]
                return max(with_time, key=lambda c: c["time"])["price"] if with_time else candidates[0]["price"]
            total_qty = sum(c["qty"] for c in candidates)
            if total_qty > 0:
                return sum(c["price"] * c["qty"] for c in candidates) / total_qty
            with_time = [c for c in candidates if c["time"] is not None]
            return max(with_time, key=lambda c: c["time"])["price"] if with_time else candidates[-1]["price"]
        except APIError as e:
            log.debug("recent_fill_price failed for %s: %s", instrument.name, e)
        return None
