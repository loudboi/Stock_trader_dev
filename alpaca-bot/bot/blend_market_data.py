"""Market-data helpers for the profitability blend.

The fixed blend was researched on Yahoo `auto_adjust=True` daily bars.  A paper
implementation must therefore request corporate-action-adjusted stock history
explicitly instead of inheriting Alpaca's raw-bar default.  This module is separate
from the existing live pullback adapter so its already-reviewed signal semantics are
not changed implicitly.
"""

from __future__ import annotations

import pandas as pd

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


def adjusted_daily_request(symbol: str, start, end,
                           feed: DataFeed = DataFeed.IEX) -> StockBarsRequest:
    """Build the explicit all-corporate-actions daily-bar request used by the blend."""
    if not symbol:
        raise ValueError("symbol is required")
    return StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment=Adjustment.ALL,
        feed=feed,
    )


def fetch_adjusted_daily(client, symbol: str, start, end,
                         feed: DataFeed = DataFeed.IEX) -> pd.DataFrame:
    """Fetch normalized adjusted OHLCV without falling back to raw history."""
    req = adjusted_daily_request(symbol, start, end, feed=feed)
    bars = client.get_stock_bars(req)
    df = getattr(bars, "df", None)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel(0)
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    cols = ["open", "high", "low", "close", "volume"]
    return df[[c for c in cols if c in df.columns]].sort_index()
