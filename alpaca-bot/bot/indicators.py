"""
bot/indicators.py
=================
Pure functions on pandas DataFrames/Series. No API calls, no side effects, so
these are trivially unit-testable. Strategy 4 only needs moving averages and ATR.

Every function expects a DataFrame with lowercase columns:
    open, high, low, close, volume
indexed by timestamp (ascending).
"""

import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    # adjust=False is the standard recursive EMA used by trading platforms.
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR (RMA smoothing of True Range)."""
    tr = true_range(df)
    # Wilder smoothing == EMA with alpha = 1/period.
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
