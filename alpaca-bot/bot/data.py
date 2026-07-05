"""
bot/data.py
===========
Backtest data sources. Live trading always uses Alpaca (bot/portfolio.py); this
module exists so the *backtester* can optionally pull decades of free daily
history from Yahoo, which Alpaca's IEX feed won't go back far enough to provide.

Fidelity note: Yahoo daily bars are split/dividend-ADJUSTED, while Alpaca returns
raw prices. Adjusted prices are the right choice for a long backtest (they handle
splits cleanly), but it does mean a Yahoo backtest is not bar-identical to live
Alpaca data — treat the deep-history run as a regime study, not a live proxy.

The pure normalizer (`normalize_ohlcv`) is separated from the network fetch so it
can be unit-tested offline.
"""

import logging

import pandas as pd

log = logging.getLogger("data")

_COLMAP = {"Open": "open", "High": "high", "Low": "low",
           "Close": "close", "Adj Close": "close", "Volume": "volume"}


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """A raw vendor frame -> lowercase OHLCV indexed by UTC timestamp (ascending).

    Tolerates yfinance quirks: a (field, ticker) MultiIndex on the columns, mixed
    capitalization, an 'Adj Close' column, and a tz-naive date index.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.copy()
    # Flatten a column MultiIndex (yfinance returns one for single tickers too).
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=_COLMAP)
    # If both 'Close' and 'Adj Close' mapped to 'close', keep the last (Adj Close).
    df = df.loc[:, ~df.columns.duplicated(keep="last")]
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[cols]
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return df.sort_index().dropna(how="any")


def clean_price_series(close: pd.Series, max_abs_return: float = 0.15) -> pd.Series:
    """Patch implausible single-day moves by carrying forward the prior price.

    Free vendor data (especially FX/futures "=X"/"=F" tickers) occasionally has
    bad ticks, and continuous futures contracts can cross zero on a real event
    (e.g. WTI crude settled at -$37.63 on 2020-04-20) where a PERCENTAGE return is
    mathematically undefined, not just large. Either case would otherwise inject a
    spurious spike into any trend/volatility calculation for weeks around it (a
    20-day realized-vol window means one bad day contaminates a month of signal).

    `max_abs_return` is a single, conservative, asset-class-agnostic bound chosen
    BEFORE looking at any backtest result — legitimate daily moves in liquid FX,
    commodity, or equity markets essentially never exceed 15% outside a handful of
    well-documented pathological events (which this targets, not conceals: the
    patched days are still visible by comparing to the raw series). Not a strategy
    parameter; do not tune this per backtest.
    """
    px = close.copy().astype(float)
    for i in range(1, len(px)):
        prev = px.iloc[i - 1]
        cur = px.iloc[i]
        if prev <= 0 or cur <= 0 or abs(cur / prev - 1.0) > max_abs_return:
            px.iloc[i] = prev
    return px


def clean_daily_data(daily_data: dict, max_abs_return: float = 0.15) -> dict:
    """Apply clean_price_series to the close column of every symbol's OHLCV frame
    (open/high/low left as-is; only close drives the strategies in bot/lab.py)."""
    out = {}
    for name, df in daily_data.items():
        if df.empty:
            out[name] = df
            continue
        clean = df.copy()
        clean["close"] = clean_price_series(clean["close"], max_abs_return)
        out[name] = clean
    return out


def load_yahoo(symbol: str, start, end) -> pd.DataFrame:
    """Daily OHLCV from Yahoo (lazy import so it's optional)."""
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "yfinance is required for --data-source yahoo. Install it with "
            "`pip install -r requirements-backtest.txt`.") from e
    raw = yf.download(symbol, start=pd.Timestamp(start).date(),
                      end=pd.Timestamp(end).date(), interval="1d",
                      auto_adjust=True, progress=False, threads=False)
    df = normalize_ohlcv(raw)
    if df.empty:
        log.warning("Yahoo returned no data for %s.", symbol)
    return df
