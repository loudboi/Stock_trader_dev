"""
bot/data.py
===========
Research/backtest data helpers.

Yahoo daily bars are adjusted and are not bar-identical to live Alpaca bars. The
optional outlier cleaner is deliberately a QUARANTINE, not a price-correction
model: suspect rows are removed as whole OHLCV bars and their timestamps are
logged. It never invents a replacement close while leaving incompatible open/high/
low values behind. Use it only after inspecting the flagged observations; genuine
market discontinuities are data, not errors.
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("data")
_COLMAP = {"Open": "open", "High": "high", "Low": "low",
           "Close": "close", "Adj Close": "close", "Volume": "volume"}
_REQUIRED = ("open", "high", "low", "close", "volume")


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=_COLMAP)
    df = df.loc[:, ~df.columns.duplicated(keep="last")]
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        log.warning("OHLCV frame missing columns: %s", ", ".join(missing))
    cols = [c for c in _REQUIRED if c in df.columns]
    df = df[cols]
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return df.sort_index().dropna(how="any")


def suspect_price_mask(close: pd.Series, max_abs_return: float = 0.15) -> pd.Series:
    """Boolean mask for non-positive prices or jumps beyond the explicit threshold.

    The threshold does not mean the observation is wrong; it means the observation
    needs review before a strategy that assumes ordinary positive percentage prices
    consumes it.
    """
    if not 0 < max_abs_return < 10:
        raise ValueError("max_abs_return must be between 0 and 10")
    px = close.astype(float)
    ret = px.pct_change(fill_method=None)
    mask = (~np.isfinite(px)) | (px <= 0) | (ret.abs() > max_abs_return)
    return pd.Series(mask, index=close.index, dtype=bool).fillna(False)


def clean_price_series(close: pd.Series, max_abs_return: float = 0.15) -> pd.Series:
    """Legacy convenience API: return the price series with suspect rows removed.

    This intentionally changes the index instead of manufacturing replacement
    prices. Call `suspect_price_mask` first if you need the exact audit list.
    """
    mask = suspect_price_mask(close, max_abs_return)
    return close.loc[~mask].copy()


def clean_daily_data(daily_data: dict, max_abs_return: float = 0.15) -> dict:
    """Quarantine suspect observations by dropping the entire OHLCV row.

    Every removed timestamp is logged. The returned DataFrame also carries
    `attrs['quarantined_timestamps']` so research output can retain provenance.
    """
    out = {}
    for name, df in daily_data.items():
        if df is None or df.empty:
            out[name] = df
            continue
        if "close" not in df:
            raise ValueError(f"{name}: cannot clean a frame without close")
        mask = suspect_price_mask(df["close"], max_abs_return)
        flagged = [pd.Timestamp(ts).isoformat() for ts in df.index[mask]]
        clean = df.loc[~mask].copy()
        clean.attrs.update(df.attrs)
        clean.attrs["quarantined_timestamps"] = flagged
        if flagged:
            log.warning("%s: quarantined %d suspect OHLCV row(s): %s",
                        name, len(flagged), ", ".join(flagged))
        out[name] = clean
    return out


def load_yahoo(symbol: str, start, end) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "yfinance is required for --data-source yahoo. Install requirements-backtest.txt") from e
    raw = yf.download(symbol, start=pd.Timestamp(start).date(),
                      end=pd.Timestamp(end).date(), interval="1d",
                      auto_adjust=True, progress=False, threads=False)
    df = normalize_ohlcv(raw)
    if df.empty:
        log.warning("Yahoo returned no data for %s.", symbol)
    return df
