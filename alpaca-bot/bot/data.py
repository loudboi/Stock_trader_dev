"""
bot/data.py
===========
Research/backtest data helpers.

Yahoo daily bars are adjusted and are not bar-identical to live Alpaca bars.
Outlier handling is deliberately conservative:

* `suspect_price_mask` FLAGS non-positive prices and large one-day moves for review.
* automatic quarantine removes a row only when percentage-return math is invalid
  (non-positive price) or when a large move immediately reverses and the prices on
  either side form a normal bridge — strong evidence of an isolated bad tick.
* genuine large one-way moves are preserved. The cleaner never manufactures a
  replacement close while leaving incompatible OHLC values behind.

Every quarantine is an entire OHLCV row and both suspect/quarantined timestamps are
retained in DataFrame attrs for reproducibility.
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


def _validate_threshold(max_abs_return: float) -> None:
    if not 0 < max_abs_return < 10:
        raise ValueError("max_abs_return must be between 0 and 10")


def suspect_price_mask(close: pd.Series, max_abs_return: float = 0.15) -> pd.Series:
    """Flag observations that require review; do not imply they are erroneous."""
    _validate_threshold(max_abs_return)
    px = close.astype(float)
    ret = px.pct_change(fill_method=None)
    mask = (~np.isfinite(px)) | (px <= 0) | (ret.abs() > max_abs_return)
    return pd.Series(mask, index=close.index, dtype=bool).fillna(False)


def quarantine_price_mask(close: pd.Series, max_abs_return: float = 0.15) -> pd.Series:
    """Rows safe to auto-quarantine without erasing plausible market discontinuities.

    Non-positive/non-finite prices are unusable by percentage-return strategies.
    A positive large move is auto-quarantined only when it immediately reverses
    and the previous-to-next price bridge is itself ordinary. This catches an
    isolated vendor spike without cascading into the following normal observation.
    """
    _validate_threshold(max_abs_return)
    px = close.astype(float)
    mask = pd.Series(False, index=close.index, dtype=bool)
    invalid = (~np.isfinite(px)) | (px <= 0)
    mask.loc[invalid] = True
    for i in range(1, len(px) - 1):
        prev, cur, nxt = px.iloc[i - 1], px.iloc[i], px.iloc[i + 1]
        if not (np.isfinite(prev) and np.isfinite(cur) and np.isfinite(nxt)):
            continue
        if prev <= 0 or cur <= 0 or nxt <= 0:
            continue
        leg_in = abs(cur / prev - 1.0)
        leg_out = abs(nxt / cur - 1.0)
        bridge = abs(nxt / prev - 1.0)
        if leg_in > max_abs_return and leg_out > max_abs_return and bridge <= max_abs_return:
            mask.iloc[i] = True
    return mask


def clean_price_series(close: pd.Series, max_abs_return: float = 0.15) -> pd.Series:
    """Legacy convenience API: remove only conservatively quarantinable prices."""
    mask = quarantine_price_mask(close, max_abs_return)
    return close.loc[~mask].copy()


def clean_daily_data(daily_data: dict, max_abs_return: float = 0.15) -> dict:
    """Quarantine entire OHLCV rows while retaining a provenance audit trail."""
    out = {}
    for name, df in daily_data.items():
        if df is None or df.empty:
            out[name] = df
            continue
        if "close" not in df:
            raise ValueError(f"{name}: cannot clean a frame without close")
        suspect = suspect_price_mask(df["close"], max_abs_return)
        quarantine = quarantine_price_mask(df["close"], max_abs_return)
        suspect_ts = [pd.Timestamp(ts).isoformat() for ts in df.index[suspect]]
        quarantine_ts = [pd.Timestamp(ts).isoformat() for ts in df.index[quarantine]]
        clean = df.loc[~quarantine].copy()
        clean.attrs.update(df.attrs)
        clean.attrs["suspect_timestamps"] = suspect_ts
        clean.attrs["quarantined_timestamps"] = quarantine_ts
        if suspect_ts:
            log.warning("%s: flagged %d suspect price row(s) for review: %s",
                        name, len(suspect_ts), ", ".join(suspect_ts))
        if quarantine_ts:
            log.warning("%s: quarantined %d invalid/isolated-spike OHLCV row(s): %s",
                        name, len(quarantine_ts), ", ".join(quarantine_ts))
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
