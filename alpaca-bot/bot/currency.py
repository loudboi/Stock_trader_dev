"""
bot/currency.py
===============
Currency conversion helpers for an EUR-based investor holding USD assets.

`EURUSD=X` is USD per EUR, so a USD asset's EUR daily growth factor is
``(1 + usd_asset_return) / (1 + eurusd_return)``. For a portfolio that is only
partly invested in USD assets and keeps the remainder in EUR cash, the conversion
is performed as an exact wealth recurrence rather than multiplying an approximate
FX-return adjustment.

FX alignment is causal: observations may be forward-filled from the last known FX
close, but future quotes are never backfilled into dates before FX history starts.
"""

import numpy as np
import pandas as pd

import bot.trend_exposure as te

_ANNUAL = 252


def te_invested_fraction(daily_data: dict, ma_period: int = 200,
                         buffer: float = 0.01) -> pd.Series:
    if not daily_data:
        return pd.Series(dtype=float)
    per_symbol = {
        name: te.exposure_series(daily["close"], ma_period, buffer).shift(1).fillna(0.0)
        for name, daily in daily_data.items()
    }
    aligned = pd.concat(per_symbol, axis=1).fillna(0.0)
    # Fixed capital split: a missing sleeve is idle, not reallocated to others.
    return aligned.sum(axis=1) / len(per_symbol)


def _fraction(invested_frac, idx: pd.Index) -> pd.Series:
    if isinstance(invested_frac, (int, float, np.number)):
        frac = pd.Series(float(invested_frac), index=idx)
    else:
        frac = pd.Series(invested_frac).reindex(idx).fillna(0.0).astype(float)
    if ((frac < -1e-12) | (frac > 1 + 1e-12)).any():
        raise ValueError("invested_frac must stay within [0, 1]")
    return frac.clip(0.0, 1.0)


def align_fx_causally(fx_close: pd.Series, idx: pd.Index) -> pd.Series:
    """Align FX closes without ever using a future observation for an earlier date."""
    if fx_close is None or len(fx_close) == 0:
        raise ValueError("FX history is required for unhedged EUR conversion")
    fx = pd.Series(fx_close, dtype=float).sort_index()
    if (fx <= 0).any():
        raise ValueError("FX close must be positive")
    union = fx.index.union(pd.Index(idx)).sort_values()
    aligned = fx.reindex(union).ffill().reindex(idx)
    if aligned.isna().any():
        first_missing = aligned.index[aligned.isna()][0]
        raise ValueError(
            f"FX history starts after required portfolio date {first_missing}; "
            "refusing to backfill future FX quotes")
    return aligned


def unhedged_eur_equity(usd_returns: pd.Series, fx_close: pd.Series,
                        invested_frac=1.0, initial: float = 100_000.0) -> pd.Series:
    """Exact EUR wealth path for a USD-return stream plus EUR cash.

    `usd_returns[t]` is the whole portfolio's USD-denominated return contribution
    for day t. `invested_frac[t]` is the fraction whose capital is actually exposed
    to USD assets; the remainder sits in EUR cash. If ``f > 0``, the implied USD
    asset return is ``usd_returns / f`` and the exact EUR factor is::

        (1-f) + f * (1 + usd_returns/f) / (1 + fx_return)

    which simplifies to ``(1-f) + (f + usd_returns)/(1+fx_return)``.
    """
    if usd_returns.empty:
        return pd.Series(dtype=float)
    r = usd_returns.sort_index().astype(float)
    frac = _fraction(invested_frac, r.index)
    fx = align_fx_causally(fx_close, r.index)
    fx_ret = fx.pct_change(fill_method=None).fillna(0.0)
    if (1.0 + fx_ret <= 0).any():
        raise ValueError("invalid FX return")

    factor = pd.Series(1.0, index=r.index, dtype=float)
    exposed = frac > 0
    factor.loc[exposed] = ((1.0 - frac.loc[exposed]) +
                           (frac.loc[exposed] + r.loc[exposed]) /
                           (1.0 + fx_ret.loc[exposed]))
    # When f=0, a strategy should not report USD-asset P&L. Refuse inconsistent
    # inputs instead of silently discarding a return.
    inconsistent = (~exposed) & (r.abs() > 1e-12)
    if inconsistent.any():
        raise ValueError("non-zero USD return while invested_frac is zero")
    if (factor <= 0).any():
        raise ValueError("EUR portfolio wealth factor became non-positive")
    out = initial * factor.cumprod()
    out.attrs["initial_equity"] = initial
    return out


def hedged_eur_equity(usd_returns: pd.Series, annual_hedge_cost: float = 0.015,
                      initial: float = 100_000.0, invested_frac=1.0) -> pd.Series:
    """Currency-hedged scenario with hedge cost charged only on USD exposure."""
    if usd_returns.empty:
        return pd.Series(dtype=float)
    if annual_hedge_cost < 0:
        raise ValueError("annual_hedge_cost cannot be negative")
    r = usd_returns.sort_index().astype(float)
    frac = _fraction(invested_frac, r.index)
    hedged_returns = r - frac * (annual_hedge_cost / _ANNUAL)
    out = initial * (1.0 + hedged_returns).cumprod()
    out.attrs["initial_equity"] = initial
    return out
