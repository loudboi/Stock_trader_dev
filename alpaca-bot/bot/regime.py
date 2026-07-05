"""
bot/regime.py
=============
Free macro-regime indicators, for conditioning/combining strategies in bot/lab.py
on the broader macro backdrop (not just price action of the traded universe).

Four indicators, each with FIXED, CONVENTIONAL thresholds chosen from standard
industry/academic usage — decided before running any backtest, not fit to make
one look good:

  vix_regime()      VIX level: "calm" < 15, "elevated" > 25 (widely-cited industry
                    cutoffs; VIX's own long-run median is ~17-19).
  curve_regime()    10Y minus 13-week Treasury yield (^TNX - ^IRX). Negative =
                    inverted, the classic (NY Fed-documented) recession signal.
  credit_regime()   HYG (high-yield) relative performance vs IEF (duration-matched
                    Treasury) over a trailing window — a "flight to quality"/credit-
                    spread-widening proxy. HYG only exists from 2007-04-11.
  vix_term_regime() VIX / VIX3M ratio: > 1.0 is backwardation (near-term fear
                    exceeds the 3-month view — a well-documented stress signal
                    distinct from the VIX LEVEL check above), < 1.0 is the normal
                    contango state. ^VIX3M only exists from 2007-01-03.

All fetched via bot.data.load_yahoo (free, daily). Pure classification functions
operate on a price/yield Series so they're unit-testable without network.
"""

import logging

import numpy as np
import pandas as pd

from bot.data import load_yahoo

log = logging.getLogger("regime")

VIX_CALM = 15.0
VIX_ELEVATED = 25.0
CREDIT_LOOKBACK = 60


def fetch_vix(start, end) -> pd.Series:
    df = load_yahoo("^VIX", start, end)
    return df["close"] if not df.empty else pd.Series(dtype=float)


def fetch_curve(start, end) -> pd.Series:
    """10Y - 13-week Treasury yield, in percentage points (^TNX/^IRX are already
    quoted in yield %, e.g. 4.25 = 4.25%)."""
    tnx = load_yahoo("^TNX", start, end)["close"]
    irx = load_yahoo("^IRX", start, end)["close"]
    df = pd.concat([tnx.rename("tnx"), irx.rename("irx")], axis=1).ffill().dropna()
    return (df["tnx"] - df["irx"]).rename("curve")


def fetch_credit(start, end) -> pd.Series:
    """Trailing relative return of HYG vs IEF (positive = credit outperforming /
    risk-on; negative = credit stress). HYG only exists from 2007-04-11."""
    hyg = load_yahoo("HYG", start, end)["close"]
    ief = load_yahoo("IEF", start, end)["close"]
    df = pd.concat([hyg.rename("hyg"), ief.rename("ief")], axis=1).ffill().dropna()
    rel = (df["hyg"].pct_change(CREDIT_LOOKBACK) - df["ief"].pct_change(CREDIT_LOOKBACK))
    return rel.rename("credit")


def fetch_vix_term_ratio(start, end) -> pd.Series:
    """VIX / VIX3M. ^VIX3M only exists from 2007-01-03."""
    vix = load_yahoo("^VIX", start, end)["close"]
    vix3m = load_yahoo("^VIX3M", start, end)["close"]
    df = pd.concat([vix.rename("vix"), vix3m.rename("vix3m")], axis=1).ffill().dropna()
    return (df["vix"] / df["vix3m"]).rename("vix_term")


# --------------------------------------------------------------------------- #
# Pure classifiers (testable offline)
# --------------------------------------------------------------------------- #
def vix_regime(vix: pd.Series, calm=VIX_CALM, elevated=VIX_ELEVATED) -> pd.Series:
    """+1 calm (VIX < calm), -1 elevated (VIX > elevated), 0 neutral in between."""
    out = pd.Series(0, index=vix.index, dtype=int)
    out[vix < calm] = 1
    out[vix > elevated] = -1
    return out


def curve_regime(curve: pd.Series) -> pd.Series:
    """+1 normal (positive/upward-sloping curve), -1 inverted."""
    return pd.Series(np.where(curve >= 0, 1, -1), index=curve.index)


def credit_regime(credit: pd.Series, threshold=0.0) -> pd.Series:
    """+1 credit outperforming Treasuries (risk-on), -1 credit stress (risk-off)."""
    return pd.Series(np.where(credit >= threshold, 1, -1), index=credit.index)


def vix_term_regime(ratio: pd.Series, backwardation=1.0) -> pd.Series:
    """-1 backwardation (ratio > 1.0, stress -- near-term fear exceeds the 3-month
    view), +1 contango (normal/calm). A structural signal, distinct from the VIX
    LEVEL check in vix_regime()."""
    return pd.Series(np.where(ratio > backwardation, -1, 1), index=ratio.index)


def align_to_panel(regime: pd.Series, panel_index: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill a regime series onto the traded panel's index (macro data
    often has a different calendar/holidays than the traded universe)."""
    return regime.reindex(panel_index, method="ffill")
