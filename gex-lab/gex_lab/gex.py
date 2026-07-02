"""
gex_lab/gex.py
==============
The GEX math and level computation. Pure functions operate on a "chain" DataFrame
(columns: strike, kind ['C'/'P'], oi, iv, T) so they're testable offline; the
yfinance fetch is separated out.

GEX convention (naive, and the single biggest assumption): dealers are LONG calls
and SHORT puts, so a call strike contributes +gamma·OI and a put strike −gamma·OI.
Dollar GEX per strike is gamma · OI · 100 · spot² · 0.01 (≈ $ hedging per 1% move).
"""

import logging
from math import pi, sqrt

import numpy as np
import pandas as pd

log = logging.getLogger("gex")

RISK_FREE = 0.04
_CHAIN_COLS = ["strike", "kind", "oi", "iv", "T"]


def _norm_pdf(x):
    return np.exp(-x * x / 2.0) / sqrt(2 * pi)


def bs_gamma(S, K, T, sigma, r=RISK_FREE):
    """Black-Scholes gamma (same for calls and puts). 0 for degenerate inputs."""
    S, K, T, sigma = float(S), float(K), float(T), float(sigma)
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + sigma * sigma / 2.0) * T) / (sigma * sqrt(T))
    return float(_norm_pdf(d1) / (S * sigma * sqrt(T)))


def _sign(kind):
    return 1.0 if kind == "C" else -1.0


def net_gex_by_strike(chain: pd.DataFrame, spot: float, r=RISK_FREE) -> pd.Series:
    """Dollar GEX per strike at the current spot (calls +, puts −), $ per 1% move."""
    if chain.empty:
        return pd.Series(dtype=float)
    g = chain.apply(lambda x: bs_gamma(spot, x["strike"], x["T"], x["iv"], r), axis=1)
    gex = g * chain["oi"] * 100.0 * spot * spot * 0.01 * chain["kind"].map(_sign)
    return gex.groupby(chain["strike"]).sum().sort_index()


def total_gex_at(chain: pd.DataFrame, S: float, r=RISK_FREE) -> float:
    """Total dollar GEX if spot were S (gammas recomputed at S). Used for the flip."""
    if chain.empty:
        return 0.0
    g = chain.apply(lambda x: bs_gamma(S, x["strike"], x["T"], x["iv"], r), axis=1)
    return float((g * chain["oi"] * 100.0 * S * S * 0.01 * chain["kind"].map(_sign)).sum())


def gamma_flip(chain: pd.DataFrame, spot: float, r=RISK_FREE,
               lo=0.6, hi=1.4, n=400):
    """Zero-gamma level: the spot at which total GEX crosses zero (dealers flip
    from short to long gamma). Returns the crossing nearest spot, or None."""
    if chain.empty:
        return None
    grid = np.linspace(spot * lo, spot * hi, n)
    tot = np.array([total_gex_at(chain, S, r) for S in grid])
    crossings = []
    for i in range(1, len(grid)):
        if tot[i - 1] == 0 or (tot[i - 1] < 0) != (tot[i] < 0):
            # linear interpolate the zero crossing
            x0, x1, y0, y1 = grid[i - 1], grid[i], tot[i - 1], tot[i]
            crossings.append(x0 if y1 == y0 else x0 - y0 * (x1 - x0) / (y1 - y0))
    if not crossings:
        return None
    return float(min(crossings, key=lambda x: abs(x - spot)))


def center_of_mass(chain: pd.DataFrame, kind: str):
    """Open-interest-weighted average strike for one side ('C' or 'P')."""
    side = chain[chain["kind"] == kind]
    if side.empty or side["oi"].sum() == 0:
        return None
    return float((side["strike"] * side["oi"]).sum() / side["oi"].sum())


def compute_levels(chain: pd.DataFrame, spot: float, r=RISK_FREE) -> dict:
    """The level set the Vol-Desk-style system keys off. Several are best-effort
    reconstructions of proprietary definitions (see notes)."""
    by = net_gex_by_strike(chain, spot, r)
    above = by[by.index > spot]
    below = by[by.index < spot]
    put_oi = chain[chain["kind"] == "P"].groupby("strike")["oi"].sum()

    pos_gex = float(above.idxmax()) if len(above) and above.max() > 0 else None   # +GEX target
    flip = gamma_flip(chain, spot, r)                                            # zeroGEX
    put_wall = float(put_oi.idxmax()) if len(put_oi) else None
    cotmp = center_of_mass(chain, "P")                                           # center of put mass
    cotmc = center_of_mass(chain, "C")                                          # center of call mass

    return {
        "spot": float(spot),
        "net_gex": float(by.sum()),
        "pos_gex": pos_gex,          # +GEX  (call-side gamma magnet above spot)
        "zero_gex": flip,            # zeroGEX (gamma flip)
        "cotmp": cotmp,              # Center Of puT Mass (put structural floor)
        "cotmc": cotmc,              # Center Of call Mass
        "put_wall": put_wall,
        # Approximations of the proprietary transition levels:
        #   pTrans ~ the gamma flip (cross into positive-gamma regime)
        #   nTrans ~ the put wall  (structural break below)
        "ptrans": flip,
        "ntrans": put_wall,
    }


# --------------------------------------------------------------------------- #
# Network fetch (kept separate so the math above is unit-tested offline)
# --------------------------------------------------------------------------- #
def fetch_chain(ticker: str, max_days: int = 45):
    """Aggregate near-dated option chains from yfinance into a chain DataFrame.
    Returns (chain_df, spot). chain_df has columns strike/kind/oi/iv/T."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    hist = t.history(period="1d")
    if hist.empty:
        return pd.DataFrame(columns=_CHAIN_COLS), None
    spot = float(hist["Close"].iloc[-1])
    today = pd.Timestamp.now(tz="UTC").tz_localize(None)
    rows = []
    for e in (t.options or []):
        T = max((pd.Timestamp(e) - today).days, 0) / 365.0
        if T <= 0 or T > max_days / 365.0:
            continue
        try:
            ch = t.option_chain(e)
        except Exception as ex:  # noqa: BLE001
            log.debug("chain fetch failed for %s %s: %s", ticker, e, ex)
            continue
        for df, kind in ((ch.calls, "C"), (ch.puts, "P")):
            for _, x in df.iterrows():
                oi = x.get("openInterest", 0) or 0
                iv = x.get("impliedVolatility", 0) or 0
                if oi > 0 and iv > 0:
                    rows.append((float(x["strike"]), kind, float(oi), float(iv), T))
    return pd.DataFrame(rows, columns=_CHAIN_COLS), spot
