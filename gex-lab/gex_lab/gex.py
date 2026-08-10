"""
gex_lab/gex.py
==============
GEX math and level construction.

The dealer-position sign convention remains a major model assumption: calls are
assigned positive dealer gamma and puts negative dealer gamma. This is not an
observable dealer inventory reconstruction.

Black-Scholes gamma assumptions are explicit and configurable. The risk-free rate
and continuous dividend yield are inputs to every calculation rather than hidden
inside an unexplained fixed 4% constant.
"""

import logging
from math import exp, pi, sqrt

import numpy as np
import pandas as pd

log = logging.getLogger("gex")

DEFAULT_RISK_FREE = 0.04
DEFAULT_DIVIDEND_YIELD = 0.0
# Compatibility alias used by older callers/tests.
RISK_FREE = DEFAULT_RISK_FREE
_CHAIN_COLS = ["strike", "kind", "oi", "iv", "T"]


def _norm_pdf(x):
    return np.exp(-x * x / 2.0) / sqrt(2 * pi)


def bs_gamma(S, K, T, sigma, r=DEFAULT_RISK_FREE, q=DEFAULT_DIVIDEND_YIELD):
    """Black-Scholes-Merton spot gamma with continuous dividend yield ``q``."""
    S, K, T, sigma, r, q = map(float, (S, K, T, sigma, r, q))
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    root_t = sqrt(T)
    d1 = (np.log(S / K) + (r - q + sigma * sigma / 2.0) * T) / (sigma * root_t)
    return float(exp(-q * T) * _norm_pdf(d1) / (S * sigma * root_t))


def _sign(kind):
    if kind == "C":
        return 1.0
    if kind == "P":
        return -1.0
    raise ValueError(f"unknown option kind {kind!r}")


def _validate_chain(chain: pd.DataFrame):
    missing = [c for c in _CHAIN_COLS if c not in chain.columns]
    if missing:
        raise ValueError("chain missing columns: " + ", ".join(missing))


def net_gex_by_strike(chain: pd.DataFrame, spot: float,
                      r=DEFAULT_RISK_FREE, q=DEFAULT_DIVIDEND_YIELD) -> pd.Series:
    if chain.empty:
        return pd.Series(dtype=float)
    _validate_chain(chain)
    if spot <= 0:
        raise ValueError("spot must be positive")
    g = chain.apply(lambda x: bs_gamma(spot, x["strike"], x["T"], x["iv"], r, q), axis=1)
    signs = chain["kind"].map(_sign)
    gex = g * chain["oi"].astype(float) * 100.0 * spot * spot * 0.01 * signs
    return gex.groupby(chain["strike"]).sum().sort_index()


def total_gex_at(chain: pd.DataFrame, S: float,
                 r=DEFAULT_RISK_FREE, q=DEFAULT_DIVIDEND_YIELD) -> float:
    if chain.empty:
        return 0.0
    _validate_chain(chain)
    if S <= 0:
        raise ValueError("spot must be positive")
    g = chain.apply(lambda x: bs_gamma(S, x["strike"], x["T"], x["iv"], r, q), axis=1)
    signs = chain["kind"].map(_sign)
    return float((g * chain["oi"].astype(float) * 100.0 * S * S * 0.01 * signs).sum())


def gamma_flip(chain: pd.DataFrame, spot: float, r=DEFAULT_RISK_FREE,
               q=DEFAULT_DIVIDEND_YIELD, lo=0.6, hi=1.4, n=400):
    if chain.empty:
        return None
    if spot <= 0 or not (0 < lo < hi) or n < 2:
        raise ValueError("invalid gamma-flip grid")
    grid = np.linspace(spot * lo, spot * hi, n)
    tot = np.array([total_gex_at(chain, S, r, q) for S in grid])
    crossings = []
    for i in range(1, len(grid)):
        if tot[i - 1] == 0 or (tot[i - 1] < 0) != (tot[i] < 0):
            x0, x1, y0, y1 = grid[i - 1], grid[i], tot[i - 1], tot[i]
            crossings.append(x0 if y1 == y0 else x0 - y0 * (x1 - x0) / (y1 - y0))
    return (float(min(crossings, key=lambda x: abs(x - spot)))
            if crossings else None)


def center_of_mass(chain: pd.DataFrame, kind: str):
    if kind not in {"C", "P"}:
        raise ValueError("kind must be C or P")
    side = chain[chain["kind"] == kind]
    total = side["oi"].sum() if not side.empty else 0
    if total <= 0:
        return None
    return float((side["strike"] * side["oi"]).sum() / total)


def compute_levels(chain: pd.DataFrame, spot: float, r=DEFAULT_RISK_FREE,
                   q=DEFAULT_DIVIDEND_YIELD) -> dict:
    """Best-effort level reconstruction under the documented sign/BSM assumptions."""
    if spot <= 0:
        raise ValueError("spot must be positive")
    by = net_gex_by_strike(chain, spot, r, q)
    above = by[by.index > spot]
    put_oi = (chain[chain["kind"] == "P"].groupby("strike")["oi"].sum()
              if not chain.empty else pd.Series(dtype=float))
    pos_gex = float(above.idxmax()) if len(above) and above.max() > 0 else None
    flip = gamma_flip(chain, spot, r, q)
    put_wall = float(put_oi.idxmax()) if len(put_oi) else None
    cotmp = center_of_mass(chain, "P") if not chain.empty else None
    cotmc = center_of_mass(chain, "C") if not chain.empty else None
    return {
        "spot": float(spot),
        "net_gex": float(by.sum()),
        "pos_gex": pos_gex,
        "zero_gex": flip,
        "cotmp": cotmp,
        "cotmc": cotmc,
        "put_wall": put_wall,
        "ptrans": flip,
        "ntrans": put_wall,
        "risk_free": float(r),
        "dividend_yield": float(q),
        "dealer_sign_model": "calls_positive_puts_negative",
    }


def fetch_chain(ticker: str, max_days: int = 45):
    """Fetch near-dated yfinance chains; network access is isolated here."""
    if max_days <= 0:
        raise ValueError("max_days must be positive")
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
