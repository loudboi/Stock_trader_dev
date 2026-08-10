"""
config.py
=========
Single source of truth for Strategy 4 instruments and live risk limits.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Per-position and portfolio risk. A fully-built position targets 1% stop risk,
# while the whole managed book is capped at 5% stop risk and 100% gross notional.
# These are strategy limits, not broker buying-power limits.
RISK_PER_TRADE = 0.01
MAX_PORTFOLIO_RISK = 0.05
MAX_GROSS_EXPOSURE = 1.00


@dataclass
class Instrument:
    name: str
    api_symbol: str
    asset_class: str
    strategy: str
    timeframe: str
    can_short: bool
    qty_decimals: int = 0
    params: dict = field(default_factory=dict)


def _pullback_instrument(name, api_symbol, asset_class, qty_decimals=0):
    return Instrument(
        name=name, api_symbol=api_symbol, asset_class=asset_class,
        strategy="trend_pullback", timeframe="1Day", can_short=False,
        qty_decimals=qty_decimals,
    )


INSTRUMENTS = {
    "SPY": _pullback_instrument("SPY", "SPY", "equity"),
    "QQQ": _pullback_instrument("QQQ", "QQQ", "equity"),
    "BTC/USD": _pullback_instrument("BTC/USD", "BTC/USD", "crypto", qty_decimals=6),
    "GLD": _pullback_instrument("GLD", "GLD", "equity"),
    "USO": _pullback_instrument("USO", "USO", "equity"),
}

PULLBACK_SYMBOLS = ["SPY", "QQQ", "BTC/USD", "GLD", "USO"]
PULLBACK_UNIVERSE: dict = {}


def resolve_instrument(name: str) -> Optional[Instrument]:
    return INSTRUMENTS.get(name) or PULLBACK_UNIVERSE.get(name)


def research_instrument(name: str) -> Instrument:
    """Resolve a configured instrument or make a read-only equity data spec.

    Research CLIs accept arbitrary ordinary stock/ETF tickers. Requiring every
    benchmark ticker (TLT, IWM, EFA, sector ETFs, ...) to be added to the *live*
    trading universe made their default Alpaca commands fail. The transient spec
    is never registered in PULLBACK_UNIVERSE and therefore cannot make an unknown
    symbol live-tradable.
    """
    known = resolve_instrument(name)
    if known is not None:
        return known
    if not name or "/" in name or "=" in name or name.startswith("^"):
        raise ValueError(
            f"{name!r} is not a configured Alpaca instrument. Use Yahoo for FX/index/"
            "Yahoo-style symbols, or explicitly configure it for live Alpaca use.")
    return _pullback_instrument(name, name, "equity")


def validate_config() -> None:
    missing = [k for k, v in {
        "ALPACA_API_KEY": ALPACA_API_KEY,
        "ALPACA_SECRET_KEY": ALPACA_SECRET_KEY,
    }.items() if not v]
    if missing:
        raise RuntimeError(
            f"Missing required env vars: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your keys."
        )
    if not (0 < RISK_PER_TRADE <= MAX_PORTFOLIO_RISK <= 1):
        raise RuntimeError("Risk limits must satisfy 0 < RISK_PER_TRADE <= MAX_PORTFOLIO_RISK <= 1")
    if not (0 < MAX_GROSS_EXPOSURE <= 10):
        raise RuntimeError("MAX_GROSS_EXPOSURE must be positive")
