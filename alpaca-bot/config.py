"""
config.py
=========
Single source of truth for Strategy 4's instruments and risk setting.

Strategy 4 (phased trend-pullback) is long-only and multi-timeframe: it decides
on completed daily bars and executes in real time. It runs as its own live
process (bot/live_pullback.py) with its own backtester (bot/backtest_pullback.py).

A note on symbols: alpaca-py uses the slash form for crypto everywhere (e.g.
"BTC/USD") for both data and trading. We keep the human-readable name as the key
and store the API symbol separately so the rest of the code never has to think
about it. (Equities are the plain ticker, e.g. "SPY".)
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #
# Each fully-built position, stopped at its volatility stop, risks this fraction
# of equity. The 30/30/40 tranches split that budget (see PullbackParams).
RISK_PER_TRADE = 0.01          # 1% of total account equity


# --------------------------------------------------------------------------- #
# Instrument definitions
# --------------------------------------------------------------------------- #
@dataclass
class Instrument:
    name: str                       # human-readable key, e.g. "BTC/USD"
    api_symbol: str                 # what we send to Alpaca, e.g. "BTC/USD"
    asset_class: str                # "equity" or "crypto"
    strategy: str                   # always "trend_pullback" here
    timeframe: str                  # signal timeframe ("1Day")
    can_short: bool                 # Strategy 4 is long-only; kept for shape parity
    qty_decimals: int = 0           # 0 = whole shares, >0 = fractional
    params: dict = field(default_factory=dict)


def _pullback_instrument(name, api_symbol, asset_class, qty_decimals=0):
    return Instrument(
        name=name, api_symbol=api_symbol, asset_class=asset_class,
        strategy="trend_pullback", timeframe="1Day",
        can_short=False, qty_decimals=qty_decimals,
    )


# The default Alpaca universe Strategy 4 trades.
INSTRUMENTS = {
    "SPY": _pullback_instrument("SPY", "SPY", "equity"),
    "QQQ": _pullback_instrument("QQQ", "QQQ", "equity"),
    "BTC/USD": _pullback_instrument("BTC/USD", "BTC/USD", "crypto", qty_decimals=6),
    "GLD": _pullback_instrument("GLD", "GLD", "equity"),
    "USO": _pullback_instrument("USO", "USO", "equity"),
}


# --------------------------------------------------------------------------- #
# Strategy 4 universe
# --------------------------------------------------------------------------- #
# To EXPAND the symbols Strategy 4 trades, edit PULLBACK_SYMBOLS below. If a
# symbol is NOT one of the five already defined in INSTRUMENTS, add its metadata
# to PULLBACK_UNIVERSE first so the resolver can find it. (The IBKR path injects
# its EUR universe here at runtime via bot/ibkr_universe.register().)

PULLBACK_SYMBOLS = ["SPY", "QQQ", "BTC/USD", "GLD", "USO"]

# Extra instruments available to Strategy 4 (added at runtime or by hand).
# Example:
#   "IWM": _pullback_instrument("IWM", "IWM", "equity"),
#   "ETH/USD": _pullback_instrument("ETH/USD", "ETH/USD", "crypto", qty_decimals=6),
PULLBACK_UNIVERSE: dict = {}


def resolve_instrument(name: str) -> Optional[Instrument]:
    """Look up an instrument by name across the core set and the pullback universe."""
    return INSTRUMENTS.get(name) or PULLBACK_UNIVERSE.get(name)


def validate_config() -> None:
    """Fail fast with a clear message if keys are missing."""
    missing = [k for k, v in {
        "ALPACA_API_KEY": ALPACA_API_KEY,
        "ALPACA_SECRET_KEY": ALPACA_SECRET_KEY,
    }.items() if not v]
    if missing:
        raise RuntimeError(
            f"Missing required env vars: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your keys."
        )
