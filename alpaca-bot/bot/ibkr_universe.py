"""
bot/ibkr_universe.py
====================
EUR-denominated instrument universe for Strategy 4 on Interactive Brokers.

This is kept entirely separate from the Alpaca config. It reuses the existing
`config.Instrument` dataclass (so the strategy/executor see the same shape) but
stashes the IBKR-specific contract details under params["ibkr"], because the
Alpaca Instrument has no exchange/currency fields.

To make the existing live trader resolve these names, call register() once at
startup; it injects them into config.PULLBACK_UNIVERSE for *this process only*
(it never edits any file, and the Alpaca path leaves PULLBACK_UNIVERSE empty).

────────────────────────────────────────────────────────────────────────────
IMPORTANT — verify every contract in TWS before trading it.
IBKR symbols, primary exchanges and currencies are exact and unforgiving: a
wrong code silently resolves to the wrong instrument or fails the order. Open
TWS → right-click → "Contract Details" (or use the adapter's `describe()` which
calls reqContractDetails) and confirm symbol / primaryExchange / currency match
what's below. Treat the values here as sensible defaults, not gospel.

You also need the matching IBKR market-data subscriptions (e.g. Xetra/Germany,
Euronext) or live price + historical requests for these will be delayed/empty.
────────────────────────────────────────────────────────────────────────────
"""

import logging
import config

log = logging.getLogger("ibkr_universe")


def _eur(name, symbol, primary_exchange, sec_type="STK", exchange="SMART",
         qty_decimals=0):
    """Build a trend_pullback Instrument carrying an IBKR contract spec."""
    return config.Instrument(
        name=name,
        api_symbol=symbol,            # unused by IBKR path; kept for shape parity
        asset_class="equity",
        strategy="trend_pullback",
        timeframe="1Day",
        can_short=False,              # Strategy 4 is long-only
        qty_decimals=qty_decimals,    # European shares trade in whole units
        params={"ibkr": {
            "symbol": symbol,
            "sec_type": sec_type,
            "exchange": exchange,            # SMART routing is fine for these
            "primary_exchange": primary_exchange,
            "currency": "EUR",
        }},
    )


# --------------------------------------------------------------------------- #
# EUR index ETFs (UCITS, EUR-listed) — the "indexes"
# --------------------------------------------------------------------------- #
# Broad, liquid, and they actually trend — the three traits Strategy 4 needs.
# Three is deliberately enough: broad-Europe, eurozone-50, and Germany cover
# different drivers without piling on funds that all move together.
_INDEX_ETFS = {
    # iShares STOXX Europe 600 UCITS ETF — broad pan-European index (Xetra)
    "STOXX600": _eur("STOXX600", "EXSA", "IBIS"),
    # iShares EURO STOXX 50 UCITS ETF — eurozone blue-chip index (Xetra)
    "ESTX50":   _eur("ESTX50", "EXW1", "IBIS"),
    # iShares Core DAX UCITS ETF — German large-cap index (Xetra)
    "DAX":      _eur("DAX", "EXS1", "IBIS"),
}

# --------------------------------------------------------------------------- #
# EUR blue-chip stocks — liquid megacaps that tend to trend
# --------------------------------------------------------------------------- #
_STOCKS = {
    "SAP":  _eur("SAP",  "SAP",  "IBIS"),   # SAP SE — Xetra, software megacap
    "ASML": _eur("ASML", "ASML", "AEB"),    # ASML Holding — Euronext Amsterdam
    "SIE":  _eur("SIE",  "SIE",  "IBIS"),   # Siemens AG — Xetra
    "MC":   _eur("MC",   "MC",   "SBF"),    # LVMH — Euronext Paris
    "TTE":  _eur("TTE",  "TTE",  "SBF"),    # TotalEnergies — Euronext Paris
}

# The full EUR universe. Edit this dict (or pass --symbols) to change what
# Strategy 4 trades on IBKR.
IBKR_EUR_UNIVERSE = {**_INDEX_ETFS, **_STOCKS}

# A conservative default to actually run with: the three indexes plus two of the
# steadiest single names. Diversified, liquid, and small enough to watch.
DEFAULT_SYMBOLS = ["STOXX600", "ESTX50", "DAX", "SAP", "ASML"]


def register() -> None:
    """Inject the EUR universe into config so resolve_instrument() finds it.

    Process-local only: this mutates the in-memory dict, never the source file.
    The Alpaca runs never call this, so their PULLBACK_UNIVERSE stays empty.
    """
    config.PULLBACK_UNIVERSE.update(IBKR_EUR_UNIVERSE)
    log.info("Registered %d EUR instruments for IBKR.", len(IBKR_EUR_UNIVERSE))


def ibkr_spec(inst) -> dict:
    """Pull the IBKR contract spec out of an instrument, or raise if missing."""
    spec = (inst.params or {}).get("ibkr")
    if not spec:
        raise ValueError(
            f"{inst.name} has no IBKR contract spec. Define it in ibkr_universe.py "
            "(symbol/primary_exchange/currency) before trading it on IBKR.")
    return spec
