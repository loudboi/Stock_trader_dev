"""
bot/portfolio_ibkr.py
=====================
Interactive Brokers adapter for Strategy 4 — a drop-in alternative to
bot/portfolio.py that talks to IBKR via ib_async instead of Alpaca.

It implements the exact methods the pullback live trader calls:
    get_equity, get_position_raw, latest_price, get_historical_bars,
    submit_market_order, close_position_raw, is_tradable_now
plus live_positions / connect / disconnect for completeness.

Nothing here is imported by the Alpaca path. The Alpaca Portfolio is untouched.

Design notes specific to IBKR (these are the things that differ from Alpaca):
  • You talk to a local IB Gateway / TWS, not a cloud endpoint. Connection
    details come from env: IBKR_HOST (127.0.0.1), IBKR_PORT (4002 = paper
    Gateway), IBKR_CLIENT_ID (17). Live ports are 4001 (Gateway) / 7496 (TWS).
  • IBKR rate-limits historical data hard. So daily bars are cached on disk and
    only a short recent tail is re-fetched each cycle (merge_bar_cache).
  • There is no "close position" endpoint — closing = submitting the opposite
    market order for the current quantity.
  • avgCost from IBKR is per-share including commissions → used as avg_entry.
  • Account equity is read in your BASE currency (EUR), so there is no FX layer.

The pure helpers near the top take plain values (not ib objects) so they are
unit-tested offline without a Gateway. Live methods are thin wrappers over them.
"""

import logging
import os
from datetime import datetime, timezone

import pandas as pd

log = logging.getLogger("portfolio_ibkr")


# ════════════════════════════════════════════════════════════════════════════
# Pure helpers (no IBKR connection needed — unit-tested offline)
# ════════════════════════════════════════════════════════════════════════════
def contract_kwargs(spec: dict) -> dict:
    """Translate an ibkr_universe spec into Stock() constructor kwargs."""
    return {
        "symbol": spec["symbol"],
        "exchange": spec.get("exchange", "SMART"),
        "currency": spec.get("currency", "EUR"),
        "primaryExchange": spec.get("primary_exchange"),
    }


def map_ib_position(position: float, avg_cost: float) -> dict | None:
    """IBKR Position (qty, avgCost) -> {'side','qty','avg_entry'} or None if flat."""
    if position is None or position == 0:
        return None
    return {
        "side": "long" if position > 0 else "short",
        "qty": abs(float(position)),
        "avg_entry": float(avg_cost),
    }


def equity_from_summary(rows, base_currency: str = "EUR") -> float | None:
    """Extract NetLiquidation in the base currency from accountSummary() rows.

    Rows are objects/dicts with .tag/.value/.currency. Prefer an exact
    base-currency match; fall back to any NetLiquidation row.
    """
    def g(r, k):
        return getattr(r, k, None) if not isinstance(r, dict) else r.get(k)

    fallback = None
    for r in rows:
        if g(r, "tag") != "NetLiquidation":
            continue
        try:
            val = float(g(r, "value"))
        except (TypeError, ValueError):
            continue
        cur = (g(r, "currency") or "").upper()
        if cur == base_currency.upper():
            return val
        fallback = val if fallback is None else fallback
    return fallback


def bars_to_df(bars) -> pd.DataFrame:
    """ib_async historical bars -> OHLCV DataFrame indexed by UTC timestamp."""
    if not bars:
        return pd.DataFrame()
    rows = []
    for b in bars:
        d = getattr(b, "date", None)
        ts = pd.Timestamp(d)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        rows.append((ts, float(b.open), float(b.high), float(b.low),
                     float(b.close), float(getattr(b, "volume", 0) or 0)))
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    return df.set_index("ts").sort_index()


def merge_bar_cache(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Merge a fresh tail into cached bars; last write wins on duplicate stamps."""
    if old is None or old.empty:
        return new.sort_index()
    if new is None or new.empty:
        return old.sort_index()
    merged = pd.concat([old, new])
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.sort_index()


def duration_str(start, end) -> str:
    """IBKR durationStr covering [start, end] with slack, in 'N D' or 'N Y'."""
    days = max(1, (pd.Timestamp(end) - pd.Timestamp(start)).days + 5)
    if days <= 365:
        return f"{days} D"
    years = (days // 365) + 1
    return f"{years} Y"


def market_open_from_hours(hours_str: str, now_local) -> bool | None:
    """Best-effort parse of an IBKR liquidHours/tradingHours string.

    Format examples (semicolon-separated days):
        "20260625:0900-20260625:1730;20260626:CLOSED"
        "20260625:0900-1730;20260626:CLOSED"
    Returns True/False, or None if it can't be determined (caller should then
    fall back to allowing the trade rather than silently blocking).
    """
    if not hours_str:
        return None
    day = now_local.strftime("%Y%m%d")
    hhmm = now_local.strftime("%H%M")
    found_day = False
    for token in hours_str.split(";"):
        token = token.strip()
        if not token or ":" not in token:
            continue
        head = token.split(":", 1)[0]
        if head != day:
            continue
        found_day = True
        if "CLOSED" in token.upper():
            return False
        # token like "20260625:0900-20260625:1730" or "20260625:0900-1730"
        try:
            _, rng = token.split(":", 1)
            open_part, close_part = rng.split("-", 1)
            open_hhmm = open_part.split(":")[-1][-4:]
            close_hhmm = close_part.split(":")[-1][-4:]
            if open_hhmm <= hhmm <= close_hhmm:
                return True
        except (ValueError, IndexError):
            return None
    return False if found_day else None


# ════════════════════════════════════════════════════════════════════════════
# Live adapter
# ════════════════════════════════════════════════════════════════════════════
class IBKRPortfolio:
    """Talks to IB Gateway / TWS. Same method surface the trader expects."""

    def __init__(self, host=None, port=None, client_id=None,
                 base_currency="EUR", cache_dir="ibkr_cache", connect=True):
        try:
            from ib_async import IB
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "ib_async is required for the IBKR path. Install it with "
                "`pip install -r requirements-ibkr.txt`.") from e
        self._IB = IB
        self.ib = IB()
        self.host = host or os.getenv("IBKR_HOST", "127.0.0.1")
        self.port = int(port or os.getenv("IBKR_PORT", "4002"))   # paper Gateway
        self.client_id = int(client_id or os.getenv("IBKR_CLIENT_ID", "17"))
        self.base_currency = base_currency
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._contracts = {}     # name -> qualified contract
        self._hours = {}         # name -> (liquidHours, tzid)
        self._bar_cache = {}     # name -> df
        if connect:
            self.connect()

    # ---- connection lifecycle --------------------------------------------- #
    def connect(self):
        if not self.ib.isConnected():
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=15)
            log.info("Connected to IBKR %s:%s (clientId=%s, base=%s).",
                     self.host, self.port, self.client_id, self.base_currency)

    def _ensure(self):
        if not self.ib.isConnected():
            log.warning("IBKR connection dropped; reconnecting…")
            self.connect()

    def disconnect(self):
        if self.ib.isConnected():
            self.ib.disconnect()

    # ---- contracts -------------------------------------------------------- #
    def _contract(self, inst):
        if inst.name in self._contracts:
            return self._contracts[inst.name]
        from ib_async import Stock
        from bot.ibkr_universe import ibkr_spec
        spec = ibkr_spec(inst)
        c = Stock(**contract_kwargs(spec))
        try:
            qualified = self.ib.qualifyContracts(c)
        except Exception as e:  # noqa: BLE001
            log.error("Could not qualify contract for %s: %s", inst.name, e)
            return None
        if not qualified:
            log.error("IBKR could not resolve a contract for %s (%s). "
                      "Check symbol/primaryExchange/currency in ibkr_universe.py.",
                      inst.name, spec)
            return None
        self._contracts[inst.name] = qualified[0]
        return qualified[0]

    # ---- account / positions ---------------------------------------------- #
    def get_equity(self) -> float:
        self._ensure()
        rows = self.ib.accountSummary()
        val = equity_from_summary(rows, self.base_currency)
        if val is None:
            log.warning("No NetLiquidation found in account summary.")
            return 0.0
        return val

    def live_positions(self) -> dict:
        self._ensure()
        out = {}
        for p in self.ib.positions():
            m = map_ib_position(p.position, p.avgCost)
            if m:
                out[p.contract.localSymbol or p.contract.symbol] = m
        return out

    def get_position_raw(self, inst):
        self._ensure()
        c = self._contract(inst)
        if c is None:
            return None
        for p in self.ib.positions():
            if getattr(p.contract, "conId", None) == getattr(c, "conId", object()):
                return map_ib_position(p.position, p.avgCost)
        return None

    def position_side(self, inst) -> str:
        r = self.get_position_raw(inst)
        return r["side"] if r else "flat"

    # ---- market hours ----------------------------------------------------- #
    def is_tradable_now(self, inst) -> bool:
        self._ensure()
        c = self._contract(inst)
        if c is None:
            return False
        try:
            if inst.name not in self._hours:
                details = self.ib.reqContractDetails(c)
                if details:
                    d = details[0]
                    self._hours[inst.name] = (getattr(d, "liquidHours", "") or
                                              getattr(d, "tradingHours", ""))
                else:
                    self._hours[inst.name] = ""
            hours = self._hours[inst.name]
            verdict = market_open_from_hours(hours, datetime.now())
            # None == couldn't determine → allow (don't silently block trading);
            # IBKR will reject/queue an order if the market is genuinely closed.
            return True if verdict is None else verdict
        except Exception as e:  # noqa: BLE001
            log.warning("Trading-hours check failed for %s (%s); allowing.",
                        inst.name, e)
            return True

    # ---- bars (cached for pacing) ----------------------------------------- #
    _BAR_SIZE = {"15Min": "15 mins", "1Hour": "1 hour",
                 "4Hour": "4 hours", "1Day": "1 day"}

    def get_historical_bars(self, inst, tf_key, start, end) -> pd.DataFrame:
        self._ensure()
        c = self._contract(inst)
        if c is None:
            return pd.DataFrame()
        bar_size = self._BAR_SIZE.get(tf_key, "1 day")
        cached = self._bar_cache.get(inst.name)
        if cached is None:
            cached = self._load_disk(inst.name, bar_size)

        try:
            if cached is None or cached.empty:
                bars = self.ib.reqHistoricalData(
                    c, endDateTime="", durationStr=duration_str(start, end),
                    barSizeSetting=bar_size, whatToShow="TRADES",
                    useRTH=True, formatDate=2)
                df = bars_to_df(bars)
            else:
                # only pull a short recent tail and merge → pacing-friendly
                bars = self.ib.reqHistoricalData(
                    c, endDateTime="", durationStr="10 D",
                    barSizeSetting=bar_size, whatToShow="TRADES",
                    useRTH=True, formatDate=2)
                df = merge_bar_cache(cached, bars_to_df(bars))
        except Exception as e:  # noqa: BLE001
            log.warning("Historical fetch failed for %s (%s); using cache.",
                        inst.name, e)
            df = cached if cached is not None else pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()
        self._bar_cache[inst.name] = df
        self._save_disk(inst.name, bar_size, df)
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        if lo.tzinfo is None:
            lo = lo.tz_localize("UTC")
        if hi.tzinfo is None:
            hi = hi.tz_localize("UTC")
        return df[(df.index >= lo) & (df.index <= hi)]

    def _cache_path(self, name, bar_size):
        safe = name.replace("/", "_") + "_" + bar_size.replace(" ", "")
        return os.path.join(self.cache_dir, safe + ".csv")

    def _load_disk(self, name, bar_size):
        path = self._cache_path(name, bar_size)
        if os.path.exists(path):
            try:
                return pd.read_csv(path, index_col=0, parse_dates=True)
            except Exception:  # noqa: BLE001
                return None
        return None

    def _save_disk(self, name, bar_size, df):
        try:
            df.to_csv(self._cache_path(name, bar_size))
        except Exception as e:  # noqa: BLE001
            log.debug("Could not write bar cache for %s: %s", name, e)

    # ---- latest price ----------------------------------------------------- #
    def latest_price(self, inst):
        self._ensure()
        c = self._contract(inst)
        if c is None:
            return None
        try:
            tickers = self.ib.reqTickers(c)
            if not tickers:
                return None
            t = tickers[0]
            for v in (getattr(t, "last", None), t.marketPrice(),
                      getattr(t, "close", None)):
                if v is not None and v == v and v > 0:   # not None, not NaN, >0
                    return float(v)
        except Exception as e:  # noqa: BLE001
            log.debug("latest_price failed for %s: %s", inst.name, e)
        return None

    # ---- orders ----------------------------------------------------------- #
    def submit_market_order(self, inst, qty: float, side: str) -> bool:
        self._ensure()
        c = self._contract(inst)
        if c is None or qty <= 0:
            return False
        from ib_async import MarketOrder
        order = MarketOrder(side.upper(), qty)   # 'BUY' / 'SELL'
        try:
            self.ib.placeOrder(c, order)
            self.ib.sleep(2)                      # let it route / report status
            return True
        except Exception as e:  # noqa: BLE001
            log.error("Order failed (%s %s %s): %s", side, qty, inst.name, e)
            return False

    def close_position_raw(self, inst) -> bool:
        r = self.get_position_raw(inst)
        if not r:
            return True                            # already flat
        side = "sell" if r["side"] == "long" else "buy"
        return self.submit_market_order(inst, r["qty"], side)
