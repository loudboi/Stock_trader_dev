"""
bot/portfolio_ibkr.py
=====================
Interactive Brokers adapter for the live pullback trader.

The adapter deliberately fails closed on ambiguous account currency, contract,
market-hours, or stale-data states. Accepted market orders return an identifier;
the live trader confirms the resulting broker position before changing local
strategy state.
"""

import logging
import os
from datetime import datetime, timezone

import pandas as pd

log = logging.getLogger("portfolio_ibkr")


def contract_kwargs(spec: dict) -> dict:
    return {
        "symbol": spec["symbol"],
        "exchange": spec.get("exchange", "SMART"),
        "currency": spec.get("currency", "EUR"),
        "primaryExchange": spec.get("primary_exchange"),
    }


def map_ib_position(position: float, avg_cost: float):
    if position is None or position == 0:
        return None
    return {
        "side": "long" if position > 0 else "short",
        "qty": abs(float(position)),
        "avg_entry": float(avg_cost),
    }


def equity_from_summary(rows, base_currency: str = "EUR"):
    """Extract NetLiquidation only in the requested base currency.

    Falling back to another currency silently corrupts quantity sizing, so an
    absent exact match is represented as None and the caller refuses to trade.
    """
    def g(r, k):
        return getattr(r, k, None) if not isinstance(r, dict) else r.get(k)

    for r in rows:
        if g(r, "tag") != "NetLiquidation":
            continue
        cur = (g(r, "currency") or "").upper()
        if cur != base_currency.upper():
            continue
        try:
            return float(g(r, "value"))
        except (TypeError, ValueError):
            return None
    return None


def bars_to_df(bars) -> pd.DataFrame:
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
    if old is None or old.empty:
        return new.sort_index()
    if new is None or new.empty:
        return old.sort_index()
    merged = pd.concat([old, new])
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.sort_index()


def duration_str(start, end) -> str:
    days = max(1, (pd.Timestamp(end) - pd.Timestamp(start)).days + 5)
    if days <= 365:
        return f"{days} D"
    return f"{(days // 365) + 1} Y"


def exchange_now(tz_id: str):
    """Return exchange-local now, or None when the timezone cannot be trusted."""
    if not tz_id:
        return None
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(tz_id))
    except Exception as e:  # noqa: BLE001
        log.warning("Could not resolve exchange timezone %r: %s", tz_id, e)
        return None


def market_open_from_hours(hours_str: str, now_local):
    if not hours_str or now_local is None:
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


def _cache_max_age(tf_key: str) -> pd.Timedelta:
    # Daily caches must survive normal weekends/holidays but not an extended feed
    # outage. Intraday caches have tighter bounds.
    return {
        "15Min": pd.Timedelta(hours=2),
        "1Hour": pd.Timedelta(hours=6),
        "4Hour": pd.Timedelta(hours=12),
        "1Day": pd.Timedelta(days=5),
    }.get(tf_key, pd.Timedelta(days=5))


def cache_is_fresh(df: pd.DataFrame, tf_key: str, now=None) -> bool:
    if df is None or df.empty:
        return False
    last = pd.Timestamp(df.index[-1])
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    else:
        last = last.tz_convert("UTC")
    now_ts = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    return now_ts - last <= _cache_max_age(tf_key)


class IBKRPortfolio:
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
        self.port = int(port or os.getenv("IBKR_PORT", "4002"))
        self.client_id = int(client_id or os.getenv("IBKR_CLIENT_ID", "17"))
        self.base_currency = base_currency
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._contracts = {}
        self._hours = {}
        self._bar_cache = {}
        if connect:
            self.connect()

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
            raise RuntimeError(f"Could not qualify IBKR contract for {inst.name}: {e}") from e
        if not qualified:
            raise RuntimeError(
                f"IBKR could not resolve {inst.name}; verify symbol/exchange/currency in ibkr_universe.py.")
        self._contracts[inst.name] = qualified[0]
        return qualified[0]

    def get_equity(self) -> float:
        self._ensure()
        rows = self.ib.accountSummary()
        val = equity_from_summary(rows, self.base_currency)
        if val is None or val <= 0:
            raise RuntimeError(
                f"No positive NetLiquidation row in required base currency {self.base_currency}; "
                "refusing to size orders from an ambiguous account value.")
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
        for p in self.ib.positions():
            if getattr(p.contract, "conId", None) == getattr(c, "conId", object()):
                return map_ib_position(p.position, p.avgCost)
        return None

    def position_side(self, inst) -> str:
        r = self.get_position_raw(inst)
        return r["side"] if r else "flat"

    def is_tradable_now(self, inst) -> bool:
        self._ensure()
        c = self._contract(inst)
        try:
            if inst.name not in self._hours:
                details = self.ib.reqContractDetails(c)
                if not details:
                    log.warning("No contract-hours details for %s; refusing new trading action.", inst.name)
                    return False
                d = details[0]
                self._hours[inst.name] = (
                    getattr(d, "liquidHours", "") or getattr(d, "tradingHours", ""),
                    getattr(d, "timeZoneId", "") or "")
            hours, tz_id = self._hours[inst.name]
            verdict = market_open_from_hours(hours, exchange_now(tz_id))
            if verdict is None:
                log.warning("Could not determine trading hours for %s; refusing new trading action.",
                            inst.name)
                return False
            return verdict
        except Exception as e:  # noqa: BLE001
            log.warning("Trading-hours check failed for %s (%s); refusing.", inst.name, e)
            return False

    _BAR_SIZE = {"15Min": "15 mins", "1Hour": "1 hour",
                 "4Hour": "4 hours", "1Day": "1 day"}

    def get_historical_bars(self, inst, tf_key, start, end) -> pd.DataFrame:
        self._ensure()
        c = self._contract(inst)
        bar_size = self._BAR_SIZE.get(tf_key, "1 day")
        cached = self._bar_cache.get(inst.name)
        if cached is None:
            cached = self._load_disk(inst.name, bar_size)
        fresh_fetch_ok = False
        try:
            if cached is None or cached.empty:
                bars = self.ib.reqHistoricalData(
                    c, endDateTime="", durationStr=duration_str(start, end),
                    barSizeSetting=bar_size, whatToShow="TRADES",
                    useRTH=True, formatDate=2)
                df = bars_to_df(bars)
            else:
                bars = self.ib.reqHistoricalData(
                    c, endDateTime="", durationStr="10 D",
                    barSizeSetting=bar_size, whatToShow="TRADES",
                    useRTH=True, formatDate=2)
                df = merge_bar_cache(cached, bars_to_df(bars))
            fresh_fetch_ok = bool(bars)
        except Exception as e:  # noqa: BLE001
            log.warning("Historical fetch failed for %s (%s); considering cache.", inst.name, e)
            df = cached if cached is not None else pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()
        if not fresh_fetch_ok and not cache_is_fresh(df, tf_key):
            log.error("Cached %s data for %s is too stale to use safely.", tf_key, inst.name)
            return pd.DataFrame()
        self._bar_cache[inst.name] = df
        self._save_disk(inst.name, bar_size, df)
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        if lo.tzinfo is None:
            lo = lo.tz_localize("UTC")
        else:
            lo = lo.tz_convert("UTC")
        if hi.tzinfo is None:
            hi = hi.tz_localize("UTC")
        else:
            hi = hi.tz_convert("UTC")
        return df[(df.index >= lo) & (df.index <= hi)]

    def _cache_path(self, name, bar_size):
        safe = name.replace("/", "_") + "_" + bar_size.replace(" ", "")
        return os.path.join(self.cache_dir, safe + ".csv")

    def _load_disk(self, name, bar_size):
        path = self._cache_path(name, bar_size)
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            idx = pd.DatetimeIndex(df.index)
            df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            return df.sort_index()
        except Exception as e:  # noqa: BLE001
            log.warning("Could not read IBKR bar cache %s: %s", path, e)
            return None

    def _save_disk(self, name, bar_size, df):
        try:
            df.to_csv(self._cache_path(name, bar_size))
        except Exception as e:  # noqa: BLE001
            log.debug("Could not write bar cache for %s: %s", name, e)

    def latest_price(self, inst):
        self._ensure()
        c = self._contract(inst)
        try:
            tickers = self.ib.reqTickers(c)
            if not tickers:
                return None
            t = tickers[0]
            # Do not use previous close as a live-price fallback: that masks a
            # current market-data outage and defeats real-time stop monitoring.
            for v in (getattr(t, "last", None), t.marketPrice()):
                if v is not None and v == v and v > 0:
                    return float(v)
        except Exception as e:  # noqa: BLE001
            log.debug("latest_price failed for %s: %s", inst.name, e)
        return None

    def submit_market_order(self, inst, qty: float, side: str):
        self._ensure()
        c = self._contract(inst)
        if qty <= 0:
            return None
        from ib_async import MarketOrder
        order = MarketOrder(side.upper(), qty)
        try:
            trade = self.ib.placeOrder(c, order)
            self.ib.sleep(0.25)
            oid = getattr(getattr(trade, "order", None), "orderId", None)
            if oid is None:
                oid = getattr(order, "orderId", None)
            return str(oid) if oid is not None else f"ibkr-{int(datetime.now(timezone.utc).timestamp()*1e6)}"
        except Exception as e:  # noqa: BLE001
            log.error("Order failed (%s %s %s): %s", side, qty, inst.name, e)
            return None

    def close_position_raw(self, inst):
        r = self.get_position_raw(inst)
        if not r:
            # Returning a synthetic confirmed token lets the caller immediately
            # reconcile to flat without confusing "already flat" with rejection.
            return "already-flat"
        side = "sell" if r["side"] == "long" else "buy"
        return self.submit_market_order(inst, r["qty"], side)

    # Resting stops are still not implemented on IBKR. The live runner therefore
    # keeps this path explicitly in-process-only and the setup docs call it out.
    def submit_stop_order(self, inst, qty: float, stop_price: float):
        log.debug("%s: broker stop not implemented for IBKR; in-process stop active.", inst.name)
        return None

    def cancel_order(self, order_id) -> bool:
        # No resting strategy stops are created by this adapter, so this is only
        # called with None under normal operation.
        return not bool(order_id)

    def recent_fill_price(self, inst, side: str, order_id=None):
        """Best-effort exact/most-recent IBKR execution price."""
        try:
            self._ensure()
            c = self._contract(inst)
            con_id = getattr(c, "conId", None)
            wanted = side.lower()
            matches = []
            for fill in self.ib.fills():
                ex = getattr(fill, "execution", None)
                fc = getattr(fill, "contract", None)
                if ex is None or getattr(fc, "conId", None) != con_id:
                    continue
                ex_side = str(getattr(ex, "side", "")).lower()
                normalized = "buy" if ex_side.startswith("b") else "sell"
                if normalized != wanted:
                    continue
                if order_id not in (None, "already-flat"):
                    ex_oid = getattr(ex, "orderId", None)
                    if ex_oid is not None and str(ex_oid) != str(order_id):
                        continue
                px = getattr(ex, "price", None)
                if px is not None:
                    matches.append((getattr(ex, "time", datetime.min), float(px)))
            if matches:
                matches.sort(key=lambda x: x[0])
                return matches[-1][1]
        except Exception as e:  # noqa: BLE001
            log.debug("recent_fill_price failed for %s: %s", inst.name, e)
        return None
