"""
bot/portfolio_ibkr.py
=====================
Interactive Brokers adapter for the live pullback trader.

The adapter fails closed on ambiguous account currency, contract, market-hours,
or stale-data states. Market submissions return broker order identifiers and the
shared live trader confirms broker position changes before mutating strategy
state. Equity positions use real GTC stop orders at IBKR, so protection survives
bot/Gateway process interruptions after IBKR has accepted the stop. Cancellation
success means IBKR has reported a cancellation state, not merely that cancelOrder
was called successfully. Exact-order fill recovery aggregates every matching
execution into a volume-weighted average price. Persisted pending orders can be
reconciled after an API-session restart through completed-order/execution requests.
"""

import logging
import os
from datetime import datetime, timezone

import pandas as pd

log = logging.getLogger("portfolio_ibkr")
_CANCEL_CONFIRM_POLLS = 12
_CANCEL_POLL_SECONDS = 0.25
_IBKR_CANCEL_CONFIRMED = {"cancelled", "apicancelled"}
_IBKR_TERMINAL = {"filled", *_IBKR_CANCEL_CONFIRMED}


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
    def g(r, k):
        return getattr(r, k, None) if not isinstance(r, dict) else r.get(k)
    for r in rows:
        if g(r, "tag") != "NetLiquidation":
            continue
        if (g(r, "currency") or "").upper() != base_currency.upper():
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
        ts = pd.Timestamp(getattr(b, "date", None))
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        rows.append((ts, float(b.open), float(b.high), float(b.low),
                     float(b.close), float(getattr(b, "volume", 0) or 0)))
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).set_index("ts").sort_index()


def merge_bar_cache(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        return new.sort_index()
    if new is None or new.empty:
        return old.sort_index()
    merged = pd.concat([old, new])
    return merged[~merged.index.duplicated(keep="last")].sort_index()


def duration_str(start, end) -> str:
    days = max(1, (pd.Timestamp(end) - pd.Timestamp(start)).days + 5)
    return f"{days} D" if days <= 365 else f"{(days // 365) + 1} Y"


def exchange_now(tz_id: str):
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
    day, hhmm = now_local.strftime("%Y%m%d"), now_local.strftime("%H%M")
    found_day = False
    for token in hours_str.split(";"):
        token = token.strip()
        if not token or ":" not in token or token.split(":", 1)[0] != day:
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
    return {
        "15Min": pd.Timedelta(hours=2), "1Hour": pd.Timedelta(hours=6),
        "4Hour": pd.Timedelta(hours=12), "1Day": pd.Timedelta(days=5),
    }.get(tf_key, pd.Timedelta(days=5))


def cache_is_fresh(df: pd.DataFrame, tf_key: str, now=None) -> bool:
    if df is None or df.empty:
        return False
    last = pd.Timestamp(df.index[-1])
    last = last.tz_localize("UTC") if last.tzinfo is None else last.tz_convert("UTC")
    now_ts = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    now_ts = now_ts.tz_localize("UTC") if now_ts.tzinfo is None else now_ts.tz_convert("UTC")
    return now_ts - last <= _cache_max_age(tf_key)


class IBKRPortfolio:
    def __init__(self, host=None, port=None, client_id=None,
                 base_currency="EUR", cache_dir="ibkr_cache", connect=True):
        try:
            from ib_async import IB
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install IBKR dependencies with requirements-ibkr.txt") from e
        self._IB = IB
        self.ib = IB()
        self.host = host or os.getenv("IBKR_HOST", "127.0.0.1")
        self.port = int(port or os.getenv("IBKR_PORT", "4002"))
        self.client_id = int(client_id or os.getenv("IBKR_CLIENT_ID", "17"))
        self.base_currency = base_currency
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._contracts, self._hours, self._bar_cache = {}, {}, {}
        self._strategy_orders = {}
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
        try:
            qualified = self.ib.qualifyContracts(Stock(**contract_kwargs(spec)))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Could not qualify IBKR contract for {inst.name}: {e}") from e
        if not qualified:
            raise RuntimeError(f"IBKR could not resolve {inst.name}; verify contract metadata")
        self._contracts[inst.name] = qualified[0]
        return qualified[0]

    def get_equity(self) -> float:
        self._ensure()
        val = equity_from_summary(self.ib.accountSummary(), self.base_currency)
        if val is None or val <= 0:
            raise RuntimeError(f"No positive NetLiquidation in required currency {self.base_currency}")
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
                    return False
                d = details[0]
                self._hours[inst.name] = (
                    getattr(d, "liquidHours", "") or getattr(d, "tradingHours", ""),
                    getattr(d, "timeZoneId", "") or "")
            hours, tz_id = self._hours[inst.name]
            verdict = market_open_from_hours(hours, exchange_now(tz_id))
            if verdict is None:
                log.warning("Could not determine trading hours for %s; refusing.", inst.name)
                return False
            return verdict
        except Exception as e:  # noqa: BLE001
            log.warning("Trading-hours check failed for %s (%s); refusing.", inst.name, e)
            return False

    _BAR_SIZE = {"15Min": "15 mins", "1Hour": "1 hour", "4Hour": "4 hours", "1Day": "1 day"}

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
                bars = self.ib.reqHistoricalData(c, endDateTime="", durationStr=duration_str(start, end),
                    barSizeSetting=bar_size, whatToShow="TRADES", useRTH=True, formatDate=2)
                df = bars_to_df(bars)
            else:
                bars = self.ib.reqHistoricalData(c, endDateTime="", durationStr="10 D",
                    barSizeSetting=bar_size, whatToShow="TRADES", useRTH=True, formatDate=2)
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
        lo = lo.tz_localize("UTC") if lo.tzinfo is None else lo.tz_convert("UTC")
        hi = hi.tz_localize("UTC") if hi.tzinfo is None else hi.tz_convert("UTC")
        return df[(df.index >= lo) & (df.index <= hi)]

    def _cache_path(self, name, bar_size):
        return os.path.join(self.cache_dir,
            name.replace("/", "_") + "_" + bar_size.replace(" ", "") + ".csv")

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
            for v in (getattr(t, "last", None), t.marketPrice()):
                if v is not None and v == v and v > 0:
                    return float(v)
        except Exception as e:  # noqa: BLE001
            log.debug("latest_price failed for %s: %s", inst.name, e)
        return None

    @staticmethod
    def _order_id(trade, order):
        oid = getattr(getattr(trade, "order", None), "orderId", None)
        if oid is None:
            oid = getattr(order, "orderId", None)
        return str(oid) if oid is not None else None

    @staticmethod
    def _normalized_trade_status(trade):
        order = getattr(trade, "order", None)
        st = getattr(trade, "orderStatus", None)
        status = str(getattr(st, "status", "") or "").lower()
        filled = float(getattr(st, "filled", 0) or 0)
        remaining = float(getattr(st, "remaining", 0) or 0)
        return str(getattr(order, "orderId", "")), {
            "status": status,
            "filled_qty": filled,
            "qty": filled + remaining,
            "terminal": status in _IBKR_TERMINAL,
        }

    def submit_market_order(self, inst, qty: float, side: str):
        self._ensure()
        if qty <= 0:
            return None
        from ib_async import MarketOrder
        c, order = self._contract(inst), MarketOrder(side.upper(), qty)
        try:
            trade = self.ib.placeOrder(c, order)
            self.ib.sleep(0.25)
            oid = self._order_id(trade, order)
            if oid:
                self._strategy_orders[oid] = order
            return oid
        except Exception as e:  # noqa: BLE001
            log.error("Order failed (%s %s %s): %s", side, qty, inst.name, e)
            return None

    def order_status(self, order_id):
        """Normalized order state, including completed-order recovery after restart."""
        self._ensure()
        wanted = str(order_id)
        try:
            for trade in self.ib.trades():
                oid, status = self._normalized_trade_status(trade)
                if oid == wanted:
                    return status
            # trades() is documented as session-scoped. Persisted pending state can
            # outlive this API session, so query completed API orders before giving up.
            for trade in self.ib.reqCompletedOrders(apiOnly=True):
                oid, status = self._normalized_trade_status(trade)
                if oid == wanted:
                    return status
        except Exception as e:  # noqa: BLE001
            log.warning("Could not read IBKR order status %s: %s", order_id, e)
        return None

    def close_position_raw(self, inst):
        r = self.get_position_raw(inst)
        if not r:
            return "already-flat"
        return self.submit_market_order(inst, r["qty"], "sell" if r["side"] == "long" else "buy")

    def submit_stop_order(self, inst, qty: float, stop_price: float):
        """Place a real GTC SELL stop for a long Strategy-4 position."""
        self._ensure()
        if qty <= 0 or stop_price <= 0:
            return None
        from ib_async import StopOrder
        c = self._contract(inst)
        order = StopOrder("SELL", qty, round(float(stop_price), 2))
        order.tif = "GTC"
        try:
            trade = self.ib.placeOrder(c, order)
            self.ib.sleep(0.25)
            oid = self._order_id(trade, order)
            if not oid:
                log.error("IBKR accepted stop for %s but returned no order id.", inst.name)
                return None
            self._strategy_orders[oid] = order
            return oid
        except Exception as e:  # noqa: BLE001
            log.error("Stop order failed for %s: %s", inst.name, e)
            return None

    def cancel_order(self, order_id) -> bool:
        """Request cancellation and return True only after IBKR confirms it."""
        if not order_id:
            return True
        self._ensure()
        wanted = str(order_id)
        current = self.order_status(wanted)
        if current:
            if current["status"] in _IBKR_CANCEL_CONFIRMED:
                self._strategy_orders.pop(wanted, None)
                return True
            if current["status"] == "filled":
                self._strategy_orders.pop(wanted, None)
                return False

        order = self._strategy_orders.get(wanted)
        if order is None:
            for trade in self.ib.openTrades():
                candidate = getattr(trade, "order", None)
                if candidate is not None and str(getattr(candidate, "orderId", "")) == wanted:
                    order = candidate
                    break
        if order is None:
            log.warning("Could not find IBKR order %s to request cancellation.", order_id)
            return False
        try:
            self.ib.cancelOrder(order)
        except Exception as e:  # noqa: BLE001
            latest = self.order_status(wanted)
            if latest and latest["status"] in _IBKR_CANCEL_CONFIRMED:
                self._strategy_orders.pop(wanted, None)
                return True
            log.warning("IBKR cancel request %s failed/unconfirmed: %s", order_id, e)
            return False

        for _ in range(_CANCEL_CONFIRM_POLLS):
            self.ib.sleep(_CANCEL_POLL_SECONDS)
            status = self.order_status(wanted)
            if not status:
                continue
            if status["status"] in _IBKR_CANCEL_CONFIRMED:
                self._strategy_orders.pop(wanted, None)
                return True
            if status["status"] == "filled":
                self._strategy_orders.pop(wanted, None)
                log.warning("IBKR order %s filled before cancellation was confirmed.", order_id)
                return False
        log.warning("IBKR cancel request %s was not terminally confirmed.", order_id)
        return False

    @staticmethod
    def _matching_fills(fills, con_id, wanted_side, order_id, since_ts):
        matches = []
        for fill in fills:
            ex, fc = getattr(fill, "execution", None), getattr(fill, "contract", None)
            if ex is None or getattr(fc, "conId", None) != con_id:
                continue
            ex_side = str(getattr(ex, "side", "")).lower()
            normalized = "buy" if ex_side.startswith("b") else "sell"
            if normalized != wanted_side:
                continue
            ex_oid = getattr(ex, "orderId", None)
            if order_id not in (None, "already-flat") and str(ex_oid) != str(order_id):
                continue
            ex_time = pd.Timestamp(getattr(ex, "time", datetime.now(timezone.utc)))
            ex_time = (ex_time.tz_localize("UTC") if ex_time.tzinfo is None
                       else ex_time.tz_convert("UTC"))
            if since_ts is not None and ex_time < since_ts:
                continue
            px = getattr(ex, "price", None)
            if px is None:
                continue
            try:
                qty = float(getattr(ex, "shares", 0) or 0)
            except (TypeError, ValueError):
                qty = 0.0
            matches.append({"time": ex_time, "price": float(px),
                            "qty": max(0.0, qty), "order_id": ex_oid})
        return matches

    def recent_fill_price(self, inst, side: str, order_id=None, since=None):
        """Return VWAP for an exact order, or for the latest matching order."""
        try:
            self._ensure()
            con_id = getattr(self._contract(inst), "conId", None)
            wanted_side = side.lower()
            since_ts = pd.Timestamp(since) if since is not None else None
            if since_ts is not None:
                since_ts = (since_ts.tz_localize("UTC") if since_ts.tzinfo is None
                            else since_ts.tz_convert("UTC"))
            matches = self._matching_fills(
                self.ib.fills(), con_id, wanted_side, order_id, since_ts)
            # fills() is session-scoped. After a reconnect/restart, explicitly ask
            # IBKR for executions when the current session has no matching fill.
            if not matches:
                matches = self._matching_fills(
                    self.ib.reqExecutions(), con_id, wanted_side, order_id, since_ts)
            if not matches:
                return None

            if order_id not in (None, "already-flat"):
                chosen = matches
            else:
                latest = max(matches, key=lambda m: m["time"])
                if latest["order_id"] is None:
                    chosen = [latest]
                else:
                    chosen = [m for m in matches if str(m["order_id"]) == str(latest["order_id"])]

            weighted_qty = sum(m["qty"] for m in chosen)
            if weighted_qty > 0:
                return sum(m["price"] * m["qty"] for m in chosen) / weighted_qty
            return max(chosen, key=lambda m: m["time"])["price"]
        except Exception as e:  # noqa: BLE001
            log.debug("recent_fill_price failed for %s: %s", inst.name, e)
        return None
