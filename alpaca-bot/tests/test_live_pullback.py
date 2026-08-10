"""Offline tests for live pullback order, stop, reconciliation, and stall safety."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.strategies.trend_pullback import PullbackParams
import bot.live_pullback as lp


class FakePortfolio:
    def __init__(self, equity=100_000.0, asset_class="equity"):
        self._equity = equity
        self.asset_class = asset_class
        self.position = None
        self.stops = {}
        self.cancelled = []
        self.market_orders = []
        self._next_id = 1
        self.last_sell_fill = None
        self.last_buy_fill = None
        self.price = 100.0
        self.fail_cancel = False
        self.position_error = None

    def get_equity(self):
        return self._equity

    def get_position_raw(self, inst):
        if self.position_error:
            raise self.position_error
        return dict(self.position) if self.position else None

    def is_tradable_now(self, inst):
        return True

    def latest_price(self, inst):
        return self.price

    def get_historical_bars(self, inst, tf, start, end):
        return pd.DataFrame()

    def submit_market_order(self, inst, qty, side):
        oid = f"mkt-{self._next_id}"
        self._next_id += 1
        self.market_orders.append((oid, side, qty))
        if side == "buy":
            self._fill_buy(qty)
            self.last_buy_fill = self.price
        return oid

    def _fill_buy(self, qty):
        if self.position is None:
            self.position = {"side": "long", "qty": float(qty), "avg_entry": self.price}
        else:
            old_q = self.position["qty"]
            old_avg = self.position["avg_entry"]
            new_q = old_q + qty
            self.position["avg_entry"] = (old_avg * old_q + self.price * qty) / new_q
            self.position["qty"] = new_q

    def close_position_raw(self, inst):
        oid = f"mkt-{self._next_id}"
        self._next_id += 1
        self.market_orders.append((oid, "sell", self.position["qty"] if self.position else 0.0))
        self.last_sell_fill = self.price
        self.position = None
        return oid

    def submit_stop_order(self, inst, qty, stop_price):
        oid = f"stop-{self._next_id}"
        self._next_id += 1
        self.stops[oid] = (qty, stop_price)
        return oid

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        if self.fail_cancel:
            return False
        self.stops.pop(order_id, None)
        return True

    def recent_fill_price(self, inst, side, order_id=None, since=None):
        return self.last_buy_fill if side == "buy" else self.last_sell_fill


class DelayedFillPortfolio(FakePortfolio):
    """Accept a buy but leave broker position unchanged until the test fills it."""
    def submit_market_order(self, inst, qty, side):
        oid = f"mkt-{self._next_id}"
        self._next_id += 1
        self.market_orders.append((oid, side, qty))
        self._delayed = (qty, side)
        return oid

    def fill_delayed(self):
        qty, side = self._delayed
        if side == "buy":
            self._fill_buy(qty)
            self.last_buy_fill = self.price


def _trader(pf, symbols=("SPY",), **params):
    return lp.PullbackLiveTrader(
        pf, list(symbols), PullbackParams(**params), state_file="_pb_test_state.json")


def teardown_function(_):
    for f in ("_pb_test_state.json", "_pb_test_state.json.tmp",
              "pullback_trades.csv", "pullback_daily_pnl.csv"):
        if os.path.exists(f):
            os.remove(f)


def test_buy_tranche_sizes_and_places_resting_stop():
    pf = FakePortfolio(); pf.price = 100.0
    t = _trader(pf)
    assert t._buy_tranche("SPY", t.instruments["SPY"], 100.0, 0, 0.05, "test")
    pos = t.state["positions"]["SPY"]
    assert pos["qty"] == 60 and pos["tranches"] == 1
    assert pos["entry_orders"][0]["order_id"].startswith("mkt-")
    qty, stop_price = pf.stops[pos["stop_order_id"]]
    assert qty == 60 and abs(stop_price - 95.0) < 1e-6


def test_accepted_buy_is_not_state_fill_until_broker_confirms():
    pf = DelayedFillPortfolio(); pf.price = 100.0
    t = _trader(pf)
    old_seconds = lp.ORDER_CONFIRM_SECONDS
    lp.ORDER_CONFIRM_SECONDS = 0.0
    try:
        assert not t._buy_tranche("SPY", t.instruments["SPY"], 100.0, 0, 0.05, "test")
    finally:
        lp.ORDER_CONFIRM_SECONDS = old_seconds
    assert "SPY" in t.state["pending"]
    assert "SPY" not in t.state["positions"]
    pf.fill_delayed()
    assert t._confirm_pending("SPY", t.instruments["SPY"])
    assert "SPY" not in t.state["pending"]
    assert t.state["positions"]["SPY"]["qty"] == 60


def test_second_tranche_uses_existing_stop_distance_and_replaces_stop():
    pf = FakePortfolio(); pf.price = 100.0
    t = _trader(pf)
    inst = t.instruments["SPY"]
    t._buy_tranche("SPY", inst, 100.0, 0, 0.05, "t1")
    first_stop = t.state["positions"]["SPY"]["stop_order_id"]
    pf.price = 110.0
    # Pass a different new ATR distance; sizing/protection must keep first-stop distance.
    t._buy_tranche("SPY", inst, 110.0, 1, 0.20, "t2")
    pos = t.state["positions"]["SPY"]
    assert pos["stop_dist"] == 0.05
    assert first_stop in pf.cancelled
    assert pos["stop_order_id"] != first_stop
    assert pos["tranches"] == 2
    qty, stop_price = pf.stops[pos["stop_order_id"]]
    assert abs(qty - pos["qty"]) < 1e-9
    assert abs(stop_price - pos["avg_entry"] * 0.95) < 1e-6


def test_crypto_tranche_tracks_position_when_broker_stop_unavailable():
    pf = FakePortfolio(asset_class="crypto")
    pf.submit_stop_order = lambda inst, qty, sp: None
    t = _trader(pf)
    t._buy_tranche("SPY", t.instruments["SPY"], 100.0, 0, 0.05, "t1")
    pos = t.state["positions"]["SPY"]
    assert pos["stop_order_id"] is None and pos["qty"] == 60


def test_close_cancels_resting_stop_and_waits_for_flat_confirmation():
    pf = FakePortfolio(); t = _trader(pf); inst = t.instruments["SPY"]
    t._buy_tranche("SPY", inst, 100.0, 0, 0.05, "t1")
    stop_id = t.state["positions"]["SPY"]["stop_order_id"]
    pf.price = 94.0
    assert t._close("SPY", inst, 94.0, "volatility stop")
    assert stop_id in pf.cancelled
    assert "SPY" not in t.state["positions"] and pf.position is None


def test_close_is_blocked_when_stop_cancel_not_confirmed():
    pf = FakePortfolio(); t = _trader(pf); inst = t.instruments["SPY"]
    t._buy_tranche("SPY", inst, 100.0, 0, 0.05, "t1")
    pf.fail_cancel = True
    before = len(pf.market_orders)
    assert not t._close("SPY", inst, 94.0, "test")
    assert len(pf.market_orders) == before
    assert "SPY" in t.state["positions"]


def test_finalize_external_close_uses_recovered_fill_price():
    pf = FakePortfolio(); t = _trader(pf)
    pos = {"qty": 60.0, "avg_entry": 100.0, "stop_dist": 0.05,
           "tranches": 1, "stop_order_id": "stop-9",
           "entry_time": "2026-01-01T00:00:00+00:00", "entry_orders": []}
    t.state["positions"]["SPY"] = pos
    pf.stops["stop-9"] = (60.0, 95.0)
    pf.last_sell_fill = 94.5
    t._finalize_external_close("SPY", t.instruments["SPY"], pos)
    assert "SPY" not in t.state["positions"]
    assert "stop-9" in pf.cancelled
    assert os.path.exists("pullback_trades.csv")


def test_finalize_external_close_estimates_when_no_fill_available():
    pf = FakePortfolio(); t = _trader(pf, min_stop=0.05)
    pos = {"qty": 10.0, "avg_entry": 200.0, "stop_dist": 0.05,
           "tranches": 1, "stop_order_id": None,
           "entry_time": "2026-01-01T00:00:00+00:00", "entry_orders": []}
    t.state["positions"]["SPY"] = pos
    t._finalize_external_close("SPY", t.instruments["SPY"], pos)
    assert "SPY" not in t.state["positions"]


def test_broker_read_error_is_not_interpreted_as_flat():
    pf = FakePortfolio(); pf.position_error = RuntimeError("temporary broker outage")
    t = _trader(pf)
    with pytest.raises(RuntimeError, match="temporary broker outage"):
        t._broker_position(t.instruments["SPY"])


def test_untracked_existing_long_requires_explicit_adoption():
    pf = FakePortfolio(); pf.position = {"side": "long", "qty": 10.0, "avg_entry": 100.0}
    t = _trader(pf)
    with pytest.raises(RuntimeError, match="adopt-existing"):
        t.reconcile()


def test_state_symbol_omission_refuses_to_start():
    state = lp.PullbackLiveTrader._default_state()
    state["positions"]["QQQ"] = {"qty": 1.0}
    with open("_pb_test_state.json", "w") as f:
        import json
        json.dump(state, f)
    with pytest.raises(RuntimeError, match="omitted"):
        _trader(FakePortfolio(), symbols=("SPY",))


def test_data_stall_alerts_once_after_threshold():
    pf = FakePortfolio(); sent = []; t = _trader(pf)
    t.notifier.notify = lambda msg: sent.append(msg)
    for _ in range(lp.STALL_ALERT_AFTER - 1):
        t._mark_data("SPY", ok=False)
    assert sent == []
    t._mark_data("SPY", ok=False)
    assert len(sent) == 1 and "DATA STALL" in sent[0]
    t._mark_data("SPY", ok=False)
    assert len(sent) == 1
    t._mark_data("SPY", ok=True)
    assert t._stall["SPY"] == 0 and "SPY" not in t._stall_alerted
