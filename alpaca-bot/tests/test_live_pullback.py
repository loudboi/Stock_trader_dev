"""
Offline tests for the live runner's order/stop/reconcile machinery
(bot/live_pullback.py), driven by a FakePortfolio — no network, no broker.

Focus: the Point-2 safety logic (resting broker stop placed/replaced/cancelled,
in-process stop, broker-flat finalization) and the data-stall alert, plus tranche
sizing. The signal logic itself is covered by test_trend_pullback.py.

Run:  pytest tests/test_live_pullback.py   (or: python tests/test_live_pullback.py)
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from bot.strategies.trend_pullback import PullbackParams
import bot.live_pullback as lp


# --------------------------------------------------------------------------- #
# Fake broker with the surface the trader uses
# --------------------------------------------------------------------------- #
class FakePortfolio:
    def __init__(self, equity=100_000.0, asset_class="equity"):
        self._equity = equity
        self.asset_class = asset_class
        self.position = None        # {'side','qty','avg_entry'} or None
        self.stops = {}             # order_id -> (qty, stop_price)
        self.cancelled = []
        self.market_orders = []
        self._next_id = 1
        self.last_sell_fill = None
        self.price = 100.0

    # account / position
    def get_equity(self):
        return self._equity

    def get_position_raw(self, inst):
        return dict(self.position) if self.position else None

    def is_tradable_now(self, inst):
        return True

    def latest_price(self, inst):
        return self.price

    def get_historical_bars(self, inst, tf, start, end):
        return pd.DataFrame()       # reconcile/_current_stop_dist tolerate empty

    # orders
    def submit_market_order(self, inst, qty, side):
        self.market_orders.append((side, qty))
        if side == "buy":
            self._fill_buy(qty)
        return True

    def _fill_buy(self, qty):
        if self.position is None:
            self.position = {"side": "long", "qty": float(qty), "avg_entry": self.price}
        else:                       # weighted-average up
            old_q = self.position["qty"]
            old_avg = self.position["avg_entry"]
            new_q = old_q + qty
            self.position["avg_entry"] = (old_avg * old_q + self.price * qty) / new_q
            self.position["qty"] = new_q

    def close_position_raw(self, inst):
        self.last_sell_fill = self.price
        self.position = None
        return True

    def submit_stop_order(self, inst, qty, stop_price):
        oid = f"stop-{self._next_id}"
        self._next_id += 1
        self.stops[oid] = (qty, stop_price)
        return oid

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        self.stops.pop(order_id, None)
        return True

    def recent_fill_price(self, inst, side):
        return self.last_sell_fill


def _trader(pf, symbols=("SPY",), **params):
    t = lp.PullbackLiveTrader(pf, list(symbols), PullbackParams(**params),
                              state_file="_pb_test_state.json")
    return t


def teardown_function(_):
    for f in ("_pb_test_state.json", "_pb_test_state.json.tmp"):
        if os.path.exists(f):
            os.remove(f)


# --------------------------------------------------------------------------- #
# Tranche buys + resting stop
# --------------------------------------------------------------------------- #
def test_buy_tranche_sizes_and_places_resting_stop():
    pf = FakePortfolio()
    pf.price = 100.0
    t = _trader(pf)
    t._buy_tranche("SPY", t.instruments["SPY"], price=100.0,
                   tranche_index=0, stop_dist=0.05, how="test")
    pos = t.state["positions"]["SPY"]
    # 1% of 100k / (100*0.05) = 200 full; 30% -> 60 shares.
    assert pos["qty"] == 60 and pos["tranches"] == 1
    # A resting stop was placed at avg_entry*(1-0.05).
    assert pos["stop_order_id"] in pf.stops
    qty, stop_price = pf.stops[pos["stop_order_id"]]
    assert qty == 60 and abs(stop_price - 100.0 * 0.95) < 1e-6


def test_second_tranche_replaces_the_resting_stop():
    pf = FakePortfolio()
    pf.price = 100.0
    t = _trader(pf)
    inst = t.instruments["SPY"]
    t._buy_tranche("SPY", inst, 100.0, 0, 0.05, "t1")
    first_stop = t.state["positions"]["SPY"]["stop_order_id"]
    pf.price = 110.0
    t._buy_tranche("SPY", inst, 110.0, 1, 0.05, "t2")
    pos = t.state["positions"]["SPY"]
    assert first_stop in pf.cancelled               # old stop cancelled
    assert pos["stop_order_id"] != first_stop       # replaced with a new one
    assert pos["tranches"] == 2
    # New stop covers the full position quantity at the new average entry.
    qty, stop_price = pf.stops[pos["stop_order_id"]]
    assert abs(qty - pos["qty"]) < 1e-9
    assert abs(stop_price - pos["avg_entry"] * 0.95) < 1e-6


def test_crypto_tranche_has_no_resting_stop_but_still_tracks_position():
    pf = FakePortfolio(asset_class="crypto")
    # Crypto path returns None for the resting stop (in-process stop covers it).
    pf.submit_stop_order = lambda inst, qty, sp: None
    pf.price = 100.0
    t = _trader(pf)
    t._buy_tranche("SPY", t.instruments["SPY"], 100.0, 0, 0.05, "t1")
    pos = t.state["positions"]["SPY"]
    assert pos["stop_order_id"] is None and pos["qty"] == 60


# --------------------------------------------------------------------------- #
# In-process close cancels the resting stop
# --------------------------------------------------------------------------- #
def test_close_cancels_resting_stop_and_logs():
    pf = FakePortfolio()
    pf.price = 100.0
    t = _trader(pf)
    inst = t.instruments["SPY"]
    t._buy_tranche("SPY", inst, 100.0, 0, 0.05, "t1")
    stop_id = t.state["positions"]["SPY"]["stop_order_id"]
    pf.price = 94.0
    t._close("SPY", inst, 94.0, "volatility stop")
    assert stop_id in pf.cancelled
    assert "SPY" not in t.state["positions"]
    assert pf.position is None


# --------------------------------------------------------------------------- #
# Broker-flat detection (resting stop fired while we were away)
# --------------------------------------------------------------------------- #
def test_finalize_external_close_uses_recovered_fill_price():
    pf = FakePortfolio()
    t = _trader(pf)
    pos = {"qty": 60.0, "avg_entry": 100.0, "stop_dist": 0.05,
           "tranches": 1, "stop_order_id": "stop-9"}
    t.state["positions"]["SPY"] = pos
    pf.stops["stop-9"] = (60.0, 95.0)
    pf.last_sell_fill = 94.5                         # broker filled the stop at 94.5
    t._finalize_external_close("SPY", t.instruments["SPY"], pos)
    assert "SPY" not in t.state["positions"]
    assert "stop-9" in pf.cancelled
    assert os.path.exists("pullback_trades.csv")    # trade was logged
    os.remove("pullback_trades.csv")


def test_finalize_external_close_estimates_when_no_fill_available():
    pf = FakePortfolio()
    pf.last_sell_fill = None                         # can't recover the fill
    t = _trader(pf, min_stop=0.05)
    pos = {"qty": 10.0, "avg_entry": 200.0, "stop_dist": 0.05,
           "tranches": 1, "stop_order_id": None}
    t.state["positions"]["SPY"] = pos
    t._finalize_external_close("SPY", t.instruments["SPY"], pos)
    # Estimated exit = avg_entry * (1 - stop_dist) = 190.
    assert "SPY" not in t.state["positions"]
    if os.path.exists("pullback_trades.csv"):
        os.remove("pullback_trades.csv")


# --------------------------------------------------------------------------- #
# Data-stall alert
# --------------------------------------------------------------------------- #
def test_data_stall_alerts_once_after_threshold():
    pf = FakePortfolio()
    sent = []
    t = _trader(pf)
    t.notifier.notify = lambda msg: sent.append(msg)
    for _ in range(lp.STALL_ALERT_AFTER - 1):
        t._mark_data("SPY", ok=False)
    assert sent == []                                # not yet
    t._mark_data("SPY", ok=False)                    # crosses threshold
    assert len(sent) == 1 and "DATA STALL" in sent[0]
    t._mark_data("SPY", ok=False)                    # no duplicate spam
    assert len(sent) == 1
    t._mark_data("SPY", ok=True)                     # recovery resets
    assert t._stall["SPY"] == 0 and "SPY" not in t._stall_alerted


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        fn()
        teardown_function(None)
        print(f"{name} OK")
    print(f"\nALL {len(fns)} LIVE-PULLBACK TESTS PASSED")
