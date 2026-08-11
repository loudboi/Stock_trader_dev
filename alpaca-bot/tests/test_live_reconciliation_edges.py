"""Live reconciliation edges around closed markets, retries, and fill inference."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.live_pullback as lp
from bot.strategies.trend_pullback import PullbackParams


class QuietNotifier:
    def notify(self, message):
        pass


class EdgePortfolio:
    def __init__(self):
        self.position = None
        self.status = None
        self.cancel_calls = []
        self.cancel_result = False
        self.stops = {}
        self.next_stop = 1
        self.fill_price = None

    def get_position_raw(self, inst):
        return dict(self.position) if self.position else None

    def order_status(self, order_id):
        return dict(self.status) if self.status is not None else None

    def cancel_order(self, order_id):
        self.cancel_calls.append(str(order_id))
        return self.cancel_result

    def submit_stop_order(self, inst, qty, stop_price):
        oid = f"stop-{self.next_stop}"
        self.next_stop += 1
        self.stops[oid] = (qty, stop_price)
        return oid

    def recent_fill_price(self, inst, side, order_id=None, since=None):
        return self.fill_price

    def is_tradable_now(self, inst):
        return False

    def get_historical_bars(self, inst, tf, start, end):
        return pd.DataFrame()


def _trader(tmp_path, pf):
    return lp.PullbackLiveTrader(
        pf, ["SPY"], PullbackParams(), state_file=str(tmp_path / "state.json"),
        notifier=QuietNotifier())


def _position(qty=60.0, avg=100.0):
    return {
        "qty": qty, "avg_entry": avg, "stop_dist": 0.05,
        "tranches": 1, "last_add_price": avg, "last_price": avg,
        "stop_order_id": None, "stop_level": None, "stop_qty": 0.0,
        "entry_time": "2026-08-01T12:00:00+00:00", "entry_orders": [],
    }


def test_failed_cancel_is_retryable_even_if_old_state_says_requested(tmp_path):
    pf = EdgePortfolio()
    pf.position = {"side": "long", "qty": 60.0, "avg_entry": 101.0}
    pf.status = {"status": "new", "filled_qty": 0.0, "qty": 60.0, "terminal": False}
    t = _trader(tmp_path, pf)
    t.state["pending"]["SPY"] = {
        "type": "buy", "order_id": "buy-1", "before_qty": 0.0,
        "before_avg_entry": 0.0, "tranche_index": 0, "stop_dist": 0.05,
        "signal_price": 100.0, "requested_qty": 60.0, "how": "test",
        "submitted_at": "2026-08-10T12:00:00+00:00",
        "cancel_requested": True, "cancel_confirmed": False,
    }
    assert t._confirm_pending("SPY", t.instruments["SPY"]) is False
    assert pf.cancel_calls == ["buy-1"]
    assert t.state["pending"]["SPY"]["cancel_attempts"] == 1


def test_manual_position_change_is_reconciled_even_when_market_closed(tmp_path):
    pf = EdgePortfolio()
    pf.position = {"side": "long", "qty": 75.0, "avg_entry": 102.0}
    t = _trader(tmp_path, pf)
    pos = _position()
    t.state["positions"]["SPY"] = pos

    t._process("SPY")

    managed = t.state["positions"]["SPY"]
    assert managed["qty"] == 75.0
    assert managed["avg_entry"] == 102.0
    assert managed["tranches"] == len(t.params.tranches)
    assert managed["externally_adjusted"] is True
    assert managed["stop_order_id"] is not None


def test_terminal_add_fill_can_infer_fill_price_from_position_average(tmp_path):
    pf = EdgePortfolio()
    pf.position = {"side": "long", "qty": 80.0, "avg_entry": 102.5}
    pf.status = {"status": "filled", "filled_qty": 20.0, "qty": 20.0, "terminal": True}
    t = _trader(tmp_path, pf)
    pos = _position(qty=60.0, avg=100.0)
    t.state["positions"]["SPY"] = pos
    t.state["pending"]["SPY"] = {
        "type": "buy", "order_id": "buy-2", "before_qty": 60.0,
        "before_avg_entry": 100.0, "tranche_index": 1, "stop_dist": 0.05,
        "signal_price": 110.0, "requested_qty": 20.0, "how": "test",
        "submitted_at": "2026-08-10T12:00:00+00:00",
    }

    assert t._confirm_pending("SPY", t.instruments["SPY"]) is True
    order = t.state["positions"]["SPY"]["entry_orders"][0]
    assert abs(order["fill_price"] - 110.0) < 1e-12
    assert order["fill_price_source"] == "inferred_from_position"
    assert abs(t.state["positions"]["SPY"]["last_add_price"] - 110.0) < 1e-12
