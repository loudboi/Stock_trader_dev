"""Generic external-close fill lookup must start after the latest reconciled change."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.live_pullback as lp
from bot.strategies.trend_pullback import PullbackParams


class QuietNotifier:
    def notify(self, message):
        pass


class BoundaryPortfolio:
    def __init__(self):
        self.position = None
        self.seen_since = None
        self.stops = []

    def get_position_raw(self, inst):
        return dict(self.position) if self.position else None

    def recent_fill_price(self, inst, side, order_id=None, since=None):
        self.seen_since = since
        return None

    def cancel_order(self, order_id):
        return True

    def submit_stop_order(self, inst, qty, stop_price):
        self.stops.append((qty, stop_price))
        return f"stop-{len(self.stops)}"

    def get_equity(self):
        return 100_000.0

    def get_historical_bars(self, inst, tf, start, end):
        return pd.DataFrame()


def _trader(tmp_path, pf):
    return lp.PullbackLiveTrader(
        pf, ["SPY"], PullbackParams(), state_file=str(tmp_path / "state.json"),
        notifier=QuietNotifier())


def _position():
    return {
        "qty": 60.0, "avg_entry": 100.0, "stop_dist": 0.05,
        "tranches": 1, "last_add_price": 100.0, "last_price": 100.0,
        "stop_order_id": None, "stop_level": None, "stop_qty": 0.0,
        "entry_time": "2026-08-01T12:00:00+00:00", "entry_orders": [],
    }


def test_external_position_adjustment_resets_fill_search_boundary(tmp_path):
    pf = BoundaryPortfolio()
    t = _trader(tmp_path, pf)
    pos = _position()
    broker = {"side": "long", "qty": 50.0, "avg_entry": 100.0}
    t._apply_external_position_change("SPY", t.instruments["SPY"], pos, broker)
    assert pos["fill_search_since"] == pos["external_adjustment_at"]


def test_finalize_external_close_uses_latest_fill_search_boundary(tmp_path, monkeypatch):
    pf = BoundaryPortfolio()
    t = _trader(tmp_path, pf)
    pos = _position()
    pos["fill_search_since"] = "2026-08-10T12:34:56+00:00"
    t.state["positions"]["SPY"] = pos
    monkeypatch.setattr(lp, "PULLBACK_TRADES_CSV", str(tmp_path / "trades.csv"))
    t._finalize_external_close("SPY", t.instruments["SPY"], pos)
    assert pf.seen_since == "2026-08-10T12:34:56+00:00"
