"""Regression tests for unknown live fill prices and realized-PnL auditability."""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.live_pullback as lp
from bot.strategies.trend_pullback import PullbackParams


class NoopNotifier:
    def __init__(self):
        self.messages = []

    def notify(self, message):
        self.messages.append(message)


class MinimalPortfolio:
    def __init__(self):
        self.fill = None

    def recent_fill_price(self, inst, side, order_id=None, since=None):
        return self.fill

    def cancel_order(self, order_id):
        return True

    def get_equity(self):
        return 100_000.0

    def get_position_raw(self, inst):
        return None

    def order_status(self, order_id):
        return {"status": "filled", "terminal": True, "filled_qty": 10.0, "qty": 10.0}


def _trader(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    trades = tmp_path / "trades.csv"
    daily = tmp_path / "daily.csv"
    monkeypatch.setattr(lp, "PULLBACK_TRADES_CSV", str(trades))
    monkeypatch.setattr(lp, "PULLBACK_DAILY_PNL_CSV", str(daily))
    pf = MinimalPortfolio()
    trader = lp.PullbackLiveTrader(
        pf, ["SPY"], PullbackParams(), state_file=str(state), notifier=NoopNotifier())
    return trader, pf, trades


def _position():
    return {
        "qty": 10.0, "avg_entry": 100.0, "stop_dist": 0.05,
        "tranches": 1, "stop_order_id": None,
        "entry_time": "2026-01-01T00:00:00+00:00", "entry_orders": [],
    }


def test_external_flat_without_fill_logs_unknown_and_does_not_accrue_pnl(tmp_path, monkeypatch):
    trader, _, trades = _trader(tmp_path, monkeypatch)
    pos = _position()
    trader.state["positions"]["SPY"] = pos

    trader._finalize_external_close("SPY", trader.instruments["SPY"], pos)

    with open(trades, newline="") as f:
        row = next(csv.DictReader(f))
    assert row["exit_price"] == ""
    assert row["pnl"] == ""
    assert "fill unavailable" in row["reason"]
    assert trader.state["daily"]["realized"] == 0.0
    assert trader.state["daily"]["date"] is None
    assert "SPY" not in trader.state["positions"]


def test_strategy_close_without_recoverable_fill_does_not_use_requested_quote(tmp_path, monkeypatch):
    trader, _, trades = _trader(tmp_path, monkeypatch)
    pos = _position()
    trader.state["positions"]["SPY"] = pos
    trader.state["pending"]["SPY"] = {
        "type": "close", "order_id": "close-1", "reason": "trend exit",
        "requested_exit_price": 95.0, "position": dict(pos),
        "submitted_at": "2026-08-10T12:00:00+00:00",
    }

    assert trader._confirm_pending("SPY", trader.instruments["SPY"])

    with open(trades, newline="") as f:
        row = next(csv.DictReader(f))
    assert row["exit_price"] == ""
    assert row["pnl"] == ""
    assert row["exit_order_id"] == "close-1"
    assert trader.state["daily"]["realized"] == 0.0
