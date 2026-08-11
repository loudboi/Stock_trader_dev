"""State files should fail fast on incompatible or malformed schemas."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.live_pullback as lp
from bot.strategies.trend_pullback import PullbackParams


class QuietNotifier:
    def notify(self, message):
        pass


class DummyPortfolio:
    pass


def _load(tmp_path, payload):
    path = tmp_path / "state.json"
    path.write_text(json.dumps(payload))
    return lp.PullbackLiveTrader(
        DummyPortfolio(), ["SPY"], PullbackParams(),
        state_file=str(path), notifier=QuietNotifier())


def test_future_state_version_is_rejected(tmp_path):
    payload = lp.PullbackLiveTrader._default_state()
    payload["version"] = 99
    with pytest.raises(RuntimeError, match="newer than supported"):
        _load(tmp_path, payload)


def test_malformed_state_bucket_is_rejected(tmp_path):
    payload = lp.PullbackLiveTrader._default_state()
    payload["positions"] = []
    with pytest.raises(RuntimeError, match="positions.*object"):
        _load(tmp_path, payload)


def test_versionless_legacy_state_is_upgraded_with_defaults(tmp_path):
    payload = {
        "positions": {},
        "intents": {},
        "last_daily": {"SPY": "legacy"},
        "daily": {"date": None, "realized": 0.0},
    }
    trader = _load(tmp_path, payload)
    assert trader.state["version"] == 3
    assert trader.state["pending"] == {}
    assert trader.state["runtime"] == {"symbols": ["SPY"]}
    assert trader.state["last_daily"]["SPY"] == "legacy"

def test_older_managed_state_version_is_rejected(tmp_path):
    payload = lp.PullbackLiveTrader._default_state()
    payload["version"] = 2
    payload["positions"]["SPY"] = {"qty": 1.0}
    with pytest.raises(RuntimeError, match="explicit migration"):
        _load(tmp_path, payload)


def test_malformed_pending_record_is_rejected(tmp_path):
    payload = lp.PullbackLiveTrader._default_state()
    payload["pending"]["SPY"] = {"type": "buy", "order_id": "x"}
    with pytest.raises(RuntimeError, match="missing required field"):
        _load(tmp_path, payload)
