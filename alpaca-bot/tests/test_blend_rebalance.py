import pandas as pd
import pytest

from bot.blend_rebalance import plan_rebalance


def test_rebalance_sells_before_buys_and_never_sells_more_than_current():
    target = pd.Series({"SPY": 0.20, "QQQ": 0.50, "GLD": 0.10})
    prices = {"SPY": 100.0, "QQQ": 100.0, "GLD": 100.0}
    current = {"SPY": 40.0, "QQQ": 20.0, "GLD": 10.0}
    orders = plan_rebalance(target, 10_000.0, prices, current)
    assert [(o.symbol, o.side, o.qty) for o in orders] == [
        ("SPY", "sell", 20.0),
        ("QQQ", "buy", 30.0),
    ]
    for order in orders:
        if order.side == "sell":
            assert order.qty <= order.current_qty


def test_rebalance_floors_target_quantity_instead_of_oversizing():
    target = pd.Series({"SPY": 0.3333})
    orders = plan_rebalance(target, 10_000.0, {"SPY": 123.0}, {"SPY": 0.0})
    assert len(orders) == 1
    assert orders[0].side == "buy"
    assert orders[0].target_qty == 27.0
    assert orders[0].qty == 27.0
    assert orders[0].target_qty * orders[0].price <= 0.3333 * 10_000.0


def test_no_trade_band_is_execution_overlay_not_alpha_reweighting():
    target = pd.Series({"SPY": 0.50, "QQQ": 0.50})
    prices = {"SPY": 100.0, "QQQ": 100.0}
    current = {"SPY": 49.6, "QQQ": 50.0}
    orders = plan_rebalance(
        target, 10_000.0, prices, current, no_trade_weight_band=0.005)
    assert orders == []


def test_cash_reserve_scales_all_risky_targets_proportionally():
    target = pd.Series({"SPY": 0.60, "QQQ": 0.40})
    orders = plan_rebalance(
        target, 10_000.0, {"SPY": 100.0, "QQQ": 100.0},
        {"SPY": 0.0, "QQQ": 0.0}, cash_reserve=0.01)
    by_symbol = {o.symbol: o for o in orders}
    assert by_symbol["SPY"].target_weight == pytest.approx(0.594)
    assert by_symbol["QQQ"].target_weight == pytest.approx(0.396)
    assert sum(o.target_weight for o in orders) == pytest.approx(0.99)


def test_unknown_external_holding_fails_closed_by_default():
    target = pd.Series({"SPY": 0.50})
    with pytest.raises(ValueError, match="outside the blend universe"):
        plan_rebalance(target, 10_000.0, {"SPY": 100.0},
                       {"SPY": 0.0, "AAPL": 1.0})


def test_short_current_position_is_rejected():
    target = pd.Series({"SPY": 0.50})
    with pytest.raises(ValueError, match="short current position"):
        plan_rebalance(target, 10_000.0, {"SPY": 100.0}, {"SPY": -1.0})


def test_invalid_target_gross_or_negative_weight_is_rejected():
    with pytest.raises(ValueError, match="cannot exceed"):
        plan_rebalance(pd.Series({"SPY": 0.8, "QQQ": 0.3}), 10_000.0,
                       {"SPY": 100.0, "QQQ": 100.0}, {})
    with pytest.raises(ValueError, match="non-negative"):
        plan_rebalance(pd.Series({"SPY": -0.1}), 10_000.0,
                       {"SPY": 100.0}, {})
