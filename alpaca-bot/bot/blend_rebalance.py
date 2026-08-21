"""Pure target-to-order planning for the profitability blend.

No broker client is imported and no order can be submitted from this module.  The
planner exists so target/position arithmetic, rounding and safety invariants can be
tested before any paper-trading adapter is connected.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd


@dataclass(frozen=True)
class RebalanceOrder:
    symbol: str
    side: str
    qty: float
    current_qty: float
    target_qty: float
    price: float
    current_weight: float
    target_weight: float


def _floor_qty(qty: float, decimals: int) -> float:
    if qty < 0 or decimals < 0:
        raise ValueError("qty/decimals must be non-negative")
    factor = 10 ** decimals
    return math.floor((qty + 1e-12) * factor) / factor


def plan_rebalance(target_weights: pd.Series, equity: float, prices: dict[str, float],
                   current_qty: dict[str, float], qty_decimals: dict[str, int] | None = None,
                   no_trade_weight_band: float = 0.0, cash_reserve: float = 0.0,
                   strict_universe: bool = True) -> list[RebalanceOrder]:
    """Convert long-only target weights into deterministic sell-then-buy deltas.

    Parameters are deliberately execution-only.  `no_trade_weight_band` and
    `cash_reserve` are not alpha parameters and must not be optimized on historical
    returns.  A non-zero cash reserve scales all risky targets proportionally.
    """
    if equity <= 0 or not math.isfinite(equity):
        raise ValueError("equity must be finite and positive")
    if not 0 <= no_trade_weight_band < 1:
        raise ValueError("no_trade_weight_band must be in [0, 1)")
    if not 0 <= cash_reserve < 1:
        raise ValueError("cash_reserve must be in [0, 1)")
    if target_weights.index.has_duplicates:
        raise ValueError("target weights must have unique symbols")

    w = target_weights.astype(float).copy()
    if not np_all_finite_nonnegative(w):
        raise ValueError("target weights must be finite and non-negative")
    if float(w.sum()) > 1 + 1e-10:
        raise ValueError("target risky weights cannot exceed 100%")
    if cash_reserve:
        w *= 1.0 - cash_reserve

    universe = set(w.index)
    if strict_universe:
        outside = {s: q for s, q in current_qty.items()
                   if s not in universe and abs(float(q)) > 1e-12}
        if outside:
            raise ValueError(
                "non-zero holdings outside the blend universe require explicit manual policy: "
                + ", ".join(sorted(outside)))

    decimals = qty_decimals or {}
    orders: list[RebalanceOrder] = []
    for symbol, target_weight in w.items():
        price = float(prices.get(symbol, 0.0) or 0.0)
        current = float(current_qty.get(symbol, 0.0) or 0.0)
        if current < -1e-12:
            raise ValueError(f"{symbol}: short current position is unsupported")
        if price <= 0 or not math.isfinite(price):
            if target_weight > 1e-12 or current > 1e-12:
                raise ValueError(f"{symbol}: positive finite price required")
            continue
        d = int(decimals.get(symbol, 0))
        if d < 0:
            raise ValueError(f"{symbol}: qty decimals must be non-negative")

        current_weight = current * price / equity
        if abs(float(target_weight) - current_weight) < no_trade_weight_band:
            continue
        target_notional = float(target_weight) * equity
        target = _floor_qty(target_notional / price, d)
        delta = target - current
        if abs(delta) < 10 ** (-d) * 0.5:
            continue
        if delta > 0:
            qty = _floor_qty(delta, d)
            side = "buy"
        else:
            qty = _floor_qty(-delta, d)
            side = "sell"
        if qty <= 0:
            continue
        if side == "sell" and qty > current + 1e-9:
            raise RuntimeError(f"{symbol}: planner would sell more than current long")
        orders.append(RebalanceOrder(
            symbol=symbol, side=side, qty=qty,
            current_qty=current, target_qty=target, price=price,
            current_weight=current_weight, target_weight=float(target_weight),
        ))

    # Selling first avoids depending on broker margin/buying power to fund a rebalance.
    orders.sort(key=lambda o: (0 if o.side == "sell" else 1, o.symbol))
    return orders


def np_all_finite_nonnegative(values: pd.Series) -> bool:
    """Small dependency-free validation helper kept local to the execution planner."""
    for value in values:
        x = float(value)
        if not math.isfinite(x) or x < -1e-12:
            return False
    return True
