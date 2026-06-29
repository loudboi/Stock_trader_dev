"""
bot/risk_manager.py
===================
Risk helpers for Strategy 4. The phased trend-pullback strategy sizes each
tranche so a fully-built position, stopped at `stop_distance` below the average
entry, loses ~RISK_PER_TRADE of equity (see bot/live_pullback.py and
bot/backtest_pullback.py). The only shared primitive is quantity rounding.
"""


def round_qty(qty: float, decimals: int) -> float:
    if decimals <= 0:
        return float(int(qty))           # whole shares (floor)
    factor = 10 ** decimals
    return int(qty * factor) / factor    # floor to N decimals (never over-size)
