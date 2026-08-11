"""
bot/overnight.py
================
Historical decomposition of close-to-open versus open-to-close returns. Every
actual holding interval is charged entry+exit costs; the first overnight row has no
previous close and therefore no trade or transaction cost. Multi-asset books use a
fixed capital split, so a closed/missing market contributes 0 return rather than
having its capital silently reallocated.
"""

import argparse
import logging
from datetime import datetime, timezone

import pandas as pd

from bot.backtest_pullback import (compute_metrics, buy_hold_combined, fetch_all,
                                   plot_equity, _parse_date, INITIAL_EQUITY, SLIPPAGE)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("overnight")
RESULTS_PNG = "overnight_results.png"
# Kept as a public module constant because tests/research notebooks historically
# referenced it. It is the cost of an ACTUAL daily round trip, not a blanket
# charge applied to rows where no holding interval exists.
_DAILY_ROUNDTRIP_COST = 2 * SLIPPAGE


def overnight_return(daily: pd.DataFrame, cost_per_side: float = SLIPPAGE) -> pd.Series:
    if cost_per_side < 0:
        raise ValueError("cost_per_side cannot be negative")
    prev_close = daily["close"].shift(1)
    raw = daily["open"] / prev_close - 1.0
    out = raw.fillna(0.0)
    traded = prev_close.notna() & daily["open"].notna()
    return out - traded.astype(float) * (2 * cost_per_side)


def intraday_return(daily: pd.DataFrame, cost_per_side: float = SLIPPAGE) -> pd.Series:
    if cost_per_side < 0:
        raise ValueError("cost_per_side cannot be negative")
    valid = daily["open"].notna() & daily["close"].notna() & (daily["open"] > 0)
    raw = (daily["close"] / daily["open"] - 1.0).where(valid, 0.0)
    return raw - valid.astype(float) * (2 * cost_per_side)


def portfolio_book(daily_data: dict, kind: str, cost_per_side: float = SLIPPAGE) -> pd.Series:
    if kind not in {"overnight", "intraday"}:
        raise ValueError("kind must be overnight or intraday")
    if not daily_data:
        return pd.Series(dtype=float)
    fn = overnight_return if kind == "overnight" else intraday_return
    per_symbol = {name: fn(daily, cost_per_side) for name, daily in daily_data.items()}
    # Fixed equal capital split: a non-trading/missing market contributes zero for
    # the day; its allocation is not handed to the other instruments.
    aligned = pd.concat(per_symbol, axis=1).fillna(0.0)
    return aligned.sum(axis=1) / len(per_symbol)


def run(daily_data, begin_ts, end_ts=None, cost_per_side=SLIPPAGE):
    overnight = portfolio_book(daily_data, "overnight", cost_per_side)
    intraday = portfolio_book(daily_data, "intraday", cost_per_side)
    if end_ts is not None:
        overnight = overnight[overnight.index <= end_ts]
        intraday = intraday[intraday.index <= end_ts]
    bh = buy_hold_combined(daily_data, begin_ts)

    def m(r):
        r = r[r.index >= begin_ts]
        eq = INITIAL_EQUITY * (1 + r).cumprod(); eq.attrs["initial_equity"] = INITIAL_EQUITY
        return eq, compute_metrics([], eq)

    eq_on, m_on = m(overnight); eq_id, m_id = m(intraday); m_bh = compute_metrics([], bh)
    print(f"\nOVERNIGHT vs INTRADAY (historical; {cost_per_side:.3%}/side)")
    print(f"overnight: return={m_on['total_return']:.1%} MaxDD={m_on['max_drawdown']:.1%} Sharpe={m_on['sharpe']:.2f}")
    print(f"intraday:  return={m_id['total_return']:.1%} MaxDD={m_id['max_drawdown']:.1%} Sharpe={m_id['sharpe']:.2f}")
    print(f"B&H:       return={m_bh['total_return']:.1%} MaxDD={m_bh['max_drawdown']:.1%} Sharpe={m_bh['sharpe']:.2f}")
    plot_equity({"overnight_only": eq_on, "intraday_only": eq_id}, eq_on, RESULTS_PNG, bh)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Overnight vs intraday historical decomposition.")
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "GLD", "TLT"])
    ap.add_argument("--cost-per-side", type=float, default=SLIPPAGE)
    ap.add_argument("--months", type=int, default=240)
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca")
    args = ap.parse_args()
    if args.cost_per_side < 0 or args.months <= 0:
        ap.error("cost-per-side cannot be negative and months must be positive")
    end_dt = _parse_date(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc))
    start_dt = _parse_date(args.start) if args.start else end_dt - pd.Timedelta(days=args.months * 31)
    if start_dt >= end_dt:
        ap.error("start must precede end")
    daily_data, _, _ = fetch_all(args.symbols, "none", start_dt, end_dt, source=args.data_source)
    if len(daily_data) != len(args.symbols):
        log.error("Missing data for requested symbols; refusing silently changed universe."); return 1
    return run(daily_data, start_dt, end_dt, args.cost_per_side)


if __name__ == "__main__":
    raise SystemExit(main())
