"""
bot/combo.py
============
Combine the TWO strategies this project has independently validated as beating
buy-and-hold's Sharpe on their own — risk parity (`bot.lab.inverse_vol`) and the
200-day trend-exposure filter (`bot.trend_exposure`) — into one blended book, at a
fixed capital split (default 50/50, chosen before looking at any result).

Why these two specifically: both are long-only, both already have a standalone
edge (unlike blending in a weak strategy just because it's uncorrelated — see the
FX/commodity trend-book experiment, which failed once its own weak Sharpe was
diluted in). Despite living on the same universe, they're only moderately
correlated (~0.7-0.7 in testing) because they make different bets: risk parity is
always invested, risk-weighted; trend-exposure is binary in/out per asset.

VALIDATED FINDING (see README): the 50/50 blend's full-window Sharpe beat BOTH
pure components in BOTH universes tested. It is NOT a universal win over
buy-and-hold in every walk-forward fold (3/5 and 3/5) — like everything else in
this project, its edge over B&H is regime-dependent — but it robustly improves on
holding either single strategy alone. Research/backtest only.

    python -m bot.combo --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
    python -m bot.combo --mode walk --folds 5 --rp-weight 0.7 --symbols SPY QQQ GLD TLT --data-source yahoo --start 2005-01-01
"""

import argparse
import logging
from datetime import datetime, timezone

import pandas as pd

import bot.trend_exposure as te
from bot.lab import (build_panel, inverse_vol, combine_books, compute_metrics,
                     slice_equity, fold_bounds)
from bot.backtest_pullback import fetch_all, buy_hold_combined, _parse_date

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("combo")

RP_LOOKBACK = 20
TE_MA_PERIOD = 200
TE_BUFFER = 0.01


def compute_books(daily_data: dict) -> dict:
    """Return {'rp': ..., 'te': ...} full-history return series (unsliced)."""
    panel = build_panel(daily_data)
    rp = inverse_vol(panel, lookback=RP_LOOKBACK)
    per_symbol = {name: te.strategy_returns(daily, ma_period=TE_MA_PERIOD,
                                            buffer=TE_BUFFER, leverage=1.0, borrow_rate=0.0)
                 for name, daily in daily_data.items()}
    trend = pd.concat(per_symbol, axis=1).mean(axis=1, skipna=True)
    return {"rp": rp, "te": trend}


def print_score(daily_data, books, rp_weight, begin_ts, end_ts):
    blend = combine_books(books, weights={"rp": rp_weight, "te": 1.0 - rp_weight})
    bh = buy_hold_combined(daily_data, begin_ts).pct_change().fillna(0.0)

    def m(r):
        return compute_metrics([], slice_equity(r, begin_ts, end_ts))
    rows = [("buy_and_hold", m(bh)), ("pure risk_parity", m(books["rp"])),
            ("pure trend_exposure", m(books["te"])),
            (f"blend ({rp_weight:.0%} RP / {1-rp_weight:.0%} TE)", m(blend))]

    print(f"\nRP+TE COMBO  rp_weight={rp_weight:.0%}  (both components independently "
          f"validated; blend tests the diversification benefit)")
    print("=" * 66)
    print(f"{'':26}{'Return%':>10}{'MaxDD%':>10}{'Sharpe':>9}")
    print("-" * 66)
    for name, mm in rows:
        print(f"{name:26}{mm['total_return']*100:>10.1f}{mm['max_drawdown']*100:>10.1f}"
              f"{mm['sharpe']:>9.2f}")
    print("=" * 66)
    blend_m, rp_m, te_m, bh_m = rows[3][1], rows[1][1], rows[2][1], rows[0][1]
    print(f"Blend vs pure RP: {'better' if blend_m['sharpe'] > rp_m['sharpe'] else 'worse'} "
          f"({blend_m['sharpe']:.2f} vs {rp_m['sharpe']:.2f})")
    print(f"Blend vs pure TE: {'better' if blend_m['sharpe'] > te_m['sharpe'] else 'worse'} "
          f"({blend_m['sharpe']:.2f} vs {te_m['sharpe']:.2f})")
    print(f"Blend vs buy&hold: {'beats' if blend_m['sharpe'] > bh_m['sharpe'] else 'trails'} "
          f"on Sharpe ({blend_m['sharpe']:.2f} vs {bh_m['sharpe']:.2f})")


def print_walk(daily_data, books, rp_weight, start_dt, end_dt, folds):
    blend = combine_books(books, weights={"rp": rp_weight, "te": 1.0 - rp_weight})
    bh = buy_hold_combined(daily_data, start_dt).pct_change().fillna(0.0)

    print(f"\nRP+TE COMBO WALK-FORWARD  rp_weight={rp_weight:.0%}  {folds} folds")
    print("=" * 92)
    print(f"{'Fold':>4}  {'Window':>23}  {'B&H':>7}  {'pure RP':>8}  {'pure TE':>8}  "
          f"{'blend':>7}  beats")
    print("-" * 92)
    wins_rp = wins_te = wins_bh = 0
    for k, (fs, fe) in enumerate(fold_bounds(start_dt, end_dt, folds)):
        def m(r):
            return compute_metrics([], slice_equity(r, fs, fe))
        m_bh, m_rp, m_te, m_bl = m(bh), m(books["rp"]), m(books["te"]), m(blend)
        beats = []
        if m_bl["sharpe"] > m_rp["sharpe"]:
            wins_rp += 1
            beats.append("RP")
        if m_bl["sharpe"] > m_te["sharpe"]:
            wins_te += 1
            beats.append("TE")
        if m_bl["sharpe"] > m_bh["sharpe"]:
            wins_bh += 1
            beats.append("BH")
        print(f"{k+1:>4}  {fs.date()}->{fe.date()}  {m_bh['sharpe']:>7.2f}  "
              f"{m_rp['sharpe']:>8.2f}  {m_te['sharpe']:>8.2f}  {m_bl['sharpe']:>7.2f}  "
              f"{','.join(beats) or '-'}")
    print("-" * 92)
    print(f"Blend beat pure RP in {wins_rp}/{folds}, pure TE in {wins_te}/{folds}, "
          f"B&H in {wins_bh}/{folds} folds.")
    print("=" * 92)


def main():
    ap = argparse.ArgumentParser(description="Blend risk parity + trend-exposure (both proven).")
    ap.add_argument("--mode", choices=["score", "walk"], default="score")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rp-weight", type=float, default=0.5,
                    help="Capital fraction in risk parity; remainder goes to "
                         "trend-exposure. Default 50/50, chosen before any result.")
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "GLD", "TLT"])
    ap.add_argument("--months", type=int, default=240)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca")
    args = ap.parse_args()

    end_dt = _parse_date(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc))
    start_dt = (_parse_date(args.start) if args.start
                else end_dt - pd.Timedelta(days=int(args.months * 31)))
    log.info("Combo window: %s -> %s (%s mode, rp_weight=%.0f%%)",
             start_dt.date(), end_dt.date(), args.mode, args.rp_weight * 100)

    daily_data, _, _ = fetch_all(args.symbols, "none", start_dt, end_dt, source=args.data_source)
    if len(daily_data) < 2:
        log.error("Need >= 2 symbols with data; got %d.", len(daily_data))
        return 1

    books = compute_books(daily_data)
    if args.mode == "walk":
        print_walk(daily_data, books, args.rp_weight, start_dt, end_dt, args.folds)
    else:
        print_score(daily_data, books, args.rp_weight, start_dt, end_dt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
