"""
bot/combo.py
============
Combine the TWO strategies this project has independently validated as beating
buy-and-hold's Sharpe on their own — a risk-based portfolio (`inverse_vol` risk
parity, or `min_var` minimum-variance — see --rp-strategy) and the 200-day
trend-exposure filter (`bot.trend_exposure`) — into one blended book, at a fixed
capital split (default 50/50, chosen before looking at any result).

Why these two specifically: both are long-only, both already have a standalone
edge (unlike blending in a weak strategy just because it's uncorrelated — see the
FX/commodity trend-book experiment, which failed once its own weak Sharpe was
diluted in). Despite living on the same universe, they're only moderately
correlated (~0.7) because they make different bets: the risk-based leg is always
invested and risk-weighted; trend-exposure is binary in/out per asset.

VALIDATED FINDING (see README): the 50/50 blend's full-window Sharpe beat BOTH
pure components in BOTH universes tested, for EITHER risk-based leg. `min_var`
as the risk leg is marginally better (higher full-window Sharpe, same 5/5 record
vs B&H on universe 1) than `inverse_vol`, so it's the default — but it is NOT a
universal win over buy-and-hold in every walk-forward fold (regime-dependent,
like everything else in this project). Research/backtest only.

    python -m bot.combo --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
    python -m bot.combo --rp-strategy inverse_vol --symbols SPY QQQ GLD TLT --data-source yahoo --start 2005-01-01
    python -m bot.combo --mode walk --folds 5 --rp-weight 0.7 --symbols SPY QQQ GLD TLT --data-source yahoo --start 2005-01-01
    python -m bot.combo --leverage 1.5 --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
"""

import argparse
import logging
from datetime import datetime, timezone

import pandas as pd

import bot.trend_exposure as te
from bot.data import clean_daily_data
from bot.lab import (build_panel, inverse_vol, min_var, erc, combine_books, compute_metrics,
                     slice_equity, fold_bounds)
from bot.backtest_pullback import fetch_all, buy_hold_combined, _parse_date

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("combo")

RP_LOOKBACK = 20
COV_LOOKBACK = 60          # min_var's trailing-covariance window
TE_MA_PERIOD = 200
TE_BUFFER = 0.01
_ANNUAL = 252
_RP_STRATEGIES = {"inverse_vol": inverse_vol, "min_var": min_var, "erc": erc}


def leverage_returns(returns: pd.Series, leverage: float, borrow_rate: float = 0.06) -> pd.Series:
    """Apply a CONSTANT leverage multiplier to an already-computed return series,
    charging annual financing on the borrowed (>1x) portion — the same borrow-cost
    convention used throughout bot/lab.py. Every single-strategy leverage attempt
    in this project failed to clear its financing cost (see README); this tests
    whether starting from the higher-Sharpe RP+TE blend changes that."""
    if leverage <= 1.0:
        return returns * leverage
    financing = (leverage - 1.0) * (borrow_rate / _ANNUAL)
    return returns * leverage - financing


def compute_books(daily_data: dict, rp_strategy="min_var", rp_lookback=RP_LOOKBACK,
                  cov_lookback=COV_LOOKBACK, te_ma_period=TE_MA_PERIOD,
                  te_buffer=TE_BUFFER) -> dict:
    """Return {'rp': ..., 'te': ...} full-history return series (unsliced).

    rp_strategy: 'min_var' (default -- marginally better, validated in the
    README), 'inverse_vol' (the original risk-parity leg), or 'erc' (equal-risk-
    contribution -- looked great on universe 1 early in the project but FAILED
    universe 2 as a standalone strategy; included here to see if that fragility
    also shows up when it's paired with trend-exposure). Other overrides default
    to this module's headline (pre-chosen) parameters; they exist for robustness
    checks (does the diversification benefit survive a different lookback/MA?),
    not for tuning to a specific backtest result."""
    panel = build_panel(daily_data)
    if rp_strategy == "min_var":
        rp = min_var(panel, cov_lookback=cov_lookback)
    elif rp_strategy == "erc":
        rp = erc(panel, cov_lookback=cov_lookback)
    else:
        rp = inverse_vol(panel, lookback=rp_lookback)
    per_symbol = {name: te.strategy_returns(daily, ma_period=te_ma_period,
                                            buffer=te_buffer, leverage=1.0, borrow_rate=0.0)
                 for name, daily in daily_data.items()}
    trend = pd.concat(per_symbol, axis=1).mean(axis=1, skipna=True)
    return {"rp": rp, "te": trend}


def print_score(daily_data, books, rp_weight, begin_ts, end_ts, leverage=1.0, borrow_rate=0.06,
               rp_label="min_var"):
    blend = combine_books(books, weights={"rp": rp_weight, "te": 1.0 - rp_weight})
    levered = leverage_returns(blend, leverage, borrow_rate)
    bh = buy_hold_combined(daily_data, begin_ts).pct_change().fillna(0.0)

    def m(r):
        return compute_metrics([], slice_equity(r, begin_ts, end_ts))
    rows = [("buy_and_hold", m(bh)), (f"pure {rp_label}", m(books["rp"])),
            ("pure trend_exposure", m(books["te"])),
            (f"blend ({rp_weight:.0%} {rp_label} / {1-rp_weight:.0%} TE)", m(blend))]
    if leverage != 1.0:
        rows.append((f"blend @ {leverage:g}x (borrow {borrow_rate:.0%}/yr)", m(levered)))

    print(f"\n{rp_label.upper()}+TE COMBO  rp_weight={rp_weight:.0%}  leverage={leverage:g}x  "
          f"(both components independently validated; blend tests the diversification benefit)")
    print("=" * 66)
    print(f"{'':26}{'Return%':>10}{'MaxDD%':>10}{'Sharpe':>9}")
    print("-" * 66)
    for name, mm in rows:
        print(f"{name:26}{mm['total_return']*100:>10.1f}{mm['max_drawdown']*100:>10.1f}"
              f"{mm['sharpe']:>9.2f}")
    print("=" * 66)
    blend_m, rp_m, te_m, bh_m = rows[3][1], rows[1][1], rows[2][1], rows[0][1]
    print(f"Blend vs pure {rp_label}: {'better' if blend_m['sharpe'] > rp_m['sharpe'] else 'worse'} "
          f"({blend_m['sharpe']:.2f} vs {rp_m['sharpe']:.2f})")
    print(f"Blend vs pure TE: {'better' if blend_m['sharpe'] > te_m['sharpe'] else 'worse'} "
          f"({blend_m['sharpe']:.2f} vs {te_m['sharpe']:.2f})")
    print(f"Blend vs buy&hold: {'beats' if blend_m['sharpe'] > bh_m['sharpe'] else 'trails'} "
          f"on Sharpe ({blend_m['sharpe']:.2f} vs {bh_m['sharpe']:.2f})")
    if leverage != 1.0:
        lev_m = rows[4][1]
        beat_ret = "beats" if lev_m["total_return"] > bh_m["total_return"] else "trails"
        beat_shp = "beats" if lev_m["sharpe"] > bh_m["sharpe"] else "trails"
        print(f"Levered blend vs buy&hold: {beat_ret} on return "
              f"({lev_m['total_return']*100:.0f}% vs {bh_m['total_return']*100:.0f}%), "
              f"{beat_shp} on Sharpe ({lev_m['sharpe']:.2f} vs {bh_m['sharpe']:.2f})")


def print_walk(daily_data, books, rp_weight, start_dt, end_dt, folds, leverage=1.0, borrow_rate=0.06,
              rp_label="min_var"):
    blend = combine_books(books, weights={"rp": rp_weight, "te": 1.0 - rp_weight})
    active = leverage_returns(blend, leverage, borrow_rate) if leverage != 1.0 else blend
    bh = buy_hold_combined(daily_data, start_dt).pct_change().fillna(0.0)

    label = "blend" if leverage == 1.0 else f"blend@{leverage:g}x"
    print(f"\n{rp_label.upper()}+TE COMBO WALK-FORWARD  rp_weight={rp_weight:.0%}  "
          f"leverage={leverage:g}x  {folds} folds")
    print("=" * 92)
    print(f"{'Fold':>4}  {'Window':>23}  {'B&H':>7}  {'pure '+rp_label:>8}  {'pure TE':>8}  "
          f"{label:>10}  beats")
    print("-" * 92)
    wins_rp = wins_te = wins_bh = 0
    for k, (fs, fe) in enumerate(fold_bounds(start_dt, end_dt, folds)):
        def m(r):
            return compute_metrics([], slice_equity(r, fs, fe))
        m_bh, m_rp, m_te, m_bl = m(bh), m(books["rp"]), m(books["te"]), m(active)
        beats = []
        if m_bl["sharpe"] > m_rp["sharpe"]:
            wins_rp += 1
            beats.append(rp_label)
        if m_bl["sharpe"] > m_te["sharpe"]:
            wins_te += 1
            beats.append("TE")
        if m_bl["sharpe"] > m_bh["sharpe"]:
            wins_bh += 1
            beats.append("BH")
        print(f"{k+1:>4}  {fs.date()}->{fe.date()}  {m_bh['sharpe']:>7.2f}  "
              f"{m_rp['sharpe']:>8.2f}  {m_te['sharpe']:>8.2f}  {m_bl['sharpe']:>10.2f}  "
              f"{','.join(beats) or '-'}")
    print("-" * 92)
    print(f"{label} beat pure {rp_label} in {wins_rp}/{folds}, pure TE in {wins_te}/{folds}, "
          f"B&H in {wins_bh}/{folds} folds.")
    print("=" * 92)


def main():
    ap = argparse.ArgumentParser(description="Blend risk parity + trend-exposure (both proven).")
    ap.add_argument("--mode", choices=["score", "walk"], default="score")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rp-strategy", choices=list(_RP_STRATEGIES), default="min_var",
                    help="Risk-based leg to blend with trend-exposure. min_var is "
                         "the default (marginally better full-window Sharpe, same "
                         "validated 5/5-vs-B&H record on universe 1); inverse_vol "
                         "is the original.")
    ap.add_argument("--rp-weight", type=float, default=0.5,
                    help="Capital fraction in the risk-based leg; remainder goes to "
                         "trend-exposure. Default 50/50, chosen before any result.")
    ap.add_argument("--leverage", type=float, default=1.0,
                    help="Constant leverage applied to the blended book (1.0 = "
                         "unlevered). Every single-strategy leverage attempt in "
                         "this project failed to clear its financing cost.")
    ap.add_argument("--borrow-rate", type=float, default=0.06,
                    help="Annual financing cost on the leveraged (>1x) portion.")
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "GLD", "TLT"])
    ap.add_argument("--months", type=int, default=240)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca")
    ap.add_argument("--clean-outliers", action="store_true",
                    help="Patch implausible single-day price moves (bad ticks, or a "
                         "real zero-crossing event like WTI's 2020-04-20 negative "
                         "print) before backtesting. See bot/data.py clean_price_series; "
                         "recommended for FX (=X) / futures (=F) tickers.")
    ap.add_argument("--max-abs-return", type=float, default=0.15,
                    help="Cap used by --clean-outliers (fixed, not a tuning knob).")
    args = ap.parse_args()

    end_dt = _parse_date(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc))
    start_dt = (_parse_date(args.start) if args.start
                else end_dt - pd.Timedelta(days=int(args.months * 31)))
    log.info("Combo window: %s -> %s (%s mode, rp_weight=%.0f%%, leverage=%gx)",
             start_dt.date(), end_dt.date(), args.mode, args.rp_weight * 100, args.leverage)

    daily_data, _, _ = fetch_all(args.symbols, "none", start_dt, end_dt, source=args.data_source)
    if len(daily_data) < 2:
        log.error("Need >= 2 symbols with data; got %d.", len(daily_data))
        return 1
    if args.clean_outliers:
        daily_data = clean_daily_data(daily_data, args.max_abs_return)
        log.info("Cleaned implausible single-day moves (cap %.0f%%).", args.max_abs_return * 100)

    books = compute_books(daily_data, rp_strategy=args.rp_strategy)
    if args.mode == "walk":
        print_walk(daily_data, books, args.rp_weight, start_dt, end_dt, args.folds,
                   args.leverage, args.borrow_rate, rp_label=args.rp_strategy)
    else:
        print_score(daily_data, books, args.rp_weight, start_dt, end_dt,
                   args.leverage, args.borrow_rate, rp_label=args.rp_strategy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
