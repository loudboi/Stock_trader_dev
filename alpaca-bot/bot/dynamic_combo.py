"""
bot/dynamic_combo.py
====================
A genuinely different mechanism from every macro-overlay tested so far. Every
prior macro-regime test (bot/regime.py's regime_gated_rp / regime_leverage_rp)
GATED OR LEVERED a single already-good strategy on a macro signal, and that
failed 12/12 across 3 indicators x 2 universes (tactical overlays on top of an
already-diversified strategy just add whipsaw).

This instead uses a real-time, causal (no-lookahead) signal to shift the CAPITAL
SPLIT between the two ALREADY-good ingredients of bot/combo.py's validated blend
(a risk-based portfolio and the trend-exposure filter) — i.e. dynamic ALLOCATION
between two good strategies, not a gate/lever on one. Two signals, thresholds
fixed BEFORE looking at any result:

  breadth   Fraction of a REFERENCE universe (9 SPDR sector ETFs, independent of
            the traded universe, to avoid a self-referential signal) trading
            above its own 200-day MA. < 40% -> lean risk-based (0.7 RP / 0.3 TE,
            narrow/choppy market favors diversification); > 60% -> lean trend
            (0.3 RP / 0.7 TE, broad participation favors trend-following);
            else the already-validated 50/50.
  vix_term  VIX3M/VIX ("calmness"): > 1.10 (steep contango, calm) -> lean trend;
            < 1.0 (backwardation, stress) -> lean risk-based; else 50/50.
            ^VIX3M only exists from 2007-01-03.

IMPORTANT — this is NOT hindsight-based. Both signals are the kind of
information a real trader would have had in real time; nothing here uses
knowledge of specific historical events (no hardcoded "2008" or "2020" dates).

    python -m bot.dynamic_combo --signal breadth --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
    python -m bot.dynamic_combo --signal vix_term --symbols SPY QQQ GLD TLT --start 2007-01-01 --data-source yahoo
    python -m bot.dynamic_combo --mode walk --signal breadth --symbols SPY QQQ GLD TLT --data-source yahoo --start 2005-01-01
"""

import argparse
import logging
from datetime import datetime, timezone

import pandas as pd

from bot import regime as rg
from bot.combo import compute_books
from bot.momentum_rotation import build_panel
from bot.backtest_pullback import (compute_metrics, buy_hold_combined, fetch_all,
                                   _parse_date, INITIAL_EQUITY, SLIPPAGE, _DAILY_WARMUP_DAYS)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("dynamic_combo")

# Fixed, pre-chosen thresholds/weights -- decided before any result.
BREADTH_MA = 200
BREADTH_LOW, BREADTH_HIGH = 0.40, 0.60
VIX_TERM_LOW, VIX_TERM_HIGH = 1.00, 1.10
LOW_RP_WEIGHT, MID_RP_WEIGHT, HIGH_RP_WEIGHT = 0.3, 0.5, 0.7
DEFAULT_REFERENCE_SYMBOLS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB"]


def breadth_signal(reference_panel: pd.DataFrame, ma_period=BREADTH_MA) -> pd.Series:
    """Fraction (0..1) of the reference universe trading above its own MA each
    day -- a classic, causal market-internals gauge, computed from a universe
    INDEPENDENT of what's actually being traded (no self-reference)."""
    ma = reference_panel.rolling(ma_period, min_periods=ma_period).mean()
    above = (reference_panel > ma).astype(float)
    return above.mean(axis=1)


def dynamic_rp_weight(signal: pd.Series, target_index, low_thresh, high_thresh,
                      low_rp=LOW_RP_WEIGHT, mid_rp=MID_RP_WEIGHT, high_rp=HIGH_RP_WEIGHT) -> pd.Series:
    """Map a signal (HIGH = trend-favorable/calm, LOW = defensive/stressed) to an
    RP capital-weight band, aligned to target_index, then SHIFTED by one day
    (decide from the prior close, apply today) -- no lookahead."""
    aligned = signal.reindex(target_index, method="ffill")
    w = pd.Series(mid_rp, index=target_index)
    w[aligned < low_thresh] = high_rp     # weak/stressed signal -> lean defensive (more RP)
    w[aligned > high_thresh] = low_rp     # strong/calm signal -> lean trend (more TE)
    return w.shift(1).fillna(mid_rp)


def dynamic_blend_returns(books: dict, rp_weight: pd.Series) -> pd.Series:
    """Blend the RP/TE books with a TIME-VARYING (already-shifted) RP weight.
    Turnover cost charged on the weight's own day-to-day change, same slippage
    convention used everywhere else."""
    rp_w = rp_weight.reindex(books["rp"].index).fillna(0.5)
    te_w = 1.0 - rp_w
    gross = books["rp"] * rp_w + books["te"] * te_w
    turnover = rp_w.diff().abs().fillna(0.0) * 2 * SLIPPAGE
    return gross - turnover


def get_signal(name: str, start_ts, end_ts, reference_symbols=None, data_source="yahoo"):
    """Fetch and orient a signal so HIGH = trend-favorable/calm (the convention
    dynamic_rp_weight expects)."""
    if name == "breadth":
        ref_data, _, _ = fetch_all(reference_symbols or DEFAULT_REFERENCE_SYMBOLS, "none",
                                   start_ts - pd.Timedelta(days=_DAILY_WARMUP_DAYS), end_ts,
                                   source=data_source)
        ref_panel = build_panel(ref_data)
        return breadth_signal(ref_panel), BREADTH_LOW, BREADTH_HIGH
    if name == "vix_term":
        ratio = rg.fetch_vix_term_ratio(start_ts - pd.Timedelta(days=30), end_ts)
        calmness = 1.0 / ratio          # invert: >1 = contango/calm, <1 = backwardation/stress
        return calmness, VIX_TERM_LOW, VIX_TERM_HIGH
    raise ValueError(f"Unknown signal: {name}")


def run(daily_data, books, signal, low_thresh, high_thresh, begin_ts, end_ts):
    panel = build_panel(daily_data)
    rp_weight = dynamic_rp_weight(signal, panel.index, low_thresh, high_thresh)
    dynamic = dynamic_blend_returns(books, rp_weight)
    fixed = books["rp"] * 0.5 + books["te"] * 0.5     # the already-validated static 50/50
    bh = buy_hold_combined(daily_data, begin_ts).pct_change().fillna(0.0)

    def m(r):
        eq = INITIAL_EQUITY * (1 + r[(r.index >= begin_ts) & (r.index <= end_ts)]).cumprod()
        return compute_metrics([], eq)
    m_bh, m_rp, m_te = m(bh), m(books["rp"]), m(books["te"])
    m_fixed, m_dyn = m(fixed), m(dynamic)
    pct_rp_heavy = float((rp_weight == HIGH_RP_WEIGHT).mean())
    pct_te_heavy = float((rp_weight == LOW_RP_WEIGHT).mean())

    print(f"\nDYNAMIC RP/TE ALLOCATION  (real-time signal shifts capital between two "
          f"already-good strategies, vs the fixed 50/50)")
    print(f"Time defensive-tilted (more RP): {pct_rp_heavy:.0%}   "
          f"Time trend-tilted (more TE): {pct_te_heavy:.0%}")
    print("=" * 62)
    print(f"{'Book':24}{'Return%':>10}{'MaxDD%':>10}{'Sharpe':>9}")
    print("-" * 62)
    for name, mm in [("buy_and_hold", m_bh), ("pure risk-based", m_rp), ("pure trend_exposure", m_te),
                     ("fixed 50/50 blend", m_fixed), ("DYNAMIC blend", m_dyn)]:
        print(f"{name:24}{mm['total_return']*100:>10.1f}{mm['max_drawdown']*100:>10.1f}"
              f"{mm['sharpe']:>9.2f}")
    print("=" * 62)
    verdict = "beats" if m_dyn["sharpe"] > m_fixed["sharpe"] else "does not beat"
    print(f"Dynamic allocation {verdict} the fixed 50/50 blend "
          f"({m_dyn['sharpe']:.2f} vs {m_fixed['sharpe']:.2f}).")
    return 0


def print_walk(daily_data, books, signal, low_thresh, high_thresh, start_dt, end_dt, folds):
    panel = build_panel(daily_data)
    rp_weight = dynamic_rp_weight(signal, panel.index, low_thresh, high_thresh)
    dynamic = dynamic_blend_returns(books, rp_weight)
    fixed = books["rp"] * 0.5 + books["te"] * 0.5
    bh = buy_hold_combined(daily_data, start_dt).pct_change().fillna(0.0)

    print(f"\nDYNAMIC RP/TE WALK-FORWARD  {folds} folds")
    print("=" * 88)
    print(f"{'Fold':>4}  {'Window':>23}  {'B&H':>7}  {'fixed 50/50':>12}  {'dynamic':>8}  beats fixed?")
    print("-" * 88)
    from bot.lab import fold_bounds, slice_equity
    wins = 0
    for k, (fs, fe) in enumerate(fold_bounds(start_dt, end_dt, folds)):
        def m(r):
            return compute_metrics([], slice_equity(r, fs, fe))
        m_bh, m_fixed, m_dyn = m(bh), m(fixed), m(dynamic)
        beat = m_dyn["sharpe"] > m_fixed["sharpe"]
        wins += int(beat)
        print(f"{k+1:>4}  {fs.date()}->{fe.date()}  {m_bh['sharpe']:>7.2f}  "
              f"{m_fixed['sharpe']:>12.2f}  {m_dyn['sharpe']:>8.2f}  {'YES' if beat else 'no'}")
    print("-" * 88)
    print(f"Dynamic beat the fixed 50/50 blend in {wins}/{folds} folds.")
    print("=" * 88)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Dynamic RP/TE allocation via a real-time signal.")
    ap.add_argument("--mode", choices=["score", "walk"], default="score")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--signal", choices=["breadth", "vix_term"], required=True)
    ap.add_argument("--reference-symbols", nargs="+", default=None,
                    help="Reference universe for --signal breadth (default: 9 SPDR sectors).")
    ap.add_argument("--rp-strategy", choices=["min_var", "inverse_vol"], default="min_var")
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "GLD", "TLT"])
    ap.add_argument("--months", type=int, default=240)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca")
    args = ap.parse_args()

    end_dt = _parse_date(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc))
    start_dt = (_parse_date(args.start) if args.start
                else end_dt - pd.Timedelta(days=int(args.months * 31)))
    log.info("Dynamic combo window: %s -> %s (%s mode, signal=%s)",
             start_dt.date(), end_dt.date(), args.mode, args.signal)

    daily_data, _, _ = fetch_all(args.symbols, "none", start_dt, end_dt, source=args.data_source)
    if len(daily_data) < 2:
        log.error("Need >= 2 symbols with data; got %d.", len(daily_data))
        return 1
    books = compute_books(daily_data, rp_strategy=args.rp_strategy)

    signal, low_thresh, high_thresh = get_signal(args.signal, start_dt, end_dt,
                                                 args.reference_symbols, args.data_source)
    if args.mode == "walk":
        return print_walk(daily_data, books, signal, low_thresh, high_thresh,
                         start_dt, end_dt, args.folds)
    return run(daily_data, books, signal, low_thresh, high_thresh, start_dt, end_dt)


if __name__ == "__main__":
    raise SystemExit(main())
