"""
bot/dynamic_combo.py
====================
Historical experiment that shifts the capital split between the risk-based and
trend-exposure books using a lagged external signal. It is a fixed-rule experiment,
not a validated live strategy.

Signals:
* breadth: fraction of a reference sector universe above its 200-day MA;
* vix_term: inverse VIX3M/VIX ratio, oriented so larger means calmer.

Signal values are forward-filled only from observations already known, then shifted
one trading row before they affect allocation. Chronological-fold mode checks
consistency of the same fixed rules; no parameters are fitted per fold.
"""

import argparse
import logging
from datetime import datetime, timezone

import pandas as pd

from bot import regime as rg
from bot.combo import compute_books
from bot.momentum_rotation import build_panel
from bot.backtest_pullback import (INITIAL_EQUITY, SLIPPAGE, _DAILY_WARMUP_DAYS,
                                   _parse_date, buy_hold_combined, compute_metrics,
                                   fetch_all)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("dynamic_combo")

BREADTH_MA = 200
BREADTH_LOW, BREADTH_HIGH = 0.40, 0.60
VIX_TERM_LOW, VIX_TERM_HIGH = 1.00, 1.10
LOW_RP_WEIGHT, MID_RP_WEIGHT, HIGH_RP_WEIGHT = 0.3, 0.5, 0.7
DEFAULT_REFERENCE_SYMBOLS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB"]


def breadth_signal(reference_panel: pd.DataFrame, ma_period=BREADTH_MA) -> pd.Series:
    if reference_panel.empty or ma_period <= 1:
        raise ValueError("breadth requires a non-empty panel and ma_period > 1")
    ma = reference_panel.rolling(ma_period, min_periods=ma_period).mean()
    valid = ma.notna()
    above = (reference_panel > ma).where(valid)
    # Do not treat an asset without enough history as a negative breadth vote.
    count = valid.sum(axis=1).replace(0, pd.NA)
    return above.sum(axis=1, skipna=True).div(count).astype(float)


def dynamic_rp_weight(signal: pd.Series, target_index, low_thresh, high_thresh,
                      low_rp=LOW_RP_WEIGHT, mid_rp=MID_RP_WEIGHT,
                      high_rp=HIGH_RP_WEIGHT) -> pd.Series:
    if low_thresh >= high_thresh:
        raise ValueError("low_thresh must be below high_thresh")
    if any(not 0 <= x <= 1 for x in (low_rp, mid_rp, high_rp)):
        raise ValueError("capital weights must stay in [0, 1]")
    signal = signal.sort_index()
    # Reindex+ffill is causal; unlike bfill it never imports a future first signal.
    aligned = signal.reindex(signal.index.union(target_index)).sort_index().ffill().reindex(target_index)
    w = pd.Series(mid_rp, index=target_index, dtype=float)
    w[aligned < low_thresh] = high_rp
    w[aligned > high_thresh] = low_rp
    return w.shift(1).fillna(mid_rp)


def dynamic_blend_returns(books: dict, rp_weight: pd.Series) -> pd.Series:
    if set(books) < {"rp", "te"}:
        raise ValueError("books must contain rp and te")
    idx = books["rp"].index.union(books["te"].index).sort_values()
    rp = books["rp"].reindex(idx).fillna(0.0)
    trend = books["te"].reindex(idx).fillna(0.0)
    rp_w = rp_weight.reindex(idx).ffill().fillna(MID_RP_WEIGHT)
    if ((rp_w < 0) | (rp_w > 1)).any():
        raise ValueError("rp_weight must stay in [0, 1]")
    gross = rp * rp_w + trend * (1.0 - rp_w)
    # Moving x of capital from RP to TE means x sold + x bought -> 2x turnover.
    turnover = rp_w.diff().abs().fillna(0.0) * 2.0 * SLIPPAGE
    return gross - turnover


def get_signal(name: str, start_ts, end_ts, reference_symbols=None,
               data_source="yahoo"):
    if start_ts >= end_ts:
        raise ValueError("start must precede end")
    if name == "breadth":
        refs = list(dict.fromkeys(reference_symbols or DEFAULT_REFERENCE_SYMBOLS))
        ref_data, _, _ = fetch_all(
            refs, "none", start_ts - pd.Timedelta(days=_DAILY_WARMUP_DAYS), end_ts,
            source=data_source)
        missing = [s for s in refs if s not in ref_data]
        if missing:
            raise RuntimeError(
                "Breadth reference universe incomplete; missing: " + ", ".join(missing))
        return breadth_signal(build_panel(ref_data)), BREADTH_LOW, BREADTH_HIGH
    if name == "vix_term":
        ratio = rg.fetch_vix_term_ratio(start_ts - pd.Timedelta(days=30), end_ts)
        if ratio is None or ratio.empty:
            raise RuntimeError("VIX term-structure signal is unavailable")
        ratio = ratio.replace(0, pd.NA).dropna()
        return 1.0 / ratio, VIX_TERM_LOW, VIX_TERM_HIGH
    raise ValueError(f"unknown signal: {name}")


def _metrics(r, begin_ts, end_ts, end_inclusive=True):
    mask = r.index >= begin_ts
    mask &= (r.index <= end_ts) if end_inclusive else (r.index < end_ts)
    rr = r[mask]
    eq = INITIAL_EQUITY * (1 + rr).cumprod()
    eq.attrs["initial_equity"] = INITIAL_EQUITY
    return compute_metrics([], eq)


def run(daily_data, books, signal, low_thresh, high_thresh, begin_ts, end_ts):
    panel = build_panel(daily_data)
    rp_weight = dynamic_rp_weight(signal, panel.index, low_thresh, high_thresh)
    dynamic = dynamic_blend_returns(books, rp_weight)
    fixed = 0.5 * books["rp"] + 0.5 * books["te"]
    bh = buy_hold_combined(daily_data, begin_ts).pct_change().fillna(0.0)
    rows = [
        ("buy_and_hold", _metrics(bh, begin_ts, end_ts)),
        ("pure risk-based", _metrics(books["rp"], begin_ts, end_ts)),
        ("pure trend_exposure", _metrics(books["te"], begin_ts, end_ts)),
        ("fixed 50/50 blend", _metrics(fixed, begin_ts, end_ts)),
        ("dynamic blend", _metrics(dynamic, begin_ts, end_ts)),
    ]
    print("\nDYNAMIC RP/TE ALLOCATION — historical fixed-rule experiment")
    print(f"{'Book':24}{'Return%':>10}{'MaxDD%':>10}{'Sharpe':>9}")
    for label, m in rows:
        print(f"{label:24}{m['total_return']*100:>10.1f}{m['max_drawdown']*100:>10.1f}"
              f"{m['sharpe']:>9.2f}")
    return 0


def print_walk(daily_data, books, signal, low_thresh, high_thresh,
               start_dt, end_dt, folds):
    if folds <= 0:
        raise ValueError("folds must be positive")
    from bot.lab import fold_bounds
    panel = build_panel(daily_data)
    rp_weight = dynamic_rp_weight(signal, panel.index, low_thresh, high_thresh)
    dynamic = dynamic_blend_returns(books, rp_weight)
    fixed = 0.5 * books["rp"] + 0.5 * books["te"]
    bh = buy_hold_combined(daily_data, start_dt).pct_change().fillna(0.0)
    bounds = fold_bounds(start_dt, end_dt, folds)
    wins = 0
    print(f"\nDYNAMIC RP/TE CHRONOLOGICAL CONSISTENCY — {folds} non-overlapping folds")
    for k, (fs, fe) in enumerate(bounds):
        final = k == len(bounds) - 1
        m_bh = _metrics(bh, fs, fe, final)
        m_fixed = _metrics(fixed, fs, fe, final)
        m_dyn = _metrics(dynamic, fs, fe, final)
        beat = m_dyn["sharpe"] > m_fixed["sharpe"]
        wins += int(beat)
        print(f"{k+1}: {fs.date()}->{fe.date()} B&H={m_bh['sharpe']:.2f} "
              f"fixed={m_fixed['sharpe']:.2f} dynamic={m_dyn['sharpe']:.2f} "
              f"{'above' if beat else 'not above'} fixed")
    print(f"SUMMARY: dynamic above fixed blend Sharpe in {wins}/{folds} folds.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Dynamic RP/TE allocation historical experiment.")
    ap.add_argument("--mode", choices=["score", "consistency", "walk"], default="score",
                    help="consistency (legacy alias walk) = fixed-rule chronological folds")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--signal", choices=["breadth", "vix_term"], required=True)
    ap.add_argument("--reference-symbols", nargs="+", default=None)
    ap.add_argument("--rp-strategy", choices=["min_var", "inverse_vol", "erc"], default="min_var")
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "GLD", "TLT"])
    ap.add_argument("--months", type=int, default=240)
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca")
    args = ap.parse_args()
    if args.folds <= 0 or args.months <= 0:
        ap.error("folds/months must be positive")
    end_dt = _parse_date(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc))
    start_dt = _parse_date(args.start) if args.start else end_dt - pd.Timedelta(days=args.months * 31)
    if start_dt >= end_dt:
        ap.error("start must precede end")
    daily_data, _, _ = fetch_all(args.symbols, "none", start_dt, end_dt,
                                 source=args.data_source)
    if len(daily_data) != len(args.symbols):
        log.error("Missing requested symbol data; refusing a silently changed universe.")
        return 1
    books = compute_books(daily_data, rp_strategy=args.rp_strategy)
    try:
        signal, low, high = get_signal(args.signal, start_dt, end_dt,
                                       args.reference_symbols, args.data_source)
    except Exception as e:  # noqa: BLE001
        log.error("Signal unavailable/incomplete: %s", e)
        return 1
    if args.mode in {"consistency", "walk"}:
        return print_walk(daily_data, books, signal, low, high,
                          start_dt, end_dt, args.folds)
    return run(daily_data, books, signal, low, high, start_dt, end_dt)


if __name__ == "__main__":
    raise SystemExit(main())
