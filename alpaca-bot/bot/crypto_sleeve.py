"""
bot/crypto_sleeve.py
=====================
Historical research on adding a small trend-filtered Bitcoin sleeve to the existing
risk+trend book. Results are sample-dependent; this module does not describe a
backtest as a proven future edge.

Tax handling is deliberately explicit. As rechecked in August 2026, FURS states
that ordinary virtual-currency gains of an individual are not subject to income tax
when they are not earned in a business activity. A separate 25% crypto-gains bill
proposed in 2025 remains under consideration. ``--mode aftertax`` therefore prints
BOTH a current non-business-individual scenario (0% crypto income tax) and a
hypothetical/proposal 25% scenario; it does not claim the proposal took effect on
2026-01-01 and does not invent a grandfathering rule.
"""

import argparse
import logging
from datetime import datetime, timezone

import pandas as pd

import bot.aftertax as at
import bot.taxes as tx
import bot.trend_exposure as te
from bot.combo import combine_books, compute_books
from bot.lab import fold_bounds, slice_equity
from bot.backtest_pullback import (INITIAL_EQUITY, _parse_date, buy_hold_combined,
                                   compute_metrics, fetch_all)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("crypto_sleeve")

BTC_MA_PERIOD = 200
BTC_BUFFER = 0.01
DEFAULT_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20]


def compute_btc_sleeve(btc_daily, ma_period=BTC_MA_PERIOD,
                       buffer=BTC_BUFFER) -> pd.Series:
    if ma_period <= 1 or buffer < 0:
        raise ValueError("invalid BTC trend parameters")
    return te.strategy_returns(btc_daily, ma_period=ma_period, buffer=buffer,
                               leverage=1.0, borrow_rate=0.0)


def _common_window(blend, btc_sleeve, begin_ts, end_ts):
    common = blend.index.intersection(btc_sleeve.index)
    common = common[(common >= begin_ts) & (common <= end_ts)]
    if not len(common):
        raise ValueError("no overlapping blend/BTC history in requested window")
    return common


def _metric(r):
    eq = INITIAL_EQUITY * (1 + r).cumprod()
    eq.attrs["initial_equity"] = INITIAL_EQUITY
    return compute_metrics([], eq)


def print_score(daily_data, books, btc_daily, begin_ts, end_ts):
    blend = combine_books(books, weights={"rp": 0.5, "te": 0.5})
    btc = compute_btc_sleeve(btc_daily)
    common = _common_window(blend, btc, begin_ts, end_ts)
    blend, btc = blend.loc[common], btc.loc[common]
    bh = buy_hold_combined(daily_data, begin_ts).pct_change().fillna(0.0)
    bh = bh.reindex(common).fillna(0.0)
    corr = blend.corr(btc)

    print("\nCRYPTO SLEEVE — historical comparison")
    print(f"Correlation(blend, trend-filtered BTC) = {corr:.3f} "
          f"(window: {common[0].date()} -> {common[-1].date()})")
    m_blend = _metric(blend)
    m_bh = _metric(bh)
    print(f"buy_and_hold: Sharpe={m_bh['sharpe']:.3f} Return={m_bh['total_return']:.1%} "
          f"MaxDD={m_bh['max_drawdown']:.1%}")
    print(f"risk+trend blend (0% BTC): Sharpe={m_blend['sharpe']:.3f} "
          f"Return={m_blend['total_return']:.1%} MaxDD={m_blend['max_drawdown']:.1%}")
    for w in DEFAULT_WEIGHTS[1:]:
        mixed = (1 - w) * blend + w * btc
        mm = _metric(mixed)
        print(f"blend + {w:.0%} BTC: Sharpe={mm['sharpe']:.3f} "
              f"Return={mm['total_return']:.1%} MaxDD={mm['max_drawdown']:.1%} "
              f"{'above' if mm['sharpe'] > m_blend['sharpe'] else 'not above'} blend Sharpe")
    return corr


def print_walk(daily_data, books, btc_daily, start_dt, end_dt, folds,
               btc_weight=0.10):
    """Legacy function name; fixed-parameter chronological consistency only."""
    if folds <= 0 or not 0 <= btc_weight <= 1:
        raise ValueError("invalid folds/btc_weight")
    blend = combine_books(books, weights={"rp": 0.5, "te": 0.5})
    btc = compute_btc_sleeve(btc_daily)
    common = _common_window(blend, btc, start_dt, end_dt)
    blend, btc = blend.loc[common], btc.loc[common]
    mixed = (1 - btc_weight) * blend + btc_weight * btc

    print(f"\nCRYPTO SLEEVE CHRONOLOGICAL CONSISTENCY — legacy walk alias; "
          f"btc_weight={btc_weight:.0%}, {folds} folds")
    bounds = fold_bounds(max(start_dt, common[0]), min(end_dt, common[-1]), folds)
    wins = checks = 0
    for k, (fs, fe) in enumerate(bounds):
        final = k == len(bounds) - 1
        pure_eq = slice_equity(blend, fs, fe, end_inclusive=final)
        mix_eq = slice_equity(mixed, fs, fe, end_inclusive=final)
        if not len(pure_eq) or not len(mix_eq):
            continue
        mp, mm = compute_metrics([], pure_eq), compute_metrics([], mix_eq)
        beat = mm["sharpe"] > mp["sharpe"]
        wins += int(beat); checks += 1
        print(f"{k+1}: {fs.date()}->{fe.date()} blend={mp['sharpe']:.2f} "
              f"+BTC={mm['sharpe']:.2f} {'above' if beat else 'not above'}")
    print(f"SUMMARY: +{btc_weight:.0%} BTC above pure blend Sharpe in {wins}/{checks} folds.")
    return wins, checks


def _securities_base_after_tax(daily_data, books, common, weight):
    if weight <= 0:
        return pd.Series(0.0, index=common)
    rp = books["rp"].reindex(common).fillna(0.0)
    rp_eq = tx.after_tax_active(rp, tx.SLOVENIA_TAX_RATE,
                                initial=INITIAL_EQUITY * weight * 0.5)
    te_eq = at._te_after_tax_equity(
        daily_data, common[0], common[-1], currency="usd",
        initial=INITIAL_EQUITY * weight * 0.5)
    return rp_eq.reindex(common).ffill().add(te_eq.reindex(common).ffill(), fill_value=0.0)


def _crypto_after_tax(btc_returns, weight, tax_rate):
    if weight <= 0:
        return pd.Series(0.0, index=btc_returns.index)
    return tx.after_tax_active(btc_returns, tax_rate,
                               initial=INITIAL_EQUITY * weight)


def print_aftertax(daily_data, books, btc_daily, begin_ts, end_ts,
                   current_rate=tx.CRYPTO_TAX_RATE,
                   proposal_rate=tx.CRYPTO_PROPOSED_TAX_RATE):
    """Compare explicit current-law and hypothetical proposal crypto scenarios."""
    blend = combine_books(books, weights={"rp": 0.5, "te": 0.5})
    btc = compute_btc_sleeve(btc_daily)
    common = _common_window(blend, btc, begin_ts, end_ts)
    btc = btc.loc[common]

    print("\nCRYPTO SLEEVE TAX SCENARIOS")
    print("Current FURS non-business-individual scenario: ordinary virtual-currency "
          f"gains income-tax rate modeled at {current_rate:.0%}.")
    print("Separate 25% crypto bill: PROPOSAL/HYPOTHETICAL scenario only; not treated "
          "as enacted law. Business-activity treatment can differ.")
    print(f"{'Book':24}{'Current Shp':>14}{'Proposal25 Shp':>16}{'Current Ret%':>14}")

    for w in DEFAULT_WEIGHTS:
        base = _securities_base_after_tax(daily_data, books, common, 1 - w)
        crypto_current = _crypto_after_tax(btc, w, current_rate)
        crypto_proposal = _crypto_after_tax(btc, w, proposal_rate)
        cur_eq = base.add(crypto_current.reindex(common).ffill(), fill_value=0.0)
        prop_eq = base.add(crypto_proposal.reindex(common).ffill(), fill_value=0.0)
        cur_eq.attrs["initial_equity"] = INITIAL_EQUITY
        prop_eq.attrs["initial_equity"] = INITIAL_EQUITY
        mc, mp = compute_metrics([], cur_eq), compute_metrics([], prop_eq)
        label = "blend (0% BTC)" if w == 0 else f"blend + {w:.0%} BTC"
        print(f"{label:24}{mc['sharpe']:>14.3f}{mp['sharpe']:>16.3f}"
              f"{mc['total_return']*100:>14.1f}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Trend-filtered BTC sleeve historical research.")
    ap.add_argument("--mode", choices=["score", "consistency", "walk", "aftertax"],
                    default="score")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--btc-weight", type=float, default=0.10)
    ap.add_argument("--btc-symbol", default="BTC-USD")
    ap.add_argument("--rp-strategy", choices=["min_var", "inverse_vol", "erc"], default="min_var")
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "GLD", "TLT"])
    ap.add_argument("--months", type=int, default=240)
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca")
    args = ap.parse_args()

    if args.folds <= 0 or args.months <= 0 or not 0 <= args.btc_weight <= 1:
        ap.error("invalid folds/months/btc-weight")
    end_dt = _parse_date(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc))
    start_dt = (_parse_date(args.start) if args.start
                else end_dt - pd.Timedelta(days=args.months * 31))
    if start_dt >= end_dt:
        ap.error("start must precede end")
    daily_data, _, _ = fetch_all(args.symbols, "none", start_dt, end_dt,
                                 source=args.data_source)
    if len(daily_data) != len(args.symbols):
        log.error("Missing requested symbol data; refusing a silently changed universe.")
        return 1
    btc_data, _, _ = fetch_all([args.btc_symbol], "none", start_dt, end_dt, source="yahoo")
    if args.btc_symbol not in btc_data:
        log.error("Could not fetch %s.", args.btc_symbol); return 1
    books = compute_books(daily_data, rp_strategy=args.rp_strategy)
    btc_daily = btc_data[args.btc_symbol]
    if args.mode in {"consistency", "walk"}:
        print_walk(daily_data, books, btc_daily, start_dt, end_dt, args.folds,
                   args.btc_weight)
    elif args.mode == "aftertax":
        print_aftertax(daily_data, books, btc_daily, start_dt, end_dt)
    else:
        print_score(daily_data, books, btc_daily, start_dt, end_dt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
