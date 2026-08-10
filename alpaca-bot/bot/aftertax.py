"""
bot/aftertax.py
================
After-tax research scenarios for a Slovenian-resident individual.

This module is intentionally conservative about what the data can prove. It is not
a tax return calculator. Securities holding-period rates are modeled from actual
calendar dates; ordinary loss carry-forward is not assumed; and the trend-exposure
sleeve is taxed on its actual exit dates instead of being forced through the coarse
annual-realization proxy used for high-turnover allocation books.

Currency conversion happens before the tax scenario. EUR conversion is exact for
the modeled USD exposure and never backfills future FX observations.
"""

import argparse
import logging
from datetime import datetime, timezone

import pandas as pd

import bot.currency as cur
import bot.taxes as tx
import bot.trend_exposure as te
from bot.combo import compute_books, combine_books, TE_MA_PERIOD, TE_BUFFER
from bot.backtest_pullback import (INITIAL_EQUITY, _parse_date, buy_hold_combined,
                                   compute_metrics, fetch_all)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("aftertax")

DEFAULT_CORE_WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]


def _slice(r, begin_ts, end_ts):
    return r[(r.index >= begin_ts) & (r.index <= end_ts)]


def _currency_fraction(name, currency, te_frac=None):
    if currency not in {"eur_smart", "eur_hedged"}:
        return 1.0
    if name == "te":
        return te_frac if te_frac is not None else 0.0
    if name == "blend":
        return 0.5 if te_frac is None else 0.5 + 0.5 * te_frac
    return 1.0


def _apply_currency(r, name, currency, fx_close=None, te_frac=None,
                    hedge_cost=0.015):
    """Convert one aggregate USD return stream into the requested EUR scenario."""
    if currency == "usd":
        return r
    frac = _currency_fraction(name, currency, te_frac)
    if currency == "eur_hedged":
        eq = cur.hedged_eur_equity(r, annual_hedge_cost=hedge_cost,
                                   invested_frac=frac)
    elif currency in {"eur_naive", "eur_smart"}:
        exposure = 1.0 if currency == "eur_naive" else frac
        eq = cur.unhedged_eur_equity(r, fx_close, invested_frac=exposure)
    else:
        raise ValueError(f"unknown currency scenario {currency}")
    out = eq.pct_change().fillna(0.0)
    out.attrs.update(r.attrs)
    return out


def _te_held_by_symbol(daily_data):
    return {
        name: te.exposure_series(daily["close"], TE_MA_PERIOD, TE_BUFFER)
                .shift(1).fillna(0.0)
        for name, daily in daily_data.items()
    }


def _te_after_tax_equity(daily_data, begin_ts, end_ts, currency="usd",
                         fx_close=None, hedge_cost=0.015,
                         initial=INITIAL_EQUITY):
    """Equal-capital TE sleeve taxed at each symbol's actual exits.

    Per-symbol strategy returns already contain switch costs. Currency conversion is
    applied before exit-timed tax. ``eur_naive`` keeps flat capital in USD cash;
    that cash-FX component is included in wealth but this remains a scenario proxy,
    not a full tax treatment of every currency transaction.
    """
    held_map = _te_held_by_symbol(daily_data)
    parts = []
    n = len(daily_data)
    for name, daily in daily_data.items():
        raw = te.strategy_returns(daily, TE_MA_PERIOD, TE_BUFFER, 1.0, 0.0)
        raw = _slice(raw, begin_ts, end_ts)
        held = held_map[name].reindex(raw.index).fillna(0.0)
        if currency == "usd":
            converted = raw
        else:
            local_frac = held if currency in {"eur_smart", "eur_hedged"} else 1.0
            if currency == "eur_hedged":
                eq = cur.hedged_eur_equity(raw, hedge_cost, initial=1.0,
                                           invested_frac=local_frac)
            else:
                eq = cur.unhedged_eur_equity(raw, fx_close, invested_frac=local_frac,
                                              initial=1.0)
            converted = eq.pct_change().fillna(0.0)
        taxed = tx.after_tax_binary_strategy(converted, held, initial=initial / n)
        parts.append(taxed)
    merged = pd.concat(parts, axis=1).sort_index().ffill()
    out = merged.sum(axis=1)
    out.attrs["initial_equity"] = initial
    return out


def _metrics_from_returns(r):
    eq = INITIAL_EQUITY * (1 + r).cumprod()
    eq.attrs["initial_equity"] = INITIAL_EQUITY
    return compute_metrics([], eq)


def print_score(daily_data, books, begin_ts, end_ts, tax_rate=tx.SLOVENIA_TAX_RATE,
                currency="usd", fx_close=None, te_frac=None, hedge_cost=0.015):
    if not 0 <= tax_rate <= 1:
        raise ValueError("tax_rate must be in [0, 1]")
    blend = combine_books(books, weights={"rp": 0.5, "te": 0.5})
    bh_ret = buy_hold_combined(daily_data, begin_ts).pct_change().fillna(0.0)
    rp = _slice(books["rp"], begin_ts, end_ts)
    te_ret = _slice(books["te"], begin_ts, end_ts)
    blend = _slice(blend, begin_ts, end_ts)
    bh_ret = _slice(bh_ret, begin_ts, end_ts)

    if te_frac is None:
        te_frac = cur.te_invested_fraction(daily_data, TE_MA_PERIOD, TE_BUFFER)
    rp_c = _apply_currency(rp, "rp", currency, fx_close, te_frac, hedge_cost)
    te_c = _apply_currency(te_ret, "te", currency, fx_close, te_frac, hedge_cost)
    blend_c = _apply_currency(blend, "blend", currency, fx_close, te_frac, hedge_cost)
    bh_c = _apply_currency(bh_ret, "buy_and_hold", currency, fx_close, te_frac, hedge_cost)

    rp_post = tx.after_tax_active(rp_c, tax_rate, INITIAL_EQUITY)
    te_post = _te_after_tax_equity(daily_data, begin_ts, end_ts, currency,
                                   fx_close, hedge_cost, INITIAL_EQUITY)
    rp_half = tx.after_tax_active(rp_c, tax_rate, INITIAL_EQUITY * 0.5)
    te_half = _te_after_tax_equity(daily_data, begin_ts, end_ts, currency,
                                   fx_close, hedge_cost, INITIAL_EQUITY * 0.5)
    blend_post = rp_half.add(te_half, fill_value=0.0)
    blend_post.attrs["initial_equity"] = INITIAL_EQUITY
    bh_post = tx.after_tax_buy_hold(bh_c, initial=INITIAL_EQUITY)

    rows = [
        ("risk-based allocation", rp_c, rp_post),
        ("trend exposure", te_c, te_post),
        ("50/50 risk + trend", blend_c, blend_post),
        ("buy_and_hold", bh_c, bh_post),
    ]
    print(f"\nAFTER-TAX SCENARIO COMPARISON (currency={currency})")
    print("Securities scenario: 25%/<5y, 20%/5-10y, 15%/10-15y, 0%/15y+; "
          "calendar holding periods. This is not tax-lot accounting.")
    print(f"{'Strategy':28}{'Pre-tax Shp':>13}{'Post-tax Shp':>14}{'Post-tax Ret%':>15}")
    for name, pre, post in rows:
        pre_m = _metrics_from_returns(pre)
        post_m = compute_metrics([], post)
        print(f"{name:28}{pre_m['sharpe']:>13.3f}{post_m['sharpe']:>14.3f}"
              f"{post_m['total_return']*100:>15.1f}")

    bh_sharpe = compute_metrics([], bh_post)["sharpe"]
    print("\nCORE-SATELLITE STRESS GRID (core=B&H; satellite=annual-realization proxy)")
    for core_w in DEFAULT_CORE_WEIGHTS:
        eq = tx.after_tax_core_satellite(bh_c, blend_c, core_w, tax_rate)
        m = compute_metrics([], eq)
        print(f"{core_w:.1f}/{1-core_w:.1f}: Sharpe={m['sharpe']:.3f} "
              f"Return={m['total_return']:.1%} "
              f"{'above' if m['sharpe'] > bh_sharpe else 'not above'} pure B&H Sharpe")
    return rows


def print_checkpoints(daily_data, books, begin_ts, core_weight,
                      tax_rate=tx.SLOVENIA_TAX_RATE,
                      checkpoint_years=(10, 12, 15, 18, 21)):
    if not 0 <= core_weight <= 1:
        raise ValueError("core_weight must be in [0, 1]")
    blend = combine_books(books, weights={"rp": 0.5, "te": 0.5})
    bh = buy_hold_combined(daily_data, begin_ts).pct_change().fillna(0.0)
    bh = bh[bh.index >= begin_ts]
    blend = blend[blend.index >= begin_ts]
    print(f"\nCORE-SATELLITE CHECKPOINTS — same continuous start {begin_ts.date()}")
    checks = wins = 0
    for years in checkpoint_years:
        target = begin_ts + pd.DateOffset(years=years)
        if target > bh.index[-1]:
            continue
        bh_slice = bh[bh.index <= target]
        sat_slice = blend[blend.index <= target]
        bh_eq = tx.after_tax_buy_hold(bh_slice)
        cs_eq = tx.after_tax_core_satellite(bh_slice, sat_slice, core_weight, tax_rate)
        mb, mc = compute_metrics([], bh_eq), compute_metrics([], cs_eq)
        beat = mc["sharpe"] > mb["sharpe"]
        checks += 1; wins += int(beat)
        print(f"{target.date()}: B&H={mb['sharpe']:.3f} core-sat={mc['sharpe']:.3f} "
              f"{'above' if beat else 'not above'}")
    print(f"SUMMARY: core-satellite above pure B&H Sharpe in {wins}/{checks} checkpoints.")
    return wins, checks


def main():
    ap = argparse.ArgumentParser(description="Slovenian after-tax scenario research.")
    ap.add_argument("--mode", choices=["score", "checkpoints"], default="score")
    ap.add_argument("--core-weight", type=float, default=0.7)
    ap.add_argument("--tax-rate", type=float, default=tx.SLOVENIA_TAX_RATE,
                    help="Annual-realization proxy rate for the high-turnover sleeve")
    ap.add_argument("--currency", choices=["usd", "eur_naive", "eur_smart", "eur_hedged"],
                    default="usd")
    ap.add_argument("--hedge-cost", type=float, default=0.015)
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "GLD", "TLT"])
    ap.add_argument("--rp-strategy", choices=["min_var", "inverse_vol", "erc"], default="min_var")
    ap.add_argument("--months", type=int, default=240)
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca")
    args = ap.parse_args()

    if (args.months <= 0 or not 0 <= args.core_weight <= 1 or
            not 0 <= args.tax_rate <= 1 or args.hedge_cost < 0):
        ap.error("invalid months/core-weight/tax-rate/hedge-cost")
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
    books = compute_books(daily_data, rp_strategy=args.rp_strategy)

    if args.mode == "checkpoints":
        if args.currency != "usd":
            log.error("Checkpoint tax/currency interaction is not modeled; use --currency usd.")
            return 1
        print_checkpoints(daily_data, books, start_dt, args.core_weight, args.tax_rate)
        return 0

    fx_close = None
    if args.currency in {"eur_naive", "eur_smart"}:
        fx_data, _, _ = fetch_all(["EURUSD=X"], "none", start_dt, end_dt, source="yahoo")
        if "EURUSD=X" not in fx_data:
            log.error("EURUSD=X data unavailable; cannot perform EUR conversion.")
            return 1
        fx_close = fx_data["EURUSD=X"]["close"]
    te_frac = cur.te_invested_fraction(daily_data, TE_MA_PERIOD, TE_BUFFER)
    print_score(daily_data, books, start_dt, end_dt, args.tax_rate,
                args.currency, fx_close, te_frac, args.hedge_cost)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
