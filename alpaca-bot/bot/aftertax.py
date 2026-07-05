"""
bot/aftertax.py
================
The honest, after-tax version of "does anything beat buy-and-hold?" — because
every backtest elsewhere in this project is PRE-TAX, and Slovenian capital-gains
tax is a massive structural advantage for true buy-and-hold: a graduated
schedule (25% under 5y, 20% 5-10y, 15% 10-15y, 0% past 15y — verified against
fu.gov.si; see bot/taxes.py for the model, its sources, and its honest
simplifications) that no actively-traded strategy can fully escape.

THE HEADLINE FINDING (see README): once taxed, the min_var+TE combo (this
project's best PRE-TAX result) actually LOSES to plain buy-and-hold on Sharpe —
annual tax on realized gains erodes a strategy's edge far more than it erodes
buy-and-hold's, which pays nothing until a single deferred sale (and even then
at a discounted rate if held 5+ years). The one structure that BEATS after-tax
buy-and-hold: a CORE-SATELLITE split — most of the capital in a true, untouched
buy-and-hold core (tax-deferred), a minority in the active combo as a satellite
(taxed annually). --mode checkpoints also reports --cross-offset: letting a
leftover satellite loss shelter part of the core's eventual sale gain, which
only matters before the core itself reaches its 0% exemption (see bot/taxes.py
for why an intra-year "harvest sooner" variant was tried and dropped as a
proven no-op).

A SECOND, EQUALLY LARGE FACTOR (see bot/currency.py): every number above
implicitly assumes a USD-based investor, but the user is Slovenian (EUR-based)
and SPY/QQQ/GLD/TLT are USD-denominated. --currency {eur_naive,eur_smart,
eur_hedged} converts the same comparison into a EUR investor's REAL return.
Unhedged EUR exposure compresses every Sharpe by roughly as much as tax does,
and the min_var+TE blend's pre-tax edge over buy-and-hold nearly disappears
once currency risk is honestly priced in — see README for the combined
(currency + tax) picture.

    python -m bot.aftertax --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
    python -m bot.aftertax --currency eur_naive --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
    python -m bot.aftertax --mode checkpoints --core-weight 0.7 \
        --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
"""

import argparse
import logging
from datetime import datetime, timezone

import pandas as pd

import bot.taxes as tx
import bot.currency as cur
from bot.combo import compute_books, combine_books
from bot.backtest_pullback import (compute_metrics, buy_hold_combined, fetch_all,
                                   _parse_date, INITIAL_EQUITY)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("aftertax")

DEFAULT_CORE_WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]


def _slice(r, begin_ts, end_ts):
    return r[(r.index >= begin_ts) & (r.index <= end_ts)]


def _apply_currency(r, name, currency, fx_close=None, te_frac=None, hedge_cost=0.015):
    """currency: 'usd' (no change), 'eur_hedged' (fixed cost drag, no FX risk),
    'eur_naive' (full FX exposure always, cash assumed held in USD), or
    'eur_smart' (FX exposure only while actually invested in USD assets --
    always 1.0 for rp/buy_and_hold, which never go to cash; te_frac for the
    te leg; a 50/50 mix of the two for the blend)."""
    if currency == "usd":
        return r
    if currency == "eur_hedged":
        eq = cur.hedged_eur_equity(r, annual_hedge_cost=hedge_cost)
        return eq.pct_change().fillna(0.0)
    if currency == "eur_smart":
        if name == "te":
            frac = te_frac
        elif name == "blend":
            frac = 0.5 * 1.0 + 0.5 * te_frac
        else:
            frac = 1.0
    else:
        frac = 1.0
    eq = cur.unhedged_eur_equity(r, fx_close, invested_frac=frac)
    return eq.pct_change().fillna(0.0)


def print_score(daily_data, books, begin_ts, end_ts, tax_rate=tx.SLOVENIA_TAX_RATE,
                currency="usd", fx_close=None, te_frac=None, hedge_cost=0.015):
    blend = combine_books(books, weights={"rp": 0.5, "te": 0.5})
    bh_ret = buy_hold_combined(daily_data, begin_ts).pct_change().fillna(0.0)
    rp, te, blend, bh_ret = (_slice(books["rp"], begin_ts, end_ts),
                             _slice(books["te"], begin_ts, end_ts),
                             _slice(blend, begin_ts, end_ts),
                             _slice(bh_ret, begin_ts, end_ts))
    rp = _apply_currency(rp, "rp", currency, fx_close, te_frac, hedge_cost)
    te = _apply_currency(te, "te", currency, fx_close, te_frac, hedge_cost)
    blend = _apply_currency(blend, "blend", currency, fx_close, te_frac, hedge_cost)
    bh_ret = _apply_currency(bh_ret, "buy_and_hold", currency, fx_close, te_frac, hedge_cost)

    def pretax_m(r):
        eq = INITIAL_EQUITY * (1 + r).cumprod()
        return compute_metrics([], eq)

    hold_years = (bh_ret.index[-1] - bh_ret.index[0]).days / 365.0
    print(f"\nAFTER-TAX COMPARISON  (currency={currency}; Slovenia: {tax_rate:.0%} under 5y, "
          f"20% 5-10y, 15% 10-15y, 0% past 15y; window covers {hold_years:.1f}y)")
    print("=" * 70)
    print(f"{'Strategy':22}{'Pre-tax Shp':>13}{'Post-tax Shp':>14}{'Post-tax Ret%':>15}")
    print("-" * 70)
    rows = [("pure risk-based", rp, tx.after_tax_active(rp, tax_rate)),
            ("pure trend_exposure", te, tx.after_tax_active(te, tax_rate)),
            ("min_var+TE blend", blend, tx.after_tax_active(blend, tax_rate)),
            ("buy_and_hold", bh_ret, tx.after_tax_buy_hold(bh_ret))]
    bh_post_sharpe = None
    for name, pre_r, post_eq in rows:
        pre_m = pretax_m(pre_r)
        post_m = compute_metrics([], post_eq)
        if name == "buy_and_hold":
            bh_post_sharpe = post_m["sharpe"]
        print(f"{name:22}{pre_m['sharpe']:>13.3f}{post_m['sharpe']:>14.3f}"
              f"{post_m['total_return']*100:>15.1f}")
    print("-" * 70)

    print(f"\nCORE-SATELLITE  (core_weight in true buy-and-hold, rest in the active blend)")
    print(f"{'core/satellite':18}{'post-tax Sharpe':>17}{'post-tax Ret%':>15}{'beats pure B&H?':>17}")
    print("-" * 67)
    best_w, best_sharpe = None, -1e9
    for core_w in DEFAULT_CORE_WEIGHTS:
        eq = tx.after_tax_core_satellite(bh_ret, blend, core_w, tax_rate)
        m = compute_metrics([], eq)
        beat = "YES" if m["sharpe"] > bh_post_sharpe else "no"
        print(f"{core_w:.1f}/{1-core_w:.1f}            {m['sharpe']:>17.3f}"
              f"{m['total_return']*100:>15.1f}{beat:>17}")
        if m["sharpe"] > best_sharpe:
            best_w, best_sharpe = core_w, m["sharpe"]
    print("-" * 67)
    print(f"Best of the grid above: core_weight={best_w:.1f} -> Sharpe={best_sharpe:.3f} "
          f"(informational only -- this is a scan over a pre-defined grid, not a fit; "
          f"don't over-read the exact peak, look at the shape of the curve). Note: at "
          f"this window's {hold_years:.1f}y length the core is already past its 15y "
          f"exemption, so --cross-offset (--mode checkpoints) makes no difference here "
          f"-- it only matters before the core's own tax reaches 0%.")
    print("=" * 70)


def print_checkpoints(daily_data, books, begin_ts, core_weight, tax_rate=tx.SLOVENIA_TAX_RATE,
                      checkpoint_years=(10, 12, 15, 18, 21)):
    """Robustness check specific to a core-satellite/buy-hold structure: since the
    core's tax treatment depends on ONE continuous holding period (not
    independent folds -- there's nothing to 'walk forward' over), the honest
    stress test is to look at several DIFFERENT END DATES along that SAME
    continuous hold and see whether the core-satellite's after-tax Sharpe
    advantage over pure buy-and-hold holds up at each checkpoint, not just the
    specific end date the full backtest happens to stop at. Also reports the
    cross_offset_losses variant (a leftover satellite loss sheltering part of
    the core's gain), which only bites before the core reaches its own 0%
    exemption -- so it's most visible at the earlier checkpoints here."""
    blend = combine_books(books, weights={"rp": 0.5, "te": 0.5})
    bh_ret_full = buy_hold_combined(daily_data, begin_ts).pct_change().fillna(0.0)
    bh_ret_full = bh_ret_full[bh_ret_full.index >= begin_ts]
    blend_full = blend[blend.index >= begin_ts]

    print(f"\nCORE-SATELLITE CHECKPOINTS  core_weight={core_weight:.0%}  "
          f"(same continuous hold from {begin_ts.date()}, measured at several end dates)")
    print("=" * 78)
    print(f"{'End date':12}{'Years held':>11}{'pure B&H Shp':>14}{'core-sat Shp':>14}"
          f"{'+crossoffst':>13}{'beats?':>8}")
    print("-" * 78)
    wins = 0
    checks = 0
    for years in checkpoint_years:
        end_ts = begin_ts + pd.Timedelta(days=int(years * 365))
        if end_ts > bh_ret_full.index[-1]:
            continue
        bh_slice = bh_ret_full[bh_ret_full.index <= end_ts]
        blend_slice = blend_full[blend_full.index <= end_ts]
        bh_eq = tx.after_tax_buy_hold(bh_slice)
        cs_eq = tx.after_tax_core_satellite(bh_slice, blend_slice, core_weight, tax_rate)
        cs_eq_x = tx.after_tax_core_satellite(bh_slice, blend_slice, core_weight, tax_rate,
                                              cross_offset_losses=True)
        m_bh, m_cs, m_cs_x = (compute_metrics([], bh_eq), compute_metrics([], cs_eq),
                              compute_metrics([], cs_eq_x))
        beat = m_cs_x["sharpe"] > m_bh["sharpe"]
        wins += int(beat)
        checks += 1
        print(f"{end_ts.date()!s:12}{years:>11}{m_bh['sharpe']:>14.3f}{m_cs['sharpe']:>14.3f}"
              f"{m_cs_x['sharpe']:>13.3f}{'YES' if beat else 'no':>8}")
    print("-" * 78)
    print(f"Core-satellite (+cross-offset column) beat pure buy-and-hold in {wins}/{checks} "
          f"checkpoints.")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description="After-tax (Slovenia) comparison vs buy-and-hold.")
    ap.add_argument("--mode", choices=["score", "checkpoints"], default="score")
    ap.add_argument("--core-weight", type=float, default=0.7,
                    help="Core-satellite split for --mode checkpoints (fraction in "
                         "the untouched buy-and-hold core).")
    ap.add_argument("--tax-rate", type=float, default=tx.SLOVENIA_TAX_RATE)
    ap.add_argument("--currency", choices=["usd", "eur_naive", "eur_smart", "eur_hedged"],
                    default="usd", help="usd: no currency adjustment (default). eur_naive: "
                    "full unhedged EUR/USD exposure at all times. eur_smart: EUR/USD exposure "
                    "only while actually invested in USD assets (only affects --mode score). "
                    "eur_hedged: currency risk removed at a fixed annual cost (--hedge-cost).")
    ap.add_argument("--hedge-cost", type=float, default=0.015,
                    help="Fixed annual cost drag assumed for --currency eur_hedged.")
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "GLD", "TLT"])
    ap.add_argument("--rp-strategy", choices=["min_var", "inverse_vol", "erc"], default="min_var")
    ap.add_argument("--months", type=int, default=240)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca")
    args = ap.parse_args()

    end_dt = _parse_date(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc))
    start_dt = (_parse_date(args.start) if args.start
                else end_dt - pd.Timedelta(days=int(args.months * 31)))
    log.info("After-tax window: %s -> %s (%s mode)", start_dt.date(), end_dt.date(), args.mode)

    daily_data, _, _ = fetch_all(args.symbols, "none", start_dt, end_dt, source=args.data_source)
    if len(daily_data) < 2:
        log.error("Need >= 2 symbols with data; got %d.", len(daily_data))
        return 1
    books = compute_books(daily_data, rp_strategy=args.rp_strategy)

    if args.mode == "checkpoints":
        if args.currency != "usd":
            log.warning("--currency is not yet modeled in --mode checkpoints; ignoring.")
        print_checkpoints(daily_data, books, start_dt, args.core_weight, args.tax_rate)
    else:
        fx_close, te_frac = None, None
        if args.currency != "usd":
            # EURUSD=X is a Yahoo-style ticker -- fetch it from yahoo regardless of
            # --data-source, since alpaca doesn't carry FX pairs in this format.
            fx_data, _, _ = fetch_all(["EURUSD=X"], "none", start_dt, end_dt, source="yahoo")
            fx_close = fx_data["EURUSD=X"]["close"]
            if args.currency == "eur_smart":
                te_frac = cur.te_invested_fraction(daily_data)
        print_score(daily_data, books, start_dt, end_dt, args.tax_rate,
                   currency=args.currency, fx_close=fx_close, te_frac=te_frac,
                   hedge_cost=args.hedge_cost)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
