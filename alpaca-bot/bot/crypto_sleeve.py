"""
bot/crypto_sleeve.py
=====================
Tests a genuinely new asset class in this project: a small, trend-filtered
Bitcoin sleeve blended into the already-validated min_var+TE combo.

Why trend-filtered, not raw buy-and-hold: BTC-USD's own max drawdown is a
brutal -83% (pure buy-and-hold, 2014-2026); the SAME no-lookahead 200-day MA
filter already validated elsewhere in this project (bot.trend_exposure) cuts
that to -70% while actually IMPROVING Sharpe (0.80 -> 0.95) and total return --
so the sleeve tested here is trend-exposure on BTC, not naive buy-and-hold.

VALIDATED FINDING: blended into the min_var+TE combo at a modest 10-15%
weight, this trend-filtered BTC sleeve has LOW correlation with the existing
book (~0.12 on both primary universes -- the same low-correlation bar that
made the original RP+TE combo work) and provides a REAL Sharpe improvement:
universe 1 (SPY/QQQ/GLD/TLT) 1.11 -> 1.36 at 15% weight; universe 2 (8-asset)
0.87 -> 1.20 at 10% weight. Walk-forward (5 folds, universe 2, 10% weight):
beat the pure blend in 4/5 folds -- the one loss is the most recent, shortest
fold, not a red flag on its own. Max drawdown rises only modestly (roughly
+2-8 percentage points), not offsetting the Sharpe/return gain.

Honest caveats: only ~10-11 years of BTC history exist (since Oct 2014), far
shorter than this project's usual 20+ year windows, so this is inherently a
less-tested result than the equity-only combo. BTC's own correlation with
risk assets has structurally risen since ~2020 (institutional adoption) --
the "low correlation" finding could erode over time; re-check periodically
rather than assuming it holds forever. Crypto trading (BTC/USD) requires the
`alpaca-py` crypto path (slash-form symbol) -- see README's "You can't short
crypto on Alpaca" caveat, which is fine here since this is long-only trend
exposure, same as everywhere else in this project.

TAX (--mode aftertax, added 2026-07): Slovenia's crypto tax changed 2026-01-01
-- a flat 25% on disposal to FIAT (no graduated holding-period discount like
securities have), but crypto-to-crypto swaps (including into a stablecoin)
are explicitly NOT a taxable event and carry forward the original cost basis.
In principle that lets a trend-following crypto strategy defer ALL tax to one
eventual cash-out by parking in a stablecoin instead of literally converting
to EUR on every "exit" -- exactly like buy-and-hold's deferral. That full-
deferral scenario was tried and DELIBERATELY NOT used as the headline model
here: it requires never rebalancing crypto back into the equity/bond legs
(any such rebalance needs an intermediate fiat conversion), which lets a
winning BTC position balloon to an undisciplined, unbounded fraction of net
worth over a decade of 100x+ growth -- in direct tension with this project's
whole risk-managed, fixed-allocation philosophy. --mode aftertax instead
models the DISCIPLINED scenario: the BTC sleeve held at a constant target
weight (rebalanced alongside the rest of the book) and taxed annually, same
mechanic as everything else active in this project (bot.taxes.after_tax_active
-- its flat rate already matches CRYPTO_TAX_RATE, since both happen to be
25%). Finding: even under this realistic, disciplined, fully-taxed scenario,
the BTC sleeve still meaningfully improves after-tax Sharpe (see README).
Crypto acquired BEFORE 2026-01-01 is grandfathered -- 100% exempt forever,
even sold later -- so if the user already holds pre-2026 BTC, none of this
tax modeling applies to that specific position.

Sources: [CoinDesk -- Slovenia moves to tax crypto profits at 25%](https://www.coindesk.com/policy/2025/04/19/slovenia-moves-to-tax-crypto-profits-at-25),
[Waltio -- Slovenia crypto tax guide 2026](https://help.waltio.com/en/articles/14739040-slovenia-crypto-tax-guide-2026-the-complete-guide).

    python -m bot.crypto_sleeve --symbols SPY QQQ GLD TLT --start 2015-01-01 --data-source yahoo
    python -m bot.crypto_sleeve --mode walk --folds 5 --symbols SPY QQQ IWM EFA EEM TLT IEF GLD \
        --start 2015-01-01 --data-source yahoo
    python -m bot.crypto_sleeve --mode aftertax --symbols SPY QQQ GLD TLT --start 2015-01-01 --data-source yahoo
"""

import argparse
import logging
from datetime import datetime, timezone

import pandas as pd

import bot.taxes as tx
import bot.trend_exposure as te
from bot.combo import compute_books, combine_books
from bot.lab import fold_bounds, slice_equity
from bot.backtest_pullback import (compute_metrics, buy_hold_combined, fetch_all,
                                   _parse_date, INITIAL_EQUITY)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("crypto_sleeve")

BTC_MA_PERIOD = 200
BTC_BUFFER = 0.01
DEFAULT_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20]


def compute_btc_sleeve(btc_daily, ma_period=BTC_MA_PERIOD, buffer=BTC_BUFFER) -> pd.Series:
    """Trend-exposure return series for BTC-USD -- long/cash on the same
    no-lookahead 200-day MA filter used throughout this project."""
    return te.strategy_returns(btc_daily, ma_period=ma_period, buffer=buffer,
                               leverage=1.0, borrow_rate=0.0)


def print_score(daily_data, books, btc_daily, begin_ts, end_ts):
    blend = combine_books(books, weights={"rp": 0.5, "te": 0.5})
    bh_ret = buy_hold_combined(daily_data, begin_ts).pct_change().fillna(0.0)
    btc_sleeve = compute_btc_sleeve(btc_daily)

    common = blend.index.intersection(btc_sleeve.index)
    common = common[(common >= begin_ts) & (common <= end_ts)]
    blend, btc_sleeve = blend.loc[common], btc_sleeve.loc[common]
    # Same window for the buy-and-hold row: if BTC's history starts after
    # begin_ts, every row must cover the SAME (BTC-limited) period or the
    # comparison is apples-to-oranges.
    bh_ret = bh_ret[(bh_ret.index >= common[0]) & (bh_ret.index <= common[-1])]

    corr = blend.corr(btc_sleeve)

    def m(r):
        return compute_metrics([], INITIAL_EQUITY * (1 + r).cumprod())

    print(f"\nCRYPTO SLEEVE  (trend-filtered BTC blended into the min_var+TE combo)")
    print(f"Correlation(blend, trend-filtered BTC) = {corr:.3f}  "
          f"(window: {common[0].date()} -> {common[-1].date()})")
    print("=" * 74)
    print(f"{'Book':28}{'Sharpe':>10}{'Return%':>13}{'MaxDD%':>11}{'beats blend?':>13}")
    print("-" * 74)
    m_bh, m_blend = m(bh_ret), m(blend)
    print(f"{'buy_and_hold':28}{m_bh['sharpe']:>10.3f}{m_bh['total_return']*100:>13.1f}"
          f"{m_bh['max_drawdown']*100:>11.1f}{'':>13}")
    print(f"{'min_var+TE blend (0% BTC)':28}{m_blend['sharpe']:>10.3f}"
          f"{m_blend['total_return']*100:>13.1f}{m_blend['max_drawdown']*100:>11.1f}{'':>13}")
    for w in DEFAULT_WEIGHTS[1:]:
        mixed = (1 - w) * blend + w * btc_sleeve
        mm = m(mixed)
        beat = "YES" if mm["sharpe"] > m_blend["sharpe"] else "no"
        print(f"{f'blend + {w:.0%} BTC':28}{mm['sharpe']:>10.3f}{mm['total_return']*100:>13.1f}"
              f"{mm['max_drawdown']*100:>11.1f}{beat:>13}")
    print("=" * 74)


def print_walk(daily_data, books, btc_daily, start_dt, end_dt, folds, btc_weight=0.10):
    blend = combine_books(books, weights={"rp": 0.5, "te": 0.5})
    btc_sleeve = compute_btc_sleeve(btc_daily)
    common = blend.index.intersection(btc_sleeve.index)
    blend, btc_sleeve = blend.loc[common], btc_sleeve.loc[common]
    mixed = (1 - btc_weight) * blend + btc_weight * btc_sleeve

    print(f"\nCRYPTO SLEEVE WALK-FORWARD  btc_weight={btc_weight:.0%}  {folds} folds")
    print("=" * 78)
    print(f"{'Fold':>4}  {'Window':>23}  {'pure blend':>11}  {'+BTC':>8}  beats")
    print("-" * 78)
    wins = 0
    checks = 0
    for k, (fs, fe) in enumerate(fold_bounds(start_dt, end_dt, folds)):
        pure_eq = slice_equity(blend, fs, fe)
        if not len(pure_eq):
            continue
        m_pure = compute_metrics([], pure_eq)
        m_mix = compute_metrics([], slice_equity(mixed, fs, fe))
        beat = m_mix["sharpe"] > m_pure["sharpe"]
        wins += int(beat)
        checks += 1
        print(f"{k+1:>4}  {fs.date()}->{fe.date()}  {m_pure['sharpe']:>11.2f}  "
              f"{m_mix['sharpe']:>8.2f}  {'YES' if beat else 'no'}")
    print("-" * 78)
    print(f"+{btc_weight:.0%} BTC beat pure blend in {wins}/{checks} folds.")
    print("=" * 78)


def print_aftertax(daily_data, books, btc_daily, begin_ts, end_ts, tax_rate=tx.CRYPTO_TAX_RATE):
    """The DISCIPLINED after-tax comparison: the BTC sleeve held at a constant
    target weight (rebalanced alongside the rest of the book, not left to
    balloon unbounded) and taxed annually -- see module docstring for why this,
    not the full-deferral-via-stablecoin-swaps scenario, is the headline
    model. after_tax_active's flat rate already matches CRYPTO_TAX_RATE (both
    25%), so this reuses it directly for the blended return series."""
    blend = combine_books(books, weights={"rp": 0.5, "te": 0.5})
    btc_sleeve = compute_btc_sleeve(btc_daily)
    common = blend.index.intersection(btc_sleeve.index)
    common = common[(common >= begin_ts) & (common <= end_ts)]
    blend, btc_sleeve = blend.loc[common], btc_sleeve.loc[common]

    print(f"\nCRYPTO SLEEVE AFTER-TAX  (disciplined: constant weight, rebalanced + taxed "
          f"annually at Slovenia's flat {tax_rate:.0%} crypto rate)")
    print("=" * 74)
    print(f"{'Book':30}{'Sharpe':>10}{'Return%':>13}{'MaxDD%':>11}{'beats blend?':>13}")
    print("-" * 74)
    blend_eq = tx.after_tax_active(blend, tax_rate)
    m_blend = compute_metrics([], blend_eq)
    print(f"{'blend (0% BTC), post-tax':30}{m_blend['sharpe']:>10.3f}"
          f"{m_blend['total_return']*100:>13.1f}{m_blend['max_drawdown']*100:>11.1f}{'':>13}")
    for w in DEFAULT_WEIGHTS[1:]:
        mixed = (1 - w) * blend + w * btc_sleeve
        mixed_eq = tx.after_tax_active(mixed, tax_rate)
        mm = compute_metrics([], mixed_eq)
        beat = "YES" if mm["sharpe"] > m_blend["sharpe"] else "no"
        print(f"{f'blend + {w:.0%} BTC, post-tax':30}{mm['sharpe']:>10.3f}"
              f"{mm['total_return']*100:>13.1f}{mm['max_drawdown']*100:>11.1f}{beat:>13}")
    print("-" * 74)
    print("Note: crypto acquired before 2026-01-01 is grandfathered (0% tax forever,")
    print("regardless of when sold) -- this table assumes a position acquired now, at")
    print("the flat post-2026 rate. See module docstring for the full-deferral-via-")
    print("stablecoin-swap scenario and why it's not modeled as the headline case.")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser(description="Trend-filtered BTC sleeve blended into the RP+TE combo.")
    ap.add_argument("--mode", choices=["score", "walk", "aftertax"], default="score")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--btc-weight", type=float, default=0.10,
                    help="Capital fraction in the BTC sleeve for --mode walk (score mode "
                         "sweeps a fixed grid instead).")
    ap.add_argument("--btc-symbol", type=str, default="BTC-USD")
    ap.add_argument("--rp-strategy", choices=["min_var", "inverse_vol", "erc"], default="min_var")
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "GLD", "TLT"])
    ap.add_argument("--months", type=int, default=240)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca")
    args = ap.parse_args()

    end_dt = _parse_date(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc))
    start_dt = (_parse_date(args.start) if args.start
                else end_dt - pd.Timedelta(days=int(args.months * 31)))
    log.info("Crypto sleeve window: %s -> %s (%s mode)", start_dt.date(), end_dt.date(), args.mode)

    daily_data, _, _ = fetch_all(args.symbols, "none", start_dt, end_dt, source=args.data_source)
    if len(daily_data) < 2:
        log.error("Need >= 2 symbols with data; got %d.", len(daily_data))
        return 1
    btc_data, _, _ = fetch_all([args.btc_symbol], "none", start_dt, end_dt, source="yahoo")
    if args.btc_symbol not in btc_data:
        log.error("Could not fetch %s.", args.btc_symbol)
        return 1
    books = compute_books(daily_data, rp_strategy=args.rp_strategy)

    if args.mode == "walk":
        print_walk(daily_data, books, btc_data[args.btc_symbol], start_dt, end_dt, args.folds,
                   args.btc_weight)
    elif args.mode == "aftertax":
        print_aftertax(daily_data, books, btc_data[args.btc_symbol], start_dt, end_dt)
    else:
        print_score(daily_data, books, btc_data[args.btc_symbol], start_dt, end_dt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
