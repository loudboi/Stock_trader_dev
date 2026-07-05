"""
bot/taxes.py
============
Models Slovenian capital-gains tax on trading, because it changes the entire
"beat buy-and-hold" question. Every backtest in this project so far has been
PRE-TAX.

THE REAL SCHEDULE (verified 2026-07 against the Financial Administration of
the Republic of Slovenia, fu.gov.si, "Disposal of securities, other holdings
or investment coupons"): capital gains tax on securities is a CLIFF based on
the TOTAL holding period of the specific lot at the time of sale -- not a
flat rate, and not a marginal/bracket system like income tax (the whole gain
gets the rate for the bracket the total holding period falls into):

    holding period            rate
    ------------------------  -----
    0 - 5 years                25%
    5 - 10 years                20%
    10 - 15 years               15%
    more than 15 years           0%   (fully exempt)

This supersedes an earlier, simpler draft of this module that assumed a flat
25% for anything under 15 years -- that was wrong for anything held 5-15
years (real rate is 15-20%, not 25%) and mattered a lot for a literal
buy-and-hold position sold at, say, the 10 or 12 year mark. Losses CAN be
carried forward to future tax years (confirmed via the Doh-KDVP filing
instructions -- a taxpayer ticks "carry forward" on the specification list),
supporting the loss-carryforward mechanic already used below. No evidence of
a Slovenian wash-sale rule (a restriction on claiming a loss if you rebuy the
same/similar security shortly after) was found in the sources checked -- see
the "tax-loss harvesting" section below for where that assumption actually
matters (or doesn't) in this module.

Mechanics, because the tax treatment differs fundamentally by how a position
is actually held and traded:

  after_tax_active()          For a strategy that rebalances/trades
                              frequently (every strategy in this project
                              except literal buy-and-hold and, mostly,
                              trend-exposure). Approximates tax as an ANNUAL
                              realization event at each calendar year-end,
                              taxed at the flat 25% short-term rate (correct,
                              since an annually-realized gain was held under
                              5 years -- typically under 1) with loss
                              carryforward. A simplification of full
                              per-trade tax-lot accounting, but it captures
                              the economically dominant effect: strategies
                              that genuinely re-trade constantly (min_var,
                              inverse_vol, erc all rebalance daily) realize
                              gains at the short-term rate almost every year.

  after_tax_buy_hold()        For a position bought once and never touched.
                              No tax until the FINAL sale, at which point the
                              REAL graduated schedule above is applied based
                              on the total holding period -- 0% past 15
                              years, but also correctly gives 15%/20% credit
                              for holds in the 10-15y/5-10y bands rather than
                              assuming a flat 25% until the exemption.
                              Accepts extra_loss_shelter (see
                              after_tax_core_satellite's cross_offset_losses)
                              to shelter part of the gain with an outside
                              loss at the point of final sale.

  after_tax_exposure_based()  A trade-level version of the schedule for
                              binary in/out strategies (e.g.
                              bot.trend_exposure): each contiguous holding
                              run is taxed, AT EXIT, using the graduated rate
                              for THAT run's actual holding period (so a
                              multi-year hold gets the 15%/20% discount, not
                              a flat 25%), rather than the coarser
                              all-or-nothing exemption cliff used before this
                              update.

TAX-LOSS HARVESTING -- WHAT WAS TRIED AND WHY IT'S NOT A SEPARATE FUNCTION:
the obvious first idea -- "harvest" a loss the moment it occurs intra-year
(sell and immediately rebuy a similar instrument; no Slovenian wash-sale rule
was found in the sources checked, so this isn't restricted) instead of only
netting at year-end -- was implemented and tested, and turned out to be a
PROVABLE NO-OP given after_tax_active's own annual-netting mechanics: since
tax already applies to the whole year's NET gain (start-of-year basis to
year-end NAV) with full loss carryforward, moving the bookkeeping of a
mid-year dip earlier changes nothing about that final number -- algebraically,
taxable = nav_end - year_start_basis, always, regardless of how many times
the running cost basis got reset by an intra-year "harvest" in between. Real
intra-year timing only matters if you can apply a harvested loss against a
DIFFERENT tax lot's gain sooner than the annual cycle would allow -- which is
exactly what cross_offset_losses (below) does; a same-lot "harvest sooner"
mechanic in isolation does not, and was removed after confirming this by test.

THE REAL LOSS-HARVESTING LEVER FOUND: cross_offset_losses on
after_tax_core_satellite. Slovenian tax nets capital gains and losses across
a taxpayer's holdings within the same annual filing (Doh-KDVP) rather than
isolating each security lot -- so a loss left over in the satellite (banked
via carryforward but never absorbed by a later satellite gain) can shelter
part of the CORE's gain when the core is finally sold. This only matters
before the core reaches its own 0% exemption (past that point there's no tax
left to shelter), so it shows up in the checkpoint analysis, not the
headline 21+-year full-window result.

All of these produce an EQUITY curve (not a return series, since a tax
payment is a discrete capital deduction, not a percentage return) so they
plug directly into bot.backtest_pullback.compute_metrics for an honest
apples-to-apples AFTER-TAX Sharpe/return/drawdown comparison.

CRYPTOCURRENCY IS A COMPLETELY DIFFERENT, SEPARATE REGIME (verified 2026-07,
directly relevant to bot/crypto_sleeve.py): crypto gains were historically
UNTAXED for individuals in Slovenia, but a new law took effect 2026-01-01:

  - A FLAT 25% on disposal to FIAT (or spending it / transferring to a third
    party) -- no graduated holding-period schedule like securities have.
    CRYPTO_TAX_RATE below happens to equal SLOVENIA_TAX_RATE (both 25%), so
    after_tax_active() already models this correctly for a crypto leg with no
    changes needed -- just pass CRYPTO_TAX_RATE for clarity at the call site.
  - Crypto-to-crypto swaps (INCLUDING into a stablecoin) are explicitly NOT a
    taxable disposal -- they carry forward the original acquisition date and
    cost basis. This means a trend-following crypto strategy that parks in a
    stablecoin instead of cashing out to EUR on every "exit" defers ALL tax
    to a single eventual fiat cash-out, exactly like buy-and-hold's deferral
    (model with after_tax_buy_hold using a flat, non-graduated schedule, e.g.
    schedule=((10**9, 0.25),) -- see bot/crypto_sleeve.py for why this is
    NOT the model used for the main recommendation there: full deferral
    requires never rebalancing crypto back into other asset classes [any such
    rebalance needs an intermediate fiat conversion], which lets a winning
    crypto position balloon to an undisciplined, unbounded fraction of net
    worth -- in tension with this project's whole risk-managed-allocation
    philosophy. The disciplined (periodically rebalanced, annually taxed)
    scenario is what's actually recommended.
  - GRANDFATHERING: crypto acquired BEFORE 2026-01-01 is 100% exempt from this
    tax FOREVER, even if sold well after 2026 -- not just gains accrued before
    the law, the ENTIRE historical gain. If the user already holds crypto
    bought before 2026, it is untouched by any of this.
  - Losses carry forward to future years, same general mechanic as
    securities (after_tax_active's loss_carryforward already covers this).

Sources: Slovenia's Ministry of Finance / FURS crypto tax reform coverage,
cross-checked via CoinDesk, Waltio, and TaxRavens' 2026 guides (see
bot/crypto_sleeve.py's module docstring for direct links).
"""

import logging

import pandas as pd

log = logging.getLogger("taxes")

SLOVENIA_TAX_RATE = 0.25                # top/short-term rate (0-5y hold)
SLOVENIA_EXEMPT_DAYS = 15 * 365         # fully exempt past this many days held
CRYPTO_TAX_RATE = 0.25                  # flat, NO holding-period discount (unlike securities)
CRYPTO_LAW_EFFECTIVE_DATE = pd.Timestamp("2026-01-01", tz="UTC")   # crypto acquired
                                        # before this date is 100% exempt forever (grandfathered)

# Real, graduated cliff schedule: (upper bound in days, rate applied to the
# WHOLE gain if held that long or less). Anything held longer than the last
# entry's threshold is fully exempt (0%).
SLOVENIA_SCHEDULE = (
    (5 * 365, 0.25),
    (10 * 365, 0.20),
    (15 * 365, 0.15),
)


def slovenia_rate_for_holding(hold_days: float, schedule=SLOVENIA_SCHEDULE) -> float:
    """The tax rate that applies to a lot's ENTIRE gain given its total
    holding period in days, per the real Slovenian cliff schedule (not a
    marginal-bracket system -- the whole gain gets one rate)."""
    for upper_days, rate in schedule:
        if hold_days <= upper_days:
            return rate
    return 0.0


def _after_tax_active_full(returns: pd.Series, tax_rate: float = SLOVENIA_TAX_RATE,
                           initial: float = 100_000.0):
    """Core of after_tax_active, but also returns the final leftover loss
    carryforward (banked but never absorbed by a later gain within this lot
    alone) so after_tax_core_satellite's cross_offset_losses can apply it
    elsewhere."""
    if returns.empty:
        return pd.Series(dtype=float), 0.0
    equity = pd.Series(index=returns.index, dtype=float)
    nav = initial
    basis = initial                       # NAV as of the last tax event
    loss_carryforward = 0.0
    years = returns.index.year
    n = len(returns)
    for i in range(n):
        nav *= (1.0 + returns.iloc[i])
        equity.iloc[i] = nav
        is_year_end = (i == n - 1) or (years[i + 1] != years[i])
        if is_year_end:
            gain = nav - basis
            taxable = gain - loss_carryforward
            if taxable > 0:
                tax = tax_rate * taxable
                nav -= tax
                equity.iloc[i] = nav
                loss_carryforward = 0.0
            else:
                loss_carryforward = -taxable      # accumulate for future offset
            basis = nav
    return equity, loss_carryforward


def after_tax_active(returns: pd.Series, tax_rate: float = SLOVENIA_TAX_RATE,
                     initial: float = 100_000.0) -> pd.Series:
    """Annual realize-and-tax simulation with loss carryforward, for a
    frequently-rebalanced/traded strategy. See module docstring for the
    approximation this makes (and for why a same-lot "harvest sooner"
    variant was tried and dropped as a proven no-op)."""
    equity, _ = _after_tax_active_full(returns, tax_rate, initial)
    return equity


def after_tax_buy_hold(returns: pd.Series, schedule=SLOVENIA_SCHEDULE,
                       initial: float = 100_000.0,
                       extra_loss_shelter: float = 0.0) -> pd.Series:
    """A position bought once at the start and sold once at the end (or never
    sold, if you'd hold past the backtest window) -- taxed once, at the end,
    using the REAL graduated rate for the total holding period (0% past 15y,
    but also 15%/20% credit for 10-15y/5-10y holds rather than a flat 25%).
    extra_loss_shelter: an outside capital loss (e.g. a satellite's leftover
    carryforward, see after_tax_core_satellite's cross_offset_losses) that
    can offset part of this gain at the point of sale, per Slovenia's
    same-filing netting of gains/losses across a taxpayer's holdings."""
    if returns.empty:
        return pd.Series(dtype=float)
    equity = initial * (1.0 + returns).cumprod()
    hold_days = (equity.index[-1] - equity.index[0]).days
    rate = slovenia_rate_for_holding(hold_days, schedule)
    if rate <= 0:
        return equity                     # exempt -- no tax due, ever
    gain = equity.iloc[-1] - initial
    taxable = gain - extra_loss_shelter
    if taxable > 0:
        equity = equity.copy()
        equity.iloc[-1] -= rate * taxable
    return equity


def after_tax_exposure_based(close: pd.Series, held: pd.Series,
                             schedule=SLOVENIA_SCHEDULE,
                             initial: float = 100_000.0) -> pd.Series:
    """A MORE ACCURATE tax model for binary in/out exposure strategies (e.g.
    bot.trend_exposure), which after_tax_active over-penalizes: that blanket
    model taxes the WHOLE year's return annually regardless of whether anything
    was actually sold, which is right for a daily-rebalanced book (min_var/
    inverse_vol/erc genuinely churn positions constantly) but wrong for a
    strategy that might hold one uninterrupted position for 1-3+ years without
    a single trade. Here, tax is only realized at the actual EXIT of each
    contiguous holding period, using the REAL graduated rate for that
    specific run's holding period -- a multi-year hold gets the 15%/20%
    discount just like buy-and-hold would, not a flat 25%.

    `held` is the position series (1 = invested, 0 = cash) already decided with
    no lookahead (e.g. bot.trend_exposure.strategy_returns' internal `held`,
    reconstructable via exposure_series(...).shift(1)). `close` is the
    underlying price. A held run's gain is derived from actual close-to-close
    returns over the run (ignores the small per-switch slippage cost already
    reflected in strategy_returns -- this is a tax-TIMING model, not a P&L
    re-derivation). A position still open at the final bar is left unrealized
    (no tax), exactly like buy-and-hold's deferral."""
    idx = close.index
    asset_ret = close.pct_change().fillna(0.0)
    equity = pd.Series(index=idx, dtype=float)
    nav = initial
    entry_nav = None
    entry_date = None
    prev_held = 0.0
    for i in range(len(idx)):
        h = held.iloc[i]
        if h > 0 and prev_held == 0.0:                  # entering today
            entry_nav = nav                             # NAV just before today's move
            entry_date = idx[i]
        if h > 0:
            nav *= (1.0 + asset_ret.iloc[i])            # accrue today's return while held
        if prev_held > 0.0 and h == 0.0:                 # exited -- flat as of today
            gain = nav - entry_nav
            hold_days = (idx[i - 1] - entry_date).days if i > 0 else 0
            rate = slovenia_rate_for_holding(hold_days, schedule)
            if gain > 0 and rate > 0:
                nav -= rate * gain
            entry_nav = None
            entry_date = None
        equity.iloc[i] = nav
        prev_held = h
    return equity


def after_tax_core_satellite(core_returns: pd.Series, satellite_returns: pd.Series,
                             core_weight: float, tax_rate: float = SLOVENIA_TAX_RATE,
                             schedule=SLOVENIA_SCHEDULE, initial: float = 100_000.0,
                             cross_offset_losses: bool = False) -> pd.Series:
    """Blend a true buy-and-hold CORE (never sold -> tax-deferred, graduated
    schedule applied at the end) with an actively-traded SATELLITE (taxed
    annually), at a fixed capital split. core_weight is the fraction of
    capital in the untouched core; the rest goes to the satellite.

    cross_offset_losses=False (default): core and satellite are two SEPARATE
    tax lots that never interact -- correct if you want a conservative,
    lot-isolated estimate.

    cross_offset_losses=True: the satellite's LEFTOVER loss carryforward at
    the end of the window (banked but never absorbed by a later satellite
    gain) is applied to shelter part of the CORE's gain at its final sale --
    Slovenian tax nets capital gains/losses across a taxpayer's holdings
    within the same annual filing rather than isolating each lot, so this is
    a real technique, not a modeling convenience. Only matters if the core
    hasn't yet reached its own 0% exemption (see module docstring)."""
    sat_weight = 1.0 - core_weight
    if cross_offset_losses and sat_weight > 0:
        sat_eq, leftover_loss = _after_tax_active_full(satellite_returns, tax_rate,
                                                        initial=initial * sat_weight)
        core_eq = after_tax_buy_hold(core_returns, schedule, initial=initial * core_weight,
                                     extra_loss_shelter=leftover_loss)
    else:
        core_eq = after_tax_buy_hold(core_returns, schedule, initial=initial * core_weight)
        sat_eq = (after_tax_active(satellite_returns, tax_rate, initial=initial * sat_weight)
                  if sat_weight > 0 else None)
    if sat_weight <= 0:
        return core_eq
    return core_eq.add(sat_eq, fill_value=0.0)


def compare_after_tax(strategies: dict, buy_hold_returns: pd.Series,
                      tax_rate: float = SLOVENIA_TAX_RATE) -> dict:
    """strategies: {name: return_series} for actively-traded strategies.
    Returns {name: after_tax_equity} plus 'buy_and_hold' -- everything on the
    same initial capital, ready for compute_metrics()."""
    from bot.backtest_pullback import INITIAL_EQUITY
    out = {name: after_tax_active(r, tax_rate, INITIAL_EQUITY) for name, r in strategies.items()}
    out["buy_and_hold"] = after_tax_buy_hold(buy_hold_returns, initial=INITIAL_EQUITY)
    return out
