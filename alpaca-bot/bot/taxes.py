"""
bot/taxes.py
============
Models Slovenian capital-gains tax on trading, because it changes the entire
"beat buy-and-hold" question. Every backtest in this project so far has been
PRE-TAX. Slovenia's capital-gains tax on securities is a massive structural
advantage for true buy-and-hold: 25% on realized gains by default, decreasing
with holding period, reaching 0% once a position has been held more than 15
years (i.e. from day 15*365+1). [User-provided figures for the current schedule;
if Slovenia's intermediate brackets (commonly cited historically as declining
step-downs at 5/10/15-year marks) differ from a flat 25% below the exemption,
adjust SLOVENIA_TAX_RATE / the schedule below accordingly — see the honest
simplification note.]

HONEST SIMPLIFICATION: none of the strategies in this project hold individual
positions anywhere near 15 years (the longest-lived, trend_exposure, flips
positions every few months to a few years), so the exact intermediate brackets
between 0 and 15 years don't matter for them — they all land in the "not yet
exempt" 25% bucket. The one case where the schedule's shape WOULD matter is a
literal, never-touched buy-and-hold position, which is exactly the case handled
separately below (after_tax_buy_hold): it defers all tax to the final sale, so
if the total holding period exceeds the exemption threshold, it owes 0%.

Two models, because the tax mechanics differ fundamentally by trading style:

  after_tax_active()    For a strategy that rebalances/trades frequently (which
                        is every strategy in this project except literal
                        buy-and-hold). Approximates tax as an ANNUAL realization
                        event: at each calendar year-end, tax 25% of that year's
                        NET gain (offsetting any carried-forward losses from
                        prior years — capital losses on securities carry forward
                        in most EU tax systems, Slovenia included). This is a
                        simplification of exact per-trade tax-lot accounting
                        (which would need every individual trade's cost basis
                        and holding period), but it captures the economically
                        dominant effect: active/systematic strategies realize
                        gains constantly and pay the short-term rate on
                        virtually all of them, every year, compounding the drag.

  after_tax_buy_hold()  For a position bought once and never touched during the
                        backtest window. No tax is due until the FINAL sale (the
                        end of the backtest), at which point the schedule is
                        applied to the total gain based on the total holding
                        period — 0% if held past the exemption threshold, which
                        this project's ~20-year backtest windows comfortably
                        exceed.

Both take a return series and produce an EQUITY curve (not a return series,
since a tax payment is a discrete capital deduction, not a percentage return) so
they plug directly into bot.backtest_pullback.compute_metrics for an honest
apples-to-apples AFTER-TAX Sharpe/return/drawdown comparison.
"""

import logging

import pandas as pd

log = logging.getLogger("taxes")

SLOVENIA_TAX_RATE = 0.25
SLOVENIA_EXEMPT_DAYS = 15 * 365 + 1     # "15 years and a day" per the user


def after_tax_active(returns: pd.Series, tax_rate: float = SLOVENIA_TAX_RATE,
                     initial: float = 100_000.0) -> pd.Series:
    """Annual realize-and-tax simulation with loss carryforward, for a
    frequently-rebalanced/traded strategy. See module docstring for the
    approximation this makes."""
    if returns.empty:
        return pd.Series(dtype=float)
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
    return equity


def after_tax_buy_hold(returns: pd.Series, tax_rate: float = SLOVENIA_TAX_RATE,
                       exempt_days: int = SLOVENIA_EXEMPT_DAYS,
                       initial: float = 100_000.0) -> pd.Series:
    """A position bought once at the start and sold once at the end (or never
    sold, if you'd hold past the backtest window) -- taxed once, at the end,
    based on total gain and total holding period."""
    if returns.empty:
        return pd.Series(dtype=float)
    equity = initial * (1.0 + returns).cumprod()
    hold_days = (equity.index[-1] - equity.index[0]).days
    if hold_days > exempt_days:
        return equity                     # exempt -- no tax due, ever
    gain = equity.iloc[-1] - initial
    if gain > 0:
        equity = equity.copy()
        equity.iloc[-1] -= tax_rate * gain
    return equity


def after_tax_exposure_based(close: pd.Series, held: pd.Series,
                             tax_rate: float = SLOVENIA_TAX_RATE,
                             exempt_days: int = SLOVENIA_EXEMPT_DAYS,
                             initial: float = 100_000.0) -> pd.Series:
    """A MORE ACCURATE tax model for binary in/out exposure strategies (e.g.
    bot.trend_exposure), which after_tax_active over-penalizes: that blanket
    model taxes the WHOLE year's return annually regardless of whether anything
    was actually sold, which is right for a daily-rebalanced book (min_var/
    inverse_vol/erc genuinely churn positions constantly) but wrong for a
    strategy that might hold one uninterrupted position for 1-3+ years without
    a single trade. Here, tax is only realized at the actual EXIT of each
    contiguous holding period, based on that specific trade's real holding
    period -- an unrealized (still-open) position accrues NO tax drag, exactly
    like buy-and-hold's deferral.

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
            if gain > 0 and hold_days <= exempt_days:
                nav -= tax_rate * gain
            entry_nav = None
            entry_date = None
        equity.iloc[i] = nav
        prev_held = h
    return equity


def after_tax_core_satellite(core_returns: pd.Series, satellite_returns: pd.Series,
                             core_weight: float, tax_rate: float = SLOVENIA_TAX_RATE,
                             exempt_days: int = SLOVENIA_EXEMPT_DAYS,
                             initial: float = 100_000.0) -> pd.Series:
    """Blend a true buy-and-hold CORE (never sold -> tax-deferred, 0% if held past
    exempt_days) with an actively-traded SATELLITE (taxed annually via
    after_tax_active), at a fixed capital split -- modeled as two SEPARATE tax
    lots (the core's deferred gain is never mixed with the satellite's annual
    realizations, which is the correct tax treatment for genuinely distinct
    holdings). core_weight is the fraction of capital in the untouched core;
    the rest goes to the satellite."""
    core_eq = after_tax_buy_hold(core_returns, tax_rate, exempt_days,
                                 initial=initial * core_weight)
    sat_weight = 1.0 - core_weight
    if sat_weight <= 0:
        return core_eq
    sat_eq = after_tax_active(satellite_returns, tax_rate, initial=initial * sat_weight)
    return core_eq.add(sat_eq, fill_value=0.0)


def compare_after_tax(strategies: dict, buy_hold_returns: pd.Series,
                      tax_rate: float = SLOVENIA_TAX_RATE) -> dict:
    """strategies: {name: return_series} for actively-traded strategies.
    Returns {name: after_tax_equity} plus 'buy_and_hold' -- everything on the
    same initial capital, ready for compute_metrics()."""
    from bot.backtest_pullback import INITIAL_EQUITY
    out = {name: after_tax_active(r, tax_rate, INITIAL_EQUITY) for name, r in strategies.items()}
    out["buy_and_hold"] = after_tax_buy_hold(buy_hold_returns, tax_rate, initial=INITIAL_EQUITY)
    return out
