"""
bot/taxes.py
============
Tax-scenario helpers for a Slovenian-resident individual.

IMPORTANT: these are research approximations, not tax advice or a tax-lot engine.
The current legal assumptions were rechecked in August 2026 against FURS/PISRS:

* Securities / shares / investment coupons: gains are generally taxed at 25%,
  falling to 20% after five completed years, 15% after ten completed years, and
  exempt after fifteen completed years. The models below use calendar
  anniversaries, not ``N * 365`` approximations.
* Losses can generally offset eligible capital gains in the SAME tax year. Generic
  carry-forward of ordinary securities losses is NOT available; ZDoh-2 provides a
  narrow carry-forward exception for a specific capital-increase situation.
* A loss on securities/investment coupons is not eligible for offset when the
  taxpayer acquires substantially the same replacement capital within 30 days
  before or after the disposal (and certain related-party replacement cases also
  apply). This module does not have tax-lot data, so it does not claim to model
  that rule.
* FURS currently states that an individual's ordinary gains from virtual currencies
  are not subject to income tax when they are not earned in the course of a
  business activity. A separate 25% crypto-gains bill proposed in 2025 is still a
  proposal as of August 2026; it is NOT modeled as current law here.

The aggregate-return functions cannot infer FIFO lots, transaction dates, the 1%
normalized acquisition/disposal expenses, replacement-capital restrictions, or
whether trading rises to the level of a business. They should therefore be read as
scenario estimates only. Where the data cannot support a legal conclusion, the
code deliberately avoids inventing one.
"""

import logging
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger("taxes")

SLOVENIA_TAX_RATE = 0.25
SLOVENIA_SCHEDULE_YEARS = ((5, 0.25), (10, 0.20), (15, 0.15))
# Legacy alias retained for callers that import it. Models do NOT use these day
# cutoffs for legal holding-period classification.
SLOVENIA_SCHEDULE = tuple((years * 365, rate) for years, rate in SLOVENIA_SCHEDULE_YEARS)
SLOVENIA_EXEMPT_DAYS = 15 * 365

# Current non-business individual treatment per FURS. The proposed 25% bill is
# retained as an explicit scenario constant, not mislabeled as enacted law.
CRYPTO_TAX_RATE = 0.0
CRYPTO_PROPOSED_TAX_RATE = 0.25
CRYPTO_TAX_PROPOSAL_EFFECTIVE_DATE = pd.Timestamp("2026-01-01", tz="UTC")


def _ts(value):
    return pd.Timestamp(value)


def completed_years(acquired, disposed) -> int:
    """Whole calendar years completed between two timestamps."""
    a, d = _ts(acquired), _ts(disposed)
    if d < a:
        raise ValueError("disposed must not precede acquired")
    years = d.year - a.year
    if (d.month, d.day) < (a.month, a.day):
        years -= 1
    return max(0, years)


def slovenia_rate_for_dates(acquired, disposed) -> float:
    years = completed_years(acquired, disposed)
    if years < 5:
        return 0.25
    if years < 10:
        return 0.20
    if years < 15:
        return 0.15
    return 0.0


def slovenia_rate_for_holding(hold_days: float, schedule=SLOVENIA_SCHEDULE) -> float:
    """Compatibility helper for synthetic tests that only have a duration.

    Real models use :func:`slovenia_rate_for_dates`; a bare day count cannot
    represent calendar anniversaries exactly across leap years.
    """
    if hold_days < 0:
        raise ValueError("hold_days cannot be negative")
    # Use a tropical-year conversion rather than hard legal cutoffs at 365*n.
    years = hold_days / 365.2425
    if years < 5:
        return 0.25
    if years < 10:
        return 0.20
    if years < 15:
        return 0.15
    return 0.0


def _year_end_event(index: pd.DatetimeIndex, i: int, realize_final: bool) -> bool:
    if i + 1 < len(index):
        return index[i + 1].year != index[i].year
    if realize_final:
        return True
    # A sample that naturally finishes at the end of December represents the
    # completed tax year even without a following January observation.
    ts = index[i]
    return ts.month == 12 and ts.day >= 28


def _after_tax_active_full(returns: pd.Series, tax_rate: float = SLOVENIA_TAX_RATE,
                           initial: float = 100_000.0, realize_final: bool = False):
    """Annual-realization STRESS-TEST proxy for a high-turnover strategy.

    Positive net NAV change over each completed calendar year is taxed at
    ``tax_rate``. Negative years do not create a generic carry-forward. This is not
    tax-lot accounting; it intentionally avoids the former, incorrect assumption
    that all securities losses carry indefinitely into later years.

    Returns ``(equity, final_same_year_loss)``. The second item is diagnostic only
    and is not automatically carried into another year.
    """
    if returns.empty:
        return pd.Series(dtype=float), 0.0
    if not 0 <= tax_rate <= 1:
        raise ValueError("tax_rate must be in [0, 1]")
    r = returns.sort_index().astype(float)
    equity = pd.Series(index=r.index, dtype=float)
    nav = float(initial)
    basis = float(initial)
    final_same_year_loss = 0.0
    for i, (ts, ret) in enumerate(r.items()):
        nav *= 1.0 + ret
        equity.iloc[i] = nav
        if _year_end_event(r.index, i, realize_final):
            gain = nav - basis
            if gain > 0:
                nav -= tax_rate * gain
                equity.iloc[i] = nav
                final_same_year_loss = 0.0
            else:
                final_same_year_loss = -gain
            basis = nav
    equity.attrs["initial_equity"] = initial
    return equity, final_same_year_loss


def after_tax_active(returns: pd.Series, tax_rate: float = SLOVENIA_TAX_RATE,
                     initial: float = 100_000.0, realize_final: bool = False) -> pd.Series:
    equity, _ = _after_tax_active_full(returns, tax_rate, initial, realize_final)
    return equity


def after_tax_buy_hold(returns: pd.Series, schedule=SLOVENIA_SCHEDULE,
                       initial: float = 100_000.0, extra_loss_shelter: float = 0.0,
                       realize_final: bool = True) -> pd.Series:
    """Single acquisition and optional final disposal.

    ``extra_loss_shelter`` is kept for API compatibility but should only be used
    for a legally eligible loss realized in the SAME tax year. The function does
    not manufacture or carry such losses itself.
    """
    if returns.empty:
        return pd.Series(dtype=float)
    r = returns.sort_index().astype(float)
    equity = initial * (1.0 + r).cumprod()
    equity.attrs["initial_equity"] = initial
    if not realize_final:
        return equity
    rate = slovenia_rate_for_dates(equity.index[0], equity.index[-1])
    gain = float(equity.iloc[-1] - initial)
    taxable = max(0.0, gain - max(0.0, extra_loss_shelter))
    if rate > 0 and taxable > 0:
        equity = equity.copy()
        equity.iloc[-1] -= rate * taxable
        equity.attrs["initial_equity"] = initial
    return equity


def after_tax_exposure_based(close: pd.Series, held: pd.Series,
                             schedule=SLOVENIA_SCHEDULE,
                             initial: float = 100_000.0) -> pd.Series:
    """Tax-at-exit scenario for a binary long/cash strategy.

    Each contiguous holding run is treated as one lot and taxed at its actual
    calendar holding period when it exits. Open final positions remain unrealized.
    Loss offsets, FIFO across multiple lots, 1% normalized expenses and the 30-day
    replacement-capital rule are outside what ``close`` + ``held`` can establish.
    """
    if close.empty:
        return pd.Series(dtype=float)
    h = held.reindex(close.index).fillna(0.0).astype(float)
    asset_ret = close.astype(float).pct_change().fillna(0.0)
    equity = pd.Series(index=close.index, dtype=float)
    nav = float(initial)
    entry_nav = None
    entry_date = None
    prev_held = 0.0
    for i, ts in enumerate(close.index):
        current = h.iloc[i]
        if current > 0 and prev_held <= 0:
            entry_nav = nav
            entry_date = ts
        if current > 0:
            nav *= 1.0 + asset_ret.iloc[i]
        if prev_held > 0 and current <= 0:
            gain = nav - entry_nav
            dispose_date = close.index[i - 1]
            rate = slovenia_rate_for_dates(entry_date, dispose_date)
            if gain > 0 and rate > 0:
                nav -= rate * gain
            entry_nav = entry_date = None
        equity.iloc[i] = nav
        prev_held = current
    equity.attrs["initial_equity"] = initial
    return equity


def after_tax_core_satellite(core_returns: pd.Series, satellite_returns: pd.Series,
                             core_weight: float, tax_rate: float = SLOVENIA_TAX_RATE,
                             schedule=SLOVENIA_SCHEDULE, initial: float = 100_000.0,
                             cross_offset_losses: bool = False) -> pd.Series:
    """Fixed capital split: buy/hold core + annual-realization satellite proxy.

    The former implementation carried satellite losses across tax years and used
    them against the eventual core sale. That is not a general Slovenian rule.
    ``cross_offset_losses`` is therefore a conservative no-op for aggregate return
    data: without tax-lot/disposal records we cannot prove a same-year loss is
    legally eligible to offset the core sale.
    """
    if not 0 <= core_weight <= 1:
        raise ValueError("core_weight must be in [0, 1]")
    if cross_offset_losses:
        log.warning("cross_offset_losses ignored: aggregate returns cannot prove an eligible "
                    "same-year realized loss; generic loss carry-forward is not modeled.")
    sat_weight = 1.0 - core_weight
    core_eq = after_tax_buy_hold(core_returns, schedule, initial * core_weight)
    if sat_weight <= 0:
        return core_eq
    sat_eq = after_tax_active(satellite_returns, tax_rate, initial * sat_weight)
    if core_weight <= 0:
        return sat_eq
    out = core_eq.add(sat_eq, fill_value=0.0)
    out.attrs["initial_equity"] = initial
    return out


def compare_after_tax(strategies: dict, buy_hold_returns: pd.Series,
                      tax_rate: float = SLOVENIA_TAX_RATE) -> dict:
    from bot.backtest_pullback import INITIAL_EQUITY
    out = {name: after_tax_active(r, tax_rate, INITIAL_EQUITY)
           for name, r in strategies.items()}
    out["buy_and_hold"] = after_tax_buy_hold(buy_hold_returns, initial=INITIAL_EQUITY)
    return out
