"""
bot/taxes.py
============
Tax-scenario helpers for a Slovenian-resident individual.

IMPORTANT: these are research approximations, not tax advice or a tax-lot engine.
The current legal assumptions were rechecked in August 2026 against FURS/PISRS:

* Securities / shares / investment coupons: gains are generally taxed at 25%,
  falling to 20% after five completed years, 15% after ten completed years, and
  exempt after fifteen completed years. Calendar anniversaries are used.
* Eligible capital losses generally offset capital gains in the SAME tax year.
  Generic carry-forward of ordinary securities losses is not modeled; ZDoh-2 has
  a narrow special carry-forward case tied to a particular capital increase.
* Certain losses cannot offset gains when substantially identical replacement
  capital is acquired within 30 days before or after disposal (and some related-
  party replacement cases also apply). This project lacks the lot data needed to
  enforce those rules.
* FURS currently states that an individual's ordinary gains from virtual currencies
  are not subject to income tax when they are not earned in a business activity.
  The separate 25% crypto-gains bill proposed in 2025 remains a proposal as of
  August 2026 and is not represented as enacted law here.

These functions cannot infer FIFO lots, normalized acquisition/disposal expenses,
replacement-capital restrictions, all same-year cross-lot offsets, or whether an
activity constitutes a business. Results are scenario estimates only.
"""

import logging

import pandas as pd

log = logging.getLogger("taxes")

SLOVENIA_TAX_RATE = 0.25
SLOVENIA_SCHEDULE_YEARS = ((5, 0.25), (10, 0.20), (15, 0.15))
SLOVENIA_SCHEDULE = tuple((years * 365, rate) for years, rate in SLOVENIA_SCHEDULE_YEARS)
SLOVENIA_EXEMPT_DAYS = 15 * 365  # compatibility constant, not a legal date cutoff

CRYPTO_TAX_RATE = 0.0
CRYPTO_PROPOSED_TAX_RATE = 0.25
CRYPTO_TAX_PROPOSAL_EFFECTIVE_DATE = pd.Timestamp("2026-01-01", tz="UTC")


def _ts(value):
    return pd.Timestamp(value)


def completed_years(acquired, disposed) -> int:
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
    """Compatibility helper when callers only have a duration, not transaction dates."""
    if hold_days < 0:
        raise ValueError("hold_days cannot be negative")
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
    ts = index[i]
    return ts.month == 12 and ts.day >= 28


def _after_tax_active_full(returns: pd.Series, tax_rate: float = SLOVENIA_TAX_RATE,
                           initial: float = 100_000.0, realize_final: bool = False):
    """Annual-realization stress-test proxy for a high-turnover strategy.

    Positive net NAV change in each completed calendar year is taxed at
    ``tax_rate``. Negative years do not create a generic future-year loss credit.
    """
    if returns.empty:
        return pd.Series(dtype=float), 0.0
    if not 0 <= tax_rate <= 1:
        raise ValueError("tax_rate must be in [0, 1]")
    r = returns.sort_index().astype(float)
    equity = pd.Series(index=r.index, dtype=float)
    nav = basis = float(initial)
    final_same_year_loss = 0.0
    for i, (_, ret) in enumerate(r.items()):
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
    """One acquisition and optional final disposal, using the calendar holding period."""
    if returns.empty:
        return pd.Series(dtype=float)
    r = returns.sort_index().astype(float)
    equity = initial * (1.0 + r).cumprod()
    equity.attrs["initial_equity"] = initial
    if not realize_final:
        return equity
    rate = slovenia_rate_for_dates(equity.index[0], equity.index[-1])
    gain = float(equity.iloc[-1] - initial)
    # Compatibility hook only: callers must independently establish that a loss is
    # legally eligible in the same year. The model itself never carries one forward.
    taxable = max(0.0, gain - max(0.0, extra_loss_shelter))
    if rate > 0 and taxable > 0:
        equity = equity.copy()
        equity.iloc[-1] -= rate * taxable
        equity.attrs["initial_equity"] = initial
    return equity


def after_tax_binary_strategy(returns: pd.Series, held: pd.Series,
                              initial: float = 100_000.0) -> pd.Series:
    """Tax a binary long/cash strategy only when an actual holding run exits.

    Unlike :func:`after_tax_exposure_based`, this accepts the strategy's actual
    return series, so entry/exit slippage, financing and currency conversion already
    present in returns remain in P&L. ``held`` is the lagged 0/1 exposure that
    generated those returns. An open final position remains unrealized.

    Same-year loss netting across different lots and the 30-day replacement rule
    still require tax-lot data and are intentionally not invented here.
    """
    if returns.empty:
        return pd.Series(dtype=float)
    r = returns.sort_index().astype(float)
    h = held.reindex(r.index).fillna(0.0).astype(float)
    if ((h < -1e-12) | (h > 1 + 1e-12)).any():
        raise ValueError("held must stay within [0, 1]")
    equity = pd.Series(index=r.index, dtype=float)
    nav = float(initial)
    entry_nav = None
    entry_date = None
    prev = 0.0
    for i, (ts, ret) in enumerate(r.items()):
        current = float(h.iloc[i])
        if current > 0 and prev <= 0:
            entry_nav = nav
            entry_date = ts
        # Always apply the supplied return: an exit row can contain transaction cost
        # even though current exposure is already zero.
        nav *= 1.0 + ret
        if prev > 0 and current <= 0:
            gain = nav - entry_nav
            rate = slovenia_rate_for_dates(entry_date, ts)
            if gain > 0 and rate > 0:
                nav -= rate * gain
            entry_nav = entry_date = None
        equity.iloc[i] = nav
        prev = current
    equity.attrs["initial_equity"] = initial
    return equity


def after_tax_exposure_based(close: pd.Series, held: pd.Series,
                             schedule=SLOVENIA_SCHEDULE,
                             initial: float = 100_000.0) -> pd.Series:
    """Compatibility price-based binary strategy tax model.

    Prefer :func:`after_tax_binary_strategy` when an actual strategy-return series
    exists because it preserves trading costs and other return adjustments.
    """
    if close.empty:
        return pd.Series(dtype=float)
    h = held.reindex(close.index).fillna(0.0).astype(float)
    returns = h * close.astype(float).pct_change().fillna(0.0)
    return after_tax_binary_strategy(returns, h, initial)


def after_tax_core_satellite(core_returns: pd.Series, satellite_returns: pd.Series,
                             core_weight: float, tax_rate: float = SLOVENIA_TAX_RATE,
                             schedule=SLOVENIA_SCHEDULE, initial: float = 100_000.0,
                             cross_offset_losses: bool = False) -> pd.Series:
    """Fixed split: buy/hold core plus annual-realization satellite stress proxy."""
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
