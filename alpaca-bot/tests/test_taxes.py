"""
Offline tests for the Slovenian capital-gains tax model (bot/taxes.py). No
network.

Run:  pytest tests/test_taxes.py    (or: python tests/test_taxes.py)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.taxes as tx


def _daily_returns(vals, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(vals), freq="D")
    return pd.Series(vals, index=idx)


# --------------------------------------------------------------------------- #
# slovenia_rate_for_holding (the real graduated schedule)
# --------------------------------------------------------------------------- #
def test_slovenia_rate_top_bracket_under_5_years():
    assert tx.slovenia_rate_for_holding(1) == 0.25
    assert tx.slovenia_rate_for_holding(5 * 365) == 0.25


def test_slovenia_rate_5_to_10_years():
    assert tx.slovenia_rate_for_holding(5 * 365 + 1) == 0.20
    assert tx.slovenia_rate_for_holding(10 * 365) == 0.20


def test_slovenia_rate_10_to_15_years():
    assert tx.slovenia_rate_for_holding(10 * 365 + 1) == 0.15
    assert tx.slovenia_rate_for_holding(15 * 365) == 0.15


def test_slovenia_rate_exempt_past_15_years():
    assert tx.slovenia_rate_for_holding(15 * 365 + 1) == 0.0
    assert tx.slovenia_rate_for_holding(30 * 365) == 0.0


# --------------------------------------------------------------------------- #
# after_tax_active
# --------------------------------------------------------------------------- #
def test_after_tax_active_taxes_a_single_year_gain():
    # A full calendar year (Jan 1 - Dec 31) with a flat +10% total return.
    idx = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    daily_r = 1.10 ** (1 / len(idx)) - 1
    r = pd.Series(daily_r, index=idx)
    eq = tx.after_tax_active(r, tax_rate=0.25, initial=100_000.0)
    gross_gain = 100_000.0 * 0.10
    expected_final = 100_000.0 + gross_gain - 0.25 * gross_gain
    assert abs(eq.iloc[-1] - expected_final) < 1.0


def test_after_tax_active_no_tax_on_a_losing_year():
    idx = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    daily_r = 0.90 ** (1 / len(idx)) - 1     # -10% for the year
    r = pd.Series(daily_r, index=idx)
    eq = tx.after_tax_active(r, tax_rate=0.25, initial=100_000.0)
    assert abs(eq.iloc[-1] - 90_000.0) < 1.0     # no tax on a loss


def test_after_tax_active_carries_losses_forward():
    # Year 1: -20%. Year 2: +20% (nominal) -- should offset against the carried
    # loss, so year 2's tax is on (gain - carried_loss), not the full gain.
    idx1 = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    idx2 = pd.date_range("2021-01-01", "2021-12-31", freq="D")
    r1 = pd.Series(0.80 ** (1 / len(idx1)) - 1, index=idx1)
    r2 = pd.Series(1.20 ** (1 / len(idx2)) - 1, index=idx2)
    r = pd.concat([r1, r2])
    eq = tx.after_tax_active(r, tax_rate=0.25, initial=100_000.0)
    # After year 1: 80,000 (loss, no tax). Year 2 nominal gain = 80,000*0.20=16,000.
    # Carried loss = 20,000 > 16,000 gain -> fully offset, NO tax due this year.
    expected_year2_end = 80_000.0 * 1.20
    assert abs(eq.iloc[-1] - expected_year2_end) < 1.0


def test_after_tax_active_partial_loss_offset():
    # Year 1: -10,000 loss on 100k (-10%). Year 2: +30,000 nominal gain on 90k
    # (+33.33%). Carried loss (10,000) offsets part of the gain; tax applies to
    # the remaining (30,000 - 10,000) = 20,000.
    idx1 = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    idx2 = pd.date_range("2021-01-01", "2021-12-31", freq="D")
    r1 = pd.Series(0.90 ** (1 / len(idx1)) - 1, index=idx1)
    total_growth_yr2 = 120_000.0 / 90_000.0
    r2 = pd.Series(total_growth_yr2 ** (1 / len(idx2)) - 1, index=idx2)
    r = pd.concat([r1, r2])
    eq = tx.after_tax_active(r, tax_rate=0.25, initial=100_000.0)
    nav_before_tax_yr2 = 120_000.0
    taxable = nav_before_tax_yr2 - 90_000.0 - 10_000.0     # gain minus carried loss
    expected = nav_before_tax_yr2 - 0.25 * taxable
    assert abs(eq.iloc[-1] - expected) < 1.0


def test_after_tax_active_empty_series():
    assert tx.after_tax_active(pd.Series(dtype=float)).empty


# --------------------------------------------------------------------------- #
# after_tax_buy_hold
# --------------------------------------------------------------------------- #
def test_after_tax_buy_hold_taxed_at_top_rate_under_5_years():
    idx = pd.date_range("2020-01-01", periods=400, freq="D")   # ~1 year, under 5y
    growth = 1.5
    r = pd.Series(growth ** (1 / len(idx)) - 1, index=idx)
    eq = tx.after_tax_buy_hold(r, initial=100_000.0)
    gross_final = 100_000.0 * growth
    expected = gross_final - 0.25 * (gross_final - 100_000.0)
    assert abs(eq.iloc[-1] - expected) < 1.0
    # No tax deducted anywhere except the final bar.
    pre_tax = 100_000.0 * (1 + r).cumprod()
    assert np.allclose(eq.iloc[:-1].values, pre_tax.iloc[:-1].values)


def test_after_tax_buy_hold_discounted_rate_for_5_to_10_year_hold():
    idx = pd.date_range("2010-01-01", periods=7 * 365, freq="D")   # 7 years
    growth = 1.5
    r = pd.Series(growth ** (1 / len(idx)) - 1, index=idx)
    eq = tx.after_tax_buy_hold(r, initial=100_000.0)
    gross_final = 100_000.0 * growth
    expected = gross_final - 0.20 * (gross_final - 100_000.0)     # 5-10y bracket = 20%
    assert abs(eq.iloc[-1] - expected) < 1.0


def test_after_tax_buy_hold_discounted_rate_for_10_to_15_year_hold():
    idx = pd.date_range("2010-01-01", periods=12 * 365, freq="D")   # 12 years
    growth = 2.0
    r = pd.Series(growth ** (1 / len(idx)) - 1, index=idx)
    eq = tx.after_tax_buy_hold(r, initial=100_000.0)
    gross_final = 100_000.0 * growth
    expected = gross_final - 0.15 * (gross_final - 100_000.0)      # 10-15y bracket = 15%
    assert abs(eq.iloc[-1] - expected) < 1.0


def test_after_tax_buy_hold_exempt_past_threshold():
    idx = pd.date_range("2005-01-01", periods=16 * 365, freq="D")   # > 15 years
    r = pd.Series(0.0002, index=idx)
    eq = tx.after_tax_buy_hold(r, initial=100_000.0)
    gross = 100_000.0 * (1.0002) ** len(idx)
    assert abs(eq.iloc[-1] - gross) < 1.0     # fully exempt -- no tax deducted


def test_after_tax_buy_hold_no_tax_on_a_net_loss():
    idx = pd.date_range("2020-01-01", periods=400, freq="D")
    r = pd.Series(0.8 ** (1 / len(idx)) - 1, index=idx)   # net loss
    eq = tx.after_tax_buy_hold(r, initial=100_000.0)
    gross = 100_000.0 * 0.8
    assert abs(eq.iloc[-1] - gross) < 1.0     # losses aren't taxed


def test_after_tax_buy_hold_empty_series():
    assert tx.after_tax_buy_hold(pd.Series(dtype=float)).empty


# --------------------------------------------------------------------------- #
# after_tax_exposure_based
# --------------------------------------------------------------------------- #
def test_exposure_based_matches_raw_growth_for_one_uninterrupted_holding():
    # Held for the entire series (never exits) -> fully unrealized, no tax at all
    # (matches buy-and-hold's deferral for an open position).
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    close = pd.Series(100 * 1.001 ** np.arange(100), index=idx)
    held = pd.Series(1.0, index=idx)
    eq = tx.after_tax_exposure_based(close, held, initial=100_000.0)
    expected = 100_000.0 * (close / close.iloc[0])
    assert np.allclose(eq.values, expected.values, rtol=1e-9)


def test_exposure_based_taxes_only_at_exit_not_during_holding():
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    close = pd.Series(np.linspace(100, 150, 20), index=idx)   # steady gain
    held = pd.Series([0.0] * 2 + [1.0] * 10 + [0.0] * 8, index=idx)   # hold days 2-11, exit day 12
    eq = tx.after_tax_exposure_based(close, held, initial=100_000.0)
    # No tax deducted WHILE held -- equity should track the raw price ratio
    # exactly during the holding window (entry at day index 2's close level).
    entry_nav = eq.iloc[1]                      # NAV the day before entry (flat, unchanged)
    for i in range(2, 11):                      # while still held (before exit day)
        expected_untaxed = entry_nav * (close.iloc[i] / close.iloc[1])
        assert abs(eq.iloc[i] - expected_untaxed) < 1e-6
    # After exit (day 12 onward), tax should have been deducted exactly once,
    # at the top (under-5y) rate since this trade lasted only 10 days.
    pre_tax_exit_nav = entry_nav * (close.iloc[11] / close.iloc[1])
    gain = pre_tax_exit_nav - entry_nav
    expected_after_tax = pre_tax_exit_nav - 0.25 * gain
    assert abs(eq.iloc[12] - expected_after_tax) < 1e-6
    assert abs(eq.iloc[-1] - expected_after_tax) < 1e-6   # stays flat/untaxed after, in cash


def test_exposure_based_no_tax_on_a_losing_trade():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    close = pd.Series(np.linspace(100, 80, 10), index=idx)     # steady decline
    held = pd.Series([0.0] + [1.0] * 5 + [0.0] * 4, index=idx)
    eq = tx.after_tax_exposure_based(close, held, initial=100_000.0)
    entry_nav = eq.iloc[0]
    exit_nav = entry_nav * (close.iloc[5] / close.iloc[0])
    assert abs(eq.iloc[6] - exit_nav) < 1e-6      # no tax deducted on the loss


def test_exposure_based_flat_the_whole_time_stays_flat():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    close = pd.Series(np.linspace(100, 120, 10), index=idx)
    held = pd.Series(0.0, index=idx)
    eq = tx.after_tax_exposure_based(close, held, initial=100_000.0)
    assert (eq == 100_000.0).all()


def test_exposure_based_multiple_trades_each_taxed_independently():
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    close = pd.Series(100 + np.concatenate([np.linspace(0, 10, 10),   # trade 1: +gain
                                            np.linspace(10, 10, 5),   # flat/cash
                                            np.linspace(10, 5, 10),   # trade 2: -loss
                                            np.linspace(5, 5, 5)]), index=idx)
    held = pd.Series([1.0] * 10 + [0.0] * 5 + [1.0] * 10 + [0.0] * 5, index=idx)
    eq = tx.after_tax_exposure_based(close, held, initial=100_000.0)
    # Trade 1 gained -> taxed; trade 2 lost -> not taxed. Final NAV should be
    # strictly below the no-tax equivalent (some tax was paid on trade 1).
    no_tax_eq = tx.after_tax_exposure_based(close, held, schedule=((10 ** 9, 0.0),),
                                            initial=100_000.0)
    assert eq.iloc[-1] < no_tax_eq.iloc[-1]


def test_exposure_based_gets_discounted_rate_for_a_multi_year_trade():
    idx = pd.date_range("2010-01-01", periods=8 * 365, freq="D")   # 8-year holding trade
    close = pd.Series(100 * 1.0002 ** np.arange(len(idx)), index=idx)
    held = pd.Series(1.0, index=idx)
    held.iloc[-1] = 0.0     # exit on the very last day
    eq = tx.after_tax_exposure_based(close, held, initial=100_000.0)
    entry_nav = 100_000.0
    pre_tax_exit_nav = entry_nav * (close.iloc[-2] / close.iloc[0])
    gain = pre_tax_exit_nav - entry_nav
    expected = pre_tax_exit_nav - 0.20 * gain     # 8 years -> 5-10y bracket = 20%, not 25%
    assert abs(eq.iloc[-1] - expected) < 1.0


# --------------------------------------------------------------------------- #
# after_tax_core_satellite
# --------------------------------------------------------------------------- #
def test_core_satellite_all_core_matches_pure_buy_hold():
    idx = pd.date_range("2005-01-01", periods=20 * 365, freq="D")   # 20y, past exemption
    core_r = pd.Series(0.0002, index=idx)
    sat_r = pd.Series(0.0005, index=idx)
    combined = tx.after_tax_core_satellite(core_r, sat_r, core_weight=1.0)
    pure_bh = tx.after_tax_buy_hold(core_r)
    assert np.allclose(combined.values, pure_bh.values)


def test_core_satellite_all_satellite_matches_pure_active():
    idx = pd.date_range("2020-01-01", periods=400, freq="D")
    core_r = pd.Series(0.0002, index=idx)
    sat_r = pd.Series(0.0008, index=idx)
    combined = tx.after_tax_core_satellite(core_r, sat_r, core_weight=0.0)
    pure_active = tx.after_tax_active(sat_r)
    assert np.allclose(combined.values, pure_active.values)


def test_core_satellite_splits_capital_proportionally():
    idx = pd.date_range("2005-01-01", periods=20 * 365, freq="D")
    core_r = pd.Series(0.0002, index=idx)     # never sold -> untaxed
    sat_r = pd.Series(0.0, index=idx)          # flat -> no gain/loss/tax either way
    combined = tx.after_tax_core_satellite(core_r, sat_r, core_weight=0.7, initial=100_000.0)
    core_only = tx.after_tax_buy_hold(core_r, initial=70_000.0)
    # Satellite is flat, so combined should equal the core's contribution + the
    # untouched 30k satellite principal.
    assert np.allclose(combined.values, (core_only + 30_000.0).values, rtol=1e-9)


def test_core_satellite_cross_offset_matches_isolated_when_satellite_has_no_leftover_loss():
    # Satellite ends UP overall (no leftover loss carryforward) -> cross-offset
    # has nothing to shelter the core with, so it should match the isolated
    # (default) model exactly.
    idx = pd.date_range("2010-01-01", periods=8 * 365, freq="D")   # 8y, core not yet exempt
    core_r = pd.Series(0.0003, index=idx)
    sat_r = pd.Series(0.0006, index=idx)     # steadily positive -> ends with no leftover loss
    isolated = tx.after_tax_core_satellite(core_r, sat_r, core_weight=0.7)
    crossed = tx.after_tax_core_satellite(core_r, sat_r, core_weight=0.7,
                                          cross_offset_losses=True)
    assert np.allclose(isolated.values, crossed.values)


def test_core_satellite_cross_offset_shelters_core_gain_with_leftover_satellite_loss():
    # Satellite: a big loss in year 1 that's never fully offset by later
    # satellite gains (small later gain) -> leftover loss carryforward at the
    # end. Core: a real gain, held only 8 years (still taxable, 20% bracket).
    # Cross-offset should leave MORE final equity than the isolated model,
    # since the leftover satellite loss shelters part of the core's gain.
    idx = pd.date_range("2010-01-01", periods=8 * 365, freq="D")
    core_r = pd.Series(0.0004, index=idx)
    idx_y1 = pd.date_range("2010-01-01", "2010-12-31", freq="D")
    idx_rest = pd.date_range("2011-01-01", periods=len(idx) - len(idx_y1), freq="D")
    sat_r = pd.concat([pd.Series(0.60 ** (1 / len(idx_y1)) - 1, index=idx_y1),      # -40% year 1
                       pd.Series(0.0001, index=idx_rest)])                         # flat-ish after
    isolated = tx.after_tax_core_satellite(core_r, sat_r, core_weight=0.7)
    crossed = tx.after_tax_core_satellite(core_r, sat_r, core_weight=0.7,
                                          cross_offset_losses=True)
    assert crossed.iloc[-1] > isolated.iloc[-1] + 1.0


def test_core_satellite_cross_offset_no_benefit_once_core_is_fully_exempt():
    # Same setup as above, but held past the 15-year exemption -- the core
    # already owes 0% tax, so there's nothing left for a leftover loss to
    # shelter; cross-offset should match the isolated model exactly.
    idx = pd.date_range("2000-01-01", periods=16 * 365, freq="D")
    core_r = pd.Series(0.0002, index=idx)
    idx_y1 = pd.date_range("2000-01-01", "2000-12-31", freq="D")
    idx_rest = pd.date_range("2001-01-01", periods=len(idx) - len(idx_y1), freq="D")
    sat_r = pd.concat([pd.Series(0.60 ** (1 / len(idx_y1)) - 1, index=idx_y1),
                       pd.Series(0.0001, index=idx_rest)])
    isolated = tx.after_tax_core_satellite(core_r, sat_r, core_weight=0.7)
    crossed = tx.after_tax_core_satellite(core_r, sat_r, core_weight=0.7,
                                          cross_offset_losses=True)
    assert np.allclose(isolated.values, crossed.values)


# --------------------------------------------------------------------------- #
# compare_after_tax
# --------------------------------------------------------------------------- #
def test_compare_after_tax_includes_all_strategies_and_buy_hold():
    idx = pd.date_range("2020-01-01", periods=400, freq="D")
    r_a = pd.Series(0.0005, index=idx)
    r_bh = pd.Series(0.0004, index=idx)
    out = tx.compare_after_tax({"strat_a": r_a}, r_bh)
    assert set(out) == {"strat_a", "buy_and_hold"}
    assert len(out["strat_a"]) == len(idx) and len(out["buy_and_hold"]) == len(idx)


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        fn()
        print(f"{name} OK")
    print(f"\nALL {len(fns)} TAX TESTS PASSED")
