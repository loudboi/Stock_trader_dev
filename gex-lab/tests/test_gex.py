"""Offline tests for GEX math and model assumptions."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gex_lab import gex


def _chain(calls, puts, iv=0.3, T=0.1):
    rows = [(k, "C", oi, iv, T) for k, oi in calls.items()]
    rows += [(k, "P", oi, iv, T) for k, oi in puts.items()]
    return pd.DataFrame(rows, columns=["strike", "kind", "oi", "iv", "T"])


def test_bs_gamma_positive_peaks_near_atm_and_accepts_q():
    atm = gex.bs_gamma(100, 100, 0.1, 0.3, r=0.03, q=0.01)
    otm = gex.bs_gamma(100, 130, 0.1, 0.3, r=0.03, q=0.01)
    assert atm > 0 and atm > otm
    assert gex.bs_gamma(100, 100, 0, 0.3) == 0.0
    assert gex.bs_gamma(100, 100, 0.1, 0) == 0.0
    # Dividend yield is an explicit model input, not ignored.
    assert not np.isclose(gex.bs_gamma(100, 100, 1.0, 0.3, q=0.0),
                          gex.bs_gamma(100, 100, 1.0, 0.3, q=0.08))


def test_net_gex_sign_convention_is_explicit():
    chain = _chain({110: 1000}, {90: 1000})
    by = gex.net_gex_by_strike(chain, 100)
    assert by.loc[110] > 0 and by.loc[90] < 0
    with pytest.raises(ValueError):
        gex._sign("X")


def test_gamma_flip_between_put_and_call_mass():
    chain = _chain({105: 2000, 110: 2000}, {90: 2000, 95: 2000})
    flip = gex.gamma_flip(chain, 100, r=0.02, q=0.01)
    assert flip is not None and 90 < flip < 110


def test_center_of_mass_weights_by_oi():
    chain = _chain({100: 100, 120: 900}, {80: 900, 100: 100})
    assert abs(gex.center_of_mass(chain, "C") - 118) < 1e-6
    assert abs(gex.center_of_mass(chain, "P") - 82) < 1e-6


def test_compute_levels_records_assumptions():
    chain = _chain({105: 1500, 110: 2500, 120: 800}, {80: 900, 90: 2000, 95: 1500})
    lv = gex.compute_levels(chain, 100, r=0.025, q=0.012)
    assert lv["pos_gex"] > 100 and lv["cotmp"] < 100
    assert lv["ptrans"] == lv["zero_gex"]
    assert lv["risk_free"] == 0.025
    assert lv["dividend_yield"] == 0.012
    assert lv["dealer_sign_model"] == "calls_positive_puts_negative"


def test_empty_chain_safe_and_bad_inputs_fail():
    empty = pd.DataFrame(columns=["strike", "kind", "oi", "iv", "T"])
    assert gex.net_gex_by_strike(empty, 100).empty
    assert gex.gamma_flip(empty, 100) is None
    lv = gex.compute_levels(empty, 100)
    assert lv["pos_gex"] is None and lv["spot"] == 100.0
    with pytest.raises(ValueError):
        gex.compute_levels(empty, 0)
    with pytest.raises(ValueError):
        gex.gamma_flip(_chain({100: 1}, {90: 1}), 100, lo=2, hi=1)
