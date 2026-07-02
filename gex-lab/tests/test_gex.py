"""Offline tests for the GEX math (gex_lab/gex.py). Synthetic chains, no network."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gex_lab import gex


def _chain(calls, puts, iv=0.3, T=0.1):
    """calls/puts: {strike: oi}. Build a chain DataFrame."""
    rows = [(k, "C", oi, iv, T) for k, oi in calls.items()]
    rows += [(k, "P", oi, iv, T) for k, oi in puts.items()]
    return pd.DataFrame(rows, columns=["strike", "kind", "oi", "iv", "T"])


def test_bs_gamma_positive_and_peaks_atm():
    atm = gex.bs_gamma(100, 100, 0.1, 0.3)
    otm = gex.bs_gamma(100, 130, 0.1, 0.3)
    assert atm > 0 and atm > otm
    assert gex.bs_gamma(100, 100, 0, 0.3) == 0.0        # expired -> 0
    assert gex.bs_gamma(100, 100, 0.1, 0) == 0.0        # no vol -> 0


def test_net_gex_calls_positive_puts_negative():
    chain = _chain({110: 1000}, {90: 1000})
    by = gex.net_gex_by_strike(chain, spot=100)
    assert by.loc[110] > 0 and by.loc[90] < 0


def test_gamma_flip_between_put_and_call_mass():
    # Heavy puts below, heavy calls above -> flip somewhere in between, near spot.
    chain = _chain({105: 2000, 110: 2000}, {90: 2000, 95: 2000})
    flip = gex.gamma_flip(chain, spot=100)
    assert flip is not None and 90 < flip < 110


def test_center_of_mass_weights_by_oi():
    chain = _chain({100: 100, 120: 900}, {80: 900, 100: 100})
    assert abs(gex.center_of_mass(chain, "C") - 118.0) < 1e-6     # pulled toward 120
    assert abs(gex.center_of_mass(chain, "P") - 82.0) < 1e-6      # pulled toward 80


def test_compute_levels_shape_and_positions():
    chain = _chain({105: 1500, 110: 2500, 120: 800}, {80: 900, 90: 2000, 95: 1500})
    lv = gex.compute_levels(chain, spot=100)
    assert lv["pos_gex"] is not None and lv["pos_gex"] > 100     # +GEX target above spot
    assert lv["cotmp"] is not None and lv["cotmp"] < 100          # put mass below spot
    assert lv["zero_gex"] is not None
    assert lv["ptrans"] == lv["zero_gex"]                         # pTrans ~ flip (documented)
    for k in ("spot", "net_gex", "pos_gex", "zero_gex", "cotmp", "cotmc",
              "put_wall", "ptrans", "ntrans"):
        assert k in lv


def test_empty_chain_is_safe():
    empty = pd.DataFrame(columns=["strike", "kind", "oi", "iv", "T"])
    assert gex.net_gex_by_strike(empty, 100).empty
    assert gex.gamma_flip(empty, 100) is None
    lv = gex.compute_levels(empty, 100)
    assert lv["pos_gex"] is None and lv["spot"] == 100.0


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        fn()
        print(f"{name} OK")
    print(f"\nALL {len(fns)} GEX TESTS PASSED")
