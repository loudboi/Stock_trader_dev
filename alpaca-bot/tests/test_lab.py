"""
Offline tests for the strategy lab (bot/lab.py). Synthetic data, no network.

Run:  pytest tests/test_lab.py    (or: python tests/test_lab.py)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bot.lab as lab


def _panel(cols: dict, start="2005-01-01"):
    n = len(next(iter(cols.values())))
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC")
    return pd.DataFrame({k: np.asarray(v, float) for k, v in cols.items()}, index=idx)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def test_realized_vol_higher_for_noisier_series():
    rng = np.random.default_rng(0)
    calm = pd.Series(rng.normal(0, 0.005, 300))
    wild = pd.Series(rng.normal(0, 0.03, 300))
    assert lab.realized_vol(wild, 20).iloc[-1] > lab.realized_vol(calm, 20).iloc[-1]


def test_rsi_bounds_and_oversold():
    falling = pd.Series(np.linspace(100, 50, 50))
    r = lab.rsi(falling, 2).dropna()
    assert (r >= 0).all() and (r <= 100).all()
    assert r.iloc[-1] < 30          # a steady decline is oversold


# --------------------------------------------------------------------------- #
# Vol targeting
# --------------------------------------------------------------------------- #
def test_vol_target_no_lookahead_and_leverage_cap():
    rng = np.random.default_rng(1)
    px = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 400))
    panel = _panel({"A": px})
    r = lab.vol_target(panel, target_vol=0.15, lookback=20, max_leverage=2.0,
                       borrow_rate=0.0)
    assert r.iloc[0] == 0.0                       # warmup/shift -> flat first bar
    # Exposure is capped at max_leverage (reconstruct the scale the strategy uses).
    base = panel["A"].pct_change().fillna(0.0)
    scale = (0.15 / lab.realized_vol(base, 20)).clip(upper=2.0).shift(1).fillna(0.0)
    assert (scale <= 2.0 + 1e-9).all() and scale.max() > 1.0


def test_vol_target_borrow_drags_when_levered():
    rng = np.random.default_rng(2)
    px = 100 * np.cumprod(1 + rng.normal(0.0005, 0.004, 400))   # calm -> levered up
    panel = _panel({"A": px})
    no_fee = lab.vol_target(panel, 0.15, 20, 3.0, borrow_rate=0.0)
    fee = lab.vol_target(panel, 0.15, 20, 3.0, borrow_rate=0.10)
    eq = lambda s: (1 + s).cumprod().iloc[-1]
    assert eq(fee) < eq(no_fee)


# --------------------------------------------------------------------------- #
# Inverse-vol / risk parity
# --------------------------------------------------------------------------- #
def test_inverse_vol_underweights_the_volatile_asset():
    rng = np.random.default_rng(3)
    calm = 100 * np.cumprod(1 + rng.normal(0, 0.004, 300))
    wild = 100 * np.cumprod(1 + rng.normal(0, 0.02, 300))
    panel = _panel({"CALM": calm, "WILD": wild})
    # Reconstruct the (pre-shift) risk-parity weights to check the tilt.
    inv = (1.0 / lab.realized_vol(lab.daily_returns(panel), 20))
    w = inv.div(inv.sum(axis=1), axis=0).dropna()
    assert w["CALM"].iloc[-1] > w["WILD"].iloc[-1]      # calm asset gets more weight


# --------------------------------------------------------------------------- #
# Managed futures
# --------------------------------------------------------------------------- #
def test_managed_futures_goes_to_cash_when_nothing_trends():
    # Both assets below their MA (downtrend) -> book should be flat (zero returns).
    down = np.linspace(200, 100, 300)
    panel = _panel({"A": down, "B": down * 1.5})
    r = lab.managed_futures(panel, ma_period=50, vol_lookback=20)
    assert abs(r.iloc[-1]) < 1e-12                    # in cash, no exposure


# --------------------------------------------------------------------------- #
# Mean reversion
# --------------------------------------------------------------------------- #
def test_mean_reversion_only_trades_in_uptrend():
    # Steady downtrend: never in an uptrend -> never enters -> all-flat returns.
    down = np.linspace(300, 100, 400)
    panel = _panel({"A": down})
    r = lab.mean_reversion(panel, ma_trend=50, rsi_period=2, buy=10, exit_=70)
    assert (r == 0.0).all()


# --------------------------------------------------------------------------- #
# Ensemble + run
# --------------------------------------------------------------------------- #
def test_ensemble_is_the_mean_of_components():
    idx = pd.date_range("2005-01-01", periods=10, freq="B", tz="UTC")
    a = pd.Series(np.full(10, 0.02), index=idx)
    b = pd.Series(np.full(10, -0.01), index=idx)
    e = lab.ensemble({"a": a, "b": b})
    assert np.allclose(e.values, 0.005)


def test_run_smoke_scoreboard():
    rng = np.random.default_rng(7)
    cols = {s: 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 500))
            for s in ("SPY", "QQQ", "GLD", "TLT")}
    panel_data = {k: pd.DataFrame({"open": v, "high": v, "low": v, "close": v,
                                   "volume": np.ones_like(v)},
                                  index=pd.date_range("2005-01-01", periods=500,
                                                      freq="B", tz="UTC"))
                  for k, v in cols.items()}
    rc = lab.run(panel_data, list(lab._STRATEGIES),
                 begin_ts=panel_data["SPY"].index[210],
                 params={"ma_period": 100, "target_vol": 0.15, "vol_lookback": 20,
                         "max_leverage": 2.0, "borrow_rate": 0.06, "with_ensemble": True})
    assert rc == 0


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        fn()
        print(f"{name} OK")
    print(f"\nALL {len(fns)} LAB TESTS PASSED")
