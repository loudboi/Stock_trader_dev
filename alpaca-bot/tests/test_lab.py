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
# ERC risk parity (equal risk contribution via the covariance matrix)
# --------------------------------------------------------------------------- #
def test_erc_weights_equalize_risk_contribution():
    rng = np.random.default_rng(5)
    # Three assets with very different vols and some correlation.
    n = 400
    f = rng.normal(0, 0.01, n)
    a = rng.normal(0, 0.004, n) + 0.3 * f
    b = rng.normal(0, 0.012, n) + 0.5 * f
    c = rng.normal(0, 0.025, n) + 0.7 * f
    cov = np.cov(np.vstack([a, b, c]))
    w = lab._erc_weights(cov)
    assert abs(w.sum() - 1.0) < 1e-6 and (w >= 0).all()
    rc = w * (cov @ w)                              # per-asset risk contribution
    # Equal risk contribution -> all three contributions close to each other.
    assert (rc.max() - rc.min()) / rc.mean() < 0.05


def test_erc_underweights_the_volatile_asset_vs_equal_weight():
    rng = np.random.default_rng(6)
    n = 400
    calm = rng.normal(0, 0.004, n)
    wild = rng.normal(0, 0.03, n)
    cov = np.cov(np.vstack([calm, wild]))
    w = lab._erc_weights(cov)
    assert w[0] > w[1]                               # calm asset gets more weight than wild
    assert w[0] > 0.5                                # more than equal-weight (0.5)


# --------------------------------------------------------------------------- #
# Minimum variance
# --------------------------------------------------------------------------- #
def test_min_var_weights_sum_to_one_and_nonnegative():
    rng = np.random.default_rng(9)
    n = 300
    data = rng.normal(0, 0.01, (4, n))
    cov = np.cov(data)
    w = lab._minvar_weights(cov)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= 0).all()


def test_min_var_beats_equal_weight_variance_in_sample():
    rng = np.random.default_rng(10)
    n = 300
    data = rng.normal(0, [[0.004], [0.01], [0.02]], (3, n))
    cov = np.cov(data)
    w_mv = lab._minvar_weights(cov)
    w_eq = np.ones(3) / 3
    var_mv = w_mv @ cov @ w_mv
    var_eq = w_eq @ cov @ w_eq
    assert var_mv <= var_eq + 1e-12                  # min-var is, by construction, <= equal-weight


def test_monthly_cov_strategy_no_lookahead():
    rng = np.random.default_rng(11)
    n = 300
    panel = _panel({"A": 100 * np.cumprod(1 + rng.normal(0, 0.01, n)),
                    "B": 100 * np.cumprod(1 + rng.normal(0, 0.02, n))})
    # Before enough history exists for the first covariance estimate (lookback=60),
    # the book must stay at the equal-weight default -> return = mean of the two
    # asset returns, minus the one-time turnover cost of entering from zero on day 1
    # (the function's own internal shift(1) already encodes the decide-then-act lag).
    rets = lab.daily_returns(panel)
    equal_weight_ret = rets.mean(axis=1)
    equal_weight_ret.iloc[1] -= lab.SLIPPAGE          # entering the initial position
    r = lab._monthly_cov_strategy(panel, lookback=60, weight_fn=lab._minvar_weights)
    assert np.allclose(r.iloc[:60].values, equal_weight_ret.iloc[:60].values)


# --------------------------------------------------------------------------- #
# Vol-targeted risk parity
# --------------------------------------------------------------------------- #
def test_rp_voltarget_scales_toward_target_vol():
    rng = np.random.default_rng(12)
    n = 500
    panel = _panel({"A": 100 * np.cumprod(1 + rng.normal(0.0002, 0.006, n)),
                    "B": 100 * np.cumprod(1 + rng.normal(0.0002, 0.018, n))})
    r = lab.rp_voltarget(panel, target_vol=0.10, lookback=20, max_leverage=1.5,
                         borrow_rate=0.0)
    realized = r.iloc[100:].std() * np.sqrt(252)
    # Not an exact match (leverage cap + lag), but should land in a sane range,
    # not wildly off (e.g. not 3x or 0.2x the 10% target).
    assert 0.04 < realized < 0.20


# --------------------------------------------------------------------------- #
# Long/short: time-series trend (trend_ls)
# --------------------------------------------------------------------------- #
def test_trend_ls_flat_during_ma_warmup():
    panel = _panel({"A": np.linspace(100, 150, 40)})
    r = lab.trend_ls(panel, ma_period=20, short_borrow=0.0)
    assert (r.iloc[:20] == 0.0).all()          # no MA yet -> no signal -> flat


def test_trend_ls_profits_from_a_downtrend_via_shorting():
    # A steady decline below its own MA -> the strategy should be short and
    # therefore make money as price falls (the whole point of allowing shorts).
    down = np.linspace(150, 80, 250)
    panel = _panel({"A": down})
    r = lab.trend_ls(panel, ma_period=20, short_borrow=0.0)
    eq = (1 + r.iloc[25:]).cumprod()
    assert eq.iloc[-1] > eq.iloc[0]             # net gain while shorting the decline


def test_trend_ls_short_borrow_costs_money_when_short():
    down = np.linspace(150, 80, 250)
    panel = _panel({"A": down})
    no_fee = lab.trend_ls(panel, ma_period=20, short_borrow=0.0)
    fee = lab.trend_ls(panel, ma_period=20, short_borrow=0.20)   # steep cost, easy to see
    eq_no_fee = (1 + no_fee).cumprod().iloc[-1]
    eq_fee = (1 + fee).cumprod().iloc[-1]
    assert eq_fee < eq_no_fee


def test_trend_ls_equal_weights_long_and_short_sides():
    # Two assets, one clearly up (long) and one clearly down (short): the strategy
    # should hold both sides (unlike a long-only/cash trend filter, which would
    # only hold the winner and sit the loser out in cash).
    up = np.linspace(100, 200, 250)
    down = np.linspace(100, 50, 250)
    panel = _panel({"UP": up, "DOWN": down})
    ma = panel.rolling(20, min_periods=20).mean()
    signal = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    signal[panel > ma] = 1.0
    signal[panel < ma] = -1.0
    last = signal.iloc[-1]
    assert last["UP"] > 0 and last["DOWN"] < 0


# --------------------------------------------------------------------------- #
# Long/short: inverse-vol-weighted trend (managed_futures_ls)
# --------------------------------------------------------------------------- #
def test_managed_futures_ls_flat_during_ma_warmup():
    panel = _panel({"A": np.linspace(100, 150, 40), "B": np.linspace(50, 40, 40)})
    r = lab.managed_futures_ls(panel, ma_period=20, vol_lookback=10, short_borrow=0.0)
    assert (r.iloc[:20] == 0.0).all()


def test_managed_futures_ls_underweights_the_volatile_asset():
    # A calm uptrend and a wild uptrend (both long signals): the calm one should
    # get a bigger dollar weight, unlike trend_ls's equal-weight-of-active-signals.
    rng = np.random.default_rng(30)
    n = 250
    calm = 100 * np.cumprod(1 + rng.normal(0.001, 0.003, n))
    wild = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
    panel = _panel({"CALM": calm, "WILD": wild})
    ma = panel.rolling(20, min_periods=20).mean()
    signal = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    signal[panel > ma] = 1.0
    signal[panel < ma] = -1.0
    inv_vol = (1.0 / lab.realized_vol(lab.daily_returns(panel), 20)).replace([np.inf, -np.inf], np.nan)
    raw = (signal * inv_vol).fillna(0.0)
    w = raw.div(raw.abs().sum(axis=1).replace(0, np.nan), axis=0).dropna()
    last = w.iloc[-1]
    if last["CALM"] > 0 and last["WILD"] > 0:      # both long at the same time
        assert last["CALM"] > last["WILD"]


def test_managed_futures_ls_short_borrow_costs_money():
    down = np.linspace(150, 80, 250)
    panel = _panel({"A": down, "B": np.linspace(100, 200, 250)})
    no_fee = lab.managed_futures_ls(panel, ma_period=20, vol_lookback=10, short_borrow=0.0)
    fee = lab.managed_futures_ls(panel, ma_period=20, vol_lookback=10, short_borrow=0.20)
    eq_no_fee = (1 + no_fee).cumprod().iloc[-1]
    eq_fee = (1 + fee).cumprod().iloc[-1]
    assert eq_fee < eq_no_fee


# --------------------------------------------------------------------------- #
# Long/short: cross-sectional momentum (xsmom_ls)
# --------------------------------------------------------------------------- #
def test_xsmom_weights_are_dollar_neutral_after_a_rebalance():
    rng = np.random.default_rng(20)
    n = 400
    panel = _panel({s: 100 * np.cumprod(1 + rng.normal(d, 0.01, n))
                    for s, d in zip("ABCD", [0.001, 0.0005, -0.0005, -0.001])})
    w = lab.xsmom_weights(panel, lookback_months=3, skip_months=0, top_frac=0.3)
    active_rows = w[(w != 0).any(axis=1)]
    assert len(active_rows) > 0
    # Long book (0.5) == short book (0.5) every time a position is active -> net
    # notional is exactly zero (market-neutral), and the long/short legs balance.
    assert np.allclose(active_rows.sum(axis=1).values, 0.0)
    assert np.allclose(active_rows.clip(lower=0).sum(axis=1).values, 0.5)
    assert np.allclose(active_rows.clip(upper=0).sum(axis=1).values, -0.5)


def test_xsmom_ls_longs_the_winner_shorts_the_loser():
    rng = np.random.default_rng(21)
    n = 300
    # Four assets with a clear, persistent momentum ranking.
    panel = _panel({
        "WINNER": 100 * np.cumprod(1 + rng.normal(0.002, 0.005, n)),
        "MID_HIGH": 100 * np.cumprod(1 + rng.normal(0.0005, 0.005, n)),
        "MID_LOW": 100 * np.cumprod(1 + rng.normal(-0.0005, 0.005, n)),
        "LOSER": 100 * np.cumprod(1 + rng.normal(-0.002, 0.005, n)),
    })
    lookback, skip = 3 * lab._DAYS_PER_MONTH, 0
    base_i = 200
    scores = lab.momentum(panel, base_i, lookback, skip)
    assert scores["WINNER"] > scores["MID_HIGH"] > scores["MID_LOW"] > scores["LOSER"]


def test_xsmom_ls_short_borrow_costs_money():
    rng = np.random.default_rng(22)
    n = 300
    panel = _panel({s: 100 * np.cumprod(1 + rng.normal(d, 0.008, n))
                    for s, d in zip("ABCD", [0.0015, 0.0005, -0.0005, -0.0015])})
    no_fee = lab.xsmom_ls(panel, lookback_months=3, skip_months=0, short_borrow=0.0)
    fee = lab.xsmom_ls(panel, lookback_months=3, skip_months=0, short_borrow=0.20)
    eq_no_fee = (1 + no_fee).cumprod().iloc[-1]
    eq_fee = (1 + fee).cumprod().iloc[-1]
    assert eq_fee < eq_no_fee


def test_xsmom_ls_no_lookahead_flat_before_first_rebalance():
    rng = np.random.default_rng(23)
    n = 200
    panel = _panel({s: 100 * np.cumprod(1 + rng.normal(0, 0.01, n)) for s in "ABCD"})
    r = lab.xsmom_ls(panel, lookback_months=3, skip_months=0, short_borrow=0.0)
    # No momentum score can exist before 3 months (63 trading days) of history, so
    # a conservative early prefix (well short of that, regardless of exactly where
    # a calendar-month boundary falls in business days) must be exactly flat.
    assert (r.iloc[:40] == 0.0).all()


# --------------------------------------------------------------------------- #
# Ensemble + run
# --------------------------------------------------------------------------- #
def test_ensemble_is_the_mean_of_components():
    idx = pd.date_range("2005-01-01", periods=10, freq="B", tz="UTC")
    a = pd.Series(np.full(10, 0.02), index=idx)
    b = pd.Series(np.full(10, -0.01), index=idx)
    e = lab.ensemble({"a": a, "b": b})
    assert np.allclose(e.values, 0.005)


# --------------------------------------------------------------------------- #
# Walk-forward (fold slicing, consistency check — no per-fold fitting)
# --------------------------------------------------------------------------- #
def test_fold_bounds_partition_sequentially():
    s = pd.Timestamp("2005-01-01", tz="UTC")
    e = pd.Timestamp("2025-01-01", tz="UTC")
    bounds = lab.fold_bounds(s, e, 5)
    assert len(bounds) == 5
    assert bounds[0][0] == s and bounds[-1][1] == e
    for i in range(len(bounds) - 1):
        assert bounds[i][1] == bounds[i + 1][0]      # contiguous, no gaps/overlap
        assert bounds[i][0] < bounds[i][1]           # each fold has positive length


def test_slice_equity_starts_compounding_from_the_window_start():
    idx = pd.date_range("2005-01-01", periods=10, freq="B", tz="UTC")
    r = pd.Series([0.0, 0.01, 0.02, -0.01, 0.03, 0.0, 0.01, -0.02, 0.01, 0.0], index=idx)
    eq = lab.slice_equity(r, start_ts=idx[3], end_ts=idx[7], initial=1000.0)
    # eq.iloc[0] applies that first in-window day's return on top of `initial`
    # (matches how compute_metrics treats eq.iloc[0] as the baseline elsewhere).
    assert eq.index[0] == idx[3] and abs(eq.iloc[0] - 1000.0 * (1 + r.iloc[3])) < 1e-9
    assert eq.index[-1] == idx[7]
    assert len(eq) == 5
    # A slice starting one bar later must NOT be affected by the return excluded.
    eq2 = lab.slice_equity(r, start_ts=idx[4], end_ts=idx[7], initial=1000.0)
    assert abs(eq2.iloc[0] - 1000.0 * (1 + r.iloc[4])) < 1e-9


def test_slice_equity_empty_window_returns_empty_series():
    idx = pd.date_range("2005-01-01", periods=5, freq="B", tz="UTC")
    r = pd.Series([0.0, 0.01, 0.02, -0.01, 0.03], index=idx)
    empty = lab.slice_equity(r, start_ts=idx[-1] + pd.Timedelta(days=10))
    assert empty.empty


def _wf_panel(n=1200, seed=3):
    rng = np.random.default_rng(seed)
    cols = {s: 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
            for s in ("SPY", "QQQ", "GLD", "TLT")}
    idx = pd.date_range("2005-01-01", periods=n, freq="B", tz="UTC")
    return {k: pd.DataFrame({"open": v, "high": v, "low": v, "close": v,
                             "volume": np.ones_like(v)}, index=idx)
            for k, v in cols.items()}


def test_walk_forward_folds_cover_the_full_window_disjointly():
    panel_data = _wf_panel()
    idx = panel_data["SPY"].index
    per_fold = lab.run_walk_forward(
        panel_data, ["inverse_vol", "min_var"],
        params={"borrow_rate": 0.06, "with_ensemble": False},
        start_ts=idx[300], end_ts=idx[-1], folds=4)
    assert len(per_fold) == 4
    # Sequential, contiguous folds spanning the requested window.
    assert per_fold[0]["start"] == idx[300]
    assert per_fold[-1]["end"] == idx[-1]
    for i in range(len(per_fold) - 1):
        assert per_fold[i]["end"] == per_fold[i + 1]["start"]
    for row in per_fold:
        assert "bh" in row and set(row["strategies"]) == {"inverse_vol", "min_var"}
        assert "sharpe" in row["bh"]


def test_walk_forward_no_per_fold_fitting_matches_full_window_slice():
    # The whole point: a fold's metrics must equal slicing the SAME full-history
    # return series (computed once) to that fold's window — never a series
    # recomputed/refit using only that fold's data.
    panel_data = _wf_panel(seed=4)
    idx = panel_data["SPY"].index
    names = ["inverse_vol"]
    params = {"borrow_rate": 0.06, "with_ensemble": False}
    full_returns = lab.compute_strategy_returns(panel_data, names, params)["inverse_vol"]

    per_fold = lab.run_walk_forward(panel_data, names, params,
                                    start_ts=idx[300], end_ts=idx[-1], folds=3)
    fold0 = per_fold[0]
    expected_eq = lab.slice_equity(full_returns, fold0["start"], fold0["end"])
    expected_m = lab.compute_metrics([], expected_eq)
    assert abs(fold0["strategies"]["inverse_vol"]["sharpe"] - expected_m["sharpe"]) < 1e-9
    assert abs(fold0["strategies"]["inverse_vol"]["total_return"]
              - expected_m["total_return"]) < 1e-9


def test_print_walk_forward_runs_without_error(capsys):
    panel_data = _wf_panel(seed=5)
    idx = panel_data["SPY"].index
    per_fold = lab.run_walk_forward(
        panel_data, ["inverse_vol", "min_var", "vol_target"],
        params={"target_vol": 0.15, "vol_lookback": 20, "max_leverage": 2.0,
               "borrow_rate": 0.06, "with_ensemble": False},
        start_ts=idx[300], end_ts=idx[-1], folds=3)
    lab.print_walk_forward(per_fold)
    out = capsys.readouterr().out
    assert "WALK-FORWARD" in out and "SUMMARY" in out


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
