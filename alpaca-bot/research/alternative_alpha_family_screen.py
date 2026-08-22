from __future__ import annotations

import itertools
import json
from math import sqrt

import numpy as np
import pandas as pd
import yfinance as yf

START = "2004-01-01"
END = "2025-01-02"  # Deliberately excludes the burned 2025-2026 interval.
TRADING_COST = 0.0020
BORROW_SPREAD = 0.015

TSMOM_UNIVERSE = ["QQQ", "SPY", "IWM", "EFA", "EEM", "IEF", "TLT", "GLD", "DBC"]
ALL_TICKERS = sorted(set(TSMOM_UNIVERSE + ["QLD", "^VIX"]))

DEV_REGIMES = [
    ("2007-2009", "2007-01-01", "2009-12-31"),
    ("2010-2014", "2010-01-01", "2014-12-31"),
    ("2015-2019", "2015-01-01", "2019-12-31"),
    ("2020-2024", "2020-01-01", "2024-12-31"),
]


def _normalize(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").dropna().copy()
    idx = pd.DatetimeIndex(out.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out.index = idx.normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


def load_adjusted(symbol: str) -> pd.Series:
    frame = yf.Ticker(symbol).history(
        start=START,
        end=END,
        auto_adjust=False,
        actions=False,
        repair=False,
    )
    if frame is None or frame.empty:
        raise RuntimeError(f"no history for {symbol}")
    col = "Adj Close" if "Adj Close" in frame.columns else "Close"
    out = _normalize(frame[col])
    if out.empty:
        raise RuntimeError(f"no adjusted history for {symbol}")
    return out


def load_market() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    raw = {s: load_adjusted(s) for s in ALL_TICKERS}
    qidx = raw["QQQ"].index
    prices = pd.DataFrame(index=qidx)
    for symbol, series in raw.items():
        prices[symbol] = series.reindex(qidx).ffill()

    dev_start = pd.Timestamp(DEV_REGIMES[0][1], tz="UTC")
    needed = [s for s in ALL_TICKERS if s != "DBC"]
    missing = prices.loc[prices.index >= dev_start, needed].isna().any()
    if missing.any():
        raise RuntimeError(f"missing development data: {list(missing[missing].index)}")

    fred = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF")
    fred.columns = ["date", "value"]
    fred["date"] = pd.to_datetime(fred["date"], utc=True).dt.normalize()
    fred["value"] = pd.to_numeric(fred["value"], errors="coerce") / 100.0
    fed = fred.dropna().set_index("date").value.sort_index()
    fed = fed.reindex(fed.index.union(qidx).sort_values()).ffill().reindex(qidx)
    fed = fed.shift(1).fillna(0.0).clip(lower=0.0)

    vix = prices["^VIX"].copy()
    return prices, vix, fed


def month_start_flags(index: pd.Index) -> np.ndarray:
    periods = pd.Series(index.to_period("M"), index=index)
    return periods.ne(periods.shift(1)).to_numpy()


def week_start_flags(index: pd.Index) -> np.ndarray:
    periods = pd.Series(index.to_period("W-FRI"), index=index)
    return periods.ne(periods.shift(1)).to_numpy()


def simulate_exposure(
    asset_returns: pd.Series,
    exposure: pd.Series,
    fed: pd.Series,
    cost: float = TRADING_COST,
) -> dict:
    idx = asset_returns.index
    exp = exposure.reindex(idx).ffill().fillna(0.0).astype(float)
    rate = fed.reindex(idx).ffill().fillna(0.0)
    wealth = 1.0
    prev = 0.0
    traded = 0.0
    path = []
    for ts in idx:
        e = float(exp.loc[ts])
        turnover = abs(e - prev)
        daily = e * float(asset_returns.loc[ts])
        daily -= max(e - 1.0, 0.0) * (float(rate.loc[ts]) + BORROW_SPREAD) / 252.0
        daily -= turnover * cost
        wealth *= 1.0 + daily
        if wealth <= 0 or not np.isfinite(wealth):
            wealth = np.nan
            path.append(wealth)
            break
        traded += turnover
        path.append(wealth)
        prev = e
    series = pd.Series(path, index=idx[: len(path)], dtype=float)
    return {"path": series, "turnover": traded}


def simulate_weights(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    fed: pd.Series,
    cost: float = TRADING_COST,
    synthetic_borrow: bool = True,
) -> dict:
    idx = returns.index
    w = weights.reindex(idx).ffill().fillna(0.0)
    rate = fed.reindex(idx).ffill().fillna(0.0)
    wealth = 1.0
    prev = np.zeros(w.shape[1], dtype=float)
    traded = 0.0
    path = []
    for ts in idx:
        cur = w.loc[ts].to_numpy(dtype=float)
        turnover = float(np.abs(cur - prev).sum())
        daily = float(np.dot(cur, returns.loc[ts].to_numpy(dtype=float)))
        if synthetic_borrow:
            daily -= max(float(cur.sum()) - 1.0, 0.0) * (float(rate.loc[ts]) + BORROW_SPREAD) / 252.0
        daily -= turnover * cost
        wealth *= 1.0 + daily
        if wealth <= 0 or not np.isfinite(wealth):
            wealth = np.nan
            path.append(wealth)
            break
        traded += turnover
        path.append(wealth)
        prev = cur
    series = pd.Series(path, index=idx[: len(path)], dtype=float)
    return {"path": series, "turnover": traded}


def tsmom_weights(prices: pd.DataFrame, lookback: int, target_vol: float, gross_cap: float) -> pd.DataFrame:
    p = prices[TSMOM_UNIVERSE]
    rets = p.pct_change(fill_method=None)
    flags = month_start_flags(p.index)
    out = pd.DataFrame(0.0, index=p.index, columns=p.columns)
    current = np.zeros(len(p.columns), dtype=float)

    for i, ts in enumerate(p.index):
        if flags[i] and i > max(lookback, 63):
            prior = i - 1
            momentum = p.iloc[prior] / p.iloc[prior - lookback] - 1.0
            available = momentum.replace([np.inf, -np.inf], np.nan).dropna()
            selected = available[available > 0].index.tolist()
            current = np.zeros(len(p.columns), dtype=float)
            if selected:
                base = pd.Series(0.0, index=p.columns)
                base.loc[selected] = 1.0 / len(selected)
                cov = rets.iloc[i - 63 : i].cov() * 252.0
                vec = base.to_numpy(dtype=float)
                variance = float(vec @ cov.to_numpy(dtype=float) @ vec)
                pvol = sqrt(max(variance, 0.0))
                scale = min(gross_cap, target_vol / pvol) if pvol > 1e-12 else 0.0
                current = vec * scale
        out.iloc[i] = current
    return out


def asymmetric_trend_exposure(prices: pd.Series, exit_window: int, reentry_window: int, risk_on: float) -> pd.Series:
    flags = week_start_flags(prices.index)
    prior_close = prices.shift(1)
    exit_ma = prices.shift(1).rolling(exit_window, min_periods=exit_window).mean()
    prior_high = prices.shift(2).rolling(reentry_window, min_periods=reentry_window).max()
    exposure = pd.Series(0.50, index=prices.index, dtype=float)
    state_on = False

    for i, ts in enumerate(prices.index):
        if flags[i]:
            pc = prior_close.loc[ts]
            ma = exit_ma.loc[ts]
            high = prior_high.loc[ts]
            if np.isfinite(pc) and np.isfinite(ma):
                if state_on and pc < ma:
                    state_on = False
                elif not state_on and np.isfinite(high) and pc > high:
                    state_on = True
                elif i and not np.isfinite(prior_high.iloc[i - 1]) and pc >= ma:
                    state_on = True
        exposure.loc[ts] = risk_on if state_on else 0.50
    return exposure


def panic_reversal_exposure(
    qqq: pd.Series,
    vix: pd.Series,
    threshold: float,
    hold_sessions: int,
    panic_exposure: float,
) -> pd.Series:
    prior_5d = qqq.shift(1) / qqq.shift(6) - 1.0
    prior_vix = vix.shift(1)
    out = pd.Series(1.0, index=qqq.index, dtype=float)
    remaining = 0
    for ts in qqq.index:
        if remaining <= 0:
            signal = (
                np.isfinite(prior_5d.loc[ts])
                and np.isfinite(prior_vix.loc[ts])
                and float(prior_5d.loc[ts]) <= threshold
                and float(prior_vix.loc[ts]) >= 30.0
            )
            if signal:
                remaining = hold_sessions
        if remaining > 0:
            out.loc[ts] = panic_exposure
            remaining -= 1
    return out


def qld_rotation_weights(qqq: pd.Series, lookback: int, ma_window: int) -> pd.DataFrame:
    idx = qqq.index
    flags = month_start_flags(idx)
    prior = qqq.shift(1)
    momentum = qqq.shift(1) / qqq.shift(1 + lookback) - 1.0
    ma = qqq.shift(1).rolling(ma_window, min_periods=ma_window).mean()
    out = pd.DataFrame(0.0, index=idx, columns=["QQQ", "QLD"])
    current = np.array([1.0, 0.0])
    for i, ts in enumerate(idx):
        if flags[i]:
            risk_on = (
                np.isfinite(momentum.loc[ts])
                and np.isfinite(ma.loc[ts])
                and float(momentum.loc[ts]) > 0.0
                and float(prior.loc[ts]) > float(ma.loc[ts])
            )
            current = np.array([0.0, 1.0]) if risk_on else np.array([1.0, 0.0])
        out.iloc[i] = current
    return out


def cagr(path: pd.Series, start: str, end: str) -> float:
    s = path.loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")].dropna()
    if len(s) < 2 or s.iloc[0] <= 0 or s.iloc[-1] <= 0:
        return np.nan
    years = (s.index[-1] - s.index[0]).total_seconds() / (365.2425 * 86400.0)
    return float((s.iloc[-1] / s.iloc[0]) ** (1.0 / years) - 1.0)


def max_drawdown(path: pd.Series, start: str, end: str) -> float:
    s = path.loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")].dropna()
    if s.empty:
        return np.nan
    return float((s / s.cummax() - 1.0).min())


def evaluate(label: str, family: str, params: dict, result: dict, qqq_path: pd.Series) -> dict:
    row = {"label": label, "family": family, **params}
    excesses = []
    wins = 0
    for regime, start, end in DEV_REGIMES:
        sc = cagr(result["path"], start, end)
        bc = cagr(qqq_path, start, end)
        ex = sc - bc
        row[f"{regime}_cagr"] = sc
        row[f"{regime}_qqq"] = bc
        row[f"{regime}_excess"] = ex
        if np.isfinite(ex) and ex > 0:
            wins += 1
        excesses.append(ex)
    row["wins_vs_qqq"] = wins
    row["min_excess"] = float(np.nanmin(excesses))
    row["pass_all_4"] = bool(wins == 4 and all(np.isfinite(x) for x in excesses))
    row["full_dev_cagr"] = cagr(result["path"], DEV_REGIMES[0][1], DEV_REGIMES[-1][2])
    row["full_dev_qqq"] = cagr(qqq_path, DEV_REGIMES[0][1], DEV_REGIMES[-1][2])
    row["full_dev_dd"] = max_drawdown(result["path"], DEV_REGIMES[0][1], DEV_REGIMES[-1][2])
    row["turnover_units"] = float(result["turnover"])
    return row


def main() -> None:
    prices, vix, fed = load_market()
    qqq = prices["QQQ"]
    qqq_ret = qqq.pct_change(fill_method=None).fillna(0.0)
    qqq_path = (1.0 + qqq_ret).cumprod()
    rows: list[dict] = []

    for lookback, target_vol, cap in itertools.product((126, 252), (0.20, 0.25), (1.25, 1.50)):
        weights = tsmom_weights(prices, lookback, target_vol, cap)
        rets = prices[TSMOM_UNIVERSE].pct_change(fill_method=None).fillna(0.0)
        result = simulate_weights(rets, weights, fed, synthetic_borrow=True)
        label = f"TSMOM_lb{lookback}_tv{target_vol:.2f}_cap{cap:.2f}"
        rows.append(evaluate(label, "TSMOM", {"lookback": lookback, "target_vol": target_vol, "cap": cap}, result, qqq_path))

    for exit_window, reentry_window, risk_on in itertools.product((150, 200), (20, 50), (1.25, 1.50)):
        exp = asymmetric_trend_exposure(qqq, exit_window, reentry_window, risk_on)
        result = simulate_exposure(qqq_ret, exp, fed)
        label = f"ASYM_exit{exit_window}_re{reentry_window}_on{risk_on:.2f}"
        rows.append(evaluate(label, "ASYM_TREND", {"exit_window": exit_window, "reentry_window": reentry_window, "risk_on": risk_on}, result, qqq_path))

    for threshold, hold_sessions, panic_exp in itertools.product((-0.05, -0.075), (5, 10), (1.25, 1.50)):
        exp = panic_reversal_exposure(qqq, vix, threshold, hold_sessions, panic_exp)
        result = simulate_exposure(qqq_ret, exp, fed)
        label = f"REV_thr{abs(threshold):.3f}_hold{hold_sessions}_exp{panic_exp:.2f}"
        rows.append(evaluate(label, "PANIC_REVERSAL", {"threshold": threshold, "hold_sessions": hold_sessions, "panic_exposure": panic_exp}, result, qqq_path))

    actual_rets = prices[["QQQ", "QLD"]].pct_change(fill_method=None).fillna(0.0)
    for lookback, ma_window in itertools.product((126, 252), (150, 200)):
        weights = qld_rotation_weights(qqq, lookback, ma_window)
        result = simulate_weights(actual_rets, weights, fed, synthetic_borrow=False)
        label = f"QLDROT_lb{lookback}_ma{ma_window}"
        rows.append(evaluate(label, "QQQ_QLD_ROTATION", {"lookback": lookback, "ma_window": ma_window}, result, qqq_path))

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["pass_all_4", "min_excess", "full_dev_dd"], ascending=[False, False, False]).reset_index(drop=True)
    passing = frame[frame.pass_all_4]

    print("ALTERNATIVE ACTIVE FAMILY SCREEN")
    print(f"selectionDataEnd={prices.index[-1].date()} variants={len(frame)} passiveQQQCost=0 activeCost={TRADING_COST:.2%}")
    print("2025-2026 DATA ARE NOT LOADED")
    print("\nFAMILY BESTS")
    for family, group in frame.groupby("family", sort=True):
        best = group.sort_values("min_excess", ascending=False).iloc[0]
        print(
            f"{family} best={best.label} wins={int(best.wins_vs_qqq)}/4 "
            f"minExcess={best.min_excess:+.4%} fullDev={best.full_dev_cagr:.4%} "
            f"QQQ={best.full_dev_qqq:.4%} DD={best.full_dev_dd:.2%} turnoverUnits={best.turnover_units:.2f}"
        )

    print("\nTOP 10")
    regime_cols = [f"{r[0]}_excess" for r in DEV_REGIMES]
    for _, row in frame.head(10).iterrows():
        extras = " ".join(f"{c}={row[c]:+.3%}" for c in regime_cols)
        print(
            f"{row.label} family={row.family} pass={bool(row.pass_all_4)} wins={int(row.wins_vs_qqq)}/4 "
            f"min={row.min_excess:+.3%} {extras}"
        )

    if passing.empty:
        print("\nSTAGE1_VERDICT REJECT_ALL no variant beat passive QQQ in all four development regimes")
        selected = None
    else:
        winner = passing.sort_values(["min_excess", "full_dev_dd", "turnover_units"], ascending=[False, False, True]).iloc[0]
        selected = winner.label
        print(
            f"\nSTAGE1_VERDICT SURVIVOR selected={winner.label} family={winner.family} "
            f"minExcess={winner.min_excess:+.4%}; freeze this variant before Stage 2"
        )

    frame.to_csv("alternative_alpha_family_screen.csv", index=False)
    with open("alternative_alpha_family_screen.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "selection_data_end": str(prices.index[-1].date()),
                "variant_count": int(len(frame)),
                "passing_count": int(len(passing)),
                "selected": selected,
                "rows": frame.replace({np.nan: None}).to_dict(orient="records"),
            },
            handle,
            indent=2,
        )
    print("ARTIFACTS alternative_alpha_family_screen.csv alternative_alpha_family_screen.json")


if __name__ == "__main__":
    main()
