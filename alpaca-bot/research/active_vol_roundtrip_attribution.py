from __future__ import annotations

"""Adversarial attribution for the frozen volatility-managed QQQ candidate.

This script does not select or tune strategy parameters. It consumes the continuous
external-review simulation, calibrates a constant-target QQQ control to the active
strategy's *realized* time-weighted gross exposure, decomposes two pre-specified
crash/rebound round trips, and stress-checks drifted leverage against daily intraday
lows. The event windows are diagnostics, not new validation folds.
"""

from math import log
import json

import numpy as np
import pandas as pd
import yfinance as yf

import active_vol_managed_external_review as ext


REGIMES = (
    ("2006-2009", "2006-01-01", "2009-12-31"),
    ("2010-2014", "2010-01-01", "2014-12-31"),
    ("2015-2019", "2015-01-01", "2019-12-31"),
    ("2020-2024", "2020-01-01", "2024-12-31"),
    ("2025-2026-burned", "2025-01-01", "2026-08-20"),
)

EVENT_LEGS = (
    ("GFC_CRASH", "2007-10-31", "2009-03-09"),
    ("GFC_REBOUND", "2009-03-09", "2009-12-31"),
    ("COVID_CRASH", "2020-02-19", "2020-03-23"),
    ("COVID_REBOUND", "2020-03-23", "2020-12-31"),
)


def held_exposure(result: dict) -> pd.Series:
    """Gross exposure held after each close for the next close-to-close interval.

    The continuous simulator overwrites the final row's post_exposure with zero after
    terminal liquidation. For exposure averaging, replace only that artificial final
    zero with the pre-liquidation exposure. Forced liquidations on earlier dates stay
    at zero because they are economically real in the simulated path.
    """
    frame = result["frame"]
    exposure = frame["post_exposure"].astype(float).copy()
    if not exposure.empty:
        exposure.iloc[-1] = float(frame["pre_exposure"].iloc[-1])
    return exposure.replace([np.inf, -np.inf], np.nan).dropna()


def exposure_stats(result: dict, start=None, end=None) -> dict[str, float]:
    exp = held_exposure(result)
    if start is not None:
        exp = exp.loc[pd.Timestamp(start, tz="UTC"):]
    if end is not None:
        exp = exp.loc[:pd.Timestamp(end, tz="UTC")]
    if exp.empty:
        return {"mean": np.nan, "median": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(exp.mean()),
        "median": float(exp.median()),
        "min": float(exp.min()),
        "max": float(exp.max()),
    }


def _calibration_error(result: dict, target_realized_exposure: float) -> float:
    return abs(exposure_stats(result)["mean"] - target_realized_exposure)


def realized_exposure_matched_control(data, begin, end, target_realized_exposure: float):
    """Choose a constant target whose realized drifted exposure matches the active mean.

    This is an attribution control only. It is calibrated ex post and therefore is not
    a tradable candidate or an out-of-sample strategy. The no-trade band, monthly
    cadence, financing, transaction cost and tax mechanics remain identical to the
    active simulator.
    """
    best = None
    for target in np.linspace(ext.MIN_EXPOSURE, ext.MAX_EXPOSURE, 51):
        result = ext.simulate_active(data, begin, end, constant_target=float(target))
        err = _calibration_error(result, target_realized_exposure)
        if best is None or err < best[0]:
            best = (err, float(target), result)

    _, coarse_target, _ = best
    lo = max(ext.MIN_EXPOSURE, coarse_target - 0.04)
    hi = min(ext.MAX_EXPOSURE, coarse_target + 0.04)
    for target in np.linspace(lo, hi, 41):
        result = ext.simulate_active(data, begin, end, constant_target=float(target))
        err = _calibration_error(result, target_realized_exposure)
        if err < best[0]:
            best = (err, float(target), result)
    return best[1], best[2]


def _aligned_value(path: pd.Series, date: str) -> tuple[pd.Timestamp, float]:
    ts = pd.Timestamp(date, tz="UTC")
    candidates = path.index[path.index >= ts]
    if len(candidates) == 0:
        raise ValueError(f"no observation on or after {date}")
    actual = candidates[0]
    return actual, float(path.loc[actual])


def segment_return(path: pd.Series, start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    a, av = _aligned_value(path, start)
    b, bv = _aligned_value(path, end)
    if b <= a:
        raise ValueError(f"invalid segment {start}..{end}")
    return a, b, bv / av - 1.0


def event_attribution(label: str, start: str, end: str, active: dict, matched: dict, qqq: dict) -> dict:
    a0, a1, ar = segment_return(active["path"], start, end)
    m0, m1, mr = segment_return(matched["path"], start, end)
    q0, q1, qr = segment_return(qqq["path"], start, end)
    if (a0, a1) != (m0, m1) or (a0, a1) != (q0, q1):
        raise RuntimeError("segment alignment mismatch")
    active_exp = exposure_stats(active, str(a0.date()), str(a1.date()))
    matched_exp = exposure_stats(matched, str(a0.date()), str(a1.date()))
    log_residual = log1p_safe(ar) - log1p_safe(mr)
    row = {
        "label": label,
        "start": str(a0.date()),
        "end": str(a1.date()),
        "active_return": ar,
        "matched_return": mr,
        "qqq_return": qr,
        "active_minus_matched_return_pp": ar - mr,
        "active_minus_qqq_return_pp": ar - qr,
        "log_timing_residual": log_residual,
        "active_mean_exposure": active_exp["mean"],
        "active_max_exposure": active_exp["max"],
        "matched_mean_exposure": matched_exp["mean"],
    }
    print(
        f"{label} {a0.date()}..{a1.date()} active={ar:.2%} matched={mr:.2%} QQQ={qr:.2%} "
        f"active-matched={ar-mr:+.2%} logResidual={log_residual:+.5f} "
        f"meanExp(active/matched)={active_exp['mean']:.3f}/{matched_exp['mean']:.3f} "
        f"maxActiveExp={active_exp['max']:.3f}"
    )
    return row


def log1p_safe(value: float) -> float:
    if value <= -1.0:
        return -np.inf
    return float(np.log1p(value))


def load_daily_low(index: pd.Index) -> pd.Series:
    d = yf.Ticker("QQQ").history(
        start=ext.START, end=ext.END, auto_adjust=False, actions=True, repair=False
    )
    if d is None or d.empty:
        raise RuntimeError("no QQQ history for intraday-low diagnostic")
    d = ext._normalize(d)
    low = pd.to_numeric(d["Low"], errors="coerce")
    low = low.reindex(index)
    if low.isna().any() or (low <= 0).any():
        missing = list(low[low.isna()].index[:5])
        raise RuntimeError(f"missing/invalid QQQ daily low values; sample={missing}")
    return low.astype(float)


def drifted_margin_path(active: dict, data, low: pd.Series) -> pd.DataFrame:
    """Approximate intraday margin ratio using prior-close holdings and daily low.

    The research strategy trades at the first-session close, so the day's intraday low
    is tested against positions held from the previous close. Interest, splits and the
    day's after-tax cash dividend are applied before the low diagnostic. This is still
    not broker-specific portfolio-margin simulation; it is a stricter daily-high/low
    diagnostic than the original close-only 30% check.
    """
    idx, close, divs, splits, fx, fed, _ = data
    frame = active["frame"]
    dates = frame.index
    rows = []
    for i in range(1, len(dates)):
        prev_ts = dates[i - 1]
        ts = dates[i]
        prev = frame.loc[prev_ts]
        prev_nav_usd = float(prev.nav_eur) * float(fx.loc[prev_ts])
        prev_exp = float(prev.post_exposure)
        if i == len(dates) - 1 and prev_exp == 0.0:
            # Only relevant if a caller has altered the end; normal continuous path
            # has its artificial terminal zero on the final row, not the prior row.
            pass
        prev_debt = float(prev.debt_usd)
        prev_close = float(close.loc[prev_ts, "QQQ"])
        if not np.isfinite(prev_exp) or prev_nav_usd <= 0 or prev_close <= 0:
            continue
        qty = prev_exp * prev_nav_usd / prev_close
        cash = prev_nav_usd + prev_debt - qty * prev_close
        debt = prev_debt
        if debt > 0:
            debt *= 1.0 + (float(fed.loc[ts]) + ext.BORROW_SPREAD) / 360.0

        split = float(splits["QQQ"].loc[ts])
        if split > 0 and abs(split - 1.0) > 1e-12:
            qty *= split

        dividend = float(divs["QQQ"].loc[ts])
        if dividend > 0 and qty > 0:
            cash += qty * dividend * (1.0 - ext.DIVIDEND_TAX)
            if cash > 0 and debt > 0:
                repaid = min(cash, debt)
                cash -= repaid
                debt -= repaid

        low_px = float(low.loc[ts])
        mv_low = qty * low_px
        nav_low = cash + mv_low - debt
        if mv_low <= 0:
            margin_ratio = np.nan
            gross_low = 0.0
        else:
            margin_ratio = nav_low / mv_low
            gross_low = mv_low / nav_low if nav_low > 0 else np.inf
        rows.append(
            {
                "date": ts,
                "prior_close_exposure": prev_exp,
                "low": low_px,
                "nav_at_low_usd": nav_low,
                "margin_ratio_at_low": margin_ratio,
                "gross_exposure_at_low": gross_low,
                "debt_usd": debt,
            }
        )
    return pd.DataFrame(rows).set_index("date")


def print_margin_window(name: str, margin: pd.DataFrame, start: str, end: str) -> dict:
    w = margin.loc[pd.Timestamp(start, tz="UTC"):pd.Timestamp(end, tz="UTC")]
    levered = w[w.debt_usd > 0].dropna(subset=["margin_ratio_at_low"])
    if levered.empty:
        row = {"name": name, "min_margin_ratio": np.nan, "date": None, "max_gross_at_low": np.nan}
        print(f"{name} no levered observations")
        return row
    trough = levered.margin_ratio_at_low.idxmin()
    min_ratio = float(levered.loc[trough, "margin_ratio_at_low"])
    max_gross = float(levered.gross_exposure_at_low.replace(np.inf, np.nan).max())
    breaches = {str(int(t * 100)): int((levered.margin_ratio_at_low < t).sum()) for t in (0.30, 0.35, 0.40, 0.50)}
    row = {
        "name": name,
        "min_margin_ratio": min_ratio,
        "date": str(trough.date()),
        "max_gross_at_low": max_gross,
        "breach_days": breaches,
    }
    print(
        f"{name} minMarginAtDailyLow={min_ratio:.2%} on {trough.date()} maxGrossAtLow={max_gross:.3f}x "
        f"breachDays30/35/40/50={breaches['30']}/{breaches['35']}/{breaches['40']}/{breaches['50']}"
    )
    return row


def main():
    data = ext.load_market()
    idx, _, _, _, _, _, _ = data
    begin = idx[idx >= ext.BEGIN][0]
    end = idx[idx <= ext.FINISH][-1]
    active = ext.simulate_active(data, begin, end)
    qqq = ext.simulate_passive(data, "QQQ", begin, end)

    active_realized = exposure_stats(active)["mean"]
    matched_target, matched = realized_exposure_matched_control(data, begin, end, active_realized)
    matched_realized = exposure_stats(matched)["mean"]

    print("REALIZED-EXPOSURE-MATCHED CONTROL")
    print(
        f"activeMeanRealizedGross={active_realized:.6f} constantTarget={matched_target:.6f} "
        f"controlMeanRealizedGross={matched_realized:.6f} mismatch={matched_realized-active_realized:+.6f}"
    )
    print(
        f"activeCAGR={active['metrics']['cagr']:.4%} matchedCAGR={matched['metrics']['cagr']:.4%} "
        f"timingResidualCAGR={active['metrics']['cagr']-matched['metrics']['cagr']:+.4%}"
    )

    print("\nREALIZED GROSS EXPOSURE BY REGIME")
    regime_rows = []
    for label, start, finish in REGIMES:
        stats = exposure_stats(active, start, finish)
        regime_rows.append({"label": label, **stats})
        print(
            f"{label} mean={stats['mean']:.4f} median={stats['median']:.4f} "
            f"min={stats['min']:.4f} max={stats['max']:.4f}"
        )
    apr_dec = exposure_stats(active, "2020-04-01", "2020-12-31")
    print(
        f"2020_APR_DEC mean={apr_dec['mean']:.4f} median={apr_dec['median']:.4f} "
        f"min={apr_dec['min']:.4f} max={apr_dec['max']:.4f}"
    )

    print("\nCRASH / REBOUND ATTRIBUTION")
    event_rows = [event_attribution(label, start, finish, active, matched, qqq) for label, start, finish in EVENT_LEGS]
    event_by_label = {row["label"]: row for row in event_rows}
    for prefix in ("GFC", "COVID"):
        crash = event_by_label[f"{prefix}_CRASH"]
        rebound = event_by_label[f"{prefix}_REBOUND"]
        summed = crash["log_timing_residual"] + rebound["log_timing_residual"]
        print(
            f"{prefix}_ROUNDTRIP logTimingResidual crash={crash['log_timing_residual']:+.5f} "
            f"rebound={rebound['log_timing_residual']:+.5f} sum={summed:+.5f}"
        )

    print("\nDRIFTED EXPOSURE / DAILY-LOW MARGIN DIAGNOSTIC")
    frame = active["frame"]
    pre = frame.pre_exposure.replace([np.inf, -np.inf], np.nan).dropna()
    max_pre_date = pre.idxmax()
    print(f"fullPathMaxPreTradeCloseExposure={pre.max():.4f}x on {max_pre_date.date()}")
    for day in ("2020-02-19", "2020-02-28", "2020-03-02", "2020-03-23"):
        ts = pd.Timestamp(day, tz="UTC")
        if ts in frame.index:
            row = frame.loc[ts]
            print(
                f"{day} preCloseExp={row.pre_exposure:.4f} postCloseExp={row.post_exposure:.4f} "
                f"target={row.target:.4f} debtUSD={row.debt_usd:.2f}"
            )

    low = load_daily_low(frame.index)
    margin = drifted_margin_path(active, data, low)
    margin_rows = [
        print_margin_window("GFC", margin, "2007-10-01", "2009-04-30"),
        print_margin_window("COVID", margin, "2020-02-01", "2020-04-30"),
        print_margin_window("FULL", margin, str(begin.date()), str(end.date())),
    ]

    margin.to_csv("external_review_margin_drift.csv")
    summary = {
        "active_realized_mean_gross_exposure": active_realized,
        "matched_constant_target": matched_target,
        "matched_realized_mean_gross_exposure": matched_realized,
        "active_cagr": active["metrics"]["cagr"],
        "matched_cagr": matched["metrics"]["cagr"],
        "timing_residual_cagr": active["metrics"]["cagr"] - matched["metrics"]["cagr"],
        "regime_exposure": regime_rows,
        "apr_dec_2020_exposure": apr_dec,
        "event_attribution": event_rows,
        "margin_windows": margin_rows,
    }
    with open("external_review_roundtrip.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nINTERPRETATION GUARDRAILS")
    print("Event windows are hindsight-defined diagnostics and are not new selection or validation folds.")
    print("The matched control is ex-post calibrated for attribution; it is not a proposed trading strategy.")
    print("Daily-low margin ratios are stricter than close-only checks but do not reproduce broker house-margin rules or intraday path ordering.")
    print("The 2025-2026 interval remains burned and is not used to change any parameter.")
    print("ARTIFACTS external_review_margin_drift.csv external_review_roundtrip.json")


if __name__ == "__main__":
    main()
