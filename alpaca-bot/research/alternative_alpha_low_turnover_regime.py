from __future__ import annotations

import json

import numpy as np
import pandas as pd

import alternative_alpha_qld_tax_validate as base
import alternative_alpha_qld_risk_matched_benchmark as hard

ENTRY_LOOKBACK = 252
EXIT_LOOKBACK = 126
MA_WINDOW = 200
CONFIRM_MONTHS = 2
BASE_COST = 0.0020


def build_low_turnover_events(data):
    idx, close, adj, divs, fx, _, begin, end = data
    q = adj["QQQ"]
    prior = q.shift(1)
    ma = q.shift(1).rolling(MA_WINDOW, min_periods=MA_WINDOW).mean()
    entry_mom = q.shift(1) / q.shift(1 + ENTRY_LOOKBACK) - 1.0
    exit_mom = q.shift(1) / q.shift(1 + EXIT_LOOKBACK) - 1.0
    months = pd.Series(idx.tz_localize(None).to_period("M"), index=idx)
    first = months.ne(months.shift(1))

    state = "QQQ"
    on_count = off_count = 0
    events = {}
    switches = []
    for ts in idx[first]:
        if ts < begin or ts > end:
            continue
        if not (np.isfinite(prior.loc[ts]) and np.isfinite(ma.loc[ts])):
            continue

        on = np.isfinite(entry_mom.loc[ts]) and float(prior.loc[ts]) > float(ma.loc[ts]) and float(entry_mom.loc[ts]) > 0.0
        off = np.isfinite(exit_mom.loc[ts]) and float(prior.loc[ts]) < float(ma.loc[ts]) and float(exit_mom.loc[ts]) < 0.0

        if state == "QQQ":
            on_count = on_count + 1 if on else 0
            off_count = 0
            if on_count >= CONFIRM_MONTHS:
                state = "QLD"
                switches.append((ts, "QQQ", "QLD"))
                on_count = 0
        else:
            off_count = off_count + 1 if off else 0
            on_count = 0
            if off_count >= CONFIRM_MONTHS:
                state = "QQQ"
                switches.append((ts, "QLD", "QQQ"))
                off_count = 0
        events[ts] = state

    return (idx, close, adj, divs, fx, events, begin, end), switches


def show(label, r):
    m = r["metrics"]
    risk = base.metrics(r["path"].iloc[:-1])
    print(
        f"{label} CAGR={m['cagr']:.4%} riskVolExTerminal={risk['ann_vol']:.4%} "
        f"Sharpe0ExTerminal={risk['sharpe0']:.3f} Sortino0ExTerminal={risk['sortino0']:.3f} "
        f"DDExTerminal={risk['max_dd']:.4%} CGT={r['cgt']:.2f} DivTax={r['divtax']:.2f}"
    )
    return risk


def main():
    original = base.load_market()
    data, switch_log = build_low_turnover_events(original)
    active = base.simulate_rotation(data, BASE_COST)
    qqq = base.simulate_passive(data, "QQQ")
    qld = base.simulate_passive(data, "QLD")

    print("LOW-TURNOVER QQQ/QLD REGIME — FIXED EXPLORATORY RULE")
    print("Start QQQ; monthly review. Enter QLD after 2 consecutive months: QQQ>200d MA and 252d return>0.")
    print("Exit to QQQ after 2 consecutive months: QQQ<200d MA and 126d return<0.")
    print("2025-2026 burned interval is not loaded; no parameter variants are tested.")
    print(f"switches={len(switch_log)}")
    for ts, old, new in switch_log:
        print(f"SWITCH {ts.date()} {old}->{new}")

    print("\nSTRICT CONTINUOUS EUR AFTER-TAX RESULTS")
    active_risk = show("ACTIVE", active)
    qqq_risk = show("QQQ", qqq)
    show("QLD", qld)
    print(f"ACTIVE_MINUS_QQQ_CAGR={active['metrics']['cagr']-qqq['metrics']['cagr']:+.4%}")

    matched_weight, matched = hard.calibrate_risk_match(original, active_risk["ann_vol"])
    print("\nHARD PASSIVE RISK-MATCHED BUY-AND-HOLD")
    hard.show_static("RISK_MATCH", matched)
    hard_residual = active["metrics"]["cagr"] - matched["metrics"]["cagr"]
    print(f"ACTIVE_MINUS_RISK_MATCH_CAGR={hard_residual:+.4%}")

    print("\nSWITCH-COST STRESS")
    stress = []
    for cost in (0.0020, 0.0040, 0.0080):
        r = base.simulate_rotation(data, cost)
        ex = r["metrics"]["cagr"] - qqq["metrics"]["cagr"]
        stress.append({"cost": cost, "cagr": r["metrics"]["cagr"], "excess_vs_qqq": ex})
        print(f"cost={cost:.2%} CAGR={r['metrics']['cagr']:.4%} exQQQ={ex:+.4%}")

    gates = {
        "beat_qqq": active["metrics"]["cagr"] > qqq["metrics"]["cagr"],
        "beat_risk_match": hard_residual > 0.0,
        "beat_qqq_at_40bp": stress[1]["excess_vs_qqq"] > 0.0,
    }
    passed = all(gates.values())
    print(f"\nGATES {gates}")
    print(f"LOW_TURNOVER_VERDICT {'PASS' if passed else 'FAIL'}")
    if passed:
        print("Exploratory gate passed; this warrants reproducibility and broker-parity work, not an alpha claim.")
    else:
        print("Rule rejected; do not use the burned holdout to rescue or retune it.")

    summary = {
        "holdout_loaded": False,
        "rule": {
            "entry_lookback": ENTRY_LOOKBACK,
            "exit_lookback": EXIT_LOOKBACK,
            "ma_window": MA_WINDOW,
            "confirm_months": CONFIRM_MONTHS,
        },
        "switches": [{"date": str(ts.date()), "from": old, "to": new} for ts, old, new in switch_log],
        "active": active["metrics"],
        "active_risk_ex_terminal": active_risk,
        "qqq": qqq["metrics"],
        "qqq_risk_ex_terminal": qqq_risk,
        "qld": qld["metrics"],
        "risk_matched_qld_weight": matched_weight,
        "risk_matched": matched["metrics"],
        "risk_matched_risk_ex_terminal": matched["risk_metrics_ex_terminal"],
        "active_minus_risk_match_cagr": hard_residual,
        "cost_stress": stress,
        "gates": gates,
        "verdict": "PASS" if passed else "FAIL",
    }
    with open("alternative_alpha_low_turnover_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    active["frame"].to_csv("alternative_alpha_low_turnover_path.csv")
    print("ARTIFACTS alternative_alpha_low_turnover_path.csv alternative_alpha_low_turnover_summary.json")


if __name__ == "__main__":
    main()
