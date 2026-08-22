from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

import alternative_alpha_qld_tax_validate as base


@dataclass
class StaticLot:
    qty: float
    basis_eur_per_share: float
    acquired: pd.Timestamp


def path_risk_metrics_ex_terminal(path: pd.Series) -> dict[str, float]:
    """Risk metrics excluding the artificial terminal-liquidation tax jump.

    Daily capital-gains taxes paid during the active strategy remain in its path;
    only the final evaluation liquidation is removed from risk estimation.
    """
    p = path.dropna()
    if len(p) > 1:
        p = p.iloc[:-1]
    return base.metrics(p)


def _terminal_tax(lots: deque[StaticLot], price: float, fx: float, ts: pd.Timestamp) -> tuple[float, float, float]:
    q = deque(base.Lot(l.qty, l.basis_eur_per_share, l.acquired) for l in lots)
    qty = float(sum(l.qty for l in q))
    return base._tax_on_sale(q, qty, price, fx, ts)


def simulate_static_mix(data, qld_weight: float) -> dict:
    if not (0.0 <= qld_weight <= 1.0):
        raise ValueError("qld_weight must be in [0, 1]")

    idx, close, _, divs, fx, _, begin, end = data
    dates = idx[(idx >= begin) & (idx <= end)]
    fx0 = float(fx.loc[dates[0]])
    initial_usd = base.INITIAL_EUR * fx0
    weights = {"QQQ": 1.0 - qld_weight, "QLD": qld_weight}
    qty: dict[str, float] = {}
    lots: dict[str, deque[StaticLot]] = {"QQQ": deque(), "QLD": deque()}
    dividend_tax = 0.0

    for symbol, weight in weights.items():
        p = float(close.loc[dates[0], symbol])
        shares = initial_usd * weight / p
        qty[symbol] = shares
        if shares > 0:
            lots[symbol].append(StaticLot(shares, p / fx0, dates[0]))

    pre_terminal = []
    for ts in dates:
        for symbol in ("QQQ", "QLD"):
            dv = float(divs[symbol].loc[ts])
            if dv > 0 and qty[symbol] > 0:
                gross_usd = qty[symbol] * dv
                tax_eur = gross_usd / float(fx.loc[ts]) * base.DIVIDEND_TAX
                dividend_tax += tax_eur
                net_usd = gross_usd - tax_eur * float(fx.loc[ts])
                p = float(close.loc[ts, symbol])
                add = net_usd / p
                qty[symbol] += add
                lots[symbol].append(StaticLot(add, p / float(fx.loc[ts]), ts))

        value_usd = sum(qty[s] * float(close.loc[ts, s]) for s in ("QQQ", "QLD"))
        pre_terminal.append(value_usd / float(fx.loc[ts]))

    pre_terminal_path = pd.Series(pre_terminal, index=dates, dtype=float)
    ts = dates[-1]
    terminal_tax = terminal_gains = terminal_losses = 0.0
    for symbol in ("QQQ", "QLD"):
        tax, gains, losses = _terminal_tax(
            lots[symbol],
            float(close.loc[ts, symbol]),
            float(fx.loc[ts]),
            ts,
        )
        terminal_tax += tax
        terminal_gains += gains
        terminal_losses += losses

    after_tax_path = pre_terminal_path.copy()
    after_tax_path.iloc[-1] -= terminal_tax
    return {
        "qld_weight": qld_weight,
        "path": after_tax_path,
        "pre_terminal_path": pre_terminal_path,
        "metrics": base.metrics(after_tax_path),
        "risk_metrics_ex_terminal": base.metrics(pre_terminal_path),
        "terminal_cgt_eur": terminal_tax,
        "dividend_tax_eur": dividend_tax,
        "terminal_gains_eur": terminal_gains,
        "terminal_losses_eur": terminal_losses,
    }


def active_risk_metrics_ex_terminal(active: dict) -> dict[str, float]:
    # active['path'] includes the terminal evaluation liquidation only in the final row.
    return base.metrics(active["path"].iloc[:-1])


def calibrate_risk_match(data, active_ann_vol: float) -> tuple[float, dict]:
    best: tuple[float, float, dict] | None = None
    for weight in np.linspace(0.0, 1.0, 101):
        r = simulate_static_mix(data, float(weight))
        err = abs(float(r["risk_metrics_ex_terminal"]["ann_vol"]) - active_ann_vol)
        if best is None or err < best[0]:
            best = (err, float(weight), r)

    assert best is not None
    lo = max(0.0, best[1] - 0.02)
    hi = min(1.0, best[1] + 0.02)
    for weight in np.linspace(lo, hi, 81):
        r = simulate_static_mix(data, float(weight))
        err = abs(float(r["risk_metrics_ex_terminal"]["ann_vol"]) - active_ann_vol)
        if err < best[0]:
            best = (err, float(weight), r)
    return best[1], best[2]


def show_static(label: str, result: dict) -> None:
    m = result["metrics"]
    r = result["risk_metrics_ex_terminal"]
    print(
        f"{label} qldWeight={result['qld_weight']:.4%} afterTaxCAGR={m['cagr']:.4%} "
        f"riskVolExTerminal={r['ann_vol']:.4%} Sharpe0ExTerminal={r['sharpe0']:.3f} "
        f"Sortino0ExTerminal={r['sortino0']:.3f} DDExTerminal={r['max_dd']:.4%} "
        f"terminalCGT={result['terminal_cgt_eur']:.2f} DivTax={result['dividend_tax_eur']:.2f}"
    )


def main() -> None:
    data = base.load_market()
    active = base.simulate_rotation(data, base.TRADING_COST)
    active_risk = active_risk_metrics_ex_terminal(active)

    matched_weight, matched = calibrate_risk_match(data, active_risk["ann_vol"])
    day_fraction = base.adjusted_attribution(data)["qld_day_fraction"]
    fraction_mix = simulate_static_mix(data, float(day_fraction))

    print("HARD PASSIVE BUY-AND-HOLD BENCHMARK")
    print("No discretionary trades, zero transaction cost, dividends reinvested in the same ETF, terminal tax only.")
    print("Yahoo historical Close is treated as split-adjusted; split actions are not reapplied.")
    print("The 2025-2026 burned interval is not loaded.")
    print(
        f"ACTIVE afterTaxCAGR={active['metrics']['cagr']:.4%} riskVolExTerminal={active_risk['ann_vol']:.4%} "
        f"Sharpe0ExTerminal={active_risk['sharpe0']:.3f} Sortino0ExTerminal={active_risk['sortino0']:.3f} "
        f"DDExTerminal={active_risk['max_dd']:.4%}"
    )

    print("\nRISK-MATCHED STATIC BUY-AND-HOLD MIX")
    show_static("RISK_MATCH", matched)
    print(
        f"ACTIVE_MINUS_RISK_MATCH_CAGR={active['metrics']['cagr']-matched['metrics']['cagr']:+.4%} "
        f"volMismatch={active_risk['ann_vol']-matched['risk_metrics_ex_terminal']['ann_vol']:+.4%}"
    )

    print("\nSTATIC MIX USING ACTIVE QLD DAY FRACTION")
    show_static("DAY_FRACTION_MIX", fraction_mix)
    print(f"ACTIVE_MINUS_DAY_FRACTION_MIX_CAGR={active['metrics']['cagr']-fraction_mix['metrics']['cagr']:+.4%}")

    verdict = "PASS" if active["metrics"]["cagr"] > matched["metrics"]["cagr"] else "FAIL"
    print(f"\nSTAGE2_RISK_MATCH_VERDICT {verdict}")
    if verdict == "PASS":
        print("Active beats a long-held passive QQQ/QLD blend at approximately matched realized volatility; further robustness is warranted before any holdout description.")
    else:
        print("Active does not beat a long-held passive QQQ/QLD blend at matched realized volatility; timing alpha is not established.")

    summary = {
        "holdout_loaded": False,
        "active_after_tax_cagr": active["metrics"]["cagr"],
        "active_risk_metrics_ex_terminal": active_risk,
        "risk_matched_qld_weight": matched_weight,
        "risk_matched": {
            "after_tax_metrics": matched["metrics"],
            "risk_metrics_ex_terminal": matched["risk_metrics_ex_terminal"],
            "terminal_cgt_eur": matched["terminal_cgt_eur"],
            "dividend_tax_eur": matched["dividend_tax_eur"],
        },
        "day_fraction_qld_weight": day_fraction,
        "day_fraction_mix": {
            "after_tax_metrics": fraction_mix["metrics"],
            "risk_metrics_ex_terminal": fraction_mix["risk_metrics_ex_terminal"],
            "terminal_cgt_eur": fraction_mix["terminal_cgt_eur"],
            "dividend_tax_eur": fraction_mix["dividend_tax_eur"],
        },
        "active_minus_risk_match_cagr": active["metrics"]["cagr"] - matched["metrics"]["cagr"],
        "verdict": verdict,
    }
    with open("alternative_alpha_qld_risk_match_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("ARTIFACT alternative_alpha_qld_risk_match_summary.json")


if __name__ == "__main__":
    main()
