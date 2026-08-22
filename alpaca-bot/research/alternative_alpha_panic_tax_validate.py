from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import sqrt
import json

import numpy as np
import pandas as pd
import yfinance as yf

from bot import taxes as tx
import alternative_alpha_qld_tax_validate as qld_base
import alternative_alpha_qld_risk_matched_benchmark as hard_benchmark

START = "2004-01-01"
END = "2025-01-02"  # Burned 2025-2026 interval deliberately not loaded.
BEGIN = pd.Timestamp("2007-01-01", tz="UTC")
FINISH = pd.Timestamp("2024-12-31", tz="UTC")
INITIAL_EUR = 100_000.0
PANIC_RETURN = -0.05
VIX_THRESHOLD = 30.0
PANIC_SESSIONS = 5
BASE_EXPOSURE = 1.00
PANIC_EXPOSURE = 1.25
TRADING_COST = 0.0020
BORROW_SPREAD = 0.015
DIVIDEND_TAX = 0.25


@dataclass
class Lot:
    qty: float
    basis_eur_per_share: float
    acquired: pd.Timestamp


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    idx = pd.DatetimeIndex(out.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out.index = idx.normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


def _history(symbol: str) -> pd.DataFrame:
    d = yf.Ticker(symbol).history(start=START, end=END, auto_adjust=False, actions=True, repair=False)
    if d is None or d.empty:
        raise RuntimeError(f"no history for {symbol}")
    d = _normalize(d)
    out = pd.DataFrame(index=d.index)
    out["open"] = pd.to_numeric(d["Open"], errors="coerce")
    out["close"] = pd.to_numeric(d["Close"], errors="coerce")
    out["adj"] = pd.to_numeric(d["Adj Close"], errors="coerce") if "Adj Close" in d else out["close"]
    out["div"] = pd.to_numeric(d.get("Dividends", 0.0), errors="coerce").fillna(0.0)
    return out.dropna()


def _fred(series_id: str, index: pd.Index, percent: bool = False) -> pd.Series:
    f = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}")
    f.columns = ["date", "value"]
    f["date"] = pd.to_datetime(f["date"], utc=True).dt.normalize()
    f["value"] = pd.to_numeric(f["value"], errors="coerce")
    s = f.dropna().set_index("date").value.sort_index()
    if percent:
        s = s / 100.0
    out = s.reindex(s.index.union(index).sort_values()).ffill().reindex(index)
    if out.isna().any():
        raise RuntimeError(f"invalid FRED series {series_id}")
    return out


def load_market():
    qqq = _history("QQQ")
    vix = _history("^VIX")["close"].reindex(qqq.index).ffill()
    idx = qqq.index
    fx = _fred("DEXUSEU", idx)
    fed = _fred("DFF", idx, percent=True).shift(1).fillna(0.0).clip(lower=0.0)
    begin = idx[idx >= BEGIN][0]
    end = idx[idx <= FINISH][-1]
    if qqq.loc[begin:end].isna().any().any() or vix.loc[begin:end].isna().any():
        raise RuntimeError("missing development data")

    prior_5d = qqq["adj"].shift(1) / qqq["adj"].shift(6) - 1.0
    prior_vix = vix.shift(1)
    target = pd.Series(BASE_EXPOSURE, index=idx, dtype=float)
    remaining = 0
    triggers = []
    for ts in idx:
        if remaining <= 0:
            signal = (
                np.isfinite(prior_5d.loc[ts])
                and np.isfinite(prior_vix.loc[ts])
                and float(prior_5d.loc[ts]) <= PANIC_RETURN
                and float(prior_vix.loc[ts]) >= VIX_THRESHOLD
            )
            if signal:
                remaining = PANIC_SESSIONS
                if begin <= ts <= end:
                    triggers.append(ts)
        if remaining > 0:
            target.loc[ts] = PANIC_EXPOSURE
            remaining -= 1
    return idx, qqq, vix, fx, fed, target, triggers, begin, end


def metrics(path: pd.Series) -> dict[str, float]:
    p = path.dropna()
    r = p.pct_change(fill_method=None).dropna()
    years = (p.index[-1] - p.index[0]).total_seconds() / (365.2425 * 86400)
    cagr = (float(p.iloc[-1]) / INITIAL_EUR) ** (1.0 / years) - 1.0
    vol = float(r.std(ddof=1) * sqrt(252))
    mean_ann = float(r.mean() * 252)
    down = r[r < 0]
    downvol = float(down.std(ddof=1) * sqrt(252)) if len(down) > 1 else np.nan
    return {
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe0": mean_ann / vol if vol > 0 else np.nan,
        "sortino0": mean_ann / downvol if downvol > 0 else np.nan,
        "max_dd": float((p / p.cummax() - 1.0).min()),
    }


def simulate_active(data, cost: float = TRADING_COST, spread: float = BORROW_SPREAD):
    idx, qqq, _, fx, fed, target, triggers, begin, end = data
    dates = idx[(idx >= begin) & (idx <= end)]
    fx0 = float(fx.loc[dates[0]])
    cash = INITIAL_EUR * fx0
    debt = 0.0
    qty = 0.0
    lots: deque[Lot] = deque()
    cgt = divtax = interest = traded = gains = losses = 0.0
    trades = 0
    rows = []

    def nav_at(price: float) -> float:
        return cash + qty * price - debt

    def repay_debt():
        nonlocal cash, debt
        if cash > 0 and debt > 0:
            x = min(cash, debt)
            cash -= x
            debt -= x

    def sell(amount: float, price: float, ts: pd.Timestamp):
        nonlocal cash, qty, cgt, traded, trades, gains, losses
        amount = min(max(amount, 0.0), qty)
        if amount <= 1e-12:
            return
        remaining = amount
        local_tax = 0.0
        while remaining > 1e-10 and lots:
            lot = lots[0]
            used = min(remaining, lot.qty)
            gain = used * price / float(fx.loc[ts]) - used * lot.basis_eur_per_share
            if gain > 0:
                gains += gain
                local_tax += gain * tx.slovenia_rate_for_dates(lot.acquired, ts)
            else:
                losses += -gain
            lot.qty -= used
            remaining -= used
            if lot.qty <= 1e-10:
                lots.popleft()
        if remaining > 1e-7:
            raise RuntimeError("lot underflow")
        notional = amount * price
        cash += notional * (1.0 - cost) - local_tax * float(fx.loc[ts])
        qty -= amount
        cgt += local_tax
        traded += notional
        trades += 1
        repay_debt()

    def buy(amount: float, price: float, ts: pd.Timestamp):
        nonlocal cash, debt, qty, traded, trades
        amount = max(amount, 0.0)
        if amount <= 1e-12:
            return
        notional = amount * price
        total = notional * (1.0 + cost)
        if cash >= total:
            cash -= total
        else:
            debt += total - cash
            cash = 0.0
        lots.append(Lot(amount, price / float(fx.loc[ts]), ts))
        qty += amount
        traded += notional
        trades += 1

    def rebalance(desired: float, price: float, ts: pd.Timestamp):
        n = nav_at(price)
        if n <= 0:
            raise RuntimeError("non-positive NAV")
        desired_mv = desired * n
        current_mv = qty * price
        if current_mv > desired_mv:
            sell((current_mv - desired_mv) / price, price, ts)
        elif desired_mv > current_mv:
            buy((desired_mv - current_mv) / price, price, ts)

    previous_target = None
    for j, ts in enumerate(dates):
        if j and debt > 0:
            charge = debt * (float(fed.loc[ts]) + spread) / 360.0
            debt += charge
            interest += charge

        # Yahoo historical Open/Close are split-adjusted, so no split action is reapplied.
        dv = float(qqq.loc[ts, "div"])
        if dv > 0 and qty > 0:
            gross = qty * dv
            tax_eur = gross / float(fx.loc[ts]) * DIVIDEND_TAX
            cash += gross - tax_eur * float(fx.loc[ts])
            divtax += tax_eur
            repay_debt()

        open_px = float(qqq.loc[ts, "open"])
        desired = float(target.loc[ts])
        if previous_target is None or abs(desired - previous_target) > 1e-12:
            rebalance(desired, open_px, ts)
        previous_target = desired

        close_px = float(qqq.loc[ts, "close"])
        nav_usd = nav_at(close_px)
        exposure = qty * close_px / nav_usd if nav_usd > 0 else np.nan
        margin_ratio = nav_usd / (qty * close_px) if qty > 0 else np.nan
        rows.append(
            {
                "date": ts,
                "nav_eur": nav_usd / float(fx.loc[ts]),
                "target_exposure": desired,
                "close_exposure": exposure,
                "margin_ratio": margin_ratio,
                "debt_usd": debt,
            }
        )

    # Evaluation liquidation: tax is applied, but no extra artificial transaction cost.
    ts = dates[-1]
    close_px = float(qqq.loc[ts, "close"])
    old_cost = cost
    cost = 0.0
    sell(qty, close_px, ts)
    cost = old_cost
    repay_debt()
    rows[-1]["nav_eur"] = (cash - debt) / float(fx.loc[ts])
    frame = pd.DataFrame(rows).set_index("date")
    return {
        "path": frame.nav_eur.astype(float),
        "frame": frame,
        "metrics": metrics(frame.nav_eur.astype(float)),
        "risk_metrics_ex_terminal": metrics(frame.nav_eur.astype(float).iloc[:-1]),
        "cgt": cgt,
        "divtax": divtax,
        "interest_usd": interest,
        "traded_notional_initial_x": traded / (INITIAL_EUR * fx0),
        "trades": trades,
        "trigger_count": len(triggers),
        "realized_gains_eur": gains,
        "realized_losses_eur": losses,
    }


def simulate_passive(data):
    idx, qqq, _, fx, _, _, _, begin, end = data
    dates = idx[(idx >= begin) & (idx <= end)]
    fx0 = float(fx.loc[dates[0]])
    qty = INITIAL_EUR * fx0 / float(qqq.loc[dates[0], "open"])
    lots: deque[Lot] = deque([Lot(qty, float(qqq.loc[dates[0], "open"]) / fx0, dates[0])])
    divtax = 0.0
    path = []
    for ts in dates:
        dv = float(qqq.loc[ts, "div"])
        if dv > 0:
            gross = qty * dv
            tax_eur = gross / float(fx.loc[ts]) * DIVIDEND_TAX
            divtax += tax_eur
            close_px = float(qqq.loc[ts, "close"])
            add = (gross - tax_eur * float(fx.loc[ts])) / close_px
            qty += add
            lots.append(Lot(add, close_px / float(fx.loc[ts]), ts))
        path.append(qty * float(qqq.loc[ts, "close"]) / float(fx.loc[ts]))

    ts = dates[-1]
    terminal_tax = 0.0
    p = float(qqq.loc[ts, "close"])
    for lot in lots:
        gain = lot.qty * p / float(fx.loc[ts]) - lot.qty * lot.basis_eur_per_share
        if gain > 0:
            terminal_tax += gain * tx.slovenia_rate_for_dates(lot.acquired, ts)
    path[-1] -= terminal_tax
    series = pd.Series(path, index=dates, dtype=float)
    return {
        "path": series,
        "metrics": metrics(series),
        "risk_metrics_ex_terminal": metrics(series.iloc[:-1]),
        "cgt": terminal_tax,
        "divtax": divtax,
    }


def main():
    data = load_market()
    active = simulate_active(data)
    passive = simulate_passive(data)
    print("FROZEN PANIC-REVERSAL VALIDATION")
    print("Rule: baseline 1.00x QQQ; if prior 5-session QQQ return <= -5% and prior VIX >=30, hold 1.25x for 5 sessions.")
    print("Execution: next session open using only prior-session information. Burned 2025-2026 data are not loaded.")
    print("\nSTRICT CONTINUOUS EUR AFTER-TAX RESULTS")
    for label, r in (("ACTIVE", active), ("QQQ", passive)):
        m = r["metrics"]
        rm = r["risk_metrics_ex_terminal"]
        print(
            f"{label} CAGR={m['cagr']:.4%} riskVolExTerminal={rm['ann_vol']:.4%} "
            f"Sharpe0ExTerminal={rm['sharpe0']:.3f} Sortino0ExTerminal={rm['sortino0']:.3f} "
            f"DDExTerminal={rm['max_dd']:.4%} CGT={r['cgt']:.2f} DivTax={r['divtax']:.2f}"
        )
    print(f"EXCESS_ACTIVE_VS_QQQ={active['metrics']['cagr']-passive['metrics']['cagr']:+.4%}")
    print(
        f"ACTIVE triggers={active['trigger_count']} trades={active['trades']} turnoverInitial={active['traded_notional_initial_x']:.3f}x "
        f"interestUSD={active['interest_usd']:.2f} gainsEUR={active['realized_gains_eur']:.2f} lossesEUR={active['realized_losses_eur']:.2f}"
    )

    print("\nCOST / FINANCING STRESS")
    stress = []
    for cost, spread in ((0.002, 0.015), (0.004, 0.020), (0.008, 0.030)):
        r = simulate_active(data, cost=cost, spread=spread)
        ex = r["metrics"]["cagr"] - passive["metrics"]["cagr"]
        stress.append({"cost": cost, "spread": spread, "cagr": r["metrics"]["cagr"], "excess_vs_qqq": ex})
        print(f"cost={cost:.2%} spread={spread:.2%} CAGR={r['metrics']['cagr']:.4%} exQQQ={ex:+.4%}")

    # Hard passive benchmark: long-held QQQ/QLD mix calibrated to the active strategy's
    # realized volatility. This preserves passive long-holding tax treatment.
    hard_data = qld_base.load_market()
    matched_weight, matched = hard_benchmark.calibrate_risk_match(
        hard_data, active["risk_metrics_ex_terminal"]["ann_vol"]
    )
    print("\nPASSIVE RISK-MATCHED QQQ/QLD BUY-AND-HOLD")
    print(
        f"matchedQLDWeight={matched_weight:.4%} CAGR={matched['metrics']['cagr']:.4%} "
        f"riskVolExTerminal={matched['risk_metrics_ex_terminal']['ann_vol']:.4%} "
        f"Sharpe0ExTerminal={matched['risk_metrics_ex_terminal']['sharpe0']:.3f} "
        f"DDExTerminal={matched['risk_metrics_ex_terminal']['max_dd']:.4%}"
    )
    hard_residual = active["metrics"]["cagr"] - matched["metrics"]["cagr"]
    print(f"ACTIVE_MINUS_RISK_MATCH_CAGR={hard_residual:+.4%}")

    min_margin = float(active["frame"].margin_ratio.dropna().min())
    max_close_exp = float(active["frame"].close_exposure.dropna().max())
    print(f"\nMARGIN_DIAGNOSTIC minCloseMarginRatio={min_margin:.2%} maxCloseExposure={max_close_exp:.4f}x")

    verdict = "PASS" if active["metrics"]["cagr"] > passive["metrics"]["cagr"] and hard_residual > 0 else "FAIL"
    print(f"STAGE2_PANIC_VERDICT {verdict}")
    if verdict == "FAIL":
        print("Do not open the burned holdout for selection; the frozen panic overlay has not established after-tax risk-matched alpha.")

    active["frame"].to_csv("alternative_alpha_panic_path.csv")
    summary = {
        "holdout_loaded": False,
        "rule": {
            "panic_return": PANIC_RETURN,
            "vix_threshold": VIX_THRESHOLD,
            "panic_sessions": PANIC_SESSIONS,
            "base_exposure": BASE_EXPOSURE,
            "panic_exposure": PANIC_EXPOSURE,
        },
        "active": active["metrics"],
        "active_risk_ex_terminal": active["risk_metrics_ex_terminal"],
        "passive_qqq": passive["metrics"],
        "passive_qqq_risk_ex_terminal": passive["risk_metrics_ex_terminal"],
        "active_cgt_eur": active["cgt"],
        "active_divtax_eur": active["divtax"],
        "active_interest_usd": active["interest_usd"],
        "active_trigger_count": active["trigger_count"],
        "active_turnover_initial_x": active["traded_notional_initial_x"],
        "cost_stress": stress,
        "risk_matched_qld_weight": matched_weight,
        "risk_matched_after_tax_metrics": matched["metrics"],
        "risk_matched_risk_metrics_ex_terminal": matched["risk_metrics_ex_terminal"],
        "active_minus_risk_match_cagr": hard_residual,
        "min_close_margin_ratio": min_margin,
        "max_close_exposure": max_close_exp,
        "verdict": verdict,
    }
    with open("alternative_alpha_panic_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("ARTIFACTS alternative_alpha_panic_path.csv alternative_alpha_panic_summary.json")


if __name__ == "__main__":
    main()
