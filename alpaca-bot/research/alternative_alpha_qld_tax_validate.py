from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import sqrt
import json

import numpy as np
import pandas as pd
import yfinance as yf

from bot import taxes as tx

START = "2004-01-01"
END = "2025-01-02"  # Stage 2 remains development-only; burned holdout is not loaded.
BEGIN = pd.Timestamp("2007-01-01", tz="UTC")
FINISH = pd.Timestamp("2024-12-31", tz="UTC")
INITIAL_EUR = 100_000.0
LOOKBACK = 126
MA_WINDOW = 200
TRADING_COST = 0.0020
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


def _symbol(symbol: str) -> pd.DataFrame:
    d = yf.Ticker(symbol).history(start=START, end=END, auto_adjust=False, actions=True, repair=False)
    if d is None or d.empty:
        raise RuntimeError(f"no history for {symbol}")
    d = _normalize(d)
    close = pd.to_numeric(d["Close"], errors="coerce")
    adj = pd.to_numeric(d["Adj Close"], errors="coerce") if "Adj Close" in d else close
    div = pd.to_numeric(d.get("Dividends", 0.0), errors="coerce").fillna(0.0)
    # Yahoo historical Close is already split-adjusted. Do NOT also multiply share
    # quantities by Stock Splits; doing so double-counts QLD's split history.
    return pd.DataFrame({"close": close, "adj": adj, "div": div}).dropna()


def _fred_fx(index: pd.Index) -> pd.Series:
    f = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU")
    f.columns = ["date", "value"]
    f["date"] = pd.to_datetime(f["date"], utc=True).dt.normalize()
    f["value"] = pd.to_numeric(f["value"], errors="coerce")
    s = f.dropna().set_index("date").value.sort_index()
    out = s.reindex(s.index.union(index).sort_values()).ffill().reindex(index)
    if out.isna().any():
        raise RuntimeError("invalid DEXUSEU history")
    return out


def load_market():
    raw = {s: _symbol(s) for s in ("QQQ", "QLD")}
    idx = raw["QQQ"].index
    close = pd.DataFrame({s: raw[s]["close"].reindex(idx).ffill() for s in raw})
    adj = pd.DataFrame({s: raw[s]["adj"].reindex(idx).ffill() for s in raw})
    divs = {s: raw[s]["div"].reindex(idx).fillna(0.0) for s in raw}
    fx = _fred_fx(idx)

    begin = idx[idx >= BEGIN][0]
    end = idx[idx <= FINISH][-1]
    if close.loc[begin:end, ["QQQ", "QLD"]].isna().any().any():
        raise RuntimeError("missing QQQ/QLD development prices")

    q = adj["QQQ"]
    prior = q.shift(1)
    momentum = q.shift(1) / q.shift(1 + LOOKBACK) - 1.0
    ma = q.shift(1).rolling(MA_WINDOW, min_periods=MA_WINDOW).mean()
    months = pd.Series(idx.tz_localize(None).to_period("M"), index=idx)
    events = {}
    for ts in idx[months.ne(months.shift(1))]:
        if ts < begin or ts > end:
            continue
        if np.isfinite(momentum.loc[ts]) and np.isfinite(ma.loc[ts]):
            events[ts] = "QLD" if (float(momentum.loc[ts]) > 0.0 and float(prior.loc[ts]) > float(ma.loc[ts])) else "QQQ"
    return idx, close, adj, divs, fx, events, begin, end


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


def _tax_on_sale(lots: deque[Lot], qty: float, price: float, fx: float, ts: pd.Timestamp) -> tuple[float, float, float]:
    remaining = qty
    tax = gains = losses = 0.0
    while remaining > 1e-10 and lots:
        lot = lots[0]
        used = min(remaining, lot.qty)
        gain = used * price / fx - used * lot.basis_eur_per_share
        if gain > 0:
            gains += gain
            tax += gain * tx.slovenia_rate_for_dates(lot.acquired, ts)
        else:
            losses += -gain
        lot.qty -= used
        remaining -= used
        if lot.qty <= 1e-10:
            lots.popleft()
    if remaining > 1e-7:
        raise RuntimeError("tax lot underflow")
    return tax, gains, losses


def simulate_rotation(data, switch_cost: float = TRADING_COST):
    idx, close, _, divs, fx, events, begin, end = data
    dates = idx[(idx >= begin) & (idx <= end)]
    fx0 = float(fx.loc[dates[0]])
    cash = INITIAL_EUR * fx0
    held: str | None = None
    qty = 0.0
    lots: deque[Lot] = deque()
    cgt = divtax = traded = gains = losses = 0.0
    switches = buys = sells = 0
    rows = []
    choice = events.get(dates[0], "QQQ")

    def price(symbol: str, ts: pd.Timestamp) -> float:
        return float(close.loc[ts, symbol])

    def nav_usd(ts: pd.Timestamp) -> float:
        return cash + (qty * price(held, ts) if held else 0.0)

    def buy_all(symbol: str, ts: pd.Timestamp, charge_cost: bool = True):
        nonlocal cash, held, qty, traded, buys
        p = price(symbol, ts)
        fee = switch_cost if charge_cost else 0.0
        add = cash / (p * (1.0 + fee))
        if add <= 0:
            return
        notional = add * p
        cash -= notional * (1.0 + fee)
        if charge_cost:
            traded += notional
        lots.append(Lot(add, p / float(fx.loc[ts]), ts))
        held = symbol
        qty = add
        buys += 1

    def sell_all(ts: pd.Timestamp, charge_cost: bool = True):
        nonlocal cash, held, qty, cgt, traded, sells, gains, losses
        if held is None or qty <= 0:
            return
        p = price(held, ts)
        local_tax, local_gains, local_losses = _tax_on_sale(lots, qty, p, float(fx.loc[ts]), ts)
        notional = qty * p
        fee = switch_cost if charge_cost else 0.0
        cash += notional * (1.0 - fee) - local_tax * float(fx.loc[ts])
        if charge_cost:
            traded += notional
        cgt += local_tax
        gains += local_gains
        losses += local_losses
        qty = 0.0
        held = None
        sells += 1

    for j, ts in enumerate(dates):
        if held is not None:
            dv = float(divs[held].loc[ts])
            if dv > 0 and qty > 0:
                gross = qty * dv
                tax_eur = gross / float(fx.loc[ts]) * DIVIDEND_TAX
                net_usd = gross - tax_eur * float(fx.loc[ts])
                divtax += tax_eur
                p = price(held, ts)
                add = net_usd / p  # zero-cost automatic reinvestment, same convention as passive
                qty += add
                lots.append(Lot(add, p / float(fx.loc[ts]), ts))

        if ts in events:
            choice = events[ts]

        switched = False
        if j == 0:
            buy_all(choice, ts, charge_cost=True)
        elif choice != held:
            sell_all(ts, charge_cost=True)
            buy_all(choice, ts, charge_cost=True)
            switches += 1
            switched = True

        rows.append(
            {
                "date": ts,
                "nav_eur": nav_usd(ts) / float(fx.loc[ts]),
                "held": held,
                "switch": switched,
            }
        )

    # Terminal liquidation is an evaluation event: apply tax but no artificial transaction cost.
    sell_all(dates[-1], charge_cost=False)
    rows[-1]["nav_eur"] = cash / float(fx.loc[dates[-1]])
    frame = pd.DataFrame(rows).set_index("date")
    return {
        "path": frame.nav_eur.astype(float),
        "frame": frame,
        "metrics": metrics(frame.nav_eur.astype(float)),
        "cgt": cgt,
        "divtax": divtax,
        "traded_notional_initial_x": traded / (INITIAL_EUR * fx0),
        "switches": switches,
        "buys": buys,
        "sells": sells,
        "realized_gains": gains,
        "realized_losses": losses,
    }


def simulate_passive(data, symbol: str):
    idx, close, _, divs, fx, _, begin, end = data
    dates = idx[(idx >= begin) & (idx <= end)]
    fx0 = float(fx.loc[dates[0]])
    p0 = float(close.loc[dates[0], symbol])
    qty = INITIAL_EUR * fx0 / p0
    lots: deque[Lot] = deque([Lot(qty, p0 / fx0, dates[0])])
    divtax = 0.0
    path = []

    for ts in dates:
        dv = float(divs[symbol].loc[ts])
        if dv > 0:
            gross = qty * dv
            tax_eur = gross / float(fx.loc[ts]) * DIVIDEND_TAX
            divtax += tax_eur
            p = float(close.loc[ts, symbol])
            add = (gross - tax_eur * float(fx.loc[ts])) / p
            qty += add
            lots.append(Lot(add, p / float(fx.loc[ts]), ts))
        path.append(qty * float(close.loc[ts, symbol]) / float(fx.loc[ts]))

    ts = dates[-1]
    terminal_tax, gains, losses = _tax_on_sale(lots, qty, float(close.loc[ts, symbol]), float(fx.loc[ts]), ts)
    path[-1] -= terminal_tax
    series = pd.Series(path, index=dates, dtype=float)
    return {
        "path": series,
        "metrics": metrics(series),
        "cgt": terminal_tax,
        "divtax": divtax,
        "realized_gains": gains,
        "realized_losses": losses,
    }


def adjusted_attribution(data):
    idx, _, adj, _, _, events, begin, end = data
    dates = idx[(idx >= begin) & (idx <= end)]
    qret = adj.loc[dates, "QQQ"].pct_change(fill_method=None).fillna(0.0)
    lret = adj.loc[dates, "QLD"].pct_change(fill_method=None).fillna(0.0)
    state = pd.Series(0.0, index=dates)
    choice = events.get(dates[0], "QQQ")
    for ts in dates:
        if ts in events:
            choice = events[ts]
        state.loc[ts] = 1.0 if choice == "QLD" else 0.0
    w = float(state.mean())
    strategy_ret = (1.0 - state) * qret + state * lret
    matched_ret = (1.0 - w) * qret + w * lret
    strategy_path = (1.0 + strategy_ret).cumprod() * INITIAL_EUR
    matched_path = (1.0 + matched_ret).cumprod() * INITIAL_EUR
    return {
        "qld_day_fraction": w,
        "costless_rotation_metrics": metrics(strategy_path),
        "costless_static_mix_metrics": metrics(matched_path),
    }


def show(label: str, result: dict):
    m = result["metrics"]
    print(
        f"{label} CAGR={m['cagr']:.4%} vol={m['ann_vol']:.4%} Sharpe0={m['sharpe0']:.3f} "
        f"Sortino0={m['sortino0']:.3f} DD={m['max_dd']:.4%} CGT={result['cgt']:.2f} DivTax={result['divtax']:.2f}"
    )


def main():
    data = load_market()
    _, _, _, _, _, events, begin, end = data
    print("FROZEN_RULE QLD if QQQ 126-session momentum>0 and QQQ>200-session MA; else QQQ")
    print(f"CONTINUOUS_DEVELOPMENT_ONLY {begin.date()}..{end.date()} holdoutLoaded=false events={len(events)}")
    print("PRICE_CONVENTION Yahoo historical Close is split-adjusted; split actions are not applied to quantities")

    active = simulate_rotation(data, TRADING_COST)
    qqq = simulate_passive(data, "QQQ")
    qld = simulate_passive(data, "QLD")
    print("\nSTRICT EUR AFTER-TAX RESULTS")
    show("ACTIVE", active)
    show("QQQ", qqq)
    show("QLD", qld)
    print(
        f"EXCESS active-vs-QQQ={active['metrics']['cagr']-qqq['metrics']['cagr']:+.4%} "
        f"active-vs-QLD={active['metrics']['cagr']-qld['metrics']['cagr']:+.4%}"
    )
    print(
        f"ACTIVE switches={active['switches']} tradedNotionalInitial={active['traded_notional_initial_x']:.3f}x "
        f"realizedGainsEUR={active['realized_gains']:.2f} realizedLossesEUR={active['realized_losses']:.2f}"
    )

    attr = adjusted_attribution(data)
    am = attr["costless_rotation_metrics"]
    mm = attr["costless_static_mix_metrics"]
    print("\nLEVERAGE / TIMING ATTRIBUTION — adjusted USD, costless diagnostic")
    print(f"QLD day fraction={attr['qld_day_fraction']:.4%}")
    print(f"ROTATION CAGR={am['cagr']:.4%} vol={am['ann_vol']:.4%} DD={am['max_dd']:.4%}")
    print(f"STATIC_MIX CAGR={mm['cagr']:.4%} vol={mm['ann_vol']:.4%} DD={mm['max_dd']:.4%}")
    print(f"TIMING_RESIDUAL_CAGR={am['cagr']-mm['cagr']:+.4%}")

    print("\nSWITCH-COST SENSITIVITY — continuous EUR tax path")
    stress = []
    for c in (0.0020, 0.0040, 0.0080):
        r = simulate_rotation(data, c)
        ex = r["metrics"]["cagr"] - qqq["metrics"]["cagr"]
        stress.append({"cost": c, "cagr": r["metrics"]["cagr"], "excess_vs_qqq": ex})
        print(f"cost={c:.2%} CAGR={r['metrics']['cagr']:.4%} exQQQ={ex:+.4%} DD={r['metrics']['max_dd']:.4%}")

    active["frame"].to_csv("alternative_alpha_qld_tax_path.csv")
    summary = {
        "start": str(begin.date()),
        "end": str(end.date()),
        "holdout_loaded": False,
        "price_convention": "Yahoo historical Close split-adjusted; split actions not reapplied",
        "rule": {"lookback": LOOKBACK, "ma_window": MA_WINDOW},
        "active": active["metrics"],
        "qqq": qqq["metrics"],
        "qld": qld["metrics"],
        "active_cgt_eur": active["cgt"],
        "active_divtax_eur": active["divtax"],
        "active_switches": active["switches"],
        "active_traded_notional_initial_x": active["traded_notional_initial_x"],
        "active_realized_gains_eur": active["realized_gains"],
        "active_realized_losses_eur": active["realized_losses"],
        "qqq_cgt_eur": qqq["cgt"],
        "qld_cgt_eur": qld["cgt"],
        "attribution": attr,
        "cost_stress": stress,
    }
    with open("alternative_alpha_qld_tax_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("\nLIMITATIONS active tax remains conservative: no same-year loss offsets or normalized expenses; DEXUSEU is a research FX proxy")
    print("ARTIFACTS alternative_alpha_qld_tax_path.csv alternative_alpha_qld_tax_summary.json")


if __name__ == "__main__":
    main()
