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
END = "2026-08-21"
BEGIN = pd.Timestamp("2006-01-01", tz="UTC")
FINISH = pd.Timestamp("2026-08-20", tz="UTC")
INITIAL_EUR = 100_000.0
TARGET_VOL = 0.25
LOOKBACK = 20
MIN_EXPOSURE = 0.50
MAX_EXPOSURE = 1.75
NO_TRADE_BAND = 0.20
TRADING_COST = 0.0020
BORROW_SPREAD = 0.015
DIVIDEND_TAX = 0.25
MAINT_MARGIN = 0.30


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
    split = pd.to_numeric(d.get("Stock Splits", 0.0), errors="coerce").fillna(0.0)
    out = pd.DataFrame({"close": close, "adj": adj, "div": div, "split": split}).dropna()
    return out[(out.close > 0) & (out.adj > 0)]


def _fred(series_id: str, index: pd.Index, pct: bool = False) -> pd.Series:
    f = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}")
    f.columns = ["date", "value"]
    f["date"] = pd.to_datetime(f["date"], utc=True).dt.normalize()
    f["value"] = pd.to_numeric(f["value"], errors="coerce")
    s = f.dropna().set_index("date").value.sort_index()
    if pct:
        s = s / 100.0
    out = s.reindex(s.index.union(index).sort_values()).ffill().reindex(index)
    if out.isna().any():
        raise RuntimeError(f"invalid {series_id}")
    return out


def load_market():
    raw = {s: _symbol(s) for s in ("QQQ", "SPY")}
    first = max(v.index[0] for v in raw.values())
    last = min(v.index[-1] for v in raw.values())
    idx = raw["QQQ"].index.union(raw["SPY"].index)
    idx = idx[(idx >= first) & (idx <= last)].sort_values()
    close = pd.DataFrame({s: raw[s].close.reindex(idx).ffill() for s in raw}).dropna()
    adj = pd.DataFrame({s: raw[s].adj.reindex(close.index).ffill() for s in raw}).dropna()
    idx = close.index.intersection(adj.index)
    close = close.loc[idx]
    adj = adj.loc[idx]
    divs = {s: raw[s]["div"].reindex(idx).fillna(0.0) for s in raw}
    splits = {s: raw[s].split.reindex(idx).fillna(0.0) for s in raw}
    fx = _fred("DEXUSEU", idx)
    fed = _fred("DFF", idx, pct=True).shift(1).fillna(0.0).clip(lower=0.0)

    # Frozen timing. At decision close D, this excludes the return ending at D.
    # Execution is on the next exchange session E. Thus the latest return used in
    # the signal ends two exchange sessions before E. This is intentional parity
    # with the frozen research, not a post-review timing change.
    qret = adj["QQQ"].pct_change(fill_method=None)
    qvol = qret.rolling(LOOKBACK, min_periods=LOOKBACK).std(ddof=1).shift(1) * sqrt(252)
    events = {}
    month = None
    for i, ts in enumerate(idx):
        key = (ts.year, ts.month)
        if month is None:
            month = key
            continue
        if key == month:
            continue
        month = key
        realized = float(qvol.iloc[i - 1]) if i else np.nan
        if np.isfinite(realized) and realized > 0:
            events[ts] = float(np.clip(TARGET_VOL / realized, MIN_EXPOSURE, MAX_EXPOSURE))
    return idx, close, divs, splits, fx, fed, events


def metrics(path: pd.Series) -> dict[str, float]:
    r = path.pct_change(fill_method=None).dropna()
    years = (path.index[-1] - path.index[0]).total_seconds() / (365.2425 * 86400)
    cagr = (float(path.iloc[-1]) / INITIAL_EUR) ** (1 / years) - 1
    vol = float(r.std(ddof=1) * sqrt(252))
    mean_ann = float(r.mean() * 252)
    down = r[r < 0]
    downvol = float(down.std(ddof=1) * sqrt(252)) if len(down) > 1 else np.nan
    return {
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe0": mean_ann / vol if vol > 0 else np.nan,
        "sortino0": mean_ann / downvol if downvol > 0 else np.nan,
        "max_dd": float((path / path.cummax() - 1).min()),
    }


def beta(strategy: pd.Series, benchmark: pd.Series) -> float:
    s = strategy.pct_change(fill_method=None)
    b = benchmark.pct_change(fill_method=None)
    x = pd.concat([s.rename("s"), b.rename("b")], axis=1).dropna()
    return float(x.s.cov(x.b) / x.b.var(ddof=1))


def downside_capture(strategy: pd.Series, benchmark: pd.Series) -> float:
    s = strategy.pct_change(fill_method=None)
    b = benchmark.pct_change(fill_method=None)
    x = pd.concat([s.rename("s"), b.rename("b")], axis=1).dropna()
    x = x[x.b < 0]
    sr = (1 + x.s).prod() - 1
    br = (1 + x.b).prod() - 1
    return float(sr / br) if br else np.nan


def simulate_active(data, begin, end, constant_target=None, cost=TRADING_COST, spread=BORROW_SPREAD):
    idx, close, divs, splits, fx, fed, events = data
    dates = idx[(idx >= begin) & (idx <= end)]
    fx0 = float(fx.loc[dates[0]])
    cash = INITIAL_EUR * fx0
    debt = 0.0
    qty = 0.0
    lots = deque()
    tax = divtax = interest = traded = 0.0
    realized_gains = realized_losses = 0.0
    trades = forced = 0
    rows = []

    dynamic = constant_target is None
    target = 1.0 if dynamic else float(constant_target)
    if dynamic:
        for d in sorted(events):
            if d <= dates[0]:
                target = float(events[d])
            else:
                break

    def px(ts):
        return float(close.loc[ts, "QQQ"])

    def nav(ts):
        return cash + qty * px(ts) - debt

    def repay():
        nonlocal cash, debt
        if cash > 0 and debt > 0:
            x = min(cash, debt)
            cash -= x
            debt -= x

    def sell(amount, ts):
        nonlocal cash, qty, tax, traded, trades, realized_gains, realized_losses
        amount = min(max(0.0, amount), qty)
        if amount <= 1e-12:
            return
        remain = amount
        local_tax = 0.0
        while remain > 1e-10 and lots:
            lot = lots[0]
            used = min(remain, lot.qty)
            gain = used * px(ts) / float(fx.loc[ts]) - used * lot.basis_eur_per_share
            if gain > 0:
                realized_gains += gain
                local_tax += gain * tx.slovenia_rate_for_dates(lot.acquired, ts)
            else:
                realized_losses += -gain
            lot.qty -= used
            remain -= used
            if lot.qty <= 1e-10:
                lots.popleft()
        notional = amount * px(ts)
        cash += notional * (1 - cost) - local_tax * float(fx.loc[ts])
        qty -= amount
        tax += local_tax
        traded += notional
        trades += 1
        repay()

    def buy(amount, ts):
        nonlocal cash, debt, qty, traded, trades
        amount = max(0.0, amount)
        if amount <= 1e-12:
            return
        notional = amount * px(ts)
        total = notional * (1 + cost)
        if cash >= total:
            cash -= total
        else:
            debt += total - cash
            cash = 0.0
        lots.append(Lot(amount, px(ts) / float(fx.loc[ts]), ts))
        qty += amount
        traded += notional
        trades += 1

    def rebalance(ts, desired, force=False):
        n = nav(ts)
        mv = qty * px(ts)
        current = mv / n if n > 0 else np.nan
        if not force and abs(desired - current) < NO_TRADE_BAND:
            return False
        desired_mv = max(0.0, desired * n)
        if mv > desired_mv:
            sell((mv - desired_mv) / px(ts), ts)
        else:
            buy((desired_mv - mv) / px(ts), ts)
        return True

    for j, ts in enumerate(dates):
        if j and debt > 0:
            x = debt * (float(fed.loc[ts]) + spread) / 360.0
            debt += x
            interest += x

        ratio = float(splits["QQQ"].loc[ts])
        if ratio > 0 and abs(ratio - 1) > 1e-12:
            qty *= ratio
            for lot in lots:
                lot.qty *= ratio
                lot.basis_eur_per_share /= ratio

        dv = float(divs["QQQ"].loc[ts])
        if dv > 0 and qty > 0:
            gross = qty * dv
            dt = gross / float(fx.loc[ts]) * DIVIDEND_TAX
            cash += gross - dt * float(fx.loc[ts])
            divtax += dt
            repay()

        n0 = nav(ts)
        pre_exp = qty * px(ts) / n0 if n0 > 0 else np.nan
        event_target = np.nan
        did_trade = False
        if j == 0:
            event_target = target
            did_trade = rebalance(ts, target, force=True)
        elif ts in events:
            if dynamic:
                target = float(events[ts])
            event_target = target
            did_trade = rebalance(ts, target)

        if qty > 0 and debt > 0:
            mv = qty * px(ts)
            n = nav(ts)
            if n / mv < MAINT_MARGIN:
                sell(qty, ts)
                forced += 1

        n = nav(ts)
        post_exp = qty * px(ts) / n if n > 0 else np.nan
        rows.append({
            "date": ts,
            "nav_eur": n / float(fx.loc[ts]),
            "pre_exposure": pre_exp,
            "post_exposure": post_exp,
            "target": target,
            "event_target": event_target,
            "trade": did_trade,
            "debt_usd": debt,
        })

    sell(qty, dates[-1])
    repay()
    final_eur = (cash - debt) / float(fx.loc[dates[-1]])
    rows[-1]["nav_eur"] = final_eur
    rows[-1]["post_exposure"] = 0.0
    frame = pd.DataFrame(rows).set_index("date")
    path = frame.nav_eur.astype(float)
    return {
        "path": path,
        "frame": frame,
        "metrics": metrics(path),
        "tax": tax,
        "divtax": divtax,
        "interest": interest,
        "turnover": traded / (INITIAL_EUR * fx0),
        "trades": trades,
        "forced": forced,
        "realized_gains": realized_gains,
        "realized_losses": realized_losses,
    }


def simulate_passive(data, symbol, begin, end):
    idx, close, divs, splits, fx, _, _ = data
    dates = idx[(idx >= begin) & (idx <= end)]
    fx0 = float(fx.loc[dates[0]])
    qty = INITIAL_EUR * fx0 / float(close.loc[dates[0], symbol])
    lots = deque([Lot(qty, float(close.loc[dates[0], symbol]) / fx0, dates[0])])
    divtax = 0.0
    path = []

    for ts in dates:
        ratio = float(splits[symbol].loc[ts])
        if ratio > 0 and abs(ratio - 1) > 1e-12:
            qty *= ratio
            for lot in lots:
                lot.qty *= ratio
                lot.basis_eur_per_share /= ratio
        dv = float(divs[symbol].loc[ts])
        if dv > 0:
            gross = qty * dv
            dt = gross / float(fx.loc[ts]) * DIVIDEND_TAX
            divtax += dt
            net = gross - dt * float(fx.loc[ts])
            add = net / float(close.loc[ts, symbol])
            qty += add
            lots.append(Lot(add, float(close.loc[ts, symbol]) / float(fx.loc[ts]), ts))
        path.append(qty * float(close.loc[ts, symbol]) / float(fx.loc[ts]))

    ts = dates[-1]
    p = float(close.loc[ts, symbol])
    tax = exempt_gain = taxable_gain = 0.0
    for lot in lots:
        gain = lot.qty * p / float(fx.loc[ts]) - lot.qty * lot.basis_eur_per_share
        if gain > 0:
            rate = tx.slovenia_rate_for_dates(lot.acquired, ts)
            tax += gain * rate
            if rate == 0:
                exempt_gain += gain
            else:
                taxable_gain += gain
    path[-1] -= tax
    series = pd.Series(path, index=dates, dtype=float)
    return {
        "path": series,
        "metrics": metrics(series),
        "tax": tax,
        "divtax": divtax,
        "exempt_gain": exempt_gain,
        "taxable_gain": taxable_gain,
    }


def show(label, result, qqq=None):
    m = result["metrics"]
    suffix = ""
    if qqq is not None:
        suffix = f" betaQQQ={beta(result['path'], qqq['path']):.3f} downsideCapture={downside_capture(result['path'], qqq['path']):.3f}"
    print(
        f"{label} CAGR={m['cagr']:.4%} vol={m['ann_vol']:.4%} Sharpe0={m['sharpe0']:.3f} "
        f"Sortino0={m['sortino0']:.3f} DD={m['max_dd']:.4%}{suffix}"
    )


def main():
    data = load_market()
    idx, _, _, _, _, _, events = data
    begin = idx[idx >= BEGIN][0]
    end = idx[idx <= FINISH][-1]
    print(f"CONTINUOUS dates={begin.date()}..{end.date()}")
    print("TIMING decision-close D uses returns only through D-1; execution is next exchange session E")

    active = simulate_active(data, begin, end)
    qqq = simulate_passive(data, "QQQ", begin, end)
    spy = simulate_passive(data, "SPY", begin, end)

    print("\nCONTINUOUS FULL-PATH RESULTS — passive benchmarks have zero transaction cost")
    show("ACTIVE", active, qqq)
    show("QQQ", qqq)
    show("SPY", spy)
    print(
        f"ACTIVE CGT={active['tax']:.2f} DivTax={active['divtax']:.2f} interestUSD={active['interest']:.2f} "
        f"turnover={active['turnover']:.3f}x trades={active['trades']} forced={active['forced']} "
        f"realizedGainsEUR={active['realized_gains']:.2f} realizedLossesEUR={active['realized_losses']:.2f}"
    )
    print(
        f"QQQ CGT={qqq['tax']:.2f} DivTax={qqq['divtax']:.2f} exemptTerminalGainEUR={qqq['exempt_gain']:.2f} "
        f"taxableTerminalGainEUR={qqq['taxable_gain']:.2f}"
    )
    print(
        f"EXCESS active-vs-QQQ={active['metrics']['cagr']-qqq['metrics']['cagr']:.4%} "
        f"active-vs-SPY={active['metrics']['cagr']-spy['metrics']['cagr']:.4%}"
    )

    event_targets = pd.Series({d: t for d, t in events.items() if begin <= d <= end}, dtype=float)
    mean_target = float(event_targets.mean())
    matched = simulate_active(data, begin, end, constant_target=mean_target)
    print("\nMATCHED-LEVERAGE ATTRIBUTION CONTROL")
    print(f"mean monthly target={mean_target:.6f}")
    show("CONST_MEAN_TARGET", matched, qqq)
    print(f"TIMING_RESIDUAL_CAGR={active['metrics']['cagr']-matched['metrics']['cagr']:.4%}")

    target_vol = active["metrics"]["ann_vol"]
    lo, hi = 0.50, 1.75
    for _ in range(14):
        mid = (lo + hi) / 2
        candidate = simulate_active(data, begin, end, constant_target=mid)
        if candidate["metrics"]["ann_vol"] < target_vol:
            lo = mid
        else:
            hi = mid
    vol_target = (lo + hi) / 2
    volmatched = simulate_active(data, begin, end, constant_target=vol_target)
    print("\nVOLATILITY-MATCHED ATTRIBUTION CONTROL")
    print(f"constant target={vol_target:.6f} activeAnnVol={target_vol:.4%}")
    show("CONST_VOL_MATCH", volmatched, qqq)
    print(f"RISK_MATCHED_CAGR_RESIDUAL={active['metrics']['cagr']-volmatched['metrics']['cagr']:.4%}")

    print("\nCOVID TRACE")
    covid = active["frame"].loc["2020-02-18":"2020-03-24"]
    selected = covid[covid.trade | covid.event_target.notna()].copy()
    for day in ("2020-02-19", "2020-03-02", "2020-03-23"):
        ts = pd.Timestamp(day, tz="UTC")
        if ts in covid.index:
            selected.loc[ts] = covid.loc[ts]
    if not covid.empty:
        selected.loc[covid.nav_eur.idxmin()] = covid.loc[covid.nav_eur.idxmin()]
    for ts, row in selected.sort_index().iterrows():
        et = row.event_target if np.isfinite(row.event_target) else np.nan
        print(
            f"{ts.date()} navEUR={row.nav_eur:.2f} preExp={row.pre_exposure:.4f} postExp={row.post_exposure:.4f} "
            f"target={row.target:.4f} eventTarget={et:.4f} trade={bool(row.trade)} debtUSD={row.debt_usd:.2f}"
        )
    yr = active["path"].loc["2020-01-01":"2020-12-31"]
    dd = yr / yr.cummax() - 1
    print(f"2020_DD={dd.min():.4%} trough={dd.idxmin().date()}")

    monthly_events = int(active["frame"].event_target.notna().sum())
    traded_events = int(active["frame"].trade.sum())
    print(f"\nREBALANCE_COUNTS monthlyEvents={monthly_events} tradedEventsIncludingInitial={traded_events} terminalLiquidationTrade=1")

    print("\nCONTINUOUS COST/FINANCING SENSITIVITY — passive QQQ remains zero-cost")
    for c, sp in ((0.002, 0.015), (0.004, 0.020), (0.008, 0.030)):
        r = simulate_active(data, begin, end, cost=c, spread=sp)
        print(
            f"cost={c:.3%} spread={sp:.3%} CAGR={r['metrics']['cagr']:.4%} "
            f"exQQQ={r['metrics']['cagr']-qqq['metrics']['cagr']:.4%} DD={r['metrics']['max_dd']:.4%}"
        )

    active["frame"].to_csv("external_review_active_exposure.csv")
    summary = {
        "start": str(begin.date()),
        "end": str(end.date()),
        "active": active["metrics"],
        "qqq": qqq["metrics"],
        "spy": spy["metrics"],
        "active_tax_eur": active["tax"],
        "active_dividend_tax_eur": active["divtax"],
        "active_interest_usd": active["interest"],
        "active_turnover_x": active["turnover"],
        "active_trades": active["trades"],
        "qqq_tax_eur": qqq["tax"],
        "qqq_exempt_terminal_gain_eur": qqq["exempt_gain"],
        "qqq_taxable_terminal_gain_eur": qqq["taxable_gain"],
        "mean_target": mean_target,
        "constant_mean_target": matched["metrics"],
        "vol_match_target": vol_target,
        "constant_vol_match": volmatched["metrics"],
    }
    with open("external_review_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nNOTES")
    print("The final 32-grid is not the full adaptive hypothesis space of the project; no alpha significance claim is made here.")
    print("Active CGT remains a conservative per-sale approximation: no same-year loss offsets or normalized expenses are credited.")
    print("FRED DEXUSEU remains a research FX proxy, not the legally authoritative filing source.")
    print("ARTIFACTS external_review_active_exposure.csv external_review_summary.json")


if __name__ == "__main__":
    main()
