from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from bot import taxes as tx

SYMBOLS = ["SPY", "QQQ"]
START = "2004-01-01"
END = "2026-08-21"
DEV = [
    (pd.Timestamp("2006-01-01", tz="UTC"), pd.Timestamp("2009-12-31", tz="UTC")),
    (pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2014-12-31", tz="UTC")),
    (pd.Timestamp("2015-01-01", tz="UTC"), pd.Timestamp("2019-12-31", tz="UTC")),
    (pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC")),
]
HOLD = (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-08-20", tz="UTC"))
INITIAL_EUR = 100_000.0
TARGET_VOL = 0.25
VOL_LOOKBACK = 20
MIN_EXPOSURE = 0.50
NO_TRADE_BAND = 0.20
COST = 0.0020
BORROW_SPREAD = 0.015
DIVIDEND_TAX = 0.25
MAINT_MARGIN = 0.30
FROZEN_CAPS = (1.50, 1.75)


@dataclass
class Lot:
    qty: float
    basis_eur_per_share: float
    acquired: pd.Timestamp


def normalize_index(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    out.index = idx.normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


def load_symbol(symbol: str) -> pd.DataFrame:
    d = yf.Ticker(symbol).history(
        start=START, end=END, auto_adjust=False, actions=True, repair=False
    )
    if d is None or d.empty:
        raise RuntimeError(f"no raw history for {symbol}")
    d = normalize_index(d)
    close = pd.to_numeric(d["Close"], errors="coerce")
    adj = pd.to_numeric(d["Adj Close"], errors="coerce") if "Adj Close" in d else close
    div = pd.to_numeric(d.get("Dividends", 0.0), errors="coerce").fillna(0.0)
    split = pd.to_numeric(d.get("Stock Splits", 0.0), errors="coerce").fillna(0.0)
    out = pd.DataFrame({"close": close, "adj": adj, "dividend": div, "split": split})
    out = out.dropna(subset=["close", "adj"])
    return out[(out.close > 0) & (out.adj > 0)]


def load_fx(index: pd.Index) -> pd.Series:
    f = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU")
    f.columns = ["date", "fx"]
    f["date"] = pd.to_datetime(f["date"], utc=True).dt.normalize()
    f["fx"] = pd.to_numeric(f["fx"], errors="coerce")
    raw = f.dropna().set_index("date").fx.sort_index()
    union = raw.index.union(index).sort_values()
    out = raw.reindex(union).ffill().reindex(index)
    if out.isna().any() or (out <= 0).any():
        raise RuntimeError("invalid EURUSD history")
    return out


def load_fed(index: pd.Index) -> pd.Series:
    f = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF")
    f.columns = ["date", "fed"]
    f["date"] = pd.to_datetime(f["date"], utc=True).dt.normalize()
    f["fed"] = pd.to_numeric(f["fed"], errors="coerce")
    raw = f.dropna().set_index("date").fed.sort_index() / 100.0
    union = raw.index.union(index).sort_values()
    return raw.reindex(union).ffill().reindex(index).shift(1).fillna(0.0).clip(lower=0.0)


def run():
    raw = {s: load_symbol(s) for s in SYMBOLS}
    common = max(x.index[0] for x in raw.values())
    last = min(x.index[-1] for x in raw.values())
    # Use the union of actual source trading dates, not a synthetic business-day calendar.
    idx = raw["QQQ"].index.union(raw["SPY"].index)
    idx = idx[(idx >= common) & (idx <= last)].sort_values()
    close = pd.DataFrame({s: raw[s].close.reindex(idx).ffill() for s in SYMBOLS}).dropna()
    adj = pd.DataFrame({s: raw[s].adj.reindex(close.index).ffill() for s in SYMBOLS}).dropna()
    idx = close.index.intersection(adj.index)
    close = close.loc[idx]
    adj = adj.loc[idx]
    dividends = {s: raw[s].dividend.reindex(idx).fillna(0.0) for s in SYMBOLS}
    splits = {s: raw[s].split.reindex(idx).fillna(0.0) for s in SYMBOLS}
    fx = load_fx(idx)
    fed = load_fed(idx)

    qret = adj["QQQ"].pct_change(fill_method=None)
    qvol = qret.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().shift(1) * np.sqrt(252)

    def build_events(cap: float):
        events = {}
        last_month = None
        for i, ts in enumerate(idx):
            key = (ts.year, ts.month)
            if last_month is None:
                last_month = key
                continue
            if key == last_month:
                continue
            last_month = key
            realized = float(qvol.iloc[i - 1]) if i > 0 else np.nan
            if not np.isfinite(realized) or realized <= 0:
                continue
            events[ts] = float(np.clip(TARGET_VOL / realized, MIN_EXPOSURE, cap))
        return events

    def simulate_active(cap, begin, end, cost=COST, spread=BORROW_SPREAD,
                        dividend_tax=DIVIDEND_TAX, band=NO_TRADE_BAND):
        dates = idx[(idx >= begin) & (idx <= end)]
        if len(dates) < 2:
            return None
        events = build_events(cap)
        fx0 = float(fx.loc[dates[0]])
        cash_usd = INITIAL_EUR * fx0
        debt_usd = 0.0
        qty = 0.0
        lots = deque()
        tax_eur = 0.0
        div_tax_eur = 0.0
        interest_usd = 0.0
        traded_usd = 0.0
        forced = 0
        path_eur = []

        target = 1.0
        for d in sorted(events):
            if d <= dates[0]:
                target = float(events[d])
            else:
                break

        def p(ts):
            return float(close.loc[ts, "QQQ"])

        def nav_usd(ts):
            return cash_usd + qty * p(ts) - debt_usd

        def repay():
            nonlocal cash_usd, debt_usd
            if cash_usd > 0 and debt_usd > 0:
                x = min(cash_usd, debt_usd)
                cash_usd -= x
                debt_usd -= x

        def apply_split(ts):
            nonlocal qty
            ratio = float(splits["QQQ"].loc[ts])
            if ratio > 0 and abs(ratio - 1.0) > 1e-12:
                qty *= ratio
                for lot in lots:
                    lot.qty *= ratio
                    lot.basis_eur_per_share /= ratio

        def receive_dividend(ts):
            nonlocal cash_usd, div_tax_eur
            dv = float(dividends["QQQ"].loc[ts])
            if dv <= 0 or qty <= 0:
                return
            gross_usd = qty * dv
            gross_eur = gross_usd / float(fx.loc[ts])
            dtax = max(0.0, gross_eur * dividend_tax)
            cash_usd += gross_usd - dtax * float(fx.loc[ts])
            div_tax_eur += dtax
            repay()

        def sell(q, ts):
            nonlocal cash_usd, qty, tax_eur, traded_usd
            q = min(max(0.0, q), qty)
            if q <= 1e-12:
                return
            px = p(ts)
            remain = q
            local_tax_eur = 0.0
            while remain > 1e-10 and lots:
                lot = lots[0]
                used = min(remain, lot.qty)
                disposal_eur = used * px / float(fx.loc[ts])
                basis_eur = used * lot.basis_eur_per_share
                gain = disposal_eur - basis_eur
                if gain > 0:
                    local_tax_eur += gain * tx.slovenia_rate_for_dates(lot.acquired, ts)
                lot.qty -= used
                remain -= used
                if lot.qty <= 1e-10:
                    lots.popleft()
            notional = q * px
            cash_usd += notional * (1.0 - cost) - local_tax_eur * float(fx.loc[ts])
            qty -= q
            tax_eur += local_tax_eur
            traded_usd += notional
            repay()

        def buy(q, ts):
            nonlocal cash_usd, debt_usd, qty, traded_usd
            q = max(0.0, q)
            if q <= 1e-12:
                return
            px = p(ts)
            notional = q * px
            total = notional * (1.0 + cost)
            if cash_usd >= total:
                cash_usd -= total
            else:
                debt_usd += total - cash_usd
                cash_usd = 0.0
            basis_eur_per_share = px / float(fx.loc[ts])
            qty += q
            lots.append(Lot(q, basis_eur_per_share, ts))
            traded_usd += notional

        def rebalance(ts, desired, force=False):
            n = nav_usd(ts)
            if n <= 0:
                return
            mv = qty * p(ts)
            current = mv / n
            if not force and abs(desired - current) < band:
                return
            desired_mv = max(0.0, desired * n)
            if mv > desired_mv:
                sell((mv - desired_mv) / p(ts), ts)
            else:
                buy((desired_mv - mv) / p(ts), ts)

        for j, ts in enumerate(dates):
            if j > 0 and debt_usd > 0:
                interest = debt_usd * (float(fed.loc[ts]) + spread) / 360.0
                debt_usd += interest
                interest_usd += interest

            apply_split(ts)
            receive_dividend(ts)

            if j == 0:
                rebalance(ts, target, force=True)
            elif ts in events:
                target = float(events[ts])
                rebalance(ts, target)

            if qty > 0 and debt_usd > 0:
                mv = qty * p(ts)
                n = nav_usd(ts)
                margin_ratio = n / mv if mv > 0 else -np.inf
                if margin_ratio < MAINT_MARGIN:
                    sell(qty, ts)
                    target = MIN_EXPOSURE
                    forced += 1

            n_eur = nav_usd(ts) / float(fx.loc[ts])
            path_eur.append(n_eur)
            if n_eur <= 0 or not np.isfinite(n_eur):
                return None

        sell(qty, dates[-1])
        repay()
        final_eur = (cash_usd - debt_usd) / float(fx.loc[dates[-1]])
        path_eur[-1] = final_eur
        ser = pd.Series(path_eur, index=dates)
        years = (dates[-1] - dates[0]).total_seconds() / (365.2425 * 86400)
        return {
            "cagr": (final_eur / INITIAL_EUR) ** (1.0 / years) - 1.0,
            "return": final_eur / INITIAL_EUR - 1.0,
            "dd": float((ser / ser.cummax() - 1.0).min()),
            "tax": tax_eur,
            "div_tax": div_tax_eur,
            "interest_usd": interest_usd,
            "turnover": traded_usd / (INITIAL_EUR * fx0),
            "forced": forced,
        }

    def simulate_hold(symbol, begin, end, cost=COST, dividend_tax=DIVIDEND_TAX):
        dates = idx[(idx >= begin) & (idx <= end)]
        if len(dates) < 2:
            return None
        fx0 = float(fx.loc[dates[0]])
        cash_usd = INITIAL_EUR * fx0
        p0 = float(close.loc[dates[0], symbol])
        qty = cash_usd / (p0 * (1.0 + cost))
        cash_usd -= qty * p0 * (1.0 + cost)
        lots = deque([Lot(qty, p0 / fx0, dates[0])])
        div_tax_eur = 0.0
        traded_usd = qty * p0
        path = []

        for ts in dates:
            ratio = float(splits[symbol].loc[ts])
            if ratio > 0 and abs(ratio - 1.0) > 1e-12:
                qty *= ratio
                for lot in lots:
                    lot.qty *= ratio
                    lot.basis_eur_per_share /= ratio
            dv = float(dividends[symbol].loc[ts])
            if dv > 0 and qty > 0:
                gross_usd = qty * dv
                gross_eur = gross_usd / float(fx.loc[ts])
                dtax = max(0.0, gross_eur * dividend_tax)
                net_usd = gross_usd - dtax * float(fx.loc[ts])
                div_tax_eur += dtax
                px = float(close.loc[ts, symbol])
                add = net_usd / (px * (1.0 + cost))
                if add > 0:
                    qty += add
                    lots.append(Lot(add, px / float(fx.loc[ts]), ts))
                    traded_usd += add * px
            path.append((cash_usd + qty * float(close.loc[ts, symbol])) / float(fx.loc[ts]))

        ts = dates[-1]
        px = float(close.loc[ts, symbol])
        local_tax = 0.0
        for lot in lots:
            disposal_eur = lot.qty * px / float(fx.loc[ts])
            basis_eur = lot.qty * lot.basis_eur_per_share
            gain = disposal_eur - basis_eur
            if gain > 0:
                local_tax += gain * tx.slovenia_rate_for_dates(lot.acquired, ts)
        proceeds_usd = qty * px
        final_eur = (cash_usd + proceeds_usd * (1.0 - cost)) / float(fx.loc[ts]) - local_tax
        traded_usd += proceeds_usd
        path[-1] = final_eur
        ser = pd.Series(path, index=dates)
        years = (dates[-1] - dates[0]).total_seconds() / (365.2425 * 86400)
        return {
            "cagr": (final_eur / INITIAL_EUR) ** (1.0 / years) - 1.0,
            "return": final_eur / INITIAL_EUR - 1.0,
            "dd": float((ser / ser.cummax() - 1.0).min()),
            "tax": local_tax,
            "div_tax": div_tax_eur,
            "turnover": traded_usd / (INITIAL_EUR * fx0),
        }

    print("FROZEN VOLATILITY-MANAGED STRATEGY — EUR / RAW PRICE / DIVIDEND-AWARE")
    print("Rule: QQQ, monthly, prior 20d vol, 25% target, 50% floor, 20pp no-trade band")
    for cap in FROZEN_CAPS:
        print(f"\nCAP={cap:.2f} DEVELOPMENT")
        exs, exq = [], []
        for a, b in DEV:
            m = simulate_active(cap, a, b)
            s = simulate_hold("SPY", a, b)
            q = simulate_hold("QQQ", a, b)
            exs.append(m["cagr"] - s["cagr"])
            exq.append(m["cagr"] - q["cagr"])
            print(
                f"{a.year}-{b.year} strat={m['cagr']:.2%} SPY={s['cagr']:.2%} QQQ={q['cagr']:.2%} "
                f"exSPY={exs[-1]:.2%} exQQQ={exq[-1]:.2%} DD={m['dd']:.2%} "
                f"CGT={m['tax']:.0f} DivTax={m['div_tax']:.0f} IntUSD={m['interest_usd']:.0f} turn={m['turnover']:.1f}x"
            )
        print(
            f"DEV SUMMARY winSPY={sum(x>0 for x in exs)}/4 minSPY={min(exs):.2%} "
            f"winQQQ={sum(x>0 for x in exq)}/4 minQQQ={min(exq):.2%}"
        )
        m = simulate_active(cap, *HOLD)
        s = simulate_hold("SPY", *HOLD)
        q = simulate_hold("QQQ", *HOLD)
        print(
            f"HOLD strat={m['cagr']:.2%} Ret={m['return']:.2%} SPY={s['cagr']:.2%} QQQ={q['cagr']:.2%} "
            f"exSPY={m['cagr']-s['cagr']:.2%} exQQQ={m['cagr']-q['cagr']:.2%} DD={m['dd']:.2%} "
            f"CGT={m['tax']:.0f} DivTax={m['div_tax']:.0f} IntUSD={m['interest_usd']:.0f} turn={m['turnover']:.1f}x"
        )

    # Anti-overfit test: fixed 1.50x cap, which is the lower neighboring cap that already passed
    # the development grid. No parameter is selected from the following rolling-window results.
    cap = 1.50
    print("\nFIXED 1.50x CAP ROLLING DEVELOPMENT WINDOWS — EUR / FULL TAX")
    for years in (3, 5, 7, 10):
        starts = pd.date_range(
            pd.Timestamp("2006-01-01", tz="UTC"),
            pd.Timestamp(f"{2024-years}-12-31", tz="UTC"),
            freq="QS",
        )
        exs, exq = [], []
        for a in starts:
            b = a + pd.DateOffset(years=years)
            m = simulate_active(cap, a, b)
            s = simulate_hold("SPY", a, b)
            q = simulate_hold("QQQ", a, b)
            if m and s and q:
                exs.append(m["cagr"] - s["cagr"])
                exq.append(m["cagr"] - q["cagr"])
        print(
            f"{years}y n={len(exs)} winSPY={np.mean(np.array(exs)>0):.1%} medSPY={np.median(exs):.2%} "
            f"p10SPY={np.quantile(exs,.1):.2%} minSPY={min(exs):.2%} "
            f"winQQQ={np.mean(np.array(exq)>0):.1%} medQQQ={np.median(exq):.2%} "
            f"p10QQQ={np.quantile(exq,.1):.2%} minQQQ={min(exq):.2%}"
        )

    print("\nFIXED 1.50x CAP COST / FINANCING STRESS — EUR / FULL TAX")
    for c, spread in ((0.001, 0.010), (0.002, 0.015), (0.004, 0.020), (0.008, 0.030)):
        es, eq = [], []
        for a, b in DEV:
            m = simulate_active(1.50, a, b, cost=c, spread=spread)
            s = simulate_hold("SPY", a, b, cost=c)
            q = simulate_hold("QQQ", a, b, cost=c)
            es.append(m["cagr"] - s["cagr"])
            eq.append(m["cagr"] - q["cagr"])
        h = simulate_active(1.50, *HOLD, cost=c, spread=spread)
        hs = simulate_hold("SPY", *HOLD, cost=c)
        hq = simulate_hold("QQQ", *HOLD, cost=c)
        print(
            f"cost={c:.2%} spread={spread:.2%} devWinSPY={sum(x>0 for x in es)}/4 minSPY={min(es):.2%} "
            f"devWinQQQ={sum(x>0 for x in eq)}/4 minQQQ={min(eq):.2%} "
            f"hold={h['cagr']:.2%} exSPY={h['cagr']-hs['cagr']:.2%} exQQQ={h['cagr']-hq['cagr']:.2%}"
        )

    print("\nSTART-DATE PERTURBATION — FIXED 1.50x CAP, 7Y WINDOWS")
    starts = pd.date_range(pd.Timestamp("2006-01-01", tz="UTC"), pd.Timestamp("2017-12-31", tz="UTC"), freq="6MS")
    for a in starts:
        b = a + pd.DateOffset(years=7)
        m = simulate_active(1.50, a, b)
        s = simulate_hold("SPY", a, b)
        q = simulate_hold("QQQ", a, b)
        if m and s and q:
            print(
                f"start={a.date()} strat={m['cagr']:.2%} exSPY={m['cagr']-s['cagr']:.2%} "
                f"exQQQ={m['cagr']-q['cagr']:.2%} DD={m['dd']:.2%}"
            )


if __name__ == "__main__":
    run()
