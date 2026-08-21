from collections import deque
from itertools import product
import numpy as np
import pandas as pd

from bot.backtest_pullback import fetch_all
from bot.data import clean_daily_data
from bot import taxes as tx

SYMBOLS = ["SPY", "QQQ", "GLD", "TLT", "EFA", "EEM", "IWM"]
EQUITIES = ["SPY", "QQQ", "EFA", "EEM", "IWM"]
DEFENSIVE = ["GLD", "TLT"]
START = pd.Timestamp("2004-01-01", tz="UTC")
END = pd.Timestamp("2026-08-20", tz="UTC")
DEV = [
    (pd.Timestamp("2006-01-01", tz="UTC"), pd.Timestamp("2009-12-31", tz="UTC")),
    (pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2014-12-31", tz="UTC")),
    (pd.Timestamp("2015-01-01", tz="UTC"), pd.Timestamp("2019-12-31", tz="UTC")),
    (pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC")),
]
HOLD = (pd.Timestamp("2025-01-01", tz="UTC"), END)
INITIAL = 100000.0
COST = 0.0020


def fred(series, index):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    x = pd.read_csv(url)
    x.columns = ["date", series]
    x["date"] = pd.to_datetime(x["date"], utc=True)
    x[series] = pd.to_numeric(x[series], errors="coerce")
    return x.set_index("date")[series].sort_index().reindex(index, method="ffill").shift(1)


def period_key(ts, freq):
    if freq == "M":
        return (ts.year, ts.month)
    if freq == "Q":
        return (ts.year, (ts.month - 1) // 3)
    if freq == "A":
        return (ts.year,)
    raise ValueError(freq)


def run():
    daily, _, _ = fetch_all(SYMBOLS, "none", START, END, source="yahoo")
    if set(daily) != set(SYMBOLS):
        raise RuntimeError(f"missing prices: {sorted(set(SYMBOLS)-set(daily))}")
    daily = clean_daily_data(daily)
    panel = pd.DataFrame({s: daily[s]["close"] for s in SYMBOLS}).sort_index().ffill().dropna()

    hy_oas = fred("BAMLH0A0HYM2", panel.index)
    curve = fred("T10Y3M", panel.index)
    oas_ma = hy_oas.rolling(252, min_periods=126).mean()
    oas_fast = hy_oas.rolling(63, min_periods=42).mean()
    eq_ma200 = panel[EQUITIES].rolling(200, min_periods=200).mean()
    breadth = (panel[EQUITIES] > eq_ma200).mean(axis=1).shift(1)

    def momentum_scores(i, lookback_months, skip_months=1):
        end_i = i - 1 - skip_months * 21
        start_i = end_i - lookback_months * 21
        if start_i < 0 or end_i < 0:
            return None
        return panel.iloc[end_i] / panel.iloc[start_i] - 1.0

    def choose(i, lookback, top_k, sticky, gate, current):
        scores = momentum_scores(i, lookback)
        if scores is None:
            return tuple()
        eligible = scores[scores > 0].sort_values(ascending=False)
        br = float(breadth.iloc[i]) if pd.notna(breadth.iloc[i]) else 1.0
        credit_stress = bool(pd.notna(oas_ma.iloc[i]) and hy_oas.iloc[i] > oas_ma.iloc[i]
                             and oas_fast.iloc[i] > oas_ma.iloc[i])
        inversion = bool(pd.notna(curve.iloc[i]) and curve.iloc[i] < 0)
        if gate == "none":
            stress = False
        elif gate == "breadth":
            stress = br < 0.5
        elif gate == "credit":
            stress = credit_stress
        elif gate == "combo":
            stress = credit_stress and br < 0.5
        elif gate == "macro":
            stress = credit_stress or (inversion and br < 0.5)
        else:
            raise ValueError(gate)
        if stress:
            eligible = eligible[eligible.index.isin(DEFENSIVE)]
        ranked = list(eligible.index)
        if not ranked:
            return tuple()
        keep_limit = min(len(ranked), top_k + sticky)
        retained = [s for s in current if s in ranked[:keep_limit]]
        for s in ranked:
            if s not in retained:
                retained.append(s)
            if len(retained) >= top_k:
                break
        return tuple(retained[:top_k])

    def membership_events(lookback, top_k, freq, sticky, gate):
        events = {}
        current = tuple()
        last = None
        for i, ts in enumerate(panel.index):
            key = period_key(ts, freq)
            if last is None:
                last = key
                continue
            if key != last:
                new = choose(i, lookback, top_k, sticky, gate, current)
                if new != current:
                    w = pd.Series(0.0, index=SYMBOLS)
                    if new:
                        for s in new:
                            w[s] = 1.0 / top_k
                    events[ts] = w
                    current = new
                last = key
        return events

    def simulate(events, begin, end, initial_target=None, final_liquidate=True):
        idx = panel.loc[begin:end].index
        if len(idx) < 2:
            return None
        cash = INITIAL
        qty = {s: 0.0 for s in SYMBOLS}
        lots = {s: deque() for s in SYMBOLS}
        tax_paid = 0.0
        traded = 0.0
        eq = []

        def sell(s, q, px, ts):
            nonlocal cash, tax_paid, traded
            if q <= 1e-12:
                return
            q = min(q, qty[s])
            remain = q
            tax = 0.0
            while remain > 1e-10 and lots[s]:
                lot_q, lot_px, lot_ts = lots[s][0]
                used = min(remain, lot_q)
                gain = used * (px - lot_px)
                if gain > 0:
                    tax += gain * tx.slovenia_rate_for_dates(lot_ts, ts)
                lot_q -= used
                remain -= used
                if lot_q <= 1e-10:
                    lots[s].popleft()
                else:
                    lots[s][0] = (lot_q, lot_px, lot_ts)
            notional = q * px
            fee = notional * COST
            cash += notional - fee - tax
            qty[s] -= q
            tax_paid += tax
            traded += notional

        def buy(s, q, px, ts):
            nonlocal cash, traded
            if q <= 1e-12:
                return
            unit = px * (1 + COST)
            q = min(q, cash / unit)
            if q <= 1e-12:
                return
            notional = q * px
            cash -= notional * (1 + COST)
            qty[s] += q
            lots[s].append((q, px, ts))
            traded += notional

        all_dates = sorted(events)
        target = initial_target.copy() if initial_target is not None else pd.Series(0.0, index=SYMBOLS)
        for d in all_dates:
            if d <= idx[0]:
                target = events[d].copy()
            else:
                break

        for j, ts in enumerate(idx):
            pxs = panel.loc[ts]
            rebalance = (j == 0) or (ts in events and ts > idx[0])
            if rebalance:
                if ts in events:
                    target = events[ts].copy()
                nav = cash + sum(qty[s] * pxs[s] for s in SYMBOLS)
                desired = {s: max(0.0, float(target[s])) * nav / pxs[s] for s in SYMBOLS}
                for s in SYMBOLS:
                    if qty[s] > desired[s] + 1e-10:
                        sell(s, qty[s] - desired[s], pxs[s], ts)
                nav2 = cash + sum(qty[s] * pxs[s] for s in SYMBOLS)
                desired2 = {s: max(0.0, float(target[s])) * nav2 / pxs[s] for s in SYMBOLS}
                for s in SYMBOLS:
                    if desired2[s] > qty[s] + 1e-10:
                        buy(s, desired2[s] - qty[s], pxs[s], ts)
            eq.append(cash + sum(qty[s] * pxs[s] for s in SYMBOLS))

        if final_liquidate:
            ts = idx[-1]
            pxs = panel.loc[ts]
            for s in SYMBOLS:
                sell(s, qty[s], pxs[s], ts)
            eq[-1] = cash
        ser = pd.Series(eq, index=idx)
        years = (idx[-1] - idx[0]).total_seconds() / (365.2425 * 86400)
        cagr = (ser.iloc[-1] / INITIAL) ** (1 / max(years, 1 / 365.2425)) - 1
        dd = float((ser / ser.cummax() - 1).min())
        return {"equity": ser, "cagr": cagr, "return": ser.iloc[-1] / INITIAL - 1,
                "dd": dd, "tax": tax_paid, "turnover": traded / INITIAL}

    def bench(begin, end, kind):
        if kind == "equal7":
            w = {s: 1 / len(SYMBOLS) for s in SYMBOLS}
        elif kind == "SPY":
            w = {s: (1.0 if s == "SPY" else 0.0) for s in SYMBOLS}
        elif kind == "QQQ":
            w = {s: (1.0 if s == "QQQ" else 0.0) for s in SYMBOLS}
        else:
            raise ValueError(kind)
        first = panel.loc[begin:end].index[0]
        return simulate({first: pd.Series(w)}, begin, end)

    benchmarks = {window: {k: bench(*window, k) for k in ("equal7", "SPY", "QQQ")}
                  for window in DEV + [HOLD]}

    specs = list(product((9, 12), (1, 2, 3), ("M", "Q", "A"), (0, 1),
                         ("none", "breadth", "credit", "combo", "macro")))
    rows = []
    for spec in specs:
        ev = membership_events(*spec)
        seg = [simulate(ev, *w) for w in DEV]
        excess_eq = [m["cagr"] - benchmarks[w]["equal7"]["cagr"] for m, w in zip(seg, DEV)]
        excess_spy = [m["cagr"] - benchmarks[w]["SPY"]["cagr"] for m, w in zip(seg, DEV)]
        rows.append({"spec": spec, "events": ev, "seg": seg,
                     "wins_eq": sum(x > 0 for x in excess_eq),
                     "wins_spy": sum(x > 0 for x in excess_spy),
                     "min_eq": min(excess_eq), "med_eq": float(np.median(excess_eq)),
                     "min_spy": min(excess_spy), "med_spy": float(np.median(excess_spy)),
                     "med_cagr": float(np.median([m["cagr"] for m in seg])),
                     "worst_dd": min(m["dd"] for m in seg),
                     "tax": sum(m["tax"] for m in seg),
                     "turnover": sum(m["turnover"] for m in seg)})

    rows.sort(key=lambda r: (r["wins_eq"], r["min_eq"], r["wins_spy"],
                             r["min_spy"], r["med_eq"]), reverse=True)
    print("DEVELOPMENT RANKING — conservative immediate per-sale CGT, no loss offsets, 20bp costs")
    for i, r in enumerate(rows[:25], 1):
        lb, k, f, st, g = r["spec"]
        print(f"{i:02d} lb={lb}m skip=1 k={k} freq={f} sticky={st} gate={g} "
              f"winEq={r['wins_eq']}/4 minEq={r['min_eq']:.2%} medEq={r['med_eq']:.2%} "
              f"winSPY={r['wins_spy']}/4 minSPY={r['min_spy']:.2%} medSPY={r['med_spy']:.2%} "
              f"medCAGR={r['med_cagr']:.2%} worstDD={r['worst_dd']:.2%} tax={r['tax']:.0f} turn={r['turnover']:.1f}x")

    print("\nDEVELOPMENT BENCHMARKS")
    for w in DEV:
        b = benchmarks[w]
        print(w[0].year, w[1].year,
              f"equal7={b['equal7']['cagr']:.2%} SPY={b['SPY']['cagr']:.2%} QQQ={b['QQQ']['cagr']:.2%}")

    print("\nUNTOUCHED HOLDOUT — ranking fixed above")
    for r in rows[:15]:
        m = simulate(r["events"], *HOLD)
        b = benchmarks[HOLD]
        print(f"spec={r['spec']} CAGR={m['cagr']:.2%} Ret={m['return']:.2%} DD={m['dd']:.2%} "
              f"tax={m['tax']:.0f} turn={m['turnover']:.1f}x "
              f"exEq={m['cagr']-b['equal7']['cagr']:.2%} exSPY={m['cagr']-b['SPY']['cagr']:.2%}")
    b = benchmarks[HOLD]
    print(f"HOLD BENCH equal7 CAGR={b['equal7']['cagr']:.2%} Ret={b['equal7']['return']:.2%} tax={b['equal7']['tax']:.0f}")
    print(f"HOLD BENCH SPY    CAGR={b['SPY']['cagr']:.2%} Ret={b['SPY']['return']:.2%} tax={b['SPY']['tax']:.0f}")
    print(f"HOLD BENCH QQQ    CAGR={b['QQQ']['cagr']:.2%} Ret={b['QQQ']['return']:.2%} tax={b['QQQ']['tax']:.0f}")

    winner = rows[0]["spec"]
    print("\nWINNER NEIGHBORHOOD")
    for r in rows:
        s = r["spec"]
        dist = sum(a != b for a, b in zip(s, winner))
        if dist <= 1:
            m = simulate(r["events"], *HOLD)
            print(f"dist={dist} spec={s} devWinEq={r['wins_eq']}/4 minEq={r['min_eq']:.2%} "
                  f"holdCAGR={m['cagr']:.2%} holdExEq={m['cagr']-benchmarks[HOLD]['equal7']['cagr']:.2%}")


if __name__ == "__main__":
    run()
