from collections import deque
from itertools import product

import numpy as np
import pandas as pd

from bot.backtest_pullback import fetch_all
from bot.data import clean_daily_data
from bot import taxes as tx

RISK = [
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "XLY", "XLK", "XLI", "XLF", "XLV", "XLP", "XLU", "XLE", "XLB",
]
DEFENSIVE = ["GLD", "TLT"]
SYMBOLS = RISK + DEFENSIVE
START = pd.Timestamp("2004-01-01", tz="UTC")
END = pd.Timestamp("2026-08-20", tz="UTC")
DEV = [
    (pd.Timestamp("2006-01-01", tz="UTC"), pd.Timestamp("2009-12-31", tz="UTC")),
    (pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2014-12-31", tz="UTC")),
    (pd.Timestamp("2015-01-01", tz="UTC"), pd.Timestamp("2019-12-31", tz="UTC")),
    (pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC")),
]
HOLD = (pd.Timestamp("2025-01-01", tz="UTC"), END)
INITIAL = 100_000.0
COST = 0.0020


def fred(series: str, index: pd.Index) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    x = pd.read_csv(url)
    x.columns = ["date", series]
    x["date"] = pd.to_datetime(x["date"], utc=True)
    x[series] = pd.to_numeric(x[series], errors="coerce")
    raw = x.dropna().set_index("date")[series].sort_index()
    union = raw.index.union(index).sort_values()
    return raw.reindex(union).ffill().reindex(index).shift(1)


def period_key(ts: pd.Timestamp, freq: str):
    if freq == "M":
        return (ts.year, ts.month)
    if freq == "Q":
        return (ts.year, (ts.month - 1) // 3)
    if freq == "H":
        return (ts.year, 0 if ts.month <= 6 else 1)
    raise ValueError(freq)


def run():
    daily, _, _ = fetch_all(SYMBOLS, "none", START, END, source="yahoo")
    if set(daily) != set(SYMBOLS):
        raise RuntimeError(f"missing prices: {sorted(set(SYMBOLS) - set(daily))}")
    daily = clean_daily_data(daily)
    panel = pd.DataFrame({s: daily[s]["close"] for s in SYMBOLS}).sort_index().ffill().dropna()
    ret = panel.pct_change(fill_method=None)
    ma200 = panel.rolling(200, min_periods=200).mean().shift(1)
    vol63 = ret.rolling(63, min_periods=42).std().shift(1) * np.sqrt(252)

    hy_oas = fred("BAMLH0A0HYM2", panel.index)
    oas_slow = hy_oas.rolling(252, min_periods=126).mean()
    oas_fast = hy_oas.rolling(63, min_periods=42).mean()

    def raw_momentum(i: int, months: int):
        end_i = i - 1
        start_i = end_i - months * 21
        if start_i < 0:
            return None
        return panel.iloc[end_i] / panel.iloc[start_i] - 1.0

    def scores(i: int, score_type: str):
        m12 = raw_momentum(i, 12)
        if m12 is None:
            return None
        if score_type == "mom12":
            return m12
        m6 = raw_momentum(i, 6)
        if m6 is None:
            return None
        if score_type == "dual":
            return 0.5 * m12 + 0.5 * m6
        if score_type == "riskadj":
            v = vol63.iloc[i - 1].replace(0.0, np.nan)
            return (0.5 * m12 + 0.5 * m6) / v
        raise ValueError(score_type)

    def build_events(score_type: str, top_k: int, freq: str, sticky: int,
                     trend_gate: bool, macro_gate: bool):
        events = {}
        current_risk = tuple()
        last_key = None
        for i, ts in enumerate(panel.index):
            key = period_key(ts, freq)
            if last_key is None:
                last_key = key
                continue
            if key == last_key:
                continue
            sc = scores(i, score_type)
            if sc is None:
                last_key = key
                continue

            prior_px = panel.iloc[i - 1]
            eligible = sc[RISK].copy()
            eligible = eligible[eligible > 0]
            if trend_gate:
                eligible = eligible[prior_px[eligible.index] > ma200.iloc[i - 1][eligible.index]]
            ranked = list(eligible.sort_values(ascending=False).index)
            keep_limit = min(len(ranked), top_k + sticky)
            retained = [s for s in current_risk if s in ranked[:keep_limit]]
            for s in ranked:
                if s not in retained:
                    retained.append(s)
                if len(retained) >= top_k:
                    break
            selected = tuple(retained[:top_k])

            dsc = sc[DEFENSIVE].copy()
            dsc = dsc[dsc > 0]
            if trend_gate and len(dsc):
                dsc = dsc[prior_px[dsc.index] > ma200.iloc[i - 1][dsc.index]]
            best_def = dsc.sort_values(ascending=False).index[0] if len(dsc) else None

            credit_stress = bool(
                macro_gate
                and pd.notna(oas_slow.iloc[i - 1])
                and pd.notna(oas_fast.iloc[i - 1])
                and hy_oas.iloc[i - 1] > oas_slow.iloc[i - 1]
                and oas_fast.iloc[i - 1] > oas_slow.iloc[i - 1]
            )

            w = pd.Series(0.0, index=SYMBOLS)
            risk_budget = 0.5 if credit_stress else 1.0
            if selected:
                each = risk_budget / top_k
                for s in selected:
                    w[s] = each
            unfilled = 1.0 - float(w.sum())
            if unfilled > 1e-12 and best_def is not None:
                w[best_def] += unfilled

            prev = events[next(reversed(events))] if events else pd.Series(0.0, index=SYMBOLS)
            if not w.equals(prev):
                events[ts] = w
            current_risk = selected
            last_key = key
        return events

    def simulate(events, begin, end, cost=COST):
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
            q = min(max(0.0, q), qty[s])
            if q <= 1e-12:
                return
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
            cash += notional * (1.0 - cost) - tax
            qty[s] -= q
            tax_paid += tax
            traded += notional

        def buy(s, q, px, ts):
            nonlocal cash, traded
            q = max(0.0, q)
            unit = px * (1.0 + cost)
            q = min(q, cash / unit)
            if q <= 1e-12:
                return
            notional = q * px
            cash -= notional * (1.0 + cost)
            qty[s] += q
            lots[s].append((q, px, ts))
            traded += notional

        target = pd.Series(0.0, index=SYMBOLS)
        for d in sorted(events):
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
                desired = {s: float(target[s]) * nav / pxs[s] for s in SYMBOLS}
                for s in SYMBOLS:
                    if qty[s] > desired[s] + 1e-10:
                        sell(s, qty[s] - desired[s], pxs[s], ts)
                nav2 = cash + sum(qty[s] * pxs[s] for s in SYMBOLS)
                desired2 = {s: float(target[s]) * nav2 / pxs[s] for s in SYMBOLS}
                for s in SYMBOLS:
                    if desired2[s] > qty[s] + 1e-10:
                        buy(s, desired2[s] - qty[s], pxs[s], ts)
            eq.append(cash + sum(qty[s] * pxs[s] for s in SYMBOLS))

        ts = idx[-1]
        pxs = panel.loc[ts]
        for s in SYMBOLS:
            sell(s, qty[s], pxs[s], ts)
        eq[-1] = cash
        ser = pd.Series(eq, index=idx)
        years = (idx[-1] - idx[0]).total_seconds() / (365.2425 * 86400)
        cagr = (ser.iloc[-1] / INITIAL) ** (1.0 / max(years, 1 / 365.2425)) - 1.0
        dd = float((ser / ser.cummax() - 1.0).min())
        return {
            "equity": ser,
            "cagr": cagr,
            "return": ser.iloc[-1] / INITIAL - 1.0,
            "dd": dd,
            "tax": tax_paid,
            "turnover": traded / INITIAL,
        }

    def benchmark(begin, end, symbol):
        first = panel.loc[begin:end].index[0]
        w = pd.Series(0.0, index=SYMBOLS)
        w[symbol] = 1.0
        return simulate({first: w}, begin, end)

    benchmarks = {
        w: {s: benchmark(*w, s) for s in ("SPY", "QQQ")}
        for w in DEV + [HOLD]
    }

    specs = list(product(
        ("mom12", "dual", "riskadj"),
        (1, 2, 3),
        ("Q", "H"),
        (0, 1),
        (False, True),
        (False, True),
    ))
    rows = []
    for spec in specs:
        events = build_events(*spec)
        seg = [simulate(events, *w) for w in DEV]
        ex_spy = [m["cagr"] - benchmarks[w]["SPY"]["cagr"] for m, w in zip(seg, DEV)]
        ex_qqq = [m["cagr"] - benchmarks[w]["QQQ"]["cagr"] for m, w in zip(seg, DEV)]
        rows.append({
            "spec": spec,
            "events": events,
            "seg": seg,
            "wins_spy": sum(x > 0 for x in ex_spy),
            "min_spy": min(ex_spy),
            "med_spy": float(np.median(ex_spy)),
            "wins_qqq": sum(x > 0 for x in ex_qqq),
            "min_qqq": min(ex_qqq),
            "med_qqq": float(np.median(ex_qqq)),
            "med_cagr": float(np.median([m["cagr"] for m in seg])),
            "worst_dd": min(m["dd"] for m in seg),
            "turnover": sum(m["turnover"] for m in seg),
            "tax": sum(m["tax"] for m in seg),
        })

    rows.sort(
        key=lambda r: (
            r["wins_spy"], r["min_spy"], r["wins_qqq"], r["min_qqq"], r["med_spy"]
        ),
        reverse=True,
    )

    print("ACTIVE SECTOR / CROSS-ASSET MOMENTUM — DEVELOPMENT RANKING")
    print("score, top_k, frequency, sticky, trend_gate, macro_credit_gate")
    for i, r in enumerate(rows[:30], 1):
        print(
            f"{i:02d} spec={r['spec']} winSPY={r['wins_spy']}/4 minSPY={r['min_spy']:.2%} "
            f"medSPY={r['med_spy']:.2%} winQQQ={r['wins_qqq']}/4 minQQQ={r['min_qqq']:.2%} "
            f"medQQQ={r['med_qqq']:.2%} medCAGR={r['med_cagr']:.2%} "
            f"worstDD={r['worst_dd']:.2%} tax={r['tax']:.0f} turn={r['turnover']:.1f}x"
        )

    print("\nDEVELOPMENT BENCHMARKS")
    for w in DEV:
        print(
            f"{w[0].year}-{w[1].year} SPY={benchmarks[w]['SPY']['cagr']:.2%} "
            f"QQQ={benchmarks[w]['QQQ']['cagr']:.2%}"
        )

    print("\nUNTOUCHED HOLDOUT — DEVELOPMENT ORDER FROZEN")
    for r in rows[:15]:
        m = simulate(r["events"], *HOLD)
        b = benchmarks[HOLD]
        print(
            f"spec={r['spec']} CAGR={m['cagr']:.2%} Ret={m['return']:.2%} DD={m['dd']:.2%} "
            f"tax={m['tax']:.0f} turn={m['turnover']:.1f}x "
            f"exSPY={m['cagr']-b['SPY']['cagr']:.2%} exQQQ={m['cagr']-b['QQQ']['cagr']:.2%}"
        )
    print(
        f"HOLD BENCH SPY={benchmarks[HOLD]['SPY']['cagr']:.2%} "
        f"QQQ={benchmarks[HOLD]['QQQ']['cagr']:.2%}"
    )

    winner = rows[0]
    print("\nFIXED WINNER ROLLING DEVELOPMENT WINDOWS")
    for years in (3, 5, 7, 10):
        starts = pd.date_range(
            pd.Timestamp("2006-01-01", tz="UTC"),
            pd.Timestamp(f"{2024-years}-12-31", tz="UTC"),
            freq="QS",
        )
        exs, exq = [], []
        for a in starts:
            b = a + pd.DateOffset(years=years)
            m = simulate(winner["events"], a, b)
            s = benchmark(a, b, "SPY")
            q = benchmark(a, b, "QQQ")
            if m and s and q:
                exs.append(m["cagr"] - s["cagr"])
                exq.append(m["cagr"] - q["cagr"])
        print(
            f"{years}y n={len(exs)} winSPY={np.mean(np.array(exs)>0):.1%} "
            f"medSPY={np.median(exs):.2%} p10SPY={np.quantile(exs,.1):.2%} minSPY={min(exs):.2%} "
            f"winQQQ={np.mean(np.array(exq)>0):.1%} medQQQ={np.median(exq):.2%} "
            f"p10QQQ={np.quantile(exq,.1):.2%} minQQQ={min(exq):.2%}"
        )

    print("\nWINNER COST STRESS")
    for c in (0.001, 0.002, 0.004, 0.008):
        vals = [simulate(winner["events"], *w, cost=c) for w in DEV]
        es = [m["cagr"] - benchmark(*w, "SPY")["cagr"] for m, w in zip(vals, DEV)]
        hold = simulate(winner["events"], *HOLD, cost=c)
        print(
            f"cost={c:.2%} devWinSPY={sum(x>0 for x in es)}/4 minSPY={min(es):.2%} "
            f"hold={hold['cagr']:.2%} holdExSPY={hold['cagr']-benchmarks[HOLD]['SPY']['cagr']:.2%}"
        )

    print("\nWINNER PARAMETER NEIGHBORHOOD")
    ws = winner["spec"]
    for r in rows:
        dist = sum(a != b for a, b in zip(r["spec"], ws))
        if dist <= 1:
            m = simulate(r["events"], *HOLD)
            print(
                f"dist={dist} spec={r['spec']} devWinSPY={r['wins_spy']}/4 minSPY={r['min_spy']:.2%} "
                f"hold={m['cagr']:.2%} exSPY={m['cagr']-benchmarks[HOLD]['SPY']['cagr']:.2%}"
            )


if __name__ == "__main__":
    run()
