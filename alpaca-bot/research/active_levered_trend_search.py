from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from bot.backtest_pullback import fetch_all
from bot.data import clean_daily_data
from bot import taxes as tx

SYMBOLS = ["SPY", "QQQ", "GLD"]
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
BORROW_SPREAD = 0.015
MAINT_MARGIN = 0.30


def fred(series: str, index: pd.Index) -> pd.Series:
    x = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}")
    x.columns = ["date", series]
    x["date"] = pd.to_datetime(x["date"], utc=True)
    x[series] = pd.to_numeric(x[series], errors="coerce")
    raw = x.dropna().set_index("date")[series].sort_index()
    union = raw.index.union(index).sort_values()
    return raw.reindex(union).ffill().reindex(index).shift(1).fillna(0.0)


def run():
    daily, _, _ = fetch_all(SYMBOLS, "none", START, END, source="yahoo")
    if set(daily) != set(SYMBOLS):
        raise RuntimeError(f"missing prices: {sorted(set(SYMBOLS)-set(daily))}")
    daily = clean_daily_data(daily)
    panel = pd.DataFrame({s: daily[s]["close"] for s in SYMBOLS}).sort_index().ffill().dropna()
    fed = fred("DFF", panel.index).clip(lower=0.0) / 100.0

    ma = {n: panel[["SPY", "QQQ"]].rolling(n, min_periods=n).mean().shift(1) for n in (200, 250)}

    def build_events(risk_asset: str, ma_days: int, buffer: float, leverage: float,
                     fallback: str, confirm12: bool):
        events = {}
        state = None
        last_month = None
        for i, ts in enumerate(panel.index):
            key = (ts.year, ts.month)
            if last_month is None:
                last_month = key
                continue
            if key == last_month:
                continue
            last_month = key
            if i < max(ma_days + 2, 254):
                continue
            prev_px = float(panel[risk_asset].iloc[i - 1])
            prev_ma = float(ma[ma_days][risk_asset].iloc[i - 1])
            if not np.isfinite(prev_ma) or prev_ma <= 0:
                continue
            m12 = prev_px / float(panel[risk_asset].iloc[i - 1 - 252]) - 1.0
            currently_on = state == risk_asset
            if currently_on:
                risk_on = prev_px >= prev_ma * (1.0 - buffer)
            else:
                risk_on = prev_px > prev_ma * (1.0 + buffer)
            if confirm12:
                risk_on = risk_on and m12 > 0
            new_state = risk_asset if risk_on else fallback
            if new_state != state:
                events[ts] = (new_state, leverage if new_state == risk_asset else 1.0)
                state = new_state
        return events

    @dataclass
    class Position:
        symbol: str | None = None
        qty: float = 0.0
        entry_px: float = 0.0
        entry_ts: pd.Timestamp | None = None

    def simulate(events, begin, end, cost=COST, borrow_spread=BORROW_SPREAD,
                 maintenance=MAINT_MARGIN):
        idx = panel.loc[begin:end].index
        if len(idx) < 2:
            return None
        cash = INITIAL
        debt = 0.0
        pos = Position()
        tax_paid = 0.0
        traded = 0.0
        interest_paid = 0.0
        forced = 0
        equity_path = []

        state = ("cash", 1.0)
        for d in sorted(events):
            if d <= idx[0]:
                state = events[d]
            else:
                break

        def close_position(ts):
            nonlocal cash, debt, pos, tax_paid, traded
            if pos.symbol is None or pos.qty <= 1e-12:
                if debt > 1e-8:
                    cash -= debt
                    debt = 0.0
                pos = Position()
                return
            px = float(panel.loc[ts, pos.symbol])
            proceeds = pos.qty * px
            fee = proceeds * cost
            gain = pos.qty * (px - pos.entry_px)
            tax = 0.0
            if gain > 0 and pos.entry_ts is not None:
                tax = gain * tx.slovenia_rate_for_dates(pos.entry_ts, ts)
            cash += proceeds - fee - tax - debt
            traded += proceeds
            tax_paid += tax
            debt = 0.0
            pos = Position()

        def open_state(ts, desired_state):
            nonlocal cash, debt, pos, traded
            name, lev = desired_state
            if name == "cash":
                return
            symbol = name
            nav = cash
            if nav <= 0:
                return
            gross = nav * lev
            px = float(panel.loc[ts, symbol])
            qty = gross / (px * (1.0 + cost))
            actual_notional = qty * px
            fee = actual_notional * cost
            borrow = max(0.0, actual_notional + fee - cash)
            cash = cash + borrow - actual_notional - fee
            debt = borrow
            pos = Position(symbol=symbol, qty=qty, entry_px=px, entry_ts=ts)
            traded += actual_notional

        def switch(ts, desired_state):
            current_name = "cash" if pos.symbol is None else pos.symbol
            target_name, target_lev = desired_state
            if current_name == target_name:
                # Do not rebalance merely because leverage drifted; this reduces tax/turnover.
                return
            close_position(ts)
            open_state(ts, (target_name, target_lev))

        for j, ts in enumerate(idx):
            if j > 0 and debt > 0:
                annual = float(fed.loc[ts]) + borrow_spread
                interest = debt * annual / 360.0
                debt += interest
                interest_paid += interest

            if j == 0:
                open_state(ts, state)
            elif ts in events:
                state = events[ts]
                switch(ts, state)

            if pos.symbol is not None and debt > 0:
                mv = pos.qty * float(panel.loc[ts, pos.symbol])
                nav = cash + mv - debt
                margin_ratio = nav / mv if mv > 0 else -np.inf
                if margin_ratio < maintenance:
                    close_position(ts)
                    state = ("cash", 1.0)
                    forced += 1

            nav = cash - debt
            if pos.symbol is not None:
                nav += pos.qty * float(panel.loc[ts, pos.symbol])
            equity_path.append(nav)
            if nav <= 0 or not np.isfinite(nav):
                return {
                    "equity": pd.Series(equity_path, index=idx[:len(equity_path)]),
                    "cagr": -1.0,
                    "return": -1.0,
                    "dd": -1.0,
                    "tax": tax_paid,
                    "turnover": traded / INITIAL,
                    "interest": interest_paid,
                    "forced": forced,
                }

        close_position(idx[-1])
        equity_path[-1] = cash
        ser = pd.Series(equity_path, index=idx)
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
            "interest": interest_paid,
            "forced": forced,
        }

    def benchmark(begin, end, symbol, cost=COST):
        first = panel.loc[begin:end].index[0]
        return simulate({first: (symbol, 1.0)}, begin, end, cost=cost)

    benchmarks = {w: {s: benchmark(*w, s) for s in ("SPY", "QQQ")} for w in DEV + [HOLD]}

    specs = list(product(
        ("SPY", "QQQ"),
        (200, 250),
        (0.0, 0.01),
        (1.25, 1.50),
        ("cash", "GLD"),
        (False, True),
    ))
    rows = []
    for spec in specs:
        events = build_events(*spec)
        seg = [simulate(events, *w) for w in DEV]
        exs = [m["cagr"] - benchmarks[w]["SPY"]["cagr"] for m, w in zip(seg, DEV)]
        exq = [m["cagr"] - benchmarks[w]["QQQ"]["cagr"] for m, w in zip(seg, DEV)]
        rows.append({
            "spec": spec,
            "events": events,
            "seg": seg,
            "wins_spy": sum(x > 0 for x in exs),
            "min_spy": min(exs),
            "med_spy": float(np.median(exs)),
            "wins_qqq": sum(x > 0 for x in exq),
            "min_qqq": min(exq),
            "med_qqq": float(np.median(exq)),
            "med_cagr": float(np.median([m["cagr"] for m in seg])),
            "worst_dd": min(m["dd"] for m in seg),
            "turnover": sum(m["turnover"] for m in seg),
            "tax": sum(m["tax"] for m in seg),
            "interest": sum(m["interest"] for m in seg),
            "forced": sum(m["forced"] for m in seg),
        })
    rows.sort(
        key=lambda r: (r["wins_spy"], r["min_spy"], r["wins_qqq"], r["min_qqq"], r["med_spy"]),
        reverse=True,
    )

    print("LEVERED SLOW-TREND ACTIVE STRATEGY — DEVELOPMENT RANKING")
    print("risk_asset, ma_days, buffer, leverage, fallback, confirm12")
    for i, r in enumerate(rows[:30], 1):
        print(
            f"{i:02d} spec={r['spec']} winSPY={r['wins_spy']}/4 minSPY={r['min_spy']:.2%} "
            f"medSPY={r['med_spy']:.2%} winQQQ={r['wins_qqq']}/4 minQQQ={r['min_qqq']:.2%} "
            f"medQQQ={r['med_qqq']:.2%} medCAGR={r['med_cagr']:.2%} worstDD={r['worst_dd']:.2%} "
            f"tax={r['tax']:.0f} interest={r['interest']:.0f} turn={r['turnover']:.1f}x forced={r['forced']}"
        )

    print("\nUNTOUCHED HOLDOUT — DEVELOPMENT ORDER FROZEN")
    for r in rows[:15]:
        m = simulate(r["events"], *HOLD)
        b = benchmarks[HOLD]
        print(
            f"spec={r['spec']} CAGR={m['cagr']:.2%} Ret={m['return']:.2%} DD={m['dd']:.2%} "
            f"tax={m['tax']:.0f} interest={m['interest']:.0f} turn={m['turnover']:.1f}x forced={m['forced']} "
            f"exSPY={m['cagr']-b['SPY']['cagr']:.2%} exQQQ={m['cagr']-b['QQQ']['cagr']:.2%}"
        )
    print(f"HOLD BENCH SPY={benchmarks[HOLD]['SPY']['cagr']:.2%} QQQ={benchmarks[HOLD]['QQQ']['cagr']:.2%}")

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
            f"{years}y n={len(exs)} winSPY={np.mean(np.array(exs)>0):.1%} medSPY={np.median(exs):.2%} "
            f"p10SPY={np.quantile(exs,.1):.2%} minSPY={min(exs):.2%} "
            f"winQQQ={np.mean(np.array(exq)>0):.1%} medQQQ={np.median(exq):.2%} "
            f"p10QQQ={np.quantile(exq,.1):.2%} minQQQ={min(exq):.2%}"
        )

    print("\nWINNER COST / FINANCING STRESS")
    for c, spread in ((0.001, 0.010), (0.002, 0.015), (0.004, 0.020), (0.008, 0.030)):
        vals = [simulate(winner["events"], *w, cost=c, borrow_spread=spread) for w in DEV]
        es = [m["cagr"] - benchmark(*w, "SPY", cost=c)["cagr"] for m, w in zip(vals, DEV)]
        hold = simulate(winner["events"], *HOLD, cost=c, borrow_spread=spread)
        hspy = benchmark(*HOLD, "SPY", cost=c)
        print(
            f"cost={c:.2%} spread={spread:.2%} devWinSPY={sum(x>0 for x in es)}/4 minSPY={min(es):.2%} "
            f"hold={hold['cagr']:.2%} holdExSPY={hold['cagr']-hspy['cagr']:.2%}"
        )

    print("\nWINNER PARAMETER NEIGHBORHOOD")
    ws = winner["spec"]
    for r in rows:
        dist = sum(a != b for a, b in zip(r["spec"], ws))
        if dist <= 1:
            m = simulate(r["events"], *HOLD)
            print(
                f"dist={dist} spec={r['spec']} devWinSPY={r['wins_spy']}/4 minSPY={r['min_spy']:.2%} "
                f"devWinQQQ={r['wins_qqq']}/4 minQQQ={r['min_qqq']:.2%} "
                f"hold={m['cagr']:.2%} exSPY={m['cagr']-benchmarks[HOLD]['SPY']['cagr']:.2%} "
                f"exQQQ={m['cagr']-benchmarks[HOLD]['QQQ']['cagr']:.2%}"
            )


if __name__ == "__main__":
    run()
