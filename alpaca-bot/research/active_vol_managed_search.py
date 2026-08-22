from collections import deque
from itertools import product

import numpy as np
import pandas as pd

from bot.backtest_pullback import fetch_all
from bot.data import clean_daily_data
from bot import taxes as tx

SYMBOLS = ["SPY", "QQQ"]
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
NO_TRADE_BAND = 0.20
MIN_EXPOSURE = 0.50


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
    returns = panel.pct_change(fill_method=None)
    fed = fred("DFF", panel.index).clip(lower=0.0) / 100.0
    vol = {
        n: returns.rolling(n, min_periods=n).std().shift(1) * np.sqrt(252)
        for n in (20, 60)
    }
    ma200 = panel.rolling(200, min_periods=200).mean().shift(1)

    def build_events(symbol: str, vol_lb: int, target_vol: float, cap: float, trend_cap: bool):
        events = {}
        last_month = None
        for i, ts in enumerate(panel.index):
            key = (ts.year, ts.month)
            if last_month is None:
                last_month = key
                continue
            if key == last_month:
                continue
            last_month = key
            if i < max(202, vol_lb + 2):
                continue
            realized = float(vol[vol_lb][symbol].iloc[i - 1])
            if not np.isfinite(realized) or realized <= 0:
                continue
            exposure = float(np.clip(target_vol / realized, MIN_EXPOSURE, cap))
            if trend_cap:
                prev_px = float(panel[symbol].iloc[i - 1])
                prev_ma = float(ma200[symbol].iloc[i - 1])
                if np.isfinite(prev_ma) and prev_px < prev_ma:
                    exposure = min(exposure, MIN_EXPOSURE)
            events[ts] = exposure
        return events

    def simulate(symbol: str, events, begin, end, cost=COST, borrow_spread=BORROW_SPREAD,
                 band=NO_TRADE_BAND, maintenance=MAINT_MARGIN):
        idx = panel.loc[begin:end].index
        if len(idx) < 2:
            return None
        cash = INITIAL
        debt = 0.0
        qty = 0.0
        lots = deque()
        tax_paid = 0.0
        interest_paid = 0.0
        traded = 0.0
        forced = 0
        equity_path = []

        target = 1.0
        for d in sorted(events):
            if d <= idx[0]:
                target = float(events[d])
            else:
                break

        def px_at(ts):
            return float(panel.loc[ts, symbol])

        def nav(ts):
            return cash + qty * px_at(ts) - debt

        def repay_debt():
            nonlocal cash, debt
            if cash > 0 and debt > 0:
                pay = min(cash, debt)
                cash -= pay
                debt -= pay

        def sell(q, ts):
            nonlocal cash, qty, tax_paid, traded
            q = min(max(0.0, q), qty)
            if q <= 1e-12:
                return
            px = px_at(ts)
            remain = q
            tax = 0.0
            while remain > 1e-10 and lots:
                lot_q, lot_px, lot_ts = lots[0]
                used = min(remain, lot_q)
                gain = used * (px - lot_px)
                if gain > 0:
                    tax += gain * tx.slovenia_rate_for_dates(lot_ts, ts)
                lot_q -= used
                remain -= used
                if lot_q <= 1e-10:
                    lots.popleft()
                else:
                    lots[0] = (lot_q, lot_px, lot_ts)
            notional = q * px
            cash += notional * (1.0 - cost) - tax
            qty -= q
            tax_paid += tax
            traded += notional
            repay_debt()

        def buy(q, ts):
            nonlocal cash, debt, qty, traded
            q = max(0.0, q)
            if q <= 1e-12:
                return
            px = px_at(ts)
            notional = q * px
            total = notional * (1.0 + cost)
            if cash >= total:
                cash -= total
            else:
                shortfall = total - cash
                cash = 0.0
                debt += shortfall
            qty += q
            lots.append((q, px, ts))
            traded += notional

        def rebalance(ts, desired_exposure, force=False):
            nonlocal target
            n = nav(ts)
            if n <= 0:
                return
            mv = qty * px_at(ts)
            current = mv / n if n > 0 else np.inf
            if not force and abs(desired_exposure - current) < band:
                target = desired_exposure
                return
            desired_mv = max(0.0, desired_exposure * n)
            if mv > desired_mv:
                sell((mv - desired_mv) / px_at(ts), ts)
            else:
                buy((desired_mv - mv) / px_at(ts), ts)
            target = desired_exposure

        for j, ts in enumerate(idx):
            if j > 0 and debt > 0:
                annual = float(fed.loc[ts]) + borrow_spread
                interest = debt * annual / 360.0
                debt += interest
                interest_paid += interest

            if j == 0:
                rebalance(ts, target, force=True)
            elif ts in events:
                target = float(events[ts])
                rebalance(ts, target)

            if qty > 0 and debt > 0:
                mv = qty * px_at(ts)
                n = nav(ts)
                margin_ratio = n / mv if mv > 0 else -np.inf
                if margin_ratio < maintenance:
                    sell(qty, ts)
                    target = MIN_EXPOSURE
                    forced += 1

            n = nav(ts)
            equity_path.append(n)
            if n <= 0 or not np.isfinite(n):
                return {
                    "equity": pd.Series(equity_path, index=idx[:len(equity_path)]),
                    "cagr": -1.0, "return": -1.0, "dd": -1.0,
                    "tax": tax_paid, "interest": interest_paid,
                    "turnover": traded / INITIAL, "forced": forced,
                }

        sell(qty, idx[-1])
        repay_debt()
        equity_path[-1] = cash - debt
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
            "interest": interest_paid,
            "turnover": traded / INITIAL,
            "forced": forced,
        }

    def benchmark(begin, end, symbol, cost=COST):
        idx = panel.loc[begin:end].index
        if len(idx) < 2:
            return None
        p0 = float(panel.loc[idx[0], symbol])
        qty = INITIAL / (p0 * (1.0 + cost))
        entry_notional = qty * p0
        p1 = float(panel.loc[idx[-1], symbol])
        gain = qty * (p1 - p0)
        tax = gain * tx.slovenia_rate_for_dates(idx[0], idx[-1]) if gain > 0 else 0.0
        final = qty * p1 * (1.0 - cost) - tax
        path = qty * panel.loc[idx, symbol]
        path.iloc[0] = INITIAL
        path.iloc[-1] = final
        years = (idx[-1] - idx[0]).total_seconds() / (365.2425 * 86400)
        return {
            "cagr": (final / INITIAL) ** (1.0 / years) - 1.0,
            "return": final / INITIAL - 1.0,
            "dd": float((path / path.cummax() - 1.0).min()),
            "tax": tax,
            "turnover": (entry_notional + qty * p1) / INITIAL,
        }

    benchmarks = {w: {s: benchmark(*w, s) for s in SYMBOLS} for w in DEV + [HOLD]}

    specs = list(product(
        ("SPY", "QQQ"),
        (20, 60),
        (0.20, 0.25),
        (1.50, 1.75),
        (False, True),
    ))
    rows = []
    for spec in specs:
        events = build_events(*spec)
        symbol = spec[0]
        seg = [simulate(symbol, events, *w) for w in DEV]
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
            "tax": sum(m["tax"] for m in seg),
            "interest": sum(m["interest"] for m in seg),
            "turnover": sum(m["turnover"] for m in seg),
            "forced": sum(m["forced"] for m in seg),
        })
    rows.sort(
        key=lambda r: (r["wins_spy"], r["min_spy"], r["wins_qqq"], r["min_qqq"], r["med_spy"]),
        reverse=True,
    )

    print("VOLATILITY-MANAGED ACTIVE EXPOSURE — DEVELOPMENT RANKING")
    print("symbol, vol_lookback, target_vol, leverage_cap, trend_cap")
    for i, r in enumerate(rows, 1):
        print(
            f"{i:02d} spec={r['spec']} winSPY={r['wins_spy']}/4 minSPY={r['min_spy']:.2%} "
            f"medSPY={r['med_spy']:.2%} winQQQ={r['wins_qqq']}/4 minQQQ={r['min_qqq']:.2%} "
            f"medQQQ={r['med_qqq']:.2%} medCAGR={r['med_cagr']:.2%} worstDD={r['worst_dd']:.2%} "
            f"tax={r['tax']:.0f} interest={r['interest']:.0f} turn={r['turnover']:.1f}x forced={r['forced']}"
        )

    print("\nUNTOUCHED HOLDOUT — DEVELOPMENT ORDER FROZEN")
    for r in rows[:12]:
        symbol = r["spec"][0]
        m = simulate(symbol, r["events"], *HOLD)
        b = benchmarks[HOLD]
        print(
            f"spec={r['spec']} CAGR={m['cagr']:.2%} Ret={m['return']:.2%} DD={m['dd']:.2%} "
            f"tax={m['tax']:.0f} interest={m['interest']:.0f} turn={m['turnover']:.1f}x forced={m['forced']} "
            f"exSPY={m['cagr']-b['SPY']['cagr']:.2%} exQQQ={m['cagr']-b['QQQ']['cagr']:.2%}"
        )
    print(f"HOLD BENCH SPY={benchmarks[HOLD]['SPY']['cagr']:.2%} QQQ={benchmarks[HOLD]['QQQ']['cagr']:.2%}")

    winner = rows[0]
    wsymbol = winner["spec"][0]
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
            m = simulate(wsymbol, winner["events"], a, b)
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
        vals = [simulate(wsymbol, winner["events"], *w, cost=c, borrow_spread=spread) for w in DEV]
        es = [m["cagr"] - benchmark(*w, "SPY", cost=c)["cagr"] for m, w in zip(vals, DEV)]
        hold = simulate(wsymbol, winner["events"], *HOLD, cost=c, borrow_spread=spread)
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
            symbol = r["spec"][0]
            m = simulate(symbol, r["events"], *HOLD)
            print(
                f"dist={dist} spec={r['spec']} devWinSPY={r['wins_spy']}/4 minSPY={r['min_spy']:.2%} "
                f"devWinQQQ={r['wins_qqq']}/4 minQQQ={r['min_qqq']:.2%} "
                f"hold={m['cagr']:.2%} exSPY={m['cagr']-benchmarks[HOLD]['SPY']['cagr']:.2%} "
                f"exQQQ={m['cagr']-benchmarks[HOLD]['QQQ']['cagr']:.2%}"
            )


if __name__ == "__main__":
    run()
