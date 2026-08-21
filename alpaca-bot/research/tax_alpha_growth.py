from itertools import product
import numpy as np
import pandas as pd

from bot.backtest_pullback import fetch_all
from bot.data import clean_daily_data
from bot import taxes as tx
from bot import trend_exposure as te

SYMBOLS = ["SPY", "QQQ", "GLD", "TLT", "EFA", "EEM", "IWM"]
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


def run():
    daily, _, _ = fetch_all(SYMBOLS, "none", START, END, source="yahoo")
    if set(daily) != set(SYMBOLS):
        raise RuntimeError("missing requested price history")
    daily = clean_daily_data(daily)
    panel = pd.DataFrame({s: daily[s]["close"] for s in SYMBOLS}).sort_index().ffill().dropna()

    def buyhold_sleeve(symbol, begin, end, initial):
        idx = panel.loc[begin:end].index
        px0, px1 = float(panel.loc[idx[0], symbol]), float(panel.loc[idx[-1], symbol])
        qty = initial / (px0 * (1 + COST))
        cash = initial - qty * px0 * (1 + COST)
        eq = cash + qty * panel.loc[idx, symbol]
        gain = qty * (px1 - px0)
        tax = max(gain, 0.0) * tx.slovenia_rate_for_dates(idx[0], idx[-1])
        final = cash + qty * px1 * (1 - COST) - tax
        eq = eq.copy(); eq.iloc[-1] = final
        return eq, tax

    def static_port(weights, begin, end):
        parts, tax = [], 0.0
        for s, w in weights.items():
            if w <= 0: continue
            e, t = buyhold_sleeve(s, begin, end, INITIAL * w)
            parts.append(e); tax += t
        eq = pd.concat(parts, axis=1).sum(axis=1)
        years = (eq.index[-1]-eq.index[0]).total_seconds()/(365.2425*86400)
        return {"equity":eq, "cagr":(eq.iloc[-1]/INITIAL)**(1/years)-1,
                "return":eq.iloc[-1]/INITIAL-1, "dd":float((eq/eq.cummax()-1).min()),
                "tax":tax}

    def defensive_choice(i):
        end_i = i - 1 - 21
        start_i = end_i - 12*21
        if start_i < 0: return None
        score = panel.iloc[end_i][["GLD","TLT"]] / panel.iloc[start_i][["GLD","TLT"]] - 1
        score = score[score > 0].sort_values(ascending=False)
        return None if score.empty else str(score.index[0])

    def assignment_events(risk_asset, ma, buffer, fallback):
        exposure = te.exposure_series(panel[risk_asset], ma, buffer).shift(1).fillna(0.0)
        events = {}
        current = None
        for i, ts in enumerate(panel.index):
            on = exposure.iloc[i] > 0.5
            if on:
                wanted = risk_asset
            elif fallback == "cash":
                wanted = None
            elif fallback == "GLD":
                wanted = "GLD"
            elif fallback == "best_def":
                # Choose only when risk-off assignment is entered; keep that defensive
                # asset until risk-on returns, avoiding monthly tax churn.
                wanted = current if current in {"GLD","TLT"} else defensive_choice(i)
            else:
                raise ValueError(fallback)
            if wanted != current:
                events[ts] = wanted
                current = wanted
        return events

    def switch_sleeve(events, begin, end, initial):
        idx = panel.loc[begin:end].index
        cash = initial; qty = 0.0; asset = None; basis_px = None; acquired = None
        tax_paid = traded = 0.0; eq = []
        current = None
        for d in sorted(events):
            if d <= idx[0]: current = events[d]
            else: break
        for j, ts in enumerate(idx):
            desired = events.get(ts, current)
            if j == 0: desired = current
            if desired != asset:
                if asset is not None and qty > 0:
                    px = float(panel.loc[ts, asset]); notional = qty*px
                    gain = qty*(px-basis_px)
                    tax = max(gain,0.0)*tx.slovenia_rate_for_dates(acquired, ts)
                    cash += notional*(1-COST)-tax; tax_paid += tax; traded += notional
                    qty=0.0; basis_px=acquired=None
                if desired is not None:
                    px=float(panel.loc[ts,desired]); qty=cash/(px*(1+COST))
                    notional=qty*px; cash-=notional*(1+COST); traded+=notional
                    basis_px=px; acquired=ts
                asset=desired; current=desired
            nav=cash+(qty*float(panel.loc[ts,asset]) if asset is not None else 0.0)
            eq.append(nav)
        ts=idx[-1]
        if asset is not None and qty>0:
            px=float(panel.loc[ts,asset]); notional=qty*px; gain=qty*(px-basis_px)
            tax=max(gain,0.0)*tx.slovenia_rate_for_dates(acquired,ts)
            cash+=notional*(1-COST)-tax; tax_paid+=tax; traded+=notional
            eq[-1]=cash
        ser=pd.Series(eq,index=idx)
        return ser,tax_paid,traded/initial if initial>0 else 0.0

    def trend_port(spec, begin, end):
        qshare, ma, buffer, fallback = spec
        parts=[]; tax=turn=0.0
        for asset,w in (("QQQ",qshare),("SPY",1-qshare)):
            if w<=0: continue
            e,t,tr=switch_sleeve(assignment_events(asset,ma,buffer,fallback),begin,end,INITIAL*w)
            parts.append(e); tax+=t; turn+=tr*w
        eq=pd.concat(parts,axis=1).sum(axis=1)
        years=(eq.index[-1]-eq.index[0]).total_seconds()/(365.2425*86400)
        return {"equity":eq,"cagr":(eq.iloc[-1]/INITIAL)**(1/years)-1,
                "return":eq.iloc[-1]/INITIAL-1,"dd":float((eq/eq.cummax()-1).min()),
                "tax":tax,"turnover":turn}

    eq7={s:1/len(SYMBOLS) for s in SYMBOLS}
    benchmarks={w:{"equal7":static_port(eq7,*w),"SPY":static_port({"SPY":1},*w),
                   "QQQ":static_port({"QQQ":1},*w)} for w in DEV+[HOLD]}

    static_specs={
        "50Q30S20G":{"QQQ":.5,"SPY":.3,"GLD":.2},
        "50Q25S25G":{"QQQ":.5,"SPY":.25,"GLD":.25},
        "60Q20S20G":{"QQQ":.6,"SPY":.2,"GLD":.2},
        "40Q40S20G":{"QQQ":.4,"SPY":.4,"GLD":.2},
        "50Q30S10G10T":{"QQQ":.5,"SPY":.3,"GLD":.1,"TLT":.1},
        "67Q33S":{"QQQ":2/3,"SPY":1/3},
        "75Q25S":{"QQQ":.75,"SPY":.25},
    }
    print("STATIC TAX-EFFICIENT TILTS")
    for name,wgt in static_specs.items():
        seg=[static_port(wgt,*w) for w in DEV]
        ex=[m['cagr']-benchmarks[w]['equal7']['cagr'] for m,w in zip(seg,DEV)]
        h=static_port(wgt,*HOLD)
        print(f"{name} devWins={sum(x>0 for x in ex)}/4 minEx={min(ex):.2%} medEx={np.median(ex):.2%} "
              f"hold={h['cagr']:.2%} holdExEq={h['cagr']-benchmarks[HOLD]['equal7']['cagr']:.2%} "
              f"holdExSPY={h['cagr']-benchmarks[HOLD]['SPY']['cagr']:.2%}")

    specs=list(product((.5,2/3,.75,1.0),(150,200,250),(0.0,0.01),("cash","GLD","best_def")))
    rows=[]
    for spec in specs:
        seg=[trend_port(spec,*w) for w in DEV]
        exeq=[m['cagr']-benchmarks[w]['equal7']['cagr'] for m,w in zip(seg,DEV)]
        exspy=[m['cagr']-benchmarks[w]['SPY']['cagr'] for m,w in zip(seg,DEV)]
        rows.append({"spec":spec,"seg":seg,"wins_eq":sum(x>0 for x in exeq),
                     "wins_spy":sum(x>0 for x in exspy),"min_eq":min(exeq),
                     "med_eq":float(np.median(exeq)),"min_spy":min(exspy),
                     "med_spy":float(np.median(exspy)),"worst_dd":min(m['dd'] for m in seg),
                     "tax":sum(m['tax'] for m in seg),"turn":sum(m['turnover'] for m in seg)})
    rows.sort(key=lambda r:(r['wins_eq'],r['min_eq'],r['wins_spy'],r['min_spy'],r['med_eq']),reverse=True)
    print("\nTREND-PROTECTED GROWTH DEVELOPMENT RANKING")
    for i,r in enumerate(rows[:25],1):
        print(f"{i:02d} spec={r['spec']} winEq={r['wins_eq']}/4 minEq={r['min_eq']:.2%} medEq={r['med_eq']:.2%} "
              f"winSPY={r['wins_spy']}/4 minSPY={r['min_spy']:.2%} medSPY={r['med_spy']:.2%} "
              f"worstDD={r['worst_dd']:.2%} tax={r['tax']:.0f} turn={r['turn']:.1f}x")
    print("\nUNTOUCHED HOLDOUT")
    for r in rows[:15]:
        m=trend_port(r['spec'],*HOLD); b=benchmarks[HOLD]
        print(f"spec={r['spec']} CAGR={m['cagr']:.2%} Ret={m['return']:.2%} DD={m['dd']:.2%} tax={m['tax']:.0f} "
              f"exEq={m['cagr']-b['equal7']['cagr']:.2%} exSPY={m['cagr']-b['SPY']['cagr']:.2%} exQQQ={m['cagr']-b['QQQ']['cagr']:.2%}")
    print("\nBENCH HOLD", {k:round(v['cagr']*100,2) for k,v in benchmarks[HOLD].items()})

    winner=rows[0]['spec']
    print("\nWINNER NEIGHBORHOOD")
    for r in rows:
        dist=sum(a!=b for a,b in zip(r['spec'],winner))
        if dist<=1:
            m=trend_port(r['spec'],*HOLD)
            print(f"dist={dist} spec={r['spec']} devWins={r['wins_eq']}/4 minEq={r['min_eq']:.2%} "
                  f"hold={m['cagr']:.2%} holdExEq={m['cagr']-benchmarks[HOLD]['equal7']['cagr']:.2%}")

if __name__ == "__main__":
    run()
