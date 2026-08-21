import numpy as np
import pandas as pd

from bot.backtest_pullback import fetch_all
from bot.data import clean_daily_data
from bot import taxes as tx

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
CANDIDATE = {"QQQ": 0.50, "SPY": 0.25, "GLD": 0.25}
NEIGHBORS = {
    "40Q30S30G": {"QQQ":.40,"SPY":.30,"GLD":.30},
    "45Q25S30G": {"QQQ":.45,"SPY":.25,"GLD":.30},
    "45Q30S25G": {"QQQ":.45,"SPY":.30,"GLD":.25},
    "50Q20S30G": {"QQQ":.50,"SPY":.20,"GLD":.30},
    "50Q25S25G": CANDIDATE,
    "50Q30S20G": {"QQQ":.50,"SPY":.30,"GLD":.20},
    "55Q20S25G": {"QQQ":.55,"SPY":.20,"GLD":.25},
    "55Q25S20G": {"QQQ":.55,"SPY":.25,"GLD":.20},
    "60Q20S20G": {"QQQ":.60,"SPY":.20,"GLD":.20},
}


def run():
    daily, _, _ = fetch_all(SYMBOLS, "none", START, END, source="yahoo")
    daily = clean_daily_data(daily)
    panel = pd.DataFrame({s: daily[s]["close"] for s in SYMBOLS}).sort_index().ffill().dropna()

    def simulate(weights, begin, end, cost=0.002):
        idx = panel.loc[begin:end].index
        if len(idx) < 2: return None
        eq = pd.Series(0.0,index=idx); final=0.0; tax=0.0
        for s,w in weights.items():
            if w <= 0: continue
            capital=INITIAL*w; p0=float(panel.loc[idx[0],s]); p1=float(panel.loc[idx[-1],s])
            qty=capital/(p0*(1+cost)); cash=capital-qty*p0*(1+cost)
            eq += cash + qty*panel.loc[idx,s]
            gain=qty*(p1-p0); t=max(gain,0.0)*tx.slovenia_rate_for_dates(idx[0],idx[-1])
            tax += t; final += cash + qty*p1*(1-cost) - t
        eq.iloc[-1]=final
        years=(idx[-1]-idx[0]).total_seconds()/(365.2425*86400)
        cagr=(final/INITIAL)**(1/years)-1
        dd=float((eq/eq.cummax()-1).min())
        return {"cagr":cagr,"return":final/INITIAL-1,"dd":dd,"tax":tax,"final":final}

    EQ7={s:1/7 for s in SYMBOLS}; SPY={"SPY":1.0}; QQQ={"QQQ":1.0}
    print("FIXED CANDIDATE REGIME TEST @20BP")
    for a,b in DEV:
        c=simulate(CANDIDATE,a,b); e=simulate(EQ7,a,b); s=simulate(SPY,a,b); q=simulate(QQQ,a,b)
        print(f"{a.year}-{b.year} cand={c['cagr']:.2%} eq7={e['cagr']:.2%} SPY={s['cagr']:.2%} QQQ={q['cagr']:.2%} "
              f"exEq={c['cagr']-e['cagr']:.2%} exSPY={c['cagr']-s['cagr']:.2%} exQQQ={c['cagr']-q['cagr']:.2%} DD={c['dd']:.2%}")

    print("\nCONTINUOUS DEVELOPMENT + HOLDOUT")
    for name,(a,b) in {"DEV":(pd.Timestamp('2006-01-01',tz='UTC'),pd.Timestamp('2024-12-31',tz='UTC')),"HOLD":HOLD}.items():
        c=simulate(CANDIDATE,a,b); e=simulate(EQ7,a,b); s=simulate(SPY,a,b); q=simulate(QQQ,a,b)
        print(f"{name} cand={c['cagr']:.2%} eq7={e['cagr']:.2%} SPY={s['cagr']:.2%} QQQ={q['cagr']:.2%} "
              f"exEq={c['cagr']-e['cagr']:.2%} exSPY={c['cagr']-s['cagr']:.2%} exQQQ={c['cagr']-q['cagr']:.2%} DD={c['dd']:.2%} tax={c['tax']:.0f}")

    print("\nCOST STRESS")
    for cost in (.001,.002,.004,.008):
        vals=[]
        for a,b in DEV:
            c=simulate(CANDIDATE,a,b,cost); e=simulate(EQ7,a,b,cost); s=simulate(SPY,a,b,cost)
            vals.append((c['cagr']-e['cagr'],c['cagr']-s['cagr']))
        h=simulate(CANDIDATE,*HOLD,cost); he=simulate(EQ7,*HOLD,cost); hs=simulate(SPY,*HOLD,cost)
        print(f"cost={cost*10000:.0f}bp devMinEq={min(x[0] for x in vals):.2%} devMinSPY={min(x[1] for x in vals):.2%} "
              f"holdExEq={h['cagr']-he['cagr']:.2%} holdExSPY={h['cagr']-hs['cagr']:.2%}")

    print("\nWEIGHT NEIGHBORHOOD @20BP")
    for name,w in NEIGHBORS.items():
        exeq=[]; exspy=[]
        for a,b in DEV:
            m=simulate(w,a,b); exeq.append(m['cagr']-simulate(EQ7,a,b)['cagr']); exspy.append(m['cagr']-simulate(SPY,a,b)['cagr'])
        h=simulate(w,*HOLD); he=simulate(EQ7,*HOLD); hs=simulate(SPY,*HOLD)
        print(f"{name} winEq={sum(x>0 for x in exeq)}/4 minEq={min(exeq):.2%} winSPY={sum(x>0 for x in exspy)}/4 minSPY={min(exspy):.2%} "
              f"hold={h['cagr']:.2%} holdExEq={h['cagr']-he['cagr']:.2%} holdExSPY={h['cagr']-hs['cagr']:.2%}")

    print("\nROLLING WINDOWS, QUARTERLY STARTS, DEVELOPMENT ONLY")
    for years in (3,5,7):
        starts=pd.date_range(pd.Timestamp('2006-01-01',tz='UTC'),pd.Timestamp(f'{2024-years}-12-31',tz='UTC'),freq='QS')
        eqwins=[]; spywins=[]; excess=[]; excess_spy=[]
        for a in starts:
            b=a+pd.DateOffset(years=years)
            c=simulate(CANDIDATE,a,b); e=simulate(EQ7,a,b); s=simulate(SPY,a,b)
            if c is None or e is None or s is None: continue
            excess.append(c['cagr']-e['cagr']); excess_spy.append(c['cagr']-s['cagr'])
        print(f"{years}y n={len(excess)} winEq={np.mean(np.array(excess)>0):.1%} medEq={np.median(excess):.2%} p10Eq={np.quantile(excess,.1):.2%} minEq={min(excess):.2%} "
              f"winSPY={np.mean(np.array(excess_spy)>0):.1%} medSPY={np.median(excess_spy):.2%} p10SPY={np.quantile(excess_spy,.1):.2%} minSPY={min(excess_spy):.2%}")

    print("\nSTART-DATE PERTURBATION: 10Y WINDOWS")
    for year in range(2006,2016):
        a=pd.Timestamp(f'{year}-01-01',tz='UTC'); b=a+pd.DateOffset(years=10)
        c=simulate(CANDIDATE,a,b); e=simulate(EQ7,a,b); s=simulate(SPY,a,b)
        print(f"{year} cand={c['cagr']:.2%} exEq={c['cagr']-e['cagr']:.2%} exSPY={c['cagr']-s['cagr']:.2%}")

if __name__ == '__main__': run()
