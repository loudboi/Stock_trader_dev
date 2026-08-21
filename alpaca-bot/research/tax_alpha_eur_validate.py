import numpy as np
import pandas as pd

from bot.backtest_pullback import fetch_all
from bot.data import clean_daily_data
from bot import taxes as tx

SYMBOLS=["SPY","QQQ","GLD","TLT","EFA","EEM","IWM"]
START=pd.Timestamp("2004-01-01",tz="UTC")
END=pd.Timestamp("2026-08-20",tz="UTC")
DEV=[
 (pd.Timestamp("2006-01-01",tz="UTC"),pd.Timestamp("2009-12-31",tz="UTC")),
 (pd.Timestamp("2010-01-01",tz="UTC"),pd.Timestamp("2014-12-31",tz="UTC")),
 (pd.Timestamp("2015-01-01",tz="UTC"),pd.Timestamp("2019-12-31",tz="UTC")),
 (pd.Timestamp("2020-01-01",tz="UTC"),pd.Timestamp("2024-12-31",tz="UTC")),
]
HOLD=(pd.Timestamp("2025-01-01",tz="UTC"),END)
INITIAL=100000.0
CAND={"QQQ":.60,"SPY":.20,"GLD":.20}
EQ7={s:1/7 for s in SYMBOLS}; SPY={"SPY":1.0}; QQQ={"QQQ":1.0}
NEIGHBORS={
 "55Q20S25G":{"QQQ":.55,"SPY":.20,"GLD":.25},
 "55Q25S20G":{"QQQ":.55,"SPY":.25,"GLD":.20},
 "58Q20S22G":{"QQQ":.58,"SPY":.20,"GLD":.22},
 "60Q15S25G":{"QQQ":.60,"SPY":.15,"GLD":.25},
 "60Q20S20G":CAND,
 "60Q25S15G":{"QQQ":.60,"SPY":.25,"GLD":.15},
 "62Q18S20G":{"QQQ":.62,"SPY":.18,"GLD":.20},
 "65Q15S20G":{"QQQ":.65,"SPY":.15,"GLD":.20},
}

def run():
 daily,_,_=fetch_all(SYMBOLS,"none",START,END,source="yahoo")
 daily=clean_daily_data(daily)
 panel=pd.DataFrame({s:daily[s]["close"] for s in SYMBOLS}).sort_index().ffill().dropna()
 # FRED DEXUSEU = USD per EUR. For a USD asset, EUR value = USD / DEXUSEU.
 f=pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU")
 f.columns=["date","fx"]; f["date"]=pd.to_datetime(f["date"],utc=True); f["fx"]=pd.to_numeric(f["fx"],errors="coerce")
 fx=f.dropna().set_index("date")["fx"].sort_index()
 fx=fx.reindex(fx.index.union(panel.index)).ffill().reindex(panel.index)
 if fx.isna().any() or (fx<=0).any(): raise RuntimeError("invalid EURUSD FX alignment")

 def simulate(weights,begin,end,cost=.002):
  idx=panel.loc[begin:end].index
  if len(idx)<2:return None
  x=fx.reindex(idx); eq=pd.Series(0.0,index=idx); final=0.0; tax=0.0
  for s,w in weights.items():
   if w<=0:continue
   eur_cap=INITIAL*w; p0=float(panel.loc[idx[0],s]); p1=float(panel.loc[idx[-1],s]); f0=float(x.iloc[0]); f1=float(x.iloc[-1])
   usd_cap=eur_cap*f0; qty=usd_cap/(p0*(1+cost)); eur_cash=(usd_cap-qty*p0*(1+cost))/f0
   eq += eur_cash + qty*panel.loc[idx,s]/x
   acquisition_eur=qty*p0/f0; disposal_eur=qty*p1/f1
   taxable=max(disposal_eur-acquisition_eur,0.0); t=taxable*tx.slovenia_rate_for_dates(idx[0],idx[-1])
   tax+=t; final += eur_cash + qty*p1*(1-cost)/f1 - t
  eq.iloc[-1]=final
  years=(idx[-1]-idx[0]).total_seconds()/(365.2425*86400); cagr=(final/INITIAL)**(1/years)-1
  return {"cagr":cagr,"return":final/INITIAL-1,"dd":float((eq/eq.cummax()-1).min()),"tax":tax}

 print("EUR / FX-AWARE FIXED 60Q20S20G @20BP")
 for a,b in DEV:
  c=simulate(CAND,a,b);e=simulate(EQ7,a,b);s=simulate(SPY,a,b);q=simulate(QQQ,a,b)
  print(f"{a.year}-{b.year} cand={c['cagr']:.2%} eq7={e['cagr']:.2%} SPY={s['cagr']:.2%} QQQ={q['cagr']:.2%} exEq={c['cagr']-e['cagr']:.2%} exSPY={c['cagr']-s['cagr']:.2%} exQQQ={c['cagr']-q['cagr']:.2%} DD={c['dd']:.2%}")
 print("\nEUR HOLDOUT")
 c=simulate(CAND,*HOLD);e=simulate(EQ7,*HOLD);s=simulate(SPY,*HOLD);q=simulate(QQQ,*HOLD)
 print(f"cand={c['cagr']:.2%} Ret={c['return']:.2%} DD={c['dd']:.2%} tax={c['tax']:.0f} eq7={e['cagr']:.2%} SPY={s['cagr']:.2%} QQQ={q['cagr']:.2%} exEq={c['cagr']-e['cagr']:.2%} exSPY={c['cagr']-s['cagr']:.2%} exQQQ={c['cagr']-q['cagr']:.2%}")
 print("\nEUR COST STRESS")
 for cost in (.002,.004,.008):
  de=[];ds=[]
  for a,b in DEV:
   c=simulate(CAND,a,b,cost);e=simulate(EQ7,a,b,cost);s=simulate(SPY,a,b,cost);de.append(c['cagr']-e['cagr']);ds.append(c['cagr']-s['cagr'])
  h=simulate(CAND,*HOLD,cost);he=simulate(EQ7,*HOLD,cost);hs=simulate(SPY,*HOLD,cost)
  print(f"cost={cost*10000:.0f}bp devMinEq={min(de):.2%} devMinSPY={min(ds):.2%} holdExEq={h['cagr']-he['cagr']:.2%} holdExSPY={h['cagr']-hs['cagr']:.2%}")
 print("\nEUR WEIGHT SENSITIVITY — DEVELOPMENT ONLY + FROZEN HOLD REPORT")
 for name,w in NEIGHBORS.items():
  de=[];ds=[]
  for a,b in DEV:
   m=simulate(w,a,b);de.append(m['cagr']-simulate(EQ7,a,b)['cagr']);ds.append(m['cagr']-simulate(SPY,a,b)['cagr'])
  h=simulate(w,*HOLD);he=simulate(EQ7,*HOLD);hs=simulate(SPY,*HOLD)
  print(f"{name} winEq={sum(v>0 for v in de)}/4 minEq={min(de):.2%} winSPY={sum(v>0 for v in ds)}/4 minSPY={min(ds):.2%} hold={h['cagr']:.2%} holdExEq={h['cagr']-he['cagr']:.2%} holdExSPY={h['cagr']-hs['cagr']:.2%}")
 print("\nEUR ROLLING WINDOWS — FIXED CANDIDATE, DEVELOPMENT ONLY")
 for years in (3,5,7,10):
  latest=2024-years
  starts=pd.date_range(pd.Timestamp('2006-01-01',tz='UTC'),pd.Timestamp(f'{latest}-12-31',tz='UTC'),freq='QS')
  de=[];ds=[]
  for a in starts:
   b=a+pd.DateOffset(years=years);c=simulate(CAND,a,b);e=simulate(EQ7,a,b);s=simulate(SPY,a,b)
   if c and e and s:de.append(c['cagr']-e['cagr']);ds.append(c['cagr']-s['cagr'])
  print(f"{years}y n={len(de)} winEq={np.mean(np.array(de)>0):.1%} medEq={np.median(de):.2%} p10Eq={np.quantile(de,.1):.2%} minEq={min(de):.2%} winSPY={np.mean(np.array(ds)>0):.1%} medSPY={np.median(ds):.2%} p10SPY={np.quantile(ds,.1):.2%} minSPY={min(ds):.2%}")
 # Common-inception pre-development check, not used to select the weights.
 print("\nPRE-DEVELOPMENT 2005")
 a=pd.Timestamp('2005-01-01',tz='UTC');b=pd.Timestamp('2005-12-31',tz='UTC');c=simulate(CAND,a,b);e=simulate(EQ7,a,b);s=simulate(SPY,a,b)
 print(f"cand={c['cagr']:.2%} eq7={e['cagr']:.2%} SPY={s['cagr']:.2%} exEq={c['cagr']-e['cagr']:.2%} exSPY={c['cagr']-s['cagr']:.2%}")

if __name__=='__main__':run()
