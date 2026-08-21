from dataclasses import dataclass
import numpy as np
import pandas as pd
import yfinance as yf

from bot import taxes as tx

SYMBOLS=["SPY","QQQ","GLD","TLT","EFA","EEM","IWM"]
START="2005-01-01"; END="2026-08-21"
DEV=[
 (pd.Timestamp("2006-01-01",tz="UTC"),pd.Timestamp("2009-12-31",tz="UTC")),
 (pd.Timestamp("2010-01-01",tz="UTC"),pd.Timestamp("2014-12-31",tz="UTC")),
 (pd.Timestamp("2015-01-01",tz="UTC"),pd.Timestamp("2019-12-31",tz="UTC")),
 (pd.Timestamp("2020-01-01",tz="UTC"),pd.Timestamp("2024-12-31",tz="UTC")),
]
HOLD=(pd.Timestamp("2025-01-01",tz="UTC"),pd.Timestamp("2026-08-20",tz="UTC"))
INITIAL=100000.0; COST=.002
CAND={"QQQ":.60,"SPY":.20,"GLD":.20}; EQ7={s:1/7 for s in SYMBOLS}; SPY={"SPY":1.0}; QQQ={"QQQ":1.0}

@dataclass
class Lot:
 qty: float
 basis_eur: float
 acquired: pd.Timestamp


def load_symbol(s):
 d=yf.Ticker(s).history(start=START,end=END,auto_adjust=False,actions=True,repair=False)
 if d is None or d.empty: raise RuntimeError(f"no raw history for {s}")
 idx=pd.DatetimeIndex(d.index)
 d=d.copy(); d.index=idx.tz_convert('UTC') if idx.tz is not None else idx.tz_localize('UTC')
 close=pd.to_numeric(d['Close'],errors='coerce')
 div=pd.to_numeric(d.get('Dividends',0.0),errors='coerce').fillna(0.0)
 split=pd.to_numeric(d.get('Stock Splits',0.0),errors='coerce').fillna(0.0)
 out=pd.DataFrame({'close':close,'dividend':div,'split':split}).dropna(subset=['close'])
 return out[out.close>0]


def run():
 raw={s:load_symbol(s) for s in SYMBOLS}
 common=max(d.index[0] for d in raw.values()); last=min(d.index[-1] for d in raw.values())
 idx=pd.date_range(common.normalize(),last.normalize(),freq='B',tz='UTC')
 # Keep only dates with at least one US market close; per-symbol prices carry forward over venue holidays.
 frame=pd.DataFrame({s:raw[s].close.reindex(idx).ffill() for s in SYMBOLS}).dropna()
 idx=frame.index
 # Actions only happen on their true source rows; never forward-fill dividends or splits.
 div={s:raw[s].dividend.reindex(idx).fillna(0.0) for s in SYMBOLS}
 split={s:raw[s].split.reindex(idx).fillna(0.0) for s in SYMBOLS}
 f=pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU')
 f.columns=['date','fx'];f.date=pd.to_datetime(f.date,utc=True);f.fx=pd.to_numeric(f.fx,errors='coerce')
 fx=f.dropna().set_index('date').fx.sort_index();fx=fx.reindex(fx.index.union(idx)).ffill().reindex(idx)
 if fx.isna().any():raise RuntimeError('missing FX')

 def simulate(weights,begin,end,div_tax_rate=0.0,cost=COST):
  dates=frame.loc[begin:end].index
  if len(dates)<2:return None
  total_eq=pd.Series(0.0,index=dates); total_tax=0.0; total_div_tax=0.0
  for s,w in weights.items():
   if w<=0:continue
   cap=INITIAL*w; p0=float(frame.loc[dates[0],s]); f0=float(fx.loc[dates[0]])
   usd=cap*f0; qty=usd/(p0*(1+cost)); eur_cash=(usd-qty*p0*(1+cost))/f0
   lots=[Lot(qty,qty*p0/f0,dates[0])]
   sleeve=[]
   for ts in dates:
    ratio=float(split[s].loc[ts])
    if ratio and ratio>0 and abs(ratio-1)>1e-12:
     qty*=ratio
     for lot in lots:lot.qty*=ratio
    dv=float(div[s].loc[ts])
    if dv>0 and qty>0:
     gross_usd=qty*dv; gross_eur=gross_usd/float(fx.loc[ts]); dtax=gross_eur*div_tax_rate
     total_div_tax+=dtax
     net_usd=(gross_eur-dtax)*float(fx.loc[ts]); p=float(frame.loc[ts,s])
     add_qty=net_usd/(p*(1+cost))
     if add_qty>0:
      qty+=add_qty; lots.append(Lot(add_qty,add_qty*p/float(fx.loc[ts]),ts))
    sleeve.append(eur_cash+qty*float(frame.loc[ts,s])/float(fx.loc[ts]))
   ts=dates[-1]; p1=float(frame.loc[ts,s]); f1=float(fx.loc[ts]); disposal=qty*p1/f1
   cgt=0.0
   for lot in lots:
    lot_value=lot.qty*p1/f1; gain=lot_value-lot.basis_eur
    if gain>0:cgt+=gain*tx.slovenia_rate_for_dates(lot.acquired,ts)
   final=eur_cash+qty*p1*(1-cost)/f1-cgt
   series=pd.Series(sleeve,index=dates);series.iloc[-1]=final;total_eq+=series;total_tax+=cgt
  years=(dates[-1]-dates[0]).total_seconds()/(365.2425*86400); final=float(total_eq.iloc[-1])
  return {'cagr':(final/INITIAL)**(1/years)-1,'return':final/INITIAL-1,'dd':float((total_eq/total_eq.cummax()-1).min()),'cgt':total_tax,'div_tax':total_div_tax}

 for div_rate,label in ((0.0,'CGT_ONLY_RAW_PRICE'),(.25,'CGT_PLUS_25PCT_DIVIDEND_TAX')):
  print('\n'+label)
  for a,b in DEV:
   c=simulate(CAND,a,b,div_rate);e=simulate(EQ7,a,b,div_rate);s=simulate(SPY,a,b,div_rate);q=simulate(QQQ,a,b,div_rate)
   print(f"{a.year}-{b.year} cand={c['cagr']:.2%} eq7={e['cagr']:.2%} SPY={s['cagr']:.2%} QQQ={q['cagr']:.2%} exEq={c['cagr']-e['cagr']:.2%} exSPY={c['cagr']-s['cagr']:.2%} exQQQ={c['cagr']-q['cagr']:.2%} DD={c['dd']:.2%} CGT={c['cgt']:.0f} DivTax={c['div_tax']:.0f}")
  c=simulate(CAND,*HOLD,div_rate);e=simulate(EQ7,*HOLD,div_rate);s=simulate(SPY,*HOLD,div_rate);q=simulate(QQQ,*HOLD,div_rate)
  print(f"HOLD cand={c['cagr']:.2%} Ret={c['return']:.2%} DD={c['dd']:.2%} eq7={e['cagr']:.2%} SPY={s['cagr']:.2%} QQQ={q['cagr']:.2%} exEq={c['cagr']-e['cagr']:.2%} exSPY={c['cagr']-s['cagr']:.2%} exQQQ={c['cagr']-q['cagr']:.2%} CGT={c['cgt']:.0f} DivTax={c['div_tax']:.0f}")
  print('ROLLING')
  for years in (5,7,10):
   starts=pd.date_range(pd.Timestamp('2006-01-01',tz='UTC'),pd.Timestamp(f'{2024-years}-12-31',tz='UTC'),freq='QS')
   de=[];ds=[]
   for a in starts:
    b=a+pd.DateOffset(years=years);c=simulate(CAND,a,b,div_rate);e=simulate(EQ7,a,b,div_rate);s=simulate(SPY,a,b,div_rate)
    if c and e and s:de.append(c['cagr']-e['cagr']);ds.append(c['cagr']-s['cagr'])
   print(f"{years}y n={len(de)} winEq={np.mean(np.array(de)>0):.1%} minEq={min(de):.2%} p10Eq={np.quantile(de,.1):.2%} winSPY={np.mean(np.array(ds)>0):.1%} minSPY={min(ds):.2%} p10SPY={np.quantile(ds,.1):.2%}")

if __name__=='__main__':run()
