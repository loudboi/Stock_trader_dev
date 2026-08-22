from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from bot import taxes as tx

START = "2004-01-01"
END = "2026-08-21"
INITIAL_EUR = 100_000.0
DEV = [
    (pd.Timestamp("2006-01-01", tz="UTC"), pd.Timestamp("2009-12-31", tz="UTC")),
    (pd.Timestamp("2010-01-01", tz="UTC"), pd.Timestamp("2014-12-31", tz="UTC")),
    (pd.Timestamp("2015-01-01", tz="UTC"), pd.Timestamp("2019-12-31", tz="UTC")),
    (pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2024-12-31", tz="UTC")),
]
HOLD = (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-08-20", tz="UTC"))
DIV_TAX = 0.25
MAINT = 0.30


@dataclass
class Lot:
    qty: float
    basis_eur: float
    acquired: pd.Timestamp


def norm(d):
    x = d.copy()
    i = pd.DatetimeIndex(x.index)
    i = i.tz_localize("UTC") if i.tz is None else i.tz_convert("UTC")
    x.index = i.normalize()
    return x[~x.index.duplicated(keep="last")].sort_index()


def load(s):
    d = norm(yf.Ticker(s).history(start=START, end=END, auto_adjust=False, actions=True, repair=False))
    if d.empty:
        raise RuntimeError(f"missing {s}")
    return pd.DataFrame({
        "close": pd.to_numeric(d["Close"], errors="coerce"),
        "adj": pd.to_numeric(d["Adj Close"], errors="coerce"),
        "div": pd.to_numeric(d.get("Dividends", 0.0), errors="coerce").fillna(0.0),
        "split": pd.to_numeric(d.get("Stock Splits", 0.0), errors="coerce").fillna(0.0),
    }).dropna(subset=["close", "adj"])


def fred(series, idx, pct=False):
    f = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}")
    f.columns = ["date", "v"]
    f.date = pd.to_datetime(f.date, utc=True).dt.normalize()
    f.v = pd.to_numeric(f.v, errors="coerce")
    x = f.dropna().set_index("date").v.sort_index()
    if pct:
        x = x / 100.0
    return x.reindex(x.index.union(idx)).ffill().reindex(idx)


def run():
    raw = {s: load(s) for s in ("QQQ", "SPY")}
    idx = raw["QQQ"].index.union(raw["SPY"].index).sort_values()
    idx = idx[(idx >= max(x.index[0] for x in raw.values())) & (idx <= min(x.index[-1] for x in raw.values()))]
    close = pd.DataFrame({s: raw[s].close.reindex(idx).ffill() for s in raw}).dropna()
    adj = pd.DataFrame({s: raw[s].adj.reindex(close.index).ffill() for s in raw}).dropna()
    idx = close.index.intersection(adj.index)
    close, adj = close.loc[idx], adj.loc[idx]
    div = {s: raw[s].div.reindex(idx).fillna(0.0) for s in raw}
    split = {s: raw[s].split.reindex(idx).fillna(0.0) for s in raw}
    fx = fred("DEXUSEU", idx)
    fed = fred("DFF", idx, pct=True).shift(1).fillna(0.0).clip(lower=0.0)
    if fx.isna().any() or (fx <= 0).any():
        raise RuntimeError("bad FX")

    def events(lookback=20, target_vol=.25, cap=1.75, offset=0):
        rv = adj.QQQ.pct_change(fill_method=None).rolling(lookback, min_periods=lookback).std().shift(1) * np.sqrt(252)
        out = {}
        month_groups = pd.Series(np.arange(len(idx)), index=idx).groupby([idx.year, idx.month])
        for _, rows in month_groups:
            positions = list(rows.values)
            if len(positions) <= offset:
                continue
            i = positions[offset]
            if i <= 0:
                continue
            realized = float(rv.iloc[i - 1])
            if np.isfinite(realized) and realized > 0:
                out[idx[i]] = float(np.clip(target_vol / realized, .50, cap))
        return out

    def active(begin, end, *, lookback=20, target_vol=.25, cap=1.75, offset=0,
               cost=.002, spread=.015, band=.20):
        dates = idx[(idx >= begin) & (idx <= end)]
        if len(dates) < 2:
            return None
        ev = events(lookback, target_vol, cap, offset)
        cash = INITIAL_EUR * float(fx.loc[dates[0]])
        debt = 0.0
        qty = 0.0
        lots = deque()
        cgt = dtax = interest = traded = 0.0
        path = []
        target = 1.0
        for d in sorted(ev):
            if d <= dates[0]: target = ev[d]
            else: break

        def p(ts): return float(close.loc[ts, "QQQ"])
        def nav(ts): return cash + qty * p(ts) - debt
        def repay():
            nonlocal cash, debt
            x = min(max(cash, 0.0), max(debt, 0.0))
            cash -= x; debt -= x
        def do_split(ts):
            nonlocal qty
            r = float(split["QQQ"].loc[ts])
            if r > 0 and abs(r - 1) > 1e-12:
                qty *= r
                for lot in lots:
                    lot.qty *= r; lot.basis_eur /= r
        def dividend(ts):
            nonlocal cash, dtax
            d = float(div["QQQ"].loc[ts])
            if d > 0 and qty > 0:
                gross = qty * d
                tax_eur = gross / float(fx.loc[ts]) * DIV_TAX
                cash += gross - tax_eur * float(fx.loc[ts])
                dtax += tax_eur; repay()
        def sell(q, ts):
            nonlocal cash, qty, cgt, traded
            q = min(max(q, 0.0), qty)
            if q <= 1e-12: return
            px = p(ts); remain = q; tax_eur = 0.0
            while remain > 1e-10 and lots:
                lot = lots[0]; used = min(remain, lot.qty)
                gain = used * px / float(fx.loc[ts]) - used * lot.basis_eur
                if gain > 0: tax_eur += gain * tx.slovenia_rate_for_dates(lot.acquired, ts)
                lot.qty -= used; remain -= used
                if lot.qty <= 1e-10: lots.popleft()
            notional = q * px
            cash += notional * (1-cost) - tax_eur * float(fx.loc[ts])
            qty -= q; cgt += tax_eur; traded += notional; repay()
        def buy(q, ts):
            nonlocal cash, debt, qty, traded
            q = max(q, 0.0)
            if q <= 1e-12: return
            px = p(ts); notional = q * px; total = notional * (1+cost)
            if cash >= total: cash -= total
            else: debt += total-cash; cash = 0.0
            qty += q; lots.append(Lot(q, px/float(fx.loc[ts]), ts)); traded += notional
        def rebalance(ts, desired, force=False):
            n = nav(ts)
            if n <= 0: return
            mv = qty * p(ts); current = mv/n
            if not force and abs(desired-current) < band: return
            desired_mv = desired*n
            if mv > desired_mv: sell((mv-desired_mv)/p(ts), ts)
            else: buy((desired_mv-mv)/p(ts), ts)

        forced = 0
        for j, ts in enumerate(dates):
            if j and debt > 0:
                x = debt * (float(fed.loc[ts]) + spread) / 360.0
                debt += x; interest += x
            do_split(ts); dividend(ts)
            if j == 0: rebalance(ts, target, True)
            elif ts in ev: target = ev[ts]; rebalance(ts, target)
            if qty > 0 and debt > 0:
                mv = qty*p(ts); n = nav(ts)
                if n/mv < MAINT:
                    sell(qty, ts); forced += 1; target = .50
            n_eur = nav(ts)/float(fx.loc[ts]); path.append(n_eur)
            if n_eur <= 0: return None
        sell(qty, dates[-1]); repay()
        final = (cash-debt)/float(fx.loc[dates[-1]]); path[-1] = final
        ser = pd.Series(path, index=dates)
        years = (dates[-1]-dates[0]).total_seconds()/(365.2425*86400)
        return dict(cagr=(final/INITIAL_EUR)**(1/years)-1, ret=final/INITIAL_EUR-1,
                    dd=float((ser/ser.cummax()-1).min()), cgt=cgt, dtax=dtax,
                    interest=interest, turn=traded/(INITIAL_EUR*float(fx.loc[dates[0]])), forced=forced)

    def hold(symbol, begin, end, *, cost=0.0, reinvest_cost=0.0):
        dates = idx[(idx >= begin) & (idx <= end)]
        fx0 = float(fx.loc[dates[0]]); cash = INITIAL_EUR*fx0
        p0 = float(close.loc[dates[0], symbol]); qty = cash/(p0*(1+cost)); cash -= qty*p0*(1+cost)
        lots = deque([Lot(qty, p0/fx0, dates[0])]); dtax = 0.0; path=[]
        for ts in dates:
            r=float(split[symbol].loc[ts])
            if r>0 and abs(r-1)>1e-12:
                qty*=r
                for lot in lots: lot.qty*=r; lot.basis_eur/=r
            d=float(div[symbol].loc[ts])
            if d>0 and qty>0:
                gross=qty*d; tax_eur=gross/float(fx.loc[ts])*DIV_TAX; net=gross-tax_eur*float(fx.loc[ts]);dtax+=tax_eur
                px=float(close.loc[ts,symbol]); add=net/(px*(1+reinvest_cost)); qty+=add; lots.append(Lot(add,px/float(fx.loc[ts]),ts))
            path.append((cash+qty*float(close.loc[ts,symbol]))/float(fx.loc[ts]))
        ts=dates[-1];px=float(close.loc[ts,symbol]);cgt=0.0
        for lot in lots:
            gain=lot.qty*px/float(fx.loc[ts])-lot.qty*lot.basis_eur
            if gain>0:cgt+=gain*tx.slovenia_rate_for_dates(lot.acquired,ts)
        final=(cash+qty*px*(1-cost))/float(fx.loc[ts])-cgt;path[-1]=final
        ser=pd.Series(path,index=dates);years=(dates[-1]-dates[0]).total_seconds()/(365.2425*86400)
        return dict(cagr=(final/INITIAL_EUR)**(1/years)-1,ret=final/INITIAL_EUR-1,dd=float((ser/ser.cummax()-1).min()),cgt=cgt,dtax=dtax)

    def initial_levered_hold(begin,end,lev=1.75,cost=.002,spread=.015):
        dates=idx[(idx>=begin)&(idx<=end)];fx0=float(fx.loc[dates[0]]);cash=INITIAL_EUR*fx0;debt=0.0
        p0=float(close.loc[dates[0],"QQQ"]);gross=cash*lev;qty=gross/(p0*(1+cost));total=qty*p0*(1+cost);debt=max(0,total-cash);cash=max(0,cash-total)
        lots=deque([Lot(qty,p0/fx0,dates[0])]);cgt=dtax=interest=0.0;path=[];forced=False
        for j,ts in enumerate(dates):
            if j and debt>0:
                x=debt*(float(fed.loc[ts])+spread)/360;debt+=x;interest+=x
            r=float(split["QQQ"].loc[ts])
            if r>0 and abs(r-1)>1e-12:
                qty*=r
                for lot in lots:lot.qty*=r;lot.basis_eur/=r
            d=float(div["QQQ"].loc[ts])
            if d>0 and qty>0:
                grossd=qty*d;te=grossd/float(fx.loc[ts])*DIV_TAX;net=grossd-te*float(fx.loc[ts]);dtax+=te
                pay=min(net,debt);debt-=pay;cash+=net-pay
            px=float(close.loc[ts,"QQQ"]);nav=cash+qty*px-debt;mv=qty*px
            if not forced and debt>0 and nav/mv<MAINT:
                tax_eur=0.0
                for lot in lots:
                    gain=lot.qty*px/float(fx.loc[ts])-lot.qty*lot.basis_eur
                    if gain>0:tax_eur+=gain*tx.slovenia_rate_for_dates(lot.acquired,ts)
                cash += qty*px*(1-cost)-tax_eur*float(fx.loc[ts])-debt;cgt+=tax_eur;qty=0;debt=0;lots.clear();forced=True
            path.append((cash+qty*px-debt)/float(fx.loc[ts]))
        ts=dates[-1];px=float(close.loc[ts,"QQQ"])
        if qty>0:
            tax_eur=0.0
            for lot in lots:
                gain=lot.qty*px/float(fx.loc[ts])-lot.qty*lot.basis_eur
                if gain>0:tax_eur+=gain*tx.slovenia_rate_for_dates(lot.acquired,ts)
            cash+=qty*px*(1-cost)-tax_eur*float(fx.loc[ts])-debt;cgt+=tax_eur;qty=0;debt=0
        final=cash/float(fx.loc[ts]);path[-1]=final;ser=pd.Series(path,index=dates);years=(dates[-1]-dates[0]).total_seconds()/(365.2425*86400)
        return dict(cagr=(final/INITIAL_EUR)**(1/years)-1,dd=float((ser/ser.cummax()-1).min()),forced=forced,cgt=cgt,dtax=dtax,interest=interest)

    print("HARDENED PASSIVE BENCHMARK — active pays 20bp, passive pays 0bp")
    for cap in (1.50,1.75):
        print(f"\nCAP={cap}")
        es=[];eq=[]
        for a,b in DEV:
            m=active(a,b,cap=cap);s=hold("SPY",a,b);q=hold("QQQ",a,b);es.append(m['cagr']-s['cagr']);eq.append(m['cagr']-q['cagr'])
            print(f"{a.year}-{b.year} strat={m['cagr']:.2%} SPY0={s['cagr']:.2%} QQQ0={q['cagr']:.2%} exSPY={es[-1]:.2%} exQQQ={eq[-1]:.2%}")
        h=active(*HOLD,cap=cap);hs=hold("SPY",*HOLD);hq=hold("QQQ",*HOLD)
        print(f"SUMMARY devWinSPY={sum(x>0 for x in es)}/4 minSPY={min(es):.2%} devWinQQQ={sum(x>0 for x in eq)}/4 minQQQ={min(eq):.2%}")
        print(f"HOLD strat={h['cagr']:.2%} SPY0={hs['cagr']:.2%} QQQ0={hq['cagr']:.2%} exSPY={h['cagr']-hs['cagr']:.2%} exQQQ={h['cagr']-hq['cagr']:.2%}")

    print("\nTIMING VS INITIAL 1.75x LEVERED QQQ BUY/HOLD")
    for a,b in DEV+[HOLD]:
        m=active(a,b,cap=1.75);l=initial_levered_hold(a,b,1.75)
        label="HOLD" if (a,b)==HOLD else f"{a.year}-{b.year}"
        print(f"{label} timed={m['cagr']:.2%} staticLev={l['cagr']:.2%} excess={m['cagr']-l['cagr']:.2%} staticDD={l['dd']:.2%} forced={l['forced']}")

    print("\nFROZEN RULE PERTURBATIONS — diagnostic only, never used to select")
    perturb=[
        (15,.25,1.75,0,.20),(20,.25,1.75,0,.20),(25,.25,1.75,0,.20),(30,.25,1.75,0,.20),
        (20,.225,1.75,0,.20),(20,.275,1.75,0,.20),
        (20,.25,1.75,1,.20),(20,.25,1.75,2,.20),(20,.25,1.75,4,.20),
        (20,.25,1.75,0,.15),(20,.25,1.75,0,.25),
    ]
    for spec in perturb:
        lb,tv,cap,off,band=spec;es=[];eq=[]
        for a,b in DEV:
            m=active(a,b,lookback=lb,target_vol=tv,cap=cap,offset=off,band=band);s=hold("SPY",a,b);q=hold("QQQ",a,b);es.append(m['cagr']-s['cagr']);eq.append(m['cagr']-q['cagr'])
        h=active(*HOLD,lookback=lb,target_vol=tv,cap=cap,offset=off,band=band);hq=hold("QQQ",*HOLD)
        print(f"spec={spec} winSPY={sum(x>0 for x in es)}/4 minSPY={min(es):.2%} winQQQ={sum(x>0 for x in eq)}/4 minQQQ={min(eq):.2%} hold={h['cagr']:.2%} holdExQQQ={h['cagr']-hq['cagr']:.2%}")

if __name__ == "__main__": run()
