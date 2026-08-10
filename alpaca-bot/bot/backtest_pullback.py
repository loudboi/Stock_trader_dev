"""
bot/backtest_pullback.py
========================
Multi-timeframe backtester for the phased trend-pullback strategy.

Daily bars create decisions. Intraday bars model the following session's limit
attempt and stop path. If the improving limit is not touched, the live runner does
not market-fill until the following daily evaluation, so the intraday backtest
uses the next session's open for that fallback as well.

The combined portfolio plans all decisions sharing a signal timestamp before it
walks any future execution windows. This prevents one symbol's next-session result
from leaking into another symbol's same-close position sizing.
"""

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import groupby

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from bot import indicators as ind
from bot import risk_manager as rm
from bot.strategies.trend_pullback import TrendPullbackStrategy, PullbackParams

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("backtest_pullback")

RESULTS_PNG = "backtest_pullback_results.png"
INITIAL_EQUITY = 100_000.0
SLIPPAGE = 0.0005
ANNUALIZATION = 252
_DAILY_WARMUP_DAYS = 320
_INTRA_BUFFER_DAYS = 7


def fill_price(price: float, direction: str, is_entry: bool) -> float:
    if direction == "long":
        return price * (1 + SLIPPAGE) if is_entry else price * (1 - SLIPPAGE)
    return price * (1 - SLIPPAGE) if is_entry else price * (1 + SLIPPAGE)


def _initial_equity(equity):
    return float(equity.attrs.get("initial_equity", equity.iloc[0]))


def compute_metrics(trades: list, equity: pd.Series) -> dict:
    n = len(trades)
    pnls = np.array([t["pnl"] for t in trades], dtype=float) if n else np.array([])
    wins, losses = pnls[pnls > 0], pnls[pnls < 0]
    gross_win, gross_loss = wins.sum(), abs(losses.sum())
    profit_factor = (gross_win / gross_loss if gross_loss > 0 else
                     (float("inf") if gross_win > 0 else 0.0))
    max_dd = total_return = sharpe = 0.0
    if len(equity):
        running_max = equity.cummax()
        dd = (equity - running_max) / running_max.replace(0, np.nan)
        max_dd = float(dd.min()) if len(dd) else 0.0
        initial = _initial_equity(equity)
        total_return = float(equity.iloc[-1] / initial - 1) if initial else 0.0
    if len(equity) > 2:
        daily = equity.resample("1D").last().dropna()
        initial = _initial_equity(equity)
        # Prepend the true starting NAV when the first observation already contains
        # a return, so the first modeled return is not dropped from Sharpe.
        if len(daily) and initial != daily.iloc[0]:
            lead = daily.index[0] - pd.Timedelta(days=1)
            daily = pd.concat([pd.Series([initial], index=[lead]), daily])
        rets = daily.pct_change().dropna()
        if len(rets) > 1 and rets.std(ddof=1) > 0:
            sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(ANNUALIZATION))
    return {
        "trades": n,
        "win_rate": len(wins) / n if n else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(profit_factor),
        "max_drawdown": max_dd, "sharpe": sharpe, "total_return": total_return,
    }


@dataclass
class PyramidPos:
    qty: float = 0.0
    cost: float = 0.0
    tranches: int = 0
    last_add_price: float = 0.0
    stop_dist: float = 0.05
    entry_time: object = None
    last_price: float = 0.0

    @property
    def avg_entry(self):
        return self.cost / self.qty if self.qty else 0.0


@dataclass
class PyramidBook:
    initial: float
    realized: float = 0.0
    positions: dict = field(default_factory=dict)
    trades: list = field(default_factory=list)

    def equity(self):
        unreal = sum((p.last_price - p.avg_entry) * p.qty for p in self.positions.values())
        return self.initial + self.realized + unreal

    def gross_notional(self):
        return sum(p.qty * max(p.last_price, p.avg_entry) for p in self.positions.values())

    def stop_risk(self):
        return sum(p.qty * p.avg_entry * p.stop_dist for p in self.positions.values())

    def plan_tranche_qty(self, instrument, price, fraction, stop_dist, sizing_equity,
                         capacity=None):
        if price <= 0 or stop_dist <= 0 or sizing_equity <= 0:
            return 0.0
        risk = config.RISK_PER_TRADE * sizing_equity
        desired = rm.round_qty(fraction * risk / (price * stop_dist), instrument.qty_decimals)
        if desired <= 0:
            return 0.0
        gross_used = self.gross_notional() if capacity is None else capacity["gross"]
        risk_used = self.stop_risk() if capacity is None else capacity["risk"]
        gross_room = max(0.0, config.MAX_GROSS_EXPOSURE * sizing_equity - gross_used)
        risk_room = max(0.0, config.MAX_PORTFOLIO_RISK * sizing_equity - risk_used)
        max_by_gross = gross_room / price
        max_by_risk = risk_room / (price * stop_dist)
        qty = rm.round_qty(min(desired, max_by_gross, max_by_risk), instrument.qty_decimals)
        if capacity is not None and qty > 0:
            capacity["gross"] += qty * price
            capacity["risk"] += qty * price * stop_dist
        return max(0.0, qty)

    def add_fixed_tranche(self, name, price, qty, stop_dist, ts):
        if qty <= 0 or price <= 0:
            return False
        ef = fill_price(price, "long", is_entry=True)
        pos = self.positions.get(name) or PyramidPos()
        if pos.tranches == 0:
            pos.entry_time = ts
            pos.stop_dist = stop_dist
        pos.qty += qty
        pos.cost += qty * ef
        pos.tranches += 1
        pos.last_add_price = ef
        pos.last_price = ef
        self.positions[name] = pos
        return True

    def add_tranche(self, name, instrument, price, fraction, stop_dist, ts,
                    sizing_equity=None):
        pos = self.positions.get(name)
        actual_stop = pos.stop_dist if pos else stop_dist
        eq = self.equity() if sizing_equity is None else sizing_equity
        qty = self.plan_tranche_qty(instrument, price, fraction, actual_stop, eq)
        return self.add_fixed_tranche(name, price, qty, actual_stop, ts)

    def close(self, name, exit_level, ts, reason):
        pos = self.positions.pop(name, None)
        if not pos or pos.qty <= 0:
            return
        xf = fill_price(exit_level, "long", is_entry=False)
        pnl = (xf - pos.avg_entry) * pos.qty
        self.realized += pnl
        self.trades.append({
            "instrument": name, "direction": "long", "entry_time": pos.entry_time,
            "entry_price": round(pos.avg_entry, 4), "exit_time": ts,
            "exit_price": round(xf, 4), "qty": round(pos.qty, 6),
            "pnl": round(pnl, 2),
            "return_pct": pnl / (pos.avg_entry * pos.qty) if pos.qty else 0.0,
            "tranches": pos.tranches, "exit_reason": reason,
        })


def exec_window(daily, intra, d, intraday: bool):
    if intraday:
        t0 = daily.index[d]
        if d + 1 < len(daily):
            t1 = daily.index[d + 1]
            return intra[(intra.index > t0) & (intra.index <= t1)]
        return intra[intra.index > t0]
    return daily.iloc[d + 1:d + 2]


def _decision(book, name, strat, daily, ma_f, ma_s, atr_series, d):
    price = float(daily["close"].iloc[d])
    atr_v = atr_series.iloc[d]
    atr_v = float(atr_v) if atr_v == atr_v else 0.0
    stop_dist = strat.stop_distance(atr_v, price)
    pos = book.positions.get(name)
    decision = None
    if pos:
        te = strat.trend_exit(daily, ma_f, d)
        if te:
            decision = ("exit", te[1])
    if decision is None:
        trend = strat.trend_ok(daily, ma_f, ma_s, d)
        if pos and trend and pos.tranches < len(strat.p.tranches):
            add, _ = strat.should_add(daily, ma_f, d, pos.last_add_price)
            if add:
                decision = ("add", strat.p.tranches[pos.tranches])
        elif pos is None and trend:
            ok, _ = strat.entry_signal(daily, ma_f, d)
            if ok:
                decision = ("enter", strat.p.tranches[0])
    return decision, price, stop_dist


def _fallback_bar(daily, d, intraday):
    # Intraday mode uses d+1 for the limit-attempt session. Live falls back at
    # the next daily evaluation, corresponding to the following session open.
    idx = d + (2 if intraday else 1)
    return daily.iloc[idx] if idx < len(daily) else None


def process_day(book, name, inst, strat, daily, ma_f, ma_s, atr_series, d,
                win, exec_is_intraday, decision=None, planned_qty=None,
                sizing_equity=None):
    if decision is None:
        decision, price_d, stop_dist = _decision(
            book, name, strat, daily, ma_f, ma_s, atr_series, d)
    else:
        price_d = float(daily["close"].iloc[d])
        a = atr_series.iloc[d]
        stop_dist = strat.stop_distance(float(a) if a == a else 0.0, price_d)
    ts_d = daily.index[d]
    pos = book.positions.get(name)
    if pos:
        pos.last_price = price_d

    if decision and decision[0] == "exit":
        if len(win):
            book.close(name, float(win["open"].iloc[0]), win.index[0], decision[1])
        else:
            fb = _fallback_bar(daily, d, exec_is_intraday)
            if fb is not None:
                book.close(name, float(fb["open"]), fb.name, decision[1])
        return book.equity()

    pending = decision if decision and decision[0] in ("enter", "add") else None
    if pending and planned_qty is None:
        current = book.positions.get(name)
        actual_stop = current.stop_dist if current else stop_dist
        planned_qty = book.plan_tranche_qty(
            inst, price_d, pending[1], actual_stop,
            book.equity() if sizing_equity is None else sizing_equity)
    limit = price_d * (1 - strat.p.improve_pct)
    filled = closed = False

    for j in range(len(win)):
        o, l = float(win["open"].iloc[j]), float(win["low"].iloc[j])
        wts = win.index[j]
        if name in book.positions:
            p = book.positions[name]
            stop_level = p.avg_entry * (1 - p.stop_dist)
            if l <= stop_level:
                book.close(name, min(o, stop_level), wts, "volatility stop max(5%,2xATR)")
                pending, closed = None, True
                break

        if pending and exec_is_intraday and not filled and planned_qty and l <= limit:
            entry_ref = min(o, limit) if o <= limit else limit
            current = book.positions.get(name)
            actual_stop = current.stop_dist if current else stop_dist
            book.add_fixed_tranche(name, entry_ref, planned_qty, actual_stop, wts)
            filled = True
            # If the same OHLC candle touched both the entry and protective stop,
            # daily/intraday OHLC cannot prove a favorable ordering. Use the
            # conservative fill-then-stop path rather than ignoring the stop.
            p = book.positions[name]
            stop_level = p.avg_entry * (1 - p.stop_dist)
            if l <= stop_level:
                book.close(name, min(o, stop_level), wts,
                           "volatility stop same execution candle")
                closed = True
                break

    if pending and planned_qty and not filled and not closed:
        fb = _fallback_bar(daily, d, exec_is_intraday)
        if fb is not None:
            current = book.positions.get(name)
            actual_stop = current.stop_dist if current else stop_dist
            book.add_fixed_tranche(name, float(fb["open"]), planned_qty, actual_stop, fb.name)

    if name in book.positions:
        if len(win):
            book.positions[name].last_price = float(win["close"].iloc[-1])
        else:
            book.positions[name].last_price = price_d
    return book.equity()


def _prepare(name, daily, params):
    inst = config.research_instrument(name)
    strat = TrendPullbackStrategy(inst, params)
    ma_f, ma_s = strat.moving_averages(daily)
    atrs = ind.atr(daily, params.atr_period)
    return inst, strat, ma_f, ma_s, atrs


def run_single(name, daily, intra, params, exec_is_intraday, begin_ts=None):
    inst, strat, ma_f, ma_s, atrs = _prepare(name, daily, params)
    book = PyramidBook(initial=INITIAL_EQUITY)
    begin = strat.warmup() if begin_ts is None else max(
        strat.warmup(), int(daily.index.searchsorted(begin_ts, side="left")))
    eq_t, eq_v = [], []
    for d in range(begin, len(daily)):
        win = exec_window(daily, intra, d, exec_is_intraday)
        eq_v.append(process_day(book, name, inst, strat, daily, ma_f, ma_s, atrs,
                                d, win, exec_is_intraday))
        eq_t.append(daily.index[d])
    if name in book.positions:
        book.close(name, float(daily["close"].iloc[-1]), daily.index[-1], "end of backtest")
        if eq_v:
            eq_v[-1] = book.equity()
    s = pd.Series(eq_v, index=pd.DatetimeIndex(eq_t)).sort_index()
    s.attrs["initial_equity"] = INITIAL_EQUITY
    return book.trades, s


def run_combined(daily_data, intra_data, params, exec_is_intraday, begin_ts=None):
    prepared, events = {}, []
    for name, dfd in daily_data.items():
        prepared[name] = _prepare(name, dfd, params)
        strat = prepared[name][1]
        begin = strat.warmup() if begin_ts is None else max(
            strat.warmup(), int(dfd.index.searchsorted(begin_ts, side="left")))
        for d in range(begin, len(dfd)):
            events.append((dfd.index[d], name, d))
    events.sort(key=lambda e: (e[0], e[1]))

    book = PyramidBook(initial=INITIAL_EQUITY)
    eq_t, eq_v = [], []
    for ts, grouped in groupby(events, key=lambda e: e[0]):
        group = list(grouped)
        # Mark every existing same-close position before capturing sizing equity.
        for _, name, d in group:
            if name in book.positions:
                book.positions[name].last_price = float(daily_data[name]["close"].iloc[d])
        sizing_equity = book.equity()
        capacity = {"gross": book.gross_notional(), "risk": book.stop_risk()}
        plans = []
        for _, name, d in group:
            dfd = daily_data[name]
            inst, strat, ma_f, ma_s, atrs = prepared[name]
            decision, price, stop_dist = _decision(book, name, strat, dfd, ma_f, ma_s, atrs, d)
            qty = None
            if decision and decision[0] in ("enter", "add"):
                current = book.positions.get(name)
                actual_stop = current.stop_dist if current else stop_dist
                qty = book.plan_tranche_qty(inst, price, decision[1], actual_stop,
                                             sizing_equity, capacity)
            plans.append((name, d, decision, qty))

        # Future execution paths are walked only after all same-close decisions
        # and quantities have been fixed.
        for name, d, decision, qty in plans:
            dfd = daily_data[name]
            inst, strat, ma_f, ma_s, atrs = prepared[name]
            intra = intra_data.get(name) if exec_is_intraday else None
            win = exec_window(dfd, intra, d, exec_is_intraday)
            process_day(book, name, inst, strat, dfd, ma_f, ma_s, atrs, d, win,
                        exec_is_intraday, decision=decision, planned_qty=qty,
                        sizing_equity=sizing_equity)
        eq_t.append(ts)
        eq_v.append(book.equity())

    for name, dfd in daily_data.items():
        if name in book.positions:
            book.close(name, float(dfd["close"].iloc[-1]), dfd.index[-1], "end of backtest")
    if eq_v:
        eq_v[-1] = book.equity()
    s = pd.Series(eq_v, index=pd.DatetimeIndex(eq_t)).sort_index()
    s.attrs["initial_equity"] = INITIAL_EQUITY
    return book.trades, s


def _fmt_pf(pf):
    return "inf" if pf == float("inf") else f"{pf:.2f}"


def print_summary(per_instrument, combined, signal_tf, exec_tf):
    cols = ["Instrument", "Trades", "Win%", "AvgWin", "AvgLoss", "PF",
            "MaxDD%", "Sharpe", "Return%"]
    rows = []
    for name, m in per_instrument.items():
        rows.append([name, m["trades"], f"{m['win_rate']*100:.1f}", f"{m['avg_win']:.0f}",
                     f"{m['avg_loss']:.0f}", _fmt_pf(m["profit_factor"]),
                     f"{m['max_drawdown']*100:.1f}", f"{m['sharpe']:.2f}",
                     f"{m['total_return']*100:.1f}"])
    rows.append(["PORTFOLIO", combined["trades"], f"{combined['win_rate']*100:.1f}",
                 f"{combined['avg_win']:.0f}", f"{combined['avg_loss']:.0f}",
                 _fmt_pf(combined["profit_factor"]), f"{combined['max_drawdown']*100:.1f}",
                 f"{combined['sharpe']:.2f}", f"{combined['total_return']*100:.1f}"])
    widths = [max(len(str(r[i])) for r in ([cols] + rows)) for i in range(len(cols))]
    line = "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cols))
    print("\n" + "=" * len(line))
    print(f"TREND-PULLBACK BACKTEST  signal={signal_tf}  exec={exec_tf}")
    print("=" * len(line)); print(line); print("-" * len(line))
    for r in rows:
        if r[0] == "PORTFOLIO": print("-" * len(line))
        print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))
    print("=" * len(line))


def flag_negative_sharpe(per_instrument, combined):
    flagged = [(n, m) for n, m in per_instrument.items() if m["sharpe"] < 0]
    if flagged or combined["sharpe"] < 0:
        print("\nSHARPE CHECK")
        for n, m in flagged:
            print(f"  [!] {n}: Sharpe {m['sharpe']:.2f}")
        if combined["sharpe"] < 0:
            print(f"  [!] PORTFOLIO: Sharpe {combined['sharpe']:.2f}")


def plot_equity(per_series, combined_series, path, bh_series=None):
    if combined_series is None or not len(combined_series):
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), height_ratios=[1.4, 1])
    cs = combined_series.copy(); cs.index = cs.index.tz_localize(None) if cs.index.tz else cs.index
    ax1.plot(cs.index, cs.values, lw=1.8, label="Combined portfolio")
    if bh_series is not None and len(bh_series):
        bh = bh_series.copy(); bh.index = bh.index.tz_localize(None) if bh.index.tz else bh.index
        ax1.plot(bh.index, bh.values, lw=1.4, ls=":", label="Buy & hold (equal-weight)")
    ax1.axhline(_initial_equity(combined_series), ls="--", lw=1, label="Starting equity")
    rmx = cs.cummax(); ax1.fill_between(cs.index, cs.values, rmx.values, where=cs.values < rmx.values,
                                       alpha=0.25, label="Drawdown")
    ax1.set_title("Trend-Pullback — Combined Portfolio Equity"); ax1.legend(); ax1.grid(alpha=0.3)
    for name, s in per_series.items():
        si = s.copy(); si.index = si.index.tz_localize(None) if si.index.tz else si.index
        ax2.plot(si.index, si.values, lw=1.2, label=name)
    ax2.axhline(INITIAL_EQUITY, ls="--", lw=1); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def fetch_bars(pf, instrument, tf_key, days_back):
    end = datetime.now(timezone.utc)
    return pf.get_historical_bars(instrument, tf_key, end - timedelta(days=days_back), end)


def fetch_all(symbols, exec_tf, start_dt, end_dt, source="alpaca"):
    daily_start = start_dt - timedelta(days=_DAILY_WARMUP_DAYS)
    intra_start = start_dt - timedelta(days=_INTRA_BUFFER_DAYS)
    exec_is_intraday = exec_tf != "none"
    if source == "yahoo":
        if exec_is_intraday:
            log.info("Yahoo source has no intraday history -> next-daily-bar execution.")
            exec_is_intraday = False
        from bot.data import load_yahoo
        daily_data = {}
        for name in symbols:
            d = load_yahoo(name, daily_start, end_dt)
            if d.empty:
                log.warning("No daily data for %s; skipping.", name)
            else:
                daily_data[name] = d
        return daily_data, {}, exec_is_intraday

    from bot.portfolio import Portfolio
    config.validate_config()
    pf = Portfolio(); daily_data, intra_data = {}, {}
    for name in symbols:
        try:
            inst = config.research_instrument(name)
        except ValueError as e:
            log.error("%s", e); continue
        d = pf.get_historical_bars(inst, "1Day", daily_start, end_dt)
        if d.empty:
            log.warning("No daily data for %s; skipping.", name); continue
        daily_data[name] = d
        if exec_is_intraday:
            intra_data[name] = pf.get_historical_bars(inst, exec_tf, intra_start, end_dt)
    return daily_data, intra_data, exec_is_intraday


def buy_hold_equity(daily, begin_ts=None, initial=INITIAL_EQUITY):
    df = daily[daily.index >= begin_ts] if begin_ts is not None else daily
    if df.empty:
        return pd.Series(dtype=float)
    out = (initial * df["close"] / df["close"].iloc[0]).rename("buy_hold")
    out.attrs["initial_equity"] = initial
    return out


def buy_hold_combined(daily_data, begin_ts=None, initial=INITIAL_EQUITY):
    """Equal-weight initial allocation on the first date all assets have data."""
    if not daily_data:
        return pd.Series(dtype=float)
    starts = []
    for d in daily_data.values():
        eligible = d[d.index >= begin_ts] if begin_ts is not None else d
        if eligible.empty:
            return pd.Series(dtype=float)
        starts.append(eligible.index[0])
    common_start = max(starts)
    n = len(daily_data)
    parts = [buy_hold_equity(d, common_start, initial / n) for d in daily_data.values()]
    merged = pd.concat(parts, axis=1).sort_index().ffill().dropna()
    out = merged.sum(axis=1)
    out.attrs["initial_equity"] = initial
    return out


def print_vs_benchmark(per_strat, per_bh, combined_strat, combined_bh):
    cols = ["Instrument", "Strat Ret%", "B&H Ret%", "Strat MaxDD%", "B&H MaxDD%",
            "Strat Shp", "B&H Shp", "Beat B&H?"]
    rows = []
    for name in per_strat:
        s, b = per_strat[name], per_bh[name]
        rows.append([name, f"{s['total_return']*100:.1f}", f"{b['total_return']*100:.1f}",
                     f"{s['max_drawdown']*100:.1f}", f"{b['max_drawdown']*100:.1f}",
                     f"{s['sharpe']:.2f}", f"{b['sharpe']:.2f}",
                     "YES" if s["total_return"] > b["total_return"] else "no"])
    rows.append(["PORTFOLIO", f"{combined_strat['total_return']*100:.1f}",
                 f"{combined_bh['total_return']*100:.1f}",
                 f"{combined_strat['max_drawdown']*100:.1f}", f"{combined_bh['max_drawdown']*100:.1f}",
                 f"{combined_strat['sharpe']:.2f}", f"{combined_bh['sharpe']:.2f}",
                 "YES" if combined_strat["total_return"] > combined_bh["total_return"] else "no"])
    widths = [max(len(str(r[i])) for r in ([cols] + rows)) for i in range(len(cols))]
    print("\nVS BUY-AND-HOLD (shared investable window)")
    print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cols)))
    for r in rows:
        print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))


def run_backtest(daily_data, intra_data, params, exec_is_intraday,
                 signal_tf="1Day", exec_tf="4Hour", begin_ts=None):
    if not daily_data:
        log.error("No data to backtest."); return 1
    per_instrument, per_series = {}, {}
    for name, daily in daily_data.items():
        trades, eq = run_single(name, daily, intra_data.get(name), params,
                                exec_is_intraday, begin_ts)
        per_instrument[name] = compute_metrics(trades, eq); per_series[name] = eq
    ct, ce = run_combined(daily_data, intra_data, params, exec_is_intraday, begin_ts)
    combined = compute_metrics(ct, ce)
    per_bh = {name: compute_metrics([], buy_hold_equity(daily, begin_ts))
              for name, daily in daily_data.items()}
    bh_combined_series = buy_hold_combined(daily_data, begin_ts)
    combined_bh = compute_metrics([], bh_combined_series)
    print_summary(per_instrument, combined, signal_tf, exec_tf)
    print_vs_benchmark(per_instrument, per_bh, combined, combined_bh)
    flag_negative_sharpe(per_instrument, combined)
    plot_equity(per_series, ce, RESULTS_PNG, bh_combined_series)
    return 0


def _parse_date(s):
    return pd.Timestamp(datetime.strptime(s, "%Y-%m-%d"), tz="UTC")


def main():
    ap = argparse.ArgumentParser(description="Multi-timeframe trend-pullback backtest.")
    ap.add_argument("--symbols", nargs="+", default=config.PULLBACK_SYMBOLS)
    ap.add_argument("--exec-timeframe", choices=["1Hour", "4Hour", "none"], default="4Hour")
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca")
    args = ap.parse_args()
    if args.months <= 0:
        ap.error("--months must be positive")
    end_dt = _parse_date(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc))
    start_dt = _parse_date(args.start) if args.start else end_dt - pd.Timedelta(days=args.months * 31)
    if start_dt >= end_dt:
        ap.error("start date must be before end date")
    params = PullbackParams(use_ema=args.ema)
    daily_data, intra_data, exec_is_intraday = fetch_all(
        args.symbols, args.exec_timeframe, start_dt, end_dt, source=args.data_source)
    return run_backtest(daily_data, intra_data, params, exec_is_intraday,
                        "1Day", args.exec_timeframe, begin_ts=start_dt)


if __name__ == "__main__":
    raise SystemExit(main())
