"""
bot/backtest_pullback.py
========================
Multi-timeframe backtester for the phased-entry trend-pullback strategy.

Two timeframes, by design:
  - SIGNAL timeframe (daily): decides whether a trade is valid. The trend gate,
    pullback/breakout entries, tranche adds, and the MA/structural exit all run on
    daily bars. This is the boss; intraday cannot overrule it.
  - EXECUTION timeframe (1h or 4h): only improves the *fill price*. When the daily
    chart arms an entry/add, the executor places a limit ~`improve_pct` below the
    daily close and tries to catch an intraday dip during the next session. If no
    dip comes, it still fills at the session close, so intraday never blocks a valid
    daily setup. The volatility stop is also monitored intraday.

Volatility stop: stop_distance = max(5%, 2 * ATR(14) / price), computed on daily.
Each tranche is sized so a fully-built position stopped at that distance loses ~1%
of equity, split 30/30/40. Slippage 0.05%, commission $0.

Run from the project root:

    python -m bot.backtest_pullback                              # daily signal, 4h exec
    python -m bot.backtest_pullback --exec-timeframe 1Hour
    python -m bot.backtest_pullback --exec-timeframe none        # fill at next daily bar
    python -m bot.backtest_pullback --symbols SPY QQQ GLD --months 9 --ema

The 200-period daily MA needs ~200 bars of warmup, so daily history is fetched well
ahead of the test window; intraday is fetched only for the test window.
"""

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

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

# Modelling assumptions (kept identical to live where it matters).
INITIAL_EQUITY = 100_000.0
SLIPPAGE = 0.0005          # 0.05% applied adversely to every fill
ANNUALIZATION = 252        # daily-return Sharpe annualization

_DAILY_WARMUP_DAYS = 320   # ~200 trading days for the 200-day MA, plus slack
_INTRA_BUFFER_DAYS = 7


def fill_price(price: float, direction: str, is_entry: bool) -> float:
    """Apply 0.05% slippage in the adverse direction (Strategy 4 is long-only)."""
    if direction == "long":
        return price * (1 + SLIPPAGE) if is_entry else price * (1 - SLIPPAGE)
    return price * (1 - SLIPPAGE) if is_entry else price * (1 + SLIPPAGE)


def compute_metrics(trades: list, equity: pd.Series) -> dict:
    """Trade + equity-curve statistics for the summary table."""
    n = len(trades)
    pnls = np.array([t["pnl"] for t in trades], dtype=float) if n else np.array([])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    win_rate = len(wins) / n if n else 0.0
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())
    if gross_loss == 0:
        profit_factor = float("inf") if gross_win > 0 else 0.0
    else:
        profit_factor = gross_win / gross_loss

    if len(equity) > 1:
        running_max = equity.cummax()
        dd = (equity - running_max) / running_max
        max_dd = float(dd.min())
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    else:
        max_dd, total_return = 0.0, 0.0

    # Sharpe from daily-resampled equity (risk-free = 0).
    sharpe = 0.0
    if len(equity) > 2:
        daily = equity.resample("1D").last().dropna()
        rets = daily.pct_change().dropna()
        if len(rets) > 1 and rets.std(ddof=1) > 0:
            sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(ANNUALIZATION))

    return {
        "trades": n, "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
        "profit_factor": profit_factor, "max_drawdown": max_dd, "sharpe": sharpe,
        "total_return": total_return,
    }


# --------------------------------------------------------------------------- #
# Pyramiding position + book
# --------------------------------------------------------------------------- #
@dataclass
class PyramidPos:
    qty: float = 0.0
    cost: float = 0.0
    tranches: int = 0
    last_add_price: float = 0.0
    stop_dist: float = 0.05          # fixed at the first tranche
    entry_time: object = None
    last_price: float = 0.0

    @property
    def avg_entry(self) -> float:
        return self.cost / self.qty if self.qty else 0.0


@dataclass
class PyramidBook:
    initial: float
    realized: float = 0.0
    positions: dict = field(default_factory=dict)
    trades: list = field(default_factory=list)

    def equity(self) -> float:
        unreal = sum((p.last_price - p.avg_entry) * p.qty for p in self.positions.values())
        return self.initial + self.realized + unreal

    def add_tranche(self, name, instrument, price, fraction, stop_dist, ts) -> bool:
        if price <= 0 or stop_dist <= 0:
            return False
        risk = config.RISK_PER_TRADE * self.equity()
        full_qty = risk / (price * stop_dist)        # full size risks ~1% at the stop
        tq = rm.round_qty(fraction * full_qty, instrument.qty_decimals)
        if tq <= 0:
            return False
        ef = fill_price(price, "long", is_entry=True)
        pos = self.positions.get(name) or PyramidPos()
        if pos.tranches == 0:
            pos.entry_time = ts
            pos.stop_dist = stop_dist                 # set once, when initiating
        pos.qty += tq
        pos.cost += tq * ef
        pos.tranches += 1
        pos.last_add_price = price
        pos.last_price = price
        self.positions[name] = pos
        return True

    def close(self, name, exit_level, ts, reason):
        pos = self.positions.pop(name, None)
        if not pos or pos.qty <= 0:
            return
        xf = fill_price(exit_level, "long", is_entry=False)
        pnl = (xf - pos.avg_entry) * pos.qty
        self.realized += pnl
        ret = pnl / (pos.avg_entry * pos.qty) if pos.qty else 0.0
        self.trades.append({
            "instrument": name, "direction": "long",
            "entry_time": pos.entry_time, "entry_price": round(pos.avg_entry, 4),
            "exit_time": ts, "exit_price": round(xf, 4),
            "qty": round(pos.qty, 6), "pnl": round(pnl, 2),
            "return_pct": ret, "tranches": pos.tranches, "exit_reason": reason,
        })


# --------------------------------------------------------------------------- #
# Execution window: the candles available to fill a daily decision
# --------------------------------------------------------------------------- #
def exec_window(daily, intra, d, intraday: bool):
    if intraday:
        t0 = daily.index[d]
        if d + 1 < len(daily):
            t1 = daily.index[d + 1]
            return intra[(intra.index > t0) & (intra.index <= t1)]
        return intra[intra.index > t0]
    # single-timeframe: the next daily bar is the only execution candle
    return daily.iloc[d + 1: d + 2]


# --------------------------------------------------------------------------- #
# Process one signal (daily) bar against the shared book
# --------------------------------------------------------------------------- #
def process_day(book, name, inst, strat, daily, ma_f, ma_s, atr_series, d,
                win, exec_is_intraday):
    price_d = float(daily["close"].iloc[d])
    ts_d = daily.index[d]
    a = atr_series.iloc[d]
    atr_d = float(a) if a == a else 0.0          # NaN-safe
    stop_dist = strat.stop_distance(atr_d, price_d)

    pos = book.positions.get(name)
    if pos:
        pos.last_price = price_d

    # ---- decide on the daily close ----
    decision = None  # ("exit", reason) | ("add", frac) | ("enter", frac)
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

    # ---- daily exit executes at next-session open ----
    if decision and decision[0] == "exit":
        if len(win):
            book.close(name, float(win["open"].iloc[0]), win.index[0], decision[1])
        else:
            book.close(name, price_d, ts_d, decision[1])
        return book.equity()

    pending = decision if (decision and decision[0] in ("enter", "add")) else None
    limit = price_d * (1 - strat.p.improve_pct)
    filled = closed = False

    # ---- walk the execution candles: monitor stop, try the improving limit ----
    for j in range(len(win)):
        o = float(win["open"].iloc[j])
        l = float(win["low"].iloc[j])
        wts = win.index[j]

        if name in book.positions:
            p = book.positions[name]
            stop_level = p.avg_entry * (1 - p.stop_dist)
            if l <= stop_level:
                book.close(name, min(o, stop_level), wts,
                           "volatility stop max(5%,2xATR)")
                closed = True
                pending = None
                break

        if pending and exec_is_intraday and not filled and l <= limit:
            book.add_tranche(name, inst, limit, pending[1], stop_dist, wts)  # got the dip
            filled = True

    # ---- fallback fill: take the trade anyway (don't override the daily trend) ----
    if pending and not filled and not closed:
        if exec_is_intraday:
            ref = float(win["close"].iloc[-1]) if len(win) else price_d
            tsf = win.index[-1] if len(win) else ts_d
        else:
            ref = float(win["open"].iloc[0]) if len(win) else price_d  # next daily open
            tsf = win.index[0] if len(win) else ts_d
        book.add_tranche(name, inst, ref, pending[1], stop_dist, tsf)

    # ---- mark to market with the last execution price ----
    if name in book.positions:
        last_c = float(win["close"].iloc[-1]) if len(win) else price_d
        book.positions[name].last_price = last_c
    return book.equity()


# --------------------------------------------------------------------------- #
# Passes
# --------------------------------------------------------------------------- #
def run_single(name, daily, intra, params, exec_is_intraday, begin_ts=None) -> tuple:
    inst = config.resolve_instrument(name)
    strat = TrendPullbackStrategy(inst, params)
    ma_f, ma_s = strat.moving_averages(daily)
    atr_series = ind.atr(daily, params.atr_period)
    book = PyramidBook(initial=INITIAL_EQUITY)
    warmup = strat.warmup()
    begin = warmup if begin_ts is None else max(
        warmup, int(daily.index.searchsorted(begin_ts, side="left")))

    eq_t, eq_v = [], []
    for d in range(begin, len(daily)):
        win = exec_window(daily, intra, d, exec_is_intraday)
        eq = process_day(book, name, inst, strat, daily, ma_f, ma_s, atr_series,
                         d, win, exec_is_intraday)
        eq_t.append(daily.index[d])
        eq_v.append(eq)

    if name in book.positions:
        book.close(name, float(daily["close"].iloc[-1]), daily.index[-1], "end of backtest")
        if eq_v:
            eq_v[-1] = book.equity()

    return book.trades, pd.Series(eq_v, index=pd.DatetimeIndex(eq_t)).sort_index()


def run_combined(daily_data, intra_data, params, exec_is_intraday, begin_ts=None) -> tuple:
    strategies, mas, atrs = {}, {}, {}
    for name, dfd in daily_data.items():
        s = TrendPullbackStrategy(config.resolve_instrument(name), params)
        strategies[name] = s
        mas[name] = s.moving_averages(dfd)
        atrs[name] = ind.atr(dfd, params.atr_period)

    events = []
    for name, dfd in daily_data.items():
        w = strategies[name].warmup()
        begin = w if begin_ts is None else max(
            w, int(dfd.index.searchsorted(begin_ts, side="left")))
        for d in range(begin, len(dfd)):
            events.append((dfd.index[d], name, d))
    events.sort(key=lambda e: e[0])

    book = PyramidBook(initial=INITIAL_EQUITY)
    eq_t, eq_v = [], []
    for ts, name, d in events:
        dfd = daily_data[name]
        intra = intra_data.get(name) if exec_is_intraday else None
        ma_f, ma_s = mas[name]
        win = exec_window(dfd, intra, d, exec_is_intraday)
        eq = process_day(book, name, config.resolve_instrument(name), strategies[name],
                         dfd, ma_f, ma_s, atrs[name], d, win, exec_is_intraday)
        eq_t.append(ts)
        eq_v.append(eq)

    for name, dfd in daily_data.items():
        if name in book.positions:
            book.close(name, float(dfd["close"].iloc[-1]), dfd.index[-1], "end of backtest")
    if eq_v:
        eq_v[-1] = book.equity()

    s = pd.Series(eq_v, index=pd.DatetimeIndex(eq_t)).sort_index()
    return book.trades, s[~s.index.duplicated(keep="last")]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt_pf(pf):
    return "inf" if pf == float("inf") else f"{pf:.2f}"


def print_summary(per_instrument, combined, signal_tf, exec_tf):
    cols = ["Instrument", "Trades", "Win%", "AvgWin", "AvgLoss", "PF",
            "MaxDD%", "Sharpe", "Return%"]
    rows = []
    for name, m in per_instrument.items():
        rows.append([name, m["trades"], f"{m['win_rate']*100:.1f}",
                     f"{m['avg_win']:.0f}", f"{m['avg_loss']:.0f}",
                     _fmt_pf(m["profit_factor"]), f"{m['max_drawdown']*100:.1f}",
                     f"{m['sharpe']:.2f}", f"{m['total_return']*100:.1f}"])
    rows.append(["PORTFOLIO", combined["trades"], f"{combined['win_rate']*100:.1f}",
                 f"{combined['avg_win']:.0f}", f"{combined['avg_loss']:.0f}",
                 _fmt_pf(combined["profit_factor"]), f"{combined['max_drawdown']*100:.1f}",
                 f"{combined['sharpe']:.2f}", f"{combined['total_return']*100:.1f}"])

    widths = [max(len(str(r[i])) for r in ([cols] + rows)) for i in range(len(cols))]
    line = "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cols))
    print("\n" + "=" * len(line))
    print(f"TREND-PULLBACK BACKTEST  signal={signal_tf}  exec={exec_tf}  "
          "(30/30/40, stop=max(5%,2xATR), 0.05% slippage)")
    print("=" * len(line))
    print(line)
    print("-" * len(line))
    for r in rows:
        if r[0] == "PORTFOLIO":
            print("-" * len(line))
        print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))
    print("=" * len(line))


def flag_negative_sharpe(per_instrument, combined):
    print("\nSHARPE CHECK")
    print("-" * 60)
    flagged = [(n, m) for n, m in per_instrument.items() if m["sharpe"] < 0]
    if not flagged and combined["sharpe"] >= 0:
        print("No negative Sharpe ratios. The phased trend method cleared the bar.")
    for n, m in flagged:
        print(f"  [!] {n}: Sharpe {m['sharpe']:.2f} — consider tuning the MA pair, "
              "pullback band/volume filter, add_step, atr_mult, or min_stop.")
    if combined["sharpe"] < 0:
        print(f"  [!] PORTFOLIO: Sharpe {combined['sharpe']:.2f} — net-unprofitable "
              "over this window across the chosen instruments.")


def plot_equity(per_series, combined_series, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), height_ratios=[1.4, 1])
    cs = combined_series.copy()
    cs.index = cs.index.tz_localize(None) if cs.index.tz else cs.index
    ax1.plot(cs.index, cs.values, color="#2e7d32", lw=1.8, label="Combined portfolio")
    ax1.axhline(INITIAL_EQUITY, color="#888", ls="--", lw=1, label="Starting equity")
    rmx = cs.cummax()
    ax1.fill_between(cs.index, cs.values, rmx.values, where=(cs.values < rmx.values),
                     color="#e06666", alpha=0.25, label="Drawdown")
    ax1.set_title("Trend-Pullback — Combined Portfolio Equity (daily signal / intraday fills)")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)
    for name, s in per_series.items():
        si = s.copy()
        si.index = si.index.tz_localize(None) if si.index.tz else si.index
        ax2.plot(si.index, si.values, lw=1.2, label=name)
    ax2.axhline(INITIAL_EQUITY, color="#888", ls="--", lw=1)
    ax2.set_title("Per-Instrument Standalone Equity Curves")
    ax2.set_ylabel("Equity ($)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left", fontsize=9, ncol=3)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log.info("Saved equity curve chart -> %s", path)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def fetch_bars(pf, instrument, tf_key, days_back):
    """Fetch `days_back` of history at tf_key via the Portfolio (alpaca-py)."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    return pf.get_historical_bars(instrument, tf_key, start, end)


def fetch_all(symbols, exec_tf, start_dt, end_dt):
    """Fetch daily history with ~320d of warmup lead before start_dt (so the
    200-day MA is valid at the start), and intraday only over the test window."""
    from bot.portfolio import Portfolio
    config.validate_config()
    pf = Portfolio()
    daily_start = start_dt - timedelta(days=_DAILY_WARMUP_DAYS)
    intra_start = start_dt - timedelta(days=_INTRA_BUFFER_DAYS)
    exec_is_intraday = exec_tf != "none"

    daily_data, intra_data = {}, {}
    for name in symbols:
        inst = config.resolve_instrument(name)
        log.info("Fetching %s daily...", name)
        d = pf.get_historical_bars(inst, "1Day", daily_start, end_dt)
        if d.empty:
            log.warning("No daily data for %s; skipping.", name)
            continue
        daily_data[name] = d
        log.info("  daily bars: %d (%s -> %s)", len(d), d.index[0].date(), d.index[-1].date())
        if exec_is_intraday:
            log.info("Fetching %s %s (execution)...", name, exec_tf)
            it = pf.get_historical_bars(inst, exec_tf, intra_start, end_dt)
            intra_data[name] = it
            log.info("  intraday bars: %d", len(it))
    return daily_data, intra_data, exec_is_intraday


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_backtest(daily_data, intra_data, params, exec_is_intraday,
                 signal_tf="1Day", exec_tf="4Hour", begin_ts=None):
    if not daily_data:
        log.error("No data to backtest.")
        return 1
    per_instrument, per_series = {}, {}
    for name, daily in daily_data.items():
        intra = intra_data.get(name)
        trades, eq = run_single(name, daily, intra, params, exec_is_intraday, begin_ts)
        per_instrument[name] = compute_metrics(trades, eq)
        per_series[name] = eq
        log.info("%s: %d trades, return %.1f%%, Sharpe %.2f", name,
                 per_instrument[name]["trades"],
                 per_instrument[name]["total_return"] * 100,
                 per_instrument[name]["sharpe"])

    log.info("Running combined portfolio pass...")
    ct, ce = run_combined(daily_data, intra_data, params, exec_is_intraday, begin_ts)
    combined = compute_metrics(ct, ce)

    print_summary(per_instrument, combined, signal_tf, exec_tf)
    flag_negative_sharpe(per_instrument, combined)
    plot_equity(per_series, ce, RESULTS_PNG)
    return 0


def _parse_date(s):
    return pd.Timestamp(datetime.strptime(s, "%Y-%m-%d"), tz="UTC")


def main():
    ap = argparse.ArgumentParser(description="Multi-timeframe trend-pullback backtest.")
    ap.add_argument("--symbols", nargs="+", default=config.PULLBACK_SYMBOLS)
    ap.add_argument("--exec-timeframe", choices=["1Hour", "4Hour", "none"],
                    default="4Hour", help="Execution timeframe (none = fill at next daily bar).")
    ap.add_argument("--months", type=int, default=6,
                    help="Months of history (ignored if --start given).")
    ap.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD.")
    ap.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (default today).")
    ap.add_argument("--ema", action="store_true", help="Use EMAs instead of SMAs.")
    args = ap.parse_args()

    end_dt = _parse_date(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc))
    start_dt = _parse_date(args.start) if args.start else end_dt - pd.Timedelta(days=int(args.months * 31))
    log.info("Backtest window: %s -> %s", start_dt.date(), end_dt.date())

    params = PullbackParams(use_ema=args.ema)
    daily_data, intra_data, exec_is_intraday = fetch_all(
        args.symbols, args.exec_timeframe, start_dt, end_dt)
    return run_backtest(daily_data, intra_data, params, exec_is_intraday,
                        "1Day", args.exec_timeframe, begin_ts=start_dt)


if __name__ == "__main__":
    raise SystemExit(main())
