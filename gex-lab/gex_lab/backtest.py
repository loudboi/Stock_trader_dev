"""
gex_lab/backtest.py
===================
Backtest engine for the "cross above pTrans -> ride to +GEX" thesis and the
Vol-Desk stop framework. It consumes a per-ticker daily panel with OHLC PLUS the
GEX levels for each day (ptrans, ntrans, pos_gex, cotmp) and simulates entries and
exits mechanically.

TWO HONEST CAVEATS, loudly:
  1. P&L is measured on the UNDERLYING, not options. Real single-stock option P&L
     (theta, IV crush, 5-15% spreads) will be very different. This tests whether
     the SIGNAL works (does price reach +GEX?), which is the core thesis — not the
     options trade economics.
  2. It needs HISTORICAL levels per day. Free data only gives today's chain, so
     run --demo now (synthetic) and point --snapshots at gex_lab/screen output
     once you've accumulated real nightly history (or bought option history).

Stop framework (interpretations of the described rules are documented inline):
  Stop1 close < nTrans | Stop2 -10% while below pTrans | Stop3 day-7 <50% progress
  Stop4 <10%/day progress for 3 consecutive sessions | Target T1 = +GEX.
"""

import argparse
import glob
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("gex_backtest")

_LEVEL_COLS = ["ptrans", "ntrans", "pos_gex", "cotmp"]


@dataclass
class Params:
    min_rr: float = 2.0
    min_cushion: float = 0.02
    hard_stop: float = 0.10          # Stop2
    time_stop_days: int = 7          # Stop3
    time_stop_progress: float = 0.50
    stall_days: int = 3              # Stop4
    stall_progress: float = 0.10     # <10%/day progress toward T1


def simulate(df: pd.DataFrame, p: Params = Params()) -> list:
    """Simulate one ticker. df: daily bars (open/high/low/close) + level columns
    (ptrans/ntrans/pos_gex/cotmp) aligned by date. Returns a list of trades."""
    trades = []
    pos = None
    stall = 0
    for i in range(1, len(df)):
        row, prev = df.iloc[i], df.iloc[i - 1]
        ptrans, ntrans, tgt, cotmp = (row.get(c) for c in _LEVEL_COLS)
        c = float(row["close"])

        if pos is None:
            # Entry: close CROSSES above pTrans, with R:R and cushion filters.
            if _num(ptrans) and _num(tgt) and prev["close"] <= ptrans < c and tgt > c:
                rr = (tgt - c) / (c - ptrans)
                cushion = (c - cotmp) / c if _num(cotmp) else 1.0
                if rr >= p.min_rr and cushion >= p.min_cushion:
                    pos = {"entry": c, "i0": i, "tgt": float(tgt), "last_prog": 0.0}
                    stall = 0
            continue

        # ---- manage an open position ----
        entry, i0, tgt_locked = pos["entry"], pos["i0"], pos["tgt"]
        day = i - i0
        exit_px = reason = None

        if float(row["high"]) >= tgt_locked:                 # Target T1 (+GEX)
            exit_px, reason = tgt_locked, "T1 +GEX"
        elif _num(ntrans) and c < ntrans:                    # Stop1
            exit_px, reason = c, "stop1 close<nTrans"
        elif c <= entry * (1 - p.hard_stop) and _num(ptrans) and c < ptrans:  # Stop2
            exit_px, reason = c, "stop2 -10% below pTrans"
        else:
            prog = (c - entry) / (tgt_locked - entry) if tgt_locked > entry else 0.0
            if day >= p.time_stop_days and prog < p.time_stop_progress:       # Stop3
                exit_px, reason = c, "stop3 time (<50% by day7)"
            else:
                stall = stall + 1 if (prog - pos["last_prog"]) < p.stall_progress else 0
                pos["last_prog"] = prog
                if stall >= p.stall_days:                                     # Stop4
                    exit_px, reason = c, "stop4 stalling"

        if exit_px is not None:
            trades.append(_close(df, pos, i, exit_px, reason))
            pos, stall = None, 0

    if pos is not None:                                      # close at the end
        trades.append(_close(df, pos, len(df) - 1, float(df.iloc[-1]["close"]), "end"))
    return trades


def _num(x):
    return x is not None and x == x        # not None, not NaN


def _close(df, pos, i, exit_px, reason):
    entry = pos["entry"]
    return {
        "entry_date": df.index[pos["i0"]], "exit_date": df.index[i],
        "entry": round(entry, 2), "exit": round(exit_px, 2),
        "return_pct": exit_px / entry - 1.0, "days": i - pos["i0"], "reason": reason,
    }


def metrics(trades: list) -> dict:
    n = len(trades)
    if not n:
        return {"trades": 0, "win_rate": 0.0, "avg_return": 0.0, "profit_factor": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0}
    rets = np.array([t["return_pct"] for t in trades])
    wins, losses = rets[rets > 0], rets[rets < 0]
    gl = -losses.sum()
    return {
        "trades": n,
        "win_rate": len(wins) / n,
        "avg_return": float(rets.mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(wins.sum() / gl) if gl > 0 else (float("inf") if len(wins) else 0.0),
    }


def run(panel: dict, p: Params = Params()) -> dict:
    all_trades = []
    for tk, df in panel.items():
        all_trades += simulate(df, p)
    m = metrics(all_trades)
    print("\n" + "=" * 60)
    print(f"GEX THESIS BACKTEST  (underlying P&L proxy — NOT option P&L)")
    print("=" * 60)
    print(f"names: {len(panel)}   trades: {m['trades']}")
    if m["trades"]:
        print(f"win rate:      {m['win_rate']*100:.1f}%")
        print(f"avg return:    {m['avg_return']*100:+.2f}%   (win {m['avg_win']*100:+.2f}% / "
              f"loss {m['avg_loss']*100:+.2f}%)")
        print(f"profit factor: {m['profit_factor']:.2f}")
        by_reason = pd.Series([t["reason"] for t in all_trades]).value_counts()
        print("exit reasons:  " + ", ".join(f"{k}×{v}" for k, v in by_reason.items()))
    print("=" * 60)
    print("Reminder: underlying-move proxy + (in --demo) synthetic levels. Not tradeable "
          "proof until run on REAL accumulated/purchased GEX history.")
    return m


# --------------------------------------------------------------------------- #
# Demo (synthetic) + real (accumulated snapshots) data paths
# --------------------------------------------------------------------------- #
def demo_panel(seed=0, n_names=6, days=180) -> dict:
    """Synthetic OHLC + levels so the engine is runnable/inspectable with no data."""
    rng = np.random.default_rng(seed)
    panel = {}
    idx = pd.date_range("2025-01-01", periods=days, freq="B")
    for k in range(n_names):
        drift = rng.normal(0.0006, 0.0004)
        close = 100 * np.cumprod(1 + rng.normal(drift, 0.02, days))
        df = pd.DataFrame({"open": close, "high": close * 1.01,
                           "low": close * 0.99, "close": close}, index=idx)
        # Levels loosely bracket price so entries/exits actually trigger.
        df["ptrans"] = df["close"].rolling(20, min_periods=1).mean() * 0.99
        df["ntrans"] = df["ptrans"] * 0.95
        df["pos_gex"] = df["close"].rolling(20, min_periods=1).mean() * 1.06
        df["cotmp"] = df["ptrans"] * 0.97
        panel[f"SYN{k}"] = df
    return panel


def load_real(snapshot_dir, price_source="yahoo") -> dict:
    """Build a backtest panel from accumulated screen snapshots + price history.

    Needs >= 2 dated snapshots (gex_YYYY-MM-DD.csv). Merges each ticker's daily
    levels with its OHLC bars over the covered window."""
    files = sorted(glob.glob(os.path.join(snapshot_dir, "gex_*.csv")))
    if len(files) < 2:
        log.error("Found %d snapshot(s) in %s; need >= 2. Run gex_lab.screen nightly "
                  "for a while first (it builds them).", len(files), snapshot_dir)
        return {}
    snaps = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    snaps["date"] = pd.to_datetime(snaps["date"])
    start, end = snaps["date"].min(), snaps["date"].max()
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "alpaca-bot"))
    from bot.data import load_yahoo                       # reuse the equity project's loader
    panel = {}
    for tk, g in snaps.groupby("ticker"):
        px = load_yahoo(tk, start, end + pd.Timedelta(days=1))
        if px.empty:
            continue
        lv = g.set_index("date")[_LEVEL_COLS]
        df = px.join(lv).sort_index()
        df[_LEVEL_COLS] = df[_LEVEL_COLS].ffill()          # carry levels forward between snapshots
        panel[tk] = df.dropna(subset=["close"])
    return panel


def main():
    ap = argparse.ArgumentParser(description="GEX thesis backtester (underlying proxy).")
    ap.add_argument("--demo", action="store_true", help="Run on synthetic data (no history needed).")
    ap.add_argument("--snapshots", default=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "snapshots"),
        help="Dir of accumulated gex_*.csv snapshots from gex_lab.screen.")
    args = ap.parse_args()

    if args.demo:
        log.info("Running on SYNTHETIC demo data (engine check only).")
        panel = demo_panel()
    else:
        panel = load_real(args.snapshots)
        if not panel:
            log.error("No real level history yet. Use --demo to see the engine run, and "
                      "run `python -m gex_lab.screen` nightly to accumulate snapshots.")
            return 1
    run(panel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
