"""
gex_lab/backtest.py
===================
Backtest engine for the GEX crossing thesis using UNDERLYING P&L only.

Causality rules:
* a nightly chain snapshot is never joined to that same trading date; it becomes
  eligible only on a later calendar day/session;
* a close-confirmed pTrans crossing is an end-of-day signal and enters no earlier
  than the NEXT session's open;
* stale snapshot levels expire after a small configurable age;
* when a daily bar both reaches the locked target and closes through a price stop,
  OHLC cannot prove a favorable ordering, so the adverse stop outcome is used.

The model still does not reproduce option P&L, dealer inventory, intraday path,
spread/IV/theta, or proprietary transition-level definitions.
"""

import argparse
import glob
import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("gex_backtest")

_LEVEL_COLS = ["ptrans", "ntrans", "pos_gex", "cotmp"]


@dataclass(frozen=True)
class Params:
    min_rr: float = 2.0
    min_cushion: float = 0.02
    hard_stop: float = 0.10
    time_stop_days: int = 7
    time_stop_progress: float = 0.50
    stall_days: int = 3
    stall_progress: float = 0.10

    def __post_init__(self):
        if (self.min_rr <= 0 or self.min_cushion < 0 or not 0 < self.hard_stop < 1 or
                self.time_stop_days <= 0 or not 0 <= self.time_stop_progress <= 1 or
                self.stall_days <= 0 or self.stall_progress < 0):
            raise ValueError("invalid GEX backtest parameters")


def _num(x):
    return x is not None and pd.notna(x) and np.isfinite(float(x))


def _price_stop(row, pos, p: Params):
    c = float(row["close"])
    ntrans = row.get("ntrans")
    ptrans = row.get("ptrans")
    if _num(ntrans) and c < float(ntrans):
        return c, "stop1 close<nTrans"
    if c <= pos["entry"] * (1 - p.hard_stop):
        threshold = float(ptrans) if _num(ptrans) else pos.get("ptrans")
        if _num(threshold) and c < float(threshold):
            return c, "stop2 hard-stop below pTrans"
    return None, None


def simulate(df: pd.DataFrame, p: Params = Params()) -> list:
    """Simulate one ticker from point-in-time daily OHLC+levels.

    Entry signals form on close and are filled, if still sensible, at the next
    available session open. The signal day's target/pTrans are locked for entry
    economics; management may use later point-in-time nTrans/pTrans observations.
    """
    required = {"open", "high", "low", "close", *_LEVEL_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("backtest frame missing columns: " + ", ".join(sorted(missing)))
    if len(df) < 2:
        return []
    df = df.sort_index()
    trades, pos, pending = [], None, None
    stall = 0

    for i in range(1, len(df)):
        row, prev = df.iloc[i], df.iloc[i - 1]

        # A signal observed at yesterday's/current previous close can only enter at
        # this session's open. Cancel gaps that already invalidate the thesis.
        if pos is None and pending is not None:
            o = float(row["open"])
            tgt = pending["tgt"]
            ntrans = pending.get("ntrans")
            if (o > pending["ptrans"] and o < tgt and
                    (not _num(ntrans) or o >= float(ntrans))):
                pos = {"entry": o, "i0": i, "tgt": tgt,
                       "ptrans": pending["ptrans"], "ntrans": ntrans,
                       "signal_date": pending["signal_date"], "last_prog": 0.0}
                stall = 0
            pending = None

        if pos is None:
            ptrans, tgt, cotmp = row.get("ptrans"), row.get("pos_gex"), row.get("cotmp")
            c, prev_c = float(row["close"]), float(prev["close"])
            if (_num(ptrans) and _num(tgt) and prev_c <= float(ptrans) < c and
                    float(tgt) > c):
                rr = (float(tgt) - c) / (c - float(ptrans))
                cushion = ((c - float(cotmp)) / c if _num(cotmp) and c > 0 else 1.0)
                if rr >= p.min_rr and cushion >= p.min_cushion:
                    pending = {"tgt": float(tgt), "ptrans": float(ptrans),
                               "ntrans": (float(row["ntrans"])
                                          if _num(row.get("ntrans")) else None),
                               "signal_date": df.index[i]}
            continue

        entry, i0, tgt = pos["entry"], pos["i0"], pos["tgt"]
        day = i - i0
        c = float(row["close"])
        stop_px, stop_reason = _price_stop(row, pos, p)
        target_hit = float(row["high"]) >= tgt

        # If a price stop and target both occur in one daily bar, OHLC cannot prove
        # the sequence. Do not award an automatic target win.
        if stop_reason and target_hit:
            exit_px, reason = stop_px, "ambiguous target/stop -> " + stop_reason
        elif stop_reason:
            exit_px, reason = stop_px, stop_reason
        elif target_hit:
            exit_px, reason = tgt, "T1 +GEX"
        else:
            exit_px = reason = None
            prog = (c - entry) / (tgt - entry) if tgt > entry else 0.0
            if day >= p.time_stop_days and prog < p.time_stop_progress:
                exit_px, reason = c, "stop3 time/progress"
            else:
                delta = prog - pos["last_prog"]
                stall = stall + 1 if delta < p.stall_progress else 0
                pos["last_prog"] = prog
                if stall >= p.stall_days:
                    exit_px, reason = c, "stop4 stalling"

        if exit_px is not None:
            trades.append(_close(df, pos, i, exit_px, reason))
            pos, stall = None, 0

    if pos is not None:
        trades.append(_close(df, pos, len(df) - 1,
                             float(df.iloc[-1]["close"]), "end"))
    return trades


def _close(df, pos, i, exit_px, reason):
    entry = pos["entry"]
    return {
        "signal_date": pos.get("signal_date"),
        "entry_date": df.index[pos["i0"]],
        "exit_date": df.index[i],
        "entry": round(entry, 4),
        "exit": round(float(exit_px), 4),
        "return_pct": float(exit_px) / entry - 1.0,
        "days": i - pos["i0"],
        "reason": reason,
    }


def metrics(trades: list) -> dict:
    n = len(trades)
    if not n:
        return {"trades": 0, "win_rate": 0.0, "avg_return": 0.0,
                "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
    rets = np.array([t["return_pct"] for t in trades], dtype=float)
    wins, losses = rets[rets > 0], rets[rets < 0]
    gross_loss = -losses.sum()
    return {
        "trades": n,
        "win_rate": len(wins) / n,
        "avg_return": float(rets.mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": (float(wins.sum() / gross_loss) if gross_loss > 0
                          else (float("inf") if len(wins) else 0.0)),
    }


def run(panel: dict, p: Params = Params()) -> dict:
    all_trades = []
    for _, df in panel.items():
        all_trades.extend(simulate(df, p))
    m = metrics(all_trades)
    print("\nGEX THESIS BACKTEST — UNDERLYING P&L PROXY, NOT OPTION P&L")
    print(f"names={len(panel)} trades={m['trades']} win_rate={m['win_rate']:.1%} "
          f"avg_return={m['avg_return']:+.2%} PF={m['profit_factor']:.2f}")
    if all_trades:
        reasons = pd.Series([t["reason"] for t in all_trades]).value_counts()
        print("exit reasons: " + ", ".join(f"{k}×{v}" for k, v in reasons.items()))
    return m


def demo_panel(seed=0, n_names=6, days=180) -> dict:
    rng = np.random.default_rng(seed)
    panel = {}
    idx = pd.date_range("2025-01-01", periods=days, freq="B", tz="UTC")
    for k in range(n_names):
        drift = rng.normal(0.0006, 0.0004)
        close = 100 * np.cumprod(1 + rng.normal(drift, 0.02, days))
        df = pd.DataFrame({"open": close, "high": close * 1.01,
                           "low": close * 0.99, "close": close}, index=idx)
        df["ptrans"] = df["close"].rolling(20, min_periods=1).mean() * 0.99
        df["ntrans"] = df["ptrans"] * 0.95
        df["pos_gex"] = df["close"].rolling(20, min_periods=1).mean() * 1.06
        df["cotmp"] = df["ptrans"] * 0.97
        panel[f"SYN{k}"] = df
    return panel


def load_yahoo_prices(ticker, start, end) -> pd.DataFrame:
    """Local GEX-lab price loader; avoids cross-project sys.path coupling."""
    import yfinance as yf
    raw = yf.download(ticker, start=pd.Timestamp(start).date(),
                      end=pd.Timestamp(end).date(), interval="1d",
                      auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    raw = raw.copy()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns={"Open": "open", "High": "high", "Low": "low",
                              "Close": "close", "Volume": "volume"})
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in raw]
    out = raw[cols].dropna().sort_index()
    idx = pd.DatetimeIndex(out.index)
    out.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return out


def _load_snapshots(snapshot_dir) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(snapshot_dir, "gex_*.csv")))
    if len(files) < 2:
        log.error("Found %d snapshot(s) in %s; need >= 2 point-in-time observations.",
                  len(files), snapshot_dir)
        return pd.DataFrame()
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        if "ticker" not in frame or not set(_LEVEL_COLS).issubset(frame.columns):
            log.warning("Skipping malformed snapshot %s", path)
            continue
        if "asof_utc" in frame:
            asof = pd.to_datetime(frame["asof_utc"], utc=True, errors="coerce")
        elif "date" in frame:
            # Legacy date-only snapshots were described as nightly. Treat them as
            # known only AFTER that date, never on the same session.
            asof = pd.to_datetime(frame["date"], utc=True, errors="coerce") + pd.Timedelta(hours=23, minutes=59)
        else:
            log.warning("Skipping snapshot without as-of/date: %s", path)
            continue
        frame = frame.copy()
        frame["asof_utc"] = asof
        frame["available_from"] = asof.dt.normalize() + pd.Timedelta(days=1)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    snaps = pd.concat(frames, ignore_index=True).dropna(subset=["asof_utc", "available_from"])
    # If screen was rerun on one date, the latest same-date snapshot is the one
    # carried into the next session. Earlier files remain preserved for audit.
    snaps["snapshot_date"] = snaps["asof_utc"].dt.normalize()
    snaps = snaps.sort_values("asof_utc").drop_duplicates(
        subset=["ticker", "snapshot_date"], keep="last")
    return snaps


def load_real(snapshot_dir, price_source="yahoo", price_loader=None,
              max_level_age_days=4) -> dict:
    """Build point-in-time OHLC+GEX panels from accumulated snapshots.

    ``price_source`` is explicit. Only Yahoo is implemented; unsupported sources
    fail instead of being silently ignored. Levels are joined with ``merge_asof``
    on their conservative availability date and expire when too old.
    """
    if price_source != "yahoo":
        raise ValueError("only price_source='yahoo' is implemented")
    if max_level_age_days < 1:
        raise ValueError("max_level_age_days must be >= 1")
    snaps = _load_snapshots(snapshot_dir)
    if snaps.empty:
        return {}
    loader = price_loader or load_yahoo_prices
    start = snaps["snapshot_date"].min() - pd.Timedelta(days=7)
    end = snaps["snapshot_date"].max() + pd.Timedelta(days=max_level_age_days + 3)
    panel = {}

    for tk, g in snaps.groupby("ticker"):
        px = loader(tk, start, end)
        if px is None or px.empty:
            continue
        px = px.copy().sort_index()
        idx = pd.DatetimeIndex(px.index)
        px.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        price_frame = px.reset_index().rename(columns={px.index.name or "index": "price_date"})
        if "price_date" not in price_frame:
            price_frame = price_frame.rename(columns={price_frame.columns[0]: "price_date"})
        price_frame["price_date"] = pd.to_datetime(price_frame["price_date"], utc=True)

        levels = g[["available_from", "asof_utc", *_LEVEL_COLS]].sort_values("available_from")
        levels = levels.rename(columns={"asof_utc": "level_asof_utc"})
        merged = pd.merge_asof(price_frame.sort_values("price_date"), levels,
                               left_on="price_date", right_on="available_from",
                               direction="backward")
        age = (merged["price_date"] - merged["available_from"]).dt.total_seconds() / 86400.0
        stale = age > max_level_age_days
        merged.loc[stale, _LEVEL_COLS] = np.nan
        merged = merged.set_index("price_date").drop(columns=["available_from"])
        panel[tk] = merged.sort_index()
    return panel


def main():
    ap = argparse.ArgumentParser(description="Point-in-time GEX thesis backtester.")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--snapshots", default=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "snapshots"))
    ap.add_argument("--price-source", choices=["yahoo"], default="yahoo")
    ap.add_argument("--max-level-age-days", type=int, default=4)
    args = ap.parse_args()
    if args.max_level_age_days < 1:
        ap.error("max-level-age-days must be >= 1")

    if args.demo:
        log.info("Running synthetic engine demo; not evidence for the GEX thesis.")
        panel = demo_panel()
    else:
        panel = load_real(args.snapshots, args.price_source,
                          max_level_age_days=args.max_level_age_days)
        if not panel:
            log.error("No usable point-in-time GEX history. Accumulate snapshots first.")
            return 1
    run(panel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
