"""
bot/sweep.py
============
Parameter-robustness tooling for Strategy 4, built ON TOP of the existing
backtester (bot/backtest_pullback.py) — no parallel signal logic.

Two modes, both over the combined portfolio:

  --mode grid   Evaluate every parameter combination over the whole window and
                rank them. Reports how many combos are profitable / positive
                Sharpe, so you can see whether the edge is broad or a knife-edge.

  --mode walk   Walk-forward: split the window into folds; on each fold pick the
                best params on the IN-SAMPLE half, then score those params on the
                untouched OUT-OF-SAMPLE half. Aggregated OOS stats are the honest
                read — in-sample-best that falls apart OOS is the overfit tell.

The grid is over a few high-impact PullbackParams; widen it with the flags below.
Data is fetched once (with warmup lead) and reused for every combination.

Run from the project root:

    python -m bot.sweep --mode grid --months 18
    python -m bot.sweep --mode walk --folds 4 --start 2023-01-01 --end 2026-06-01
    python -m bot.sweep --mode walk --add-step 0.02 0.03 0.05 --atr-mult 1.5 2.0 2.5
"""

import argparse
import itertools
import logging
from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd

import config
from bot.strategies.trend_pullback import PullbackParams
from bot.backtest_pullback import (run_combined, compute_metrics, fetch_all,
                                   _parse_date)

log = logging.getLogger("sweep")

# Parameter axes the grid sweeps. Each maps to a PullbackParams field.
_AXES = {
    "add_step": "add_step",
    "atr_mult": "atr_mult",
    "touch_band": "touch_band",
    "vol_contraction": "vol_contraction",
}

_DEFAULT_GRID = {
    "add_step": [0.02, 0.03, 0.04],
    "atr_mult": [1.5, 2.0, 2.5],
    "touch_band": [0.02],
    "vol_contraction": [0.90],
}


# --------------------------------------------------------------------------- #
# Pure evaluation core (testable offline with injected data)
# --------------------------------------------------------------------------- #
def build_grid(overrides: dict) -> list:
    """Cartesian product of the axes -> list of {axis: value} dicts."""
    axes = {k: (overrides.get(k) or _DEFAULT_GRID[k]) for k in _AXES}
    keys = list(axes)
    return [dict(zip(keys, combo)) for combo in itertools.product(*axes.values())]


def params_for(base: PullbackParams, combo: dict) -> PullbackParams:
    return replace(base, **{_AXES[k]: v for k, v in combo.items()})


def _slice_to(data: dict, end_ts) -> dict:
    if end_ts is None:
        return data
    return {n: df[df.index <= end_ts] for n, df in data.items()}


def evaluate(daily_data, intra_data, params, exec_is_intraday,
             begin_ts=None, end_ts=None) -> dict:
    """Run the combined backtest for one parameter set over [begin_ts, end_ts]."""
    daily = _slice_to(daily_data, end_ts)
    intra = _slice_to(intra_data, end_ts) if exec_is_intraday else {}
    trades, eq = run_combined(daily, intra, params, exec_is_intraday, begin_ts)
    return compute_metrics(trades, eq)


def run_grid(daily_data, intra_data, base, grid, exec_is_intraday,
             begin_ts=None, end_ts=None) -> list:
    """Evaluate every combo; return [(combo, metrics)] sorted by Sharpe desc."""
    results = []
    for combo in grid:
        m = evaluate(daily_data, intra_data, params_for(base, combo),
                     exec_is_intraday, begin_ts, end_ts)
        results.append((combo, m))
    results.sort(key=lambda r: r[1]["sharpe"], reverse=True)
    return results


def fold_bounds(start_ts, end_ts, folds: int) -> list:
    """Split [start, end] into `folds` equal slices; each is (is_start, split, oos_end)
    where in-sample is [is_start, split) and out-of-sample is [split, oos_end]."""
    total = (end_ts - start_ts) / folds
    out = []
    for k in range(folds):
        fs = start_ts + total * k
        fe = start_ts + total * (k + 1)
        split = fs + total / 2
        out.append((fs, split, fe))
    return out


def run_walk_forward(daily_data, intra_data, base, grid, exec_is_intraday,
                     start_ts, end_ts, folds: int) -> list:
    """For each fold: pick the best params in-sample, score them out-of-sample."""
    out = []
    for is_start, split, oos_end in fold_bounds(start_ts, end_ts, folds):
        ranked = run_grid(daily_data, intra_data, base, grid, exec_is_intraday,
                          begin_ts=is_start, end_ts=split)
        best_combo, is_metrics = ranked[0]
        oos = evaluate(daily_data, intra_data, params_for(base, best_combo),
                       exec_is_intraday, begin_ts=split, end_ts=oos_end)
        out.append({"is_window": (is_start, split), "oos_window": (split, oos_end),
                    "best": best_combo, "is": is_metrics, "oos": oos})
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _combo_str(combo):
    return " ".join(f"{k}={v}" for k, v in combo.items())


def print_grid(results, top=12):
    print("\n" + "=" * 78)
    print(f"PARAMETER GRID — {len(results)} combos, ranked by Sharpe (combined)")
    print("=" * 78)
    print(f"{'Sharpe':>7}  {'Return%':>8}  {'MaxDD%':>7}  {'Trades':>6}  Params")
    print("-" * 78)
    for combo, m in results[:top]:
        print(f"{m['sharpe']:>7.2f}  {m['total_return']*100:>8.1f}  "
              f"{m['max_drawdown']*100:>7.1f}  {m['trades']:>6}  {_combo_str(combo)}")
    pos_sharpe = sum(1 for _, m in results if m["sharpe"] > 0)
    pos_ret = sum(1 for _, m in results if m["total_return"] > 0)
    n = len(results)
    print("-" * 78)
    print(f"ROBUSTNESS: {pos_sharpe}/{n} combos positive Sharpe, "
          f"{pos_ret}/{n} profitable. A broad band of winners is a better sign "
          "than one lucky cell.")
    print("=" * 78)


def print_walk_forward(folds):
    print("\n" + "=" * 78)
    print(f"WALK-FORWARD — {len(folds)} folds (best in-sample, scored out-of-sample)")
    print("=" * 78)
    print(f"{'Fold':>4}  {'OOS window':>23}  {'IS Shp':>6}  {'OOS Shp':>7}  "
          f"{'OOS Ret%':>8}  Best params")
    print("-" * 78)
    oos_sharpes, oos_rets = [], []
    for k, f in enumerate(folds):
        ow = f"{f['oos_window'][0].date()}->{f['oos_window'][1].date()}"
        print(f"{k+1:>4}  {ow:>23}  {f['is']['sharpe']:>6.2f}  "
              f"{f['oos']['sharpe']:>7.2f}  {f['oos']['total_return']*100:>8.1f}  "
              f"{_combo_str(f['best'])}")
        oos_sharpes.append(f["oos"]["sharpe"])
        oos_rets.append(f["oos"]["total_return"])
    print("-" * 78)
    avg_s = sum(oos_sharpes) / len(oos_sharpes) if oos_sharpes else 0.0
    avg_r = sum(oos_rets) / len(oos_rets) if oos_rets else 0.0
    pos = sum(1 for s in oos_sharpes if s > 0)
    print(f"AGGREGATE OOS: mean Sharpe {avg_s:.2f}, mean return {avg_r*100:.1f}%, "
          f"{pos}/{len(folds)} folds positive. Consistent OOS > a great single backtest.")
    print("=" * 78)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    ap = argparse.ArgumentParser(description="Strategy 4 parameter sweep / walk-forward.")
    ap.add_argument("--mode", choices=["grid", "walk"], default="grid")
    ap.add_argument("--symbols", nargs="+", default=config.PULLBACK_SYMBOLS)
    ap.add_argument("--exec-timeframe", choices=["1Hour", "4Hour", "none"], default="none",
                    help="Execution timeframe (none = next-daily-bar fills; fastest sweep).")
    ap.add_argument("--months", type=int, default=18)
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--folds", type=int, default=4, help="Walk-forward folds.")
    for axis in _AXES:
        ap.add_argument(f"--{axis.replace('_', '-')}", nargs="+", type=float,
                        default=None, help=f"Grid values for {axis}.")
    args = ap.parse_args()

    end_dt = _parse_date(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc))
    start_dt = (_parse_date(args.start) if args.start
                else end_dt - pd.Timedelta(days=int(args.months * 31)))
    log.info("Sweep window: %s -> %s (%s mode)", start_dt.date(), end_dt.date(), args.mode)

    base = PullbackParams(use_ema=args.ema)
    overrides = {axis: getattr(args, axis) for axis in _AXES}
    grid = build_grid(overrides)
    log.info("Grid size: %d combinations", len(grid))

    daily_data, intra_data, exec_is_intraday = fetch_all(
        args.symbols, args.exec_timeframe, start_dt, end_dt)
    if not daily_data:
        log.error("No data fetched; aborting.")
        return 1

    if args.mode == "grid":
        results = run_grid(daily_data, intra_data, base, grid, exec_is_intraday,
                           begin_ts=start_dt)
        print_grid(results)
    else:
        folds = run_walk_forward(daily_data, intra_data, base, grid, exec_is_intraday,
                                 start_dt, end_dt, args.folds)
        print_walk_forward(folds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
