"""
bot/sweep.py
============
Parameter robustness and true fit/test walk-forward tooling for the pullback
backtester.

Grid mode ranks fixed parameter combinations over one window. Walk mode divides
each fold into an in-sample half and a disjoint out-of-sample half, picks the best
in-sample parameters, then evaluates those unchanged parameters out of sample.
Boundary timestamps are never included in both halves.
"""

import argparse
import itertools
import logging
from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd

import config
from bot.strategies.trend_pullback import PullbackParams
from bot.backtest_pullback import compute_metrics, fetch_all, run_combined, _parse_date

log = logging.getLogger("sweep")
_AXES = {"add_step": "add_step", "atr_mult": "atr_mult",
         "touch_band": "touch_band", "vol_contraction": "vol_contraction"}
_DEFAULT_GRID = {"add_step": [0.02, 0.03, 0.04], "atr_mult": [1.5, 2.0, 2.5],
                 "touch_band": [0.02], "vol_contraction": [0.90]}


def build_grid(overrides: dict) -> list:
    axes = {k: (overrides.get(k) or _DEFAULT_GRID[k]) for k in _AXES}
    if any(not vals for vals in axes.values()):
        raise ValueError("grid axes cannot be empty")
    keys = list(axes)
    return [dict(zip(keys, values)) for values in itertools.product(*axes.values())]


def params_for(base: PullbackParams, combo: dict) -> PullbackParams:
    unknown = set(combo) - set(_AXES)
    if unknown:
        raise ValueError("unknown sweep axis: " + ", ".join(sorted(unknown)))
    return replace(base, **{_AXES[k]: v for k, v in combo.items()})


def _slice_to(data: dict, end_ts, inclusive=True) -> dict:
    if end_ts is None:
        return data
    if inclusive:
        return {n: df[df.index <= end_ts] for n, df in data.items()}
    return {n: df[df.index < end_ts] for n, df in data.items()}


def evaluate(daily_data, intra_data, params, exec_is_intraday,
             begin_ts=None, end_ts=None, end_inclusive=True) -> dict:
    daily = _slice_to(daily_data, end_ts, end_inclusive)
    intra = (_slice_to(intra_data, end_ts, end_inclusive)
             if exec_is_intraday else {})
    trades, eq = run_combined(daily, intra, params, exec_is_intraday, begin_ts)
    return compute_metrics(trades, eq)


def run_grid(daily_data, intra_data, base, grid, exec_is_intraday,
             begin_ts=None, end_ts=None, end_inclusive=True) -> list:
    results = []
    for combo in grid:
        m = evaluate(daily_data, intra_data, params_for(base, combo),
                     exec_is_intraday, begin_ts, end_ts, end_inclusive)
        results.append((combo, m))
    return sorted(results, key=lambda item: item[1]["sharpe"], reverse=True)


def fold_bounds(start_ts, end_ts, folds: int) -> list:
    if folds <= 0 or start_ts >= end_ts:
        raise ValueError("folds must be positive and start precede end")
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
    out = []
    bounds = fold_bounds(start_ts, end_ts, folds)
    for k, (is_start, split, oos_end) in enumerate(bounds):
        # IS is [is_start, split), OOS begins at split. The final OOS end is
        # inclusive only on the last fold; intermediate fold ends are exclusive.
        ranked = run_grid(daily_data, intra_data, base, grid, exec_is_intraday,
                          begin_ts=is_start, end_ts=split, end_inclusive=False)
        if not ranked:
            continue
        best_combo, is_metrics = ranked[0]
        final = k == len(bounds) - 1
        oos = evaluate(daily_data, intra_data, params_for(base, best_combo),
                       exec_is_intraday, begin_ts=split, end_ts=oos_end,
                       end_inclusive=final)
        out.append({"is_window": (is_start, split),
                    "oos_window": (split, oos_end),
                    "best": best_combo, "is": is_metrics, "oos": oos})
    return out


def _combo_str(combo):
    return " ".join(f"{k}={v}" for k, v in combo.items())


def print_grid(results, top=12):
    print(f"\nPARAMETER GRID — {len(results)} combinations, historical Sharpe ranking")
    for combo, m in results[:top]:
        print(f"Sharpe={m['sharpe']:.2f} Return={m['total_return']:.1%} "
              f"MaxDD={m['max_drawdown']:.1%} Trades={m['trades']} {_combo_str(combo)}")
    n = len(results)
    if n:
        print(f"ROBUSTNESS: {sum(m['sharpe'] > 0 for _, m in results)}/{n} positive Sharpe; "
              f"{sum(m['total_return'] > 0 for _, m in results)}/{n} positive return.")


def print_walk_forward(folds):
    print(f"\nWALK-FORWARD — {len(folds)} disjoint fit/test folds")
    sharpes = []
    for k, f in enumerate(folds, 1):
        fs, fe = f["oos_window"]
        print(f"{k}: OOS {fs.date()}->{fe.date()} IS={f['is']['sharpe']:.2f} "
              f"OOS={f['oos']['sharpe']:.2f} Return={f['oos']['total_return']:.1%} "
              f"{_combo_str(f['best'])}")
        sharpes.append(f["oos"]["sharpe"])
    mean = sum(sharpes) / len(sharpes) if sharpes else 0.0
    print(f"AGGREGATE OOS: mean Sharpe {mean:.2f}; "
          f"{sum(s > 0 for s in sharpes)}/{len(sharpes)} positive folds.")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    ap = argparse.ArgumentParser(description="Pullback parameter grid / walk-forward research.")
    ap.add_argument("--mode", choices=["grid", "walk"], default="grid")
    ap.add_argument("--symbols", nargs="+", default=config.PULLBACK_SYMBOLS)
    ap.add_argument("--exec-timeframe", choices=["1Hour", "4Hour", "none"], default="none")
    ap.add_argument("--months", type=int, default=18)
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--data-source", choices=["alpaca", "yahoo"], default="alpaca")
    ap.add_argument("--folds", type=int, default=4)
    for axis in _AXES:
        ap.add_argument(f"--{axis.replace('_', '-')}", nargs="+", type=float, default=None)
    args = ap.parse_args()
    if args.months <= 0 or args.folds <= 0:
        ap.error("months and folds must be positive")

    end_dt = _parse_date(args.end) if args.end else pd.Timestamp(datetime.now(timezone.utc))
    start_dt = (_parse_date(args.start) if args.start
                else end_dt - pd.Timedelta(days=args.months * 31))
    if start_dt >= end_dt:
        ap.error("start must precede end")
    base = PullbackParams(use_ema=args.ema)
    overrides = {axis: getattr(args, axis) for axis in _AXES}
    grid = build_grid(overrides)
    daily_data, intra_data, exec_is_intraday = fetch_all(
        args.symbols, args.exec_timeframe, start_dt, end_dt, source=args.data_source)
    if len(daily_data) != len(args.symbols):
        log.error("Missing requested symbol data; refusing a silently changed universe.")
        return 1
    if args.mode == "grid":
        print_grid(run_grid(daily_data, intra_data, base, grid, exec_is_intraday,
                            begin_ts=start_dt))
    else:
        print_walk_forward(run_walk_forward(daily_data, intra_data, base, grid,
                                            exec_is_intraday, start_dt, end_dt,
                                            args.folds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
