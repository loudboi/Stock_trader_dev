"""
gex_lab/screen.py
=================
Point-in-time GEX screen. Every saved snapshot carries an exact UTC as-of timestamp
and uses a timestamped filename, so rerunning the screen on the same calendar date
never overwrites earlier observations.

The status uses only fields available from the free chain model. Dealer inventory
sign, risk-free rate and dividend yield remain explicit modeling assumptions and
are stored with each row for reproducibility.
"""

import argparse
import logging
import os
from datetime import datetime, timezone

import pandas as pd

from gex_lab.gex import (DEFAULT_DIVIDEND_YIELD, DEFAULT_RISK_FREE, compute_levels,
                         fetch_chain)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("screen")

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "snapshots")
MIN_RR = 2.0
MIN_CUSHION = 0.02
PENDING_BAND = 0.005


def _num(x):
    return x is not None and pd.notna(x)


def screen_row(ticker: str, levels: dict) -> dict:
    spot, ptrans, tgt = levels["spot"], levels["ptrans"], levels["pos_gex"]
    cotmp = levels["cotmp"]
    rr = cushion = None
    if _num(ptrans) and _num(tgt) and tgt > spot and spot > ptrans:
        rr = (tgt - spot) / (spot - ptrans)
    if _num(cotmp) and spot > 0:
        cushion = (spot - cotmp) / spot

    status = "BLOCKED"
    if _num(ptrans) and _num(tgt) and tgt > spot:
        above = spot > ptrans
        rr_ok = rr is not None and rr >= MIN_RR
        cush_ok = cushion is not None and cushion >= MIN_CUSHION
        if above and rr_ok and cush_ok:
            status = "CONFIRMED"
        elif not above and spot >= ptrans * (1 - PENDING_BAND):
            status = "PENDING"
    return {
        "ticker": ticker,
        "spot": round(float(spot), 4),
        "ptrans": round(float(ptrans), 4) if _num(ptrans) else None,
        "pos_gex": round(float(tgt), 4) if _num(tgt) else None,
        "ntrans": round(float(levels["ntrans"]), 4) if _num(levels["ntrans"]) else None,
        "cotmp": round(float(cotmp), 4) if _num(cotmp) else None,
        "rr": round(float(rr), 4) if rr is not None else None,
        "cushion_%": round(float(cushion) * 100, 4) if cushion is not None else None,
        "net_gex_$bn": round(float(levels["net_gex"]) / 1e9, 6),
        "status": status,
        "risk_free": float(levels.get("risk_free", DEFAULT_RISK_FREE)),
        "dividend_yield": float(levels.get("dividend_yield", DEFAULT_DIVIDEND_YIELD)),
        "dealer_sign_model": levels.get("dealer_sign_model", "calls_positive_puts_negative"),
    }


def screen(tickers, max_days=45, risk_free=DEFAULT_RISK_FREE,
           dividend_yield=DEFAULT_DIVIDEND_YIELD) -> pd.DataFrame:
    if max_days <= 0 or risk_free <= -1 or dividend_yield <= -1:
        raise ValueError("invalid max_days/risk_free/dividend_yield")
    rows = []
    for tk in dict.fromkeys(tickers):
        try:
            chain, spot = fetch_chain(tk, max_days)
            if chain.empty or spot is None:
                log.warning("%s: no chain data; skipping.", tk)
                continue
            levels = compute_levels(chain, spot, r=risk_free, q=dividend_yield)
            rows.append(screen_row(tk, levels))
            log.info("%s screened (spot %.2f, %s).", tk, spot, rows[-1]["status"])
        except Exception as e:  # noqa: BLE001
            log.warning("%s failed: %s", tk, e)
    return pd.DataFrame(rows)


def save_snapshot(df: pd.DataFrame, snapshot_dir=SNAPSHOT_DIR, asof=None) -> str:
    if df is None or df.empty:
        raise ValueError("cannot save an empty GEX snapshot")
    os.makedirs(snapshot_dir, exist_ok=True)
    asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp(datetime.now(timezone.utc))
    if asof.tzinfo is None:
        asof = asof.tz_localize("UTC")
    else:
        asof = asof.tz_convert("UTC")
    out = df.copy()
    out.insert(0, "asof_utc", asof.isoformat())
    out.insert(1, "date", asof.date().isoformat())
    stamp = asof.strftime("%Y-%m-%dT%H%M%S.%fZ")
    path = os.path.join(snapshot_dir, f"gex_{stamp}.csv")
    # Timestamp includes microseconds; still fail instead of overwriting if a caller
    # explicitly supplies the exact same as-of twice.
    if os.path.exists(path):
        raise FileExistsError(f"snapshot already exists: {path}")
    out.to_csv(path, index=False)
    log.info("Saved point-in-time snapshot -> %s (%d names).", path, len(out))
    return path


def main():
    ap = argparse.ArgumentParser(description="Free GEX level screen with point-in-time snapshots.")
    ap.add_argument("--tickers", nargs="+",
                    default=["NVDA", "AAPL", "TSLA", "AMD", "META", "MSFT", "AMZN"])
    ap.add_argument("--max-days", type=int, default=45)
    ap.add_argument("--risk-free", type=float, default=DEFAULT_RISK_FREE,
                    help="Continuously compounded risk-free rate used by BSM gamma")
    ap.add_argument("--dividend-yield", type=float, default=DEFAULT_DIVIDEND_YIELD,
                    help="Continuous dividend yield used by BSM gamma")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    if args.max_days <= 0 or args.risk_free <= -1 or args.dividend_yield <= -1:
        ap.error("invalid max-days/rate/dividend-yield")

    df = screen(args.tickers, args.max_days, args.risk_free, args.dividend_yield)
    if df.empty:
        log.error("No names screened.")
        return 1
    order = {"CONFIRMED": 0, "PENDING": 1, "BLOCKED": 2}
    df = df.sort_values(by=["status", "rr"],
                        key=lambda s: s.map(order) if s.name == "status" else s,
                        ascending=[True, False], na_position="last")
    print("\n" + df.to_string(index=False))
    print("\nMODEL NOTE: free-data status omits proprietary grade/dealer-flow fields. "
          "Dealer sign, BSM risk-free rate and dividend yield are assumptions.")
    if not args.no_save:
        save_snapshot(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
