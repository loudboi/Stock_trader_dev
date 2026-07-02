"""
gex_lab/screen.py
=================
Nightly GEX screen over a watchlist: compute levels per name, derive the P2P
fields (R:R to +GEX, cushion to put mass, status), print a table, and — crucially
— SAVE A DATED SNAPSHOT so you accumulate a real point-in-time GEX history that
gex_lab/backtest.py can eventually run on.

Status is a HONEST SUBSET of the Vol-Desk filters — only what free OI data
supports (spot vs pTrans, R:R, put-mass cushion). Grade/db_change/dealer-delta
need vendor data and are intentionally NOT faked here.
"""

import argparse
import logging
import os
from datetime import datetime, timezone

import pandas as pd

from gex_lab.gex import fetch_chain, compute_levels

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("screen")

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "snapshots")

# Entry thresholds (the subset we can compute honestly).
MIN_RR = 2.0
MIN_CUSHION = 0.02
PENDING_BAND = 0.005


def screen_row(ticker: str, levels: dict) -> dict:
    spot, ptrans, tgt = levels["spot"], levels["ptrans"], levels["pos_gex"]
    cotmp = levels["cotmp"]
    rr = cushion = None
    if ptrans and tgt and tgt > spot and spot > ptrans:
        rr = (tgt - spot) / (spot - ptrans)
    if cotmp:
        cushion = (spot - cotmp) / spot

    status = "BLOCKED"
    if ptrans and tgt and tgt > spot:
        above = spot > ptrans
        rr_ok = rr is not None and rr >= MIN_RR
        cush_ok = cushion is not None and cushion >= MIN_CUSHION
        if above and rr_ok and cush_ok:
            status = "CONFIRMED"
        elif not above and spot >= ptrans * (1 - PENDING_BAND):
            status = "PENDING"
    return {
        "ticker": ticker, "spot": round(spot, 2),
        "ptrans": round(ptrans, 2) if ptrans else None,
        "pos_gex": round(tgt, 2) if tgt else None,
        "ntrans": round(levels["ntrans"], 2) if levels["ntrans"] else None,
        "cotmp": round(cotmp, 2) if cotmp else None,
        "rr": round(rr, 2) if rr else None,
        "cushion_%": round(cushion * 100, 2) if cushion is not None else None,
        "net_gex_$bn": round(levels["net_gex"] / 1e9, 2),
        "status": status,
    }


def screen(tickers, max_days=45) -> pd.DataFrame:
    rows = []
    for tk in tickers:
        try:
            chain, spot = fetch_chain(tk, max_days)
            if chain.empty or spot is None:
                log.warning("%s: no chain data; skipping.", tk)
                continue
            rows.append(screen_row(tk, compute_levels(chain, spot)))
            log.info("%s screened (spot %.2f, %s).", tk, spot, rows[-1]["status"])
        except Exception as e:  # noqa: BLE001
            log.warning("%s failed: %s", tk, e)
    return pd.DataFrame(rows)


def save_snapshot(df: pd.DataFrame, snapshot_dir=SNAPSHOT_DIR) -> str:
    os.makedirs(snapshot_dir, exist_ok=True)
    date = datetime.now(timezone.utc).date().isoformat()
    df = df.copy()
    df.insert(0, "date", date)
    path = os.path.join(snapshot_dir, f"gex_{date}.csv")
    df.to_csv(path, index=False)
    log.info("Saved snapshot -> %s (%d names). This builds your GEX history.", path, len(df))
    return path


def main():
    ap = argparse.ArgumentParser(description="Free GEX levels screen (+ snapshot logging).")
    ap.add_argument("--tickers", nargs="+",
                    default=["NVDA", "AAPL", "TSLA", "AMD", "META", "MSFT", "AMZN"])
    ap.add_argument("--max-days", type=int, default=45, help="Include expiries within N days.")
    ap.add_argument("--no-save", action="store_true", help="Print only; don't log a snapshot.")
    args = ap.parse_args()

    df = screen(args.tickers, args.max_days)
    if df.empty:
        log.error("No names screened.")
        return 1
    order = {"CONFIRMED": 0, "PENDING": 1, "BLOCKED": 2}
    df = df.sort_values(by=["status", "rr"], key=lambda s: s.map(order) if s.name == "status" else s,
                        ascending=[True, False], na_position="last")
    print("\n" + df.to_string(index=False))
    print("\nNOTE: status uses only free-data filters (spot>pTrans, R:R>=2, cushion>=2%). "
          "Grade / db_change / dealer-delta need vendor data and are omitted.")
    if not args.no_save:
        save_snapshot(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
