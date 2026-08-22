from __future__ import annotations

"""Cross-check the frozen target engine against the research event targets.

No strategy parameter is selected here. For every monthly research event in the
continuous 2006-2026 review interval, recompute the target with
bot.vol_managed_targets.latest_target using only history available at the prior
completed session. Fail on any target, cadence, date, or realized-vol mismatch.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from bot.vol_managed_targets import DEFAULT_SPEC, latest_target
import active_vol_managed_external_review as ext


def load_adjusted_close() -> pd.Series:
    d = yf.Ticker("QQQ").history(
        start=ext.START, end=ext.END, auto_adjust=False, actions=True, repair=False
    )
    if d is None or d.empty:
        raise RuntimeError("no QQQ history for target parity review")
    d = ext._normalize(d)
    adjusted = pd.to_numeric(d["Adj Close"], errors="coerce") if "Adj Close" in d else pd.to_numeric(d["Close"], errors="coerce")
    adjusted = adjusted.dropna().astype(float)
    adjusted = adjusted[adjusted > 0]
    if adjusted.empty:
        raise RuntimeError("invalid adjusted QQQ history")
    return adjusted


def main():
    data = ext.load_market()
    idx, _, _, _, _, _, events = data
    adjusted = load_adjusted_close()
    begin = idx[idx >= ext.BEGIN][0]
    end = idx[idx <= ext.FINISH][-1]

    checked = 0
    max_target_error = 0.0
    max_vol_error = 0.0
    examples = []

    for execution_session, research_target in sorted(events.items()):
        if execution_session < begin or execution_session > end:
            continue
        prior = adjusted.index[adjusted.index < execution_session]
        if len(prior) == 0:
            raise RuntimeError(f"no prior completed QQQ session for {execution_session.date()}")
        decision_close = prior[-1]
        history = adjusted.loc[:decision_close]
        target = latest_target(history, next_session=execution_session, spec=DEFAULT_SPEC)
        if not target.monthly_rebalance_due:
            raise AssertionError(f"engine cadence not due for research event {execution_session.date()}")
        if target.decision_close != decision_close:
            raise AssertionError(
                f"decision close mismatch at {execution_session.date()}: engine={target.decision_close} expected={decision_close}"
            )
        if target.next_session != execution_session:
            raise AssertionError(
                f"next-session mismatch: engine={target.next_session} research={execution_session}"
            )

        target_error = abs(float(target.target_exposure) - float(research_target))
        max_target_error = max(max_target_error, target_error)

        # Independently reproduce the research vol at the decision close: returns
        # ending at decision_close are excluded by the frozen one-session signal lag.
        returns = history.pct_change(fill_method=None)
        expected_vol = float(
            returns.rolling(DEFAULT_SPEC.lookback, min_periods=DEFAULT_SPEC.lookback)
            .std(ddof=1)
            .mul(np.sqrt(DEFAULT_SPEC.annualization))
            .shift(DEFAULT_SPEC.signal_lag_sessions)
            .iloc[-1]
        )
        vol_error = abs(float(target.realized_vol) - expected_vol)
        max_vol_error = max(max_vol_error, vol_error)

        if target_error > 1e-12 or vol_error > 1e-12:
            raise AssertionError(
                f"target parity failure {execution_session.date()}: research={research_target:.12f} "
                f"engine={target.target_exposure:.12f} targetErr={target_error:.3e} volErr={vol_error:.3e}"
            )

        if execution_session.year in (2008, 2009, 2020) and execution_session.month in (1, 2, 3, 4, 5, 6, 12):
            examples.append(
                (
                    execution_session,
                    decision_close,
                    float(target.realized_vol),
                    float(target.target_exposure),
                )
            )
        checked += 1

    if checked == 0:
        raise RuntimeError("no monthly targets checked")

    print(
        f"TARGET_ENGINE_PARITY PASS events={checked} maxTargetError={max_target_error:.3e} "
        f"maxRealizedVolError={max_vol_error:.3e}"
    )
    print("TIMING: decision close D uses returns through D-1; target is executed on next exchange session E in the research simulator")
    for execution_session, decision_close, realized_vol, exposure in examples:
        print(
            f"execution={execution_session.date()} decisionClose={decision_close.date()} "
            f"realizedVol={realized_vol:.4%} targetExposure={exposure:.4f}"
        )


if __name__ == "__main__":
    main()
