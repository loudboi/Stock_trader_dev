# External review response — volatility-managed QQQ — 2026-08-22

## Status

**REJECT remains the controlling status.** The frozen strategy parameters are not being retuned. The research candidate may only be reconsidered after the blocking accounting, attribution, execution and statistical issues below are rerun and published.

The external review contains a mix of stale-checkout findings, valid blockers, and factual overstatements. None of the corrections below is sufficient to reverse the overall rejection by itself.

## Finding dispositions

### F0 — artifacts missing: STALE CHECKOUT, not a research blocker

The GitHub research branch contains the report, deterministic target engine, tests, and research validators. The external reviewer inspected a local checkout that did not contain the research branch. Reproducibility is still incomplete because generated result tables/exposure series were not committed; the external-review branch therefore adds a self-contained rerun and is designed to publish its summary/exposure trace.

### F1 — 2020 drawdown impossible: REVIEW ARITHMETIC IS WRONG; exact trace still required

The review's concrete falsification uses an incorrect intermediate price move. Public QQQ daily history shows approximately 227.81 on 2020-02-19 and 208.04 on 2020-03-02, a decline of about 8.7%, not the claimed 17.3%. Reconstructing the frozen 20-day signal from the surrounding daily closes gives approximately 29.2% annualized realized volatility for the March decision input, implying a March target near 0.86x (`0.25 / 0.292`) rather than continued 1.75x exposure. QQQ then declined roughly another 21.1% from March 2 to March 23. A strategy drawdown in the low-30% range is therefore mechanically plausible; the asserted 40-49% minimum is not established.

This does not waive the reproducibility requirement. The rerun must still publish pre/post exposure, target, debt and EUR NAV around 2020-02-19, 2020-03-02 and 2020-03-23, plus the exact 2020 drawdown path. Finding 1 is no longer treated as a blocking falsification, but failure to reconcile the exact trace would be blocking.

### F2 — leverage explains the result: VALID BLOCKER

SPY-relative outperformance is not the relevant alpha test for a QQQ strategy. QQQ is the primary passive benchmark. A constant-1.75x comparison is not a sufficient leverage control because 1.75x is a cap, not realized average exposure.

The rerun adds two attribution-only controls with the same trading/tax/financing mechanics: a constant target at the strategy's ex-post mean monthly target, and a constant target chosen to match the active strategy's realized volatility. CAGR, volatility, Sharpe, Sortino, beta, downside capture and drawdown must be reported. If timing residual return is not positive after those controls, the strategy must be described as leveraged QQQ with a crash brake, not as demonstrated volatility-timing alpha.

### F3 — execution-lag decay: VALID LIMITATION / mechanism claim downgraded

The prior parameter perturbation showed material recent QQQ-relative sensitivity to moving execution several sessions. That is inconsistent with making a strong causal claim that the result is purely generic volatility clustering. Until broker-parity and alternative month-boundary tests explain it, the mechanism language is downgraded to a hypothesis.

The frozen timing itself is not an accidental double-lag mismatch: the research intentionally computes volatility at decision close D using returns only through D-1, then executes on the next exchange session E. Thus the latest signal return ends two exchange sessions before execution. The target engine preserves that convention. The old phrase "one-session lag" was ambiguous and should always be qualified as one session relative to the decision close.

### F4 — regime resets erase long-holding tax advantage: VALID BLOCKER

The four development regimes are useful for chronological robustness but are not a fair standalone after-tax comparison to a Slovenian buy-and-hold investor, because each regime resets acquisition date and liquidates at its end. The correct primary tax comparison is one continuous 2006-to-2026 path with one initial capital base and terminal liquidation.

The rerun therefore uses a continuous path. Passive dividend reinvestments remain separate younger lots; the original principal lot can become exempt after the applicable holding period.

The review is incorrect in saying the repository implements generic Slovenian loss carryforward. `bot/taxes.py` explicitly states that ordinary eligible capital losses generally offset gains in the same tax year and that generic future-year carryforward is not modeled. The active FIFO research model is nevertheless incomplete because it taxes positive disposals immediately and does not credit eligible same-year offsets or normalized expenses. It remains a conservative scenario approximation, not filing software.

### F5 — contradiction with prior README: PARTLY STALE, methodological criticism retained

The current GitHub README deliberately avoids hard-coding old research outcomes and directs users to rerun studies. The quoted local README lines are not present on the current research branch. However, the substantive criticism is correct: a return claim should not rely on CAGR against lower-risk unlevered benchmarks. Risk-adjusted and leverage-matched comparisons are mandatory.

### F6 — multiplicity / criterion selected crash behavior: VALID BLOCKER FOR SIGNIFICANCE CLAIMS

The effective research history is larger than the final 32-variant grid. It is not defensible to use 32 as the complete hypothesis count. Because the earlier searches were adaptive and not all hypotheses were preregistered in a machine-readable ledger, a clean project-wide SPA/reality-check p-value cannot be reconstructed honestly after the fact.

Accordingly, no statistical-significance or proven-alpha claim may be made from this research. Future strategy work should use a registered hypothesis ledger and a new untouched evaluation stream rather than pretending the current holdout can be restored.

The 1.75x winner also lies at the tested leverage boundary. That is further evidence that leverage may be the primary return driver and is specifically addressed by matched-leverage controls.

### F7 — holdout too short and now burned: VALID

The 2025-2026 interval is short and contains few monthly decisions. It was useful as a chronological sanity check, but it cannot establish statistical significance. Neighbor diagnostics have also inspected the local holdout surface. It is therefore considered burned and must not support future parameter changes.

### F8 — 10-year QQQ rolling comparison: VALID INTERPRETATION

The prior 1.50x neighbor won only 47.2% of overlapping 10-year windows against QQQ with median excess near zero. Those overlapping windows have low effective independence, but they still contradict any claim of persistent QQQ dominance. SPY-relative robustness is not a substitute for QQQ-relative evidence.

### F9 — cost/financing sensitivity: VALID

The QQQ-relative edge weakened materially under higher cost and borrowing assumptions. Twenty basis points must not be described as universally conservative for a EUR retail investor. The continuous rerun includes higher cost/financing cases while passive QQQ remains zero-cost. Broker-specific FX conversion and financing must be modeled before any deployable claim.

The review's statement that 20bp costs consume 42% of the "claimed gross edge" is mislabeled: the quoted 2.34pp holdout excess was already after the modeled 20bp trading cost, taxes and financing, so it is a net modeled excess, not gross. The broader cost-sensitivity criticism remains valid.

### F10 — margin modeling: VALID BLOCKER FOR CRISIS CLAIMS

Close-only maintenance-margin checks cannot establish survival under intraday broker liquidation rules. The 2006-2009 crisis result must not be used as decisive evidence until high/low or finer-grained data and actual broker house-margin rules are tested. A 30% generic maintenance threshold is a research simplification, not an IBKR/Alpaca guarantee.

## Smaller-item dispositions

- Leveraged USD exposure can create an EURUSD contribution to active-minus-passive returns. FX attribution should be reported separately before claiming strategy alpha.
- FRED DEXUSEU is a research proxy, not the legally authoritative Slovenian filing FX source. This was already disclosed, but the sensitivity must be measured.
- The new deterministic target engine does **not** use `pct_change().fillna(0.0)`; it uses `pct_change(fill_method=None)` and fails on invalid/insufficient history. The review's missing-data hazard remains relevant to older research code, not the frozen target engine.
- The signal timing is intentionally lagged as described above; it is not silently inherited from `lab.py`. Naming/documentation should still be made unambiguous because execution timing is material.

## Required gate to overturn REJECT

1. Publish exact Feb-March 2020 exposure/equity trace and reconcile the reported drawdown.
2. Publish one continuous 2006-2026 EUR/FIFO/dividend-tax path for active, QQQ and SPY.
3. Publish matched-leverage and volatility-matched QQQ controls with risk-adjusted metrics.
4. Publish active realized gains/losses, tax, dividend tax, financing, turnover and trade counts sufficient to reconcile terminal wealth.
5. Reframe or withdraw the volatility-timing mechanism if the matched-control timing residual is not robust.
6. Replace generic close-only margin survival with broker-relevant intraday testing before relying on crisis results.
7. Treat 2025-2026 as burned and make no further parameter selections from it.
8. Make no project-wide statistical-significance claim without a defensible hypothesis ledger/multiple-testing design.

Until all eight gates pass, this strategy remains rejected for deployment and unproven as alpha.
