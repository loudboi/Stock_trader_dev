# Repository audit repair ledger

This file records the intent of the `agent/fix-repo-audit` repair branch. It is not
a substitute for reviewing the diff, broker paper-testing, or running CI. It exists
so an independent reviewer can verify that fixes match concrete failure modes rather
than merely checking whether tests are green.

## Live trading invariants

- Broker/API position-read failures are never represented as a valid flat position.
- Submitted/accepted orders do not mutate strategy position state until the exact bot
  order reaches a terminal broker state and its filled quantity can be attributed.
- A broker position change by itself is never proof that a pending bot order filled.
- Before a market submission, an intent-to-submit record is persisted. If the broker
  outcome is ambiguous because the response is lost or the process dies, the symbol
  stays blocked for manual reconciliation instead of automatically resubmitting.
- A persisted pending strategy order is reconciled before an existing broker long is
  classified as an untracked/manual position after restart.
- Unresolvable old pending orders generate a critical reconciliation alert rather
  than leaving an unexplained broker long silently unmanaged.
- Protective-stop cancellation must be terminally confirmed before a replacement
  stop or a market close can create another SELL.
- If the broker is already flat, state cannot forget a protective SELL unless its
  cancellation or other terminal state is confirmed. This closes the orphan-stop
  paths found by the independent review.
- Alpaca equity and IBKR managed longs use broker-resident protective stops where the
  broker path supports them. Initial non-crypto stop-placement failure is a critical
  operator-visible condition; IBKR stop orders are GTC.
- IBKR `Inactive` is treated as a terminal non-working order state, and the adapter's
  `already-flat` close sentinel normalizes to a terminal zero-fill state so it cannot
  become a permanently unresolved fake pending order.
- Failed cancellation attempts are retryable after a short process-local throttle;
  stale persisted cancellation intent does not suppress retries after restart.
- A missing live quote is a data stall, not permission to reuse a previous daily
  close as a real-time stop price. Crypto soft-stop evaluation occurs before the
  daily-history gate because crypto has no broker-resident stop in this runner.
- Existing untracked longs require explicit `--adopt-existing`; shorts are refused by
  the long-only strategy.
- Manual/external changes or overlap are marked as requiring manual reconciliation,
  re-protected conservatively, and frozen against additional strategy pyramiding.
  Discretionary strategy closes are blocked while ownership is ambiguous.
- A managed symbol cannot silently disappear from a later `--symbols` invocation.
- Paper/live runtimes use separate state/output files and a process lock prevents two
  processes from sharing one runtime state file.
- Per-symbol risk sizing is additionally constrained by managed-book stop-risk and
  gross-exposure limits. Unresolved pending BUY quantities reserve gross/risk capacity.
- Trade state/logging retains exact broker order identifiers and exact fill prices
  when recoverable. An external/manual close is not assigned a realized price from a
  generic nearby SELL; only an exact protective stop confirmed filled can supply that
  exit P&L. Otherwise exit price/P&L remain blank and daily realized P&L is untouched.
- Live state is schema version 3. Future versions, malformed buckets, malformed
  pending records, and older/versionless files containing managed trading state fail
  closed and require explicit migration/reconciliation. Empty legacy state can be
  upgraded safely.
- Live warmup configuration is shared through `config.py`; the live runner no longer
  imports the plotting/backtest module solely to obtain a constant.

## Strategy and backtest correctness

- Pullback-to-MA is a two-sided band rather than accepting arbitrary crashes far
  below the MA.
- Volume baseline uses the stated number of bars.
- Later tranches use the stop distance actually protecting the existing position.
- Combined-portfolio same-close decisions are all planned before any later execution
  window is processed.
- An explicit precomputed “no action” cannot be recomputed after another symbol's
  future fill; a dedicated sentinel and regression test enforce this.
- Intraday execution windows begin only in the session after the close that produced
  the signal. Bars from the signal's own session cannot fill the order.
- If the next-session limit attempt does not fill, the later market fallback
  re-validates trend state before entering, matching the live runner's daily update.
- If one OHLC execution bar is compatible with both a new entry and its protective
  stop, the conservative fill-then-stop outcome is used.
- US-equity 4-hour bars are anchored to the regular session with DST-aware exchange
  timezone handling.
- Buy-and-hold comparisons begin only once all requested sleeves are investable.
- The equity metric retains the true starting NAV; the first modeled return is not
  silently dropped.
- Sharpe's annual risk-free rate is explicit (`--risk-free`), with impossible
  `<= -100%` inputs rejected.
- True parameter walk-forward in `bot/sweep.py` uses disjoint in-sample and
  out-of-sample boundaries.

## Research methodology

- Research-only equity symbols can be resolved without being predeclared live
  instruments, while missing requested data causes the command to refuse a silently
  changed universe.
- Long-only minimum-variance weights use a constrained optimizer rather than an
  unconstrained solution followed by clipping.
- Turnover costs on scaled/levered books are computed from actual resulting weights.
- Cross-sectional momentum validates fractions/counts so long and short selections
  cannot overlap accidentally.
- “walk” compatibility modes that do no per-fold fitting are labeled chronological
  consistency; true fit/test walk-forward remains a distinct tool.
- Turn-of-month calendar exposure is applied to the actual intended last/first
  trading-day window.
- Overnight/intraday books do not charge a round trip before a trade can exist.
- Fixed-capital multi-asset books keep missing-market capital idle instead of
  implicitly reallocating it through `mean(skipna=True)`.
- After-tax multi-sleeve equity begins only once every requested sleeve has an actual
  value; staggered inception cannot fabricate a capital jump or drawdown.
- Vendor outlier handling quarantines an entire unusable OHLCV row and records
  provenance instead of silently replacing only `close` and creating an internally
  inconsistent candle.

## Tax and currency scenarios

The tax modules are scenario research, not filing software. Current legal assumptions
were independently rechecked in August 2026 and must be rechecked again before use.

- Slovenian securities holding-period rates use calendar anniversaries rather than
  `N*365` approximations for the default legal scenario.
- Compatibility/custom `schedule=` arguments are now validated and actually honored;
  custom duration schedules no longer silently produce baseline results.
- Ordinary securities loss handling does not assume a generic future-year
  carry-forward; aggregate returns also cannot enforce all same-year lot and
  30-day replacement-capital restrictions.
- Final partial years are not automatically treated as realized merely because a
  backtest stops midyear.
- Trend-exposure securities are taxed on modeled holding exits rather than forcing
  the entire sleeve through a coarse annual-realization proxy.
- Current FURS non-business-individual virtual-currency treatment and the separate
  proposed 25% crypto regime are represented as distinct scenarios; the proposal is
  not described as enacted law.
- EUR/USD alignment is causal: known FX can be forward-filled but a future first FX
  quote is never backfilled into earlier dates.
- Partial USD exposure uses an exact EUR-cash/USD-asset wealth recurrence and
  preserves transaction-cost rows even when end-of-row market exposure is zero.

## GEX lab

- Every new snapshot stores an exact UTC as-of timestamp and uses a unique timestamped
  filename; reruns on one date do not overwrite prior observations.
- A snapshot is not usable to explain its own trading date. Legacy date-only nightly
  snapshots are also made available only to later sessions.
- Snapshot/price joins are UTC-safe and use point-in-time `merge_asof` semantics;
  stale levels expire.
- A close-confirmed pTrans crossing enters no earlier than the next session open, and
  a next open at/below the locked pTrans invalidates the pending entry.
- A daily bar compatible with both target and price stop uses the adverse stop result
  rather than automatic target-first profit.
- Black-Scholes-Merton risk-free and dividend-yield assumptions are explicit and
  stored in snapshot output.
- The backtester has a local Yahoo loader and no longer mutates `sys.path` to import
  the other project.
- Unsupported price sources fail explicitly rather than being ignored.

## User/deployment/reproducibility

- Root and component READMEs explain which code is live-capable versus research-only
  and no longer hard-code historical performance as a permanent “proven edge.”
- The VPS install keeps an actual Git clone, so the documented update command works.
- Setup/deploy docs align with CI-supported Python 3.11/3.12 and current broker-stop,
  adoption, custom-IBKR-port, and state-safety behavior.
- Runtime/development dependencies have major-version ceilings; exact historical
  reproduction additionally records `python -m pip freeze`.
- `REPRODUCIBILITY.md` defines the minimum code/environment/data/assumption record for
  research and deployments.
- CI uses current Node-24 GitHub Action releases and emits a readable pytest failure
  tail as a GitHub annotation.
- One-time repair workflows used during the review were removed after their tested
  changes landed; `.github/workflows/ci.yml` is the only retained workflow.

## Independent review remediation

The independent Claude pass found real blockers rather than merely confirming the
first audit. The repair branch specifically added regression coverage for:

1. failed protective-stop cancellation during broker-flat teardown;
2. the pending-BUY/broker-flat stop race;
3. next-session intraday backtest causality and fallback trend re-validation;
4. live-state version/type/pending-record validation;
5. IBKR rejected (`Inactive`) and `already-flat` terminal handling;
6. unresolved restart order history and operator alerts;
7. ambiguous submit outcomes and duplicate-submit prevention;
8. pending-order portfolio capacity reservation;
9. crypto soft-stop operation during daily-history stalls;
10. exact-stop-only external exit P&L attribution;
11. custom tax-schedule sensitivity; and
12. staggered-inception after-tax sleeve aggregation.

One review claim is intentionally not implemented verbatim: IBKR API `orderId` is
client-scoped, but the review's specific premise that a Gateway restart necessarily
reuses prior IDs conflicts with IBKR's documented persistent order-id sequence.
Cross-client/order-ownership behavior therefore remains an integration-test and
operator-configuration concern rather than being “fixed” by inventing a different
identifier contract without broker evidence. `permId` remains the preferable stable
cross-client identifier if this project later supports multi-client ownership.

## Remaining validation gates

A green unit-test/CI suite is not evidence of live-money readiness. Before any real
capital deployment, run the documented paper-broker scenarios for both adapters,
including delayed/rejected/partial orders, manual overlap, stop-trigger while offline,
failed cancellation/replacement, restart with pending orders, stale quotes, untracked
longs, external shorts, and the reconcile-to-close already-flat race. Tax assumptions
must be rechecked against current primary legal sources at the time results are used.
