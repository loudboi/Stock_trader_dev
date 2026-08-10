# Repository audit repair ledger

This file records the intent of the `agent/fix-repo-audit` repair branch. It is not
a substitute for reviewing the diff or running CI. It exists so an independent
reviewer can verify that the fixes match the original failure modes rather than just
checking whether tests are green.

## Live trading invariants

- Broker/API position-read failures are never represented as a valid flat position.
- Submitted/accepted orders do not mutate position state until broker position state
  confirms execution.
- A persisted pending strategy order is reconciled before an existing broker long is
  classified as an untracked/manual position after restart.
- Protective-stop cancellation must be confirmed before a replacement stop or a
  close that could otherwise create a duplicate sell.
- Alpaca equity and IBKR managed longs use broker-resident protective stops where the
  broker path supports them. IBKR stop orders are GTC.
- A missing live quote is a data stall, not permission to reuse a previous daily
  close as a real-time stop price.
- Existing untracked longs require explicit `--adopt-existing`; shorts are refused by
  the long-only strategy.
- Manual/external changes to a managed broker position are adopted as broker truth,
  re-protected, and conservatively frozen as fully built so stale tranche history
  cannot cause another automatic add.
- A managed symbol cannot silently disappear from a later `--symbols` invocation.
- Paper/live runtimes use separate state/output files and a process lock prevents two
  processes from sharing one runtime state file.
- Per-symbol risk sizing is additionally constrained by managed-book stop-risk and
  gross-exposure limits.
- Trade state/logging retains broker order identifiers and actual fill prices when
  the adapter can recover them.

## Strategy and backtest correctness

- Pullback-to-MA is a two-sided band rather than accepting arbitrary crashes far
  below the MA.
- Volume baseline uses the stated number of bars.
- Later tranches use the stop distance actually protecting the existing position.
- Combined-portfolio same-close decisions are all planned before any later execution
  window is processed.
- An explicit precomputed “no action” cannot be recomputed after another symbol's
  future fill; a dedicated sentinel and regression test enforce this.
- If one OHLC execution bar is compatible with both a new entry and its protective
  stop, the conservative fill-then-stop outcome is used.
- Intraday no-dip fallback timing follows the later live daily evaluation rather than
  an earlier fictional fill.
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
- Vendor outlier handling quarantines an entire unusable OHLCV row and records
  provenance instead of silently replacing only `close` and creating an internally
  inconsistent candle.

## Tax and currency scenarios

The tax modules are scenario research, not filing software. Current legal assumptions
were independently rechecked in August 2026 and must be rechecked again before use.

- Slovenian securities holding-period rates use calendar anniversaries rather than
  `N*365` approximations.
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
- A close-confirmed pTrans crossing enters no earlier than the next session open.
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

## Independent-review focus

An independent reviewer should prioritize:

1. pending/partial broker execution and protective-stop races;
2. restart reconciliation and manual broker-position changes;
3. same-close combined-backtest causality and same-bar execution ambiguity;
4. tax-lot assumptions versus what aggregate return data can actually prove;
5. exact EUR conversion around entry/exit transaction-cost rows;
6. GEX snapshot availability timestamps and close-signal execution lag;
7. accidental degradation of test coverage or docs that claim more than CI/data can
   support.
