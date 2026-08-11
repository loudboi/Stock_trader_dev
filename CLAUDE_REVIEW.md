# Independent review brief

Review PR #1 (`agent/fix-repo-audit` -> `main`) as if you do **not** trust the repair
ledger or the existing tests. Do not make changes on the first pass. Produce a
prioritized list of concrete defects, regressions, assumptions, missing tests, and
user-facing inconsistencies, with file/function references and a short explanation
of how each issue can fail in practice.

The repair branch was created from an earlier repository audit. `AUDIT_FIXES.md`
records the intended remedies, but treat every statement in it as a claim to verify,
not evidence that the implementation is correct.

## Highest-priority review areas

### 1. Live broker/order/state semantics

Audit both Alpaca and IBKR paths end to end:

- API/read failure vs genuine flat position;
- submitted/accepted vs partially filled vs fully filled vs cancelled/rejected;
- market-order timeout/cancellation behavior;
- partial buy and partial close reconciliation;
- restart with a persisted pending order;
- exact broker order/fill attribution;
- protective-stop cancellation/replacement races and double-sell risk;
- protection of a remainder after partial close;
- existing untracked long adoption and external short refusal;
- manual broker quantity/average-price changes;
- process/state locking and symbol-list changes;
- paper/live state separation and custom IBKR port classification;
- broker-resident IBKR stop lifecycle and reconnect behavior;
- account-currency and stale-data fail-closed behavior.

Try to construct failure timelines rather than reviewing each function in isolation.
Pay particular attention to whether a position change is ever used as stronger proof
of order completion than the broker order state actually supports.

### 2. Backtest causality and execution fidelity

Verify that no future execution result can affect another same-close decision or
position size. Specifically inspect:

- `run_combined` same-timestamp planning;
- the explicit no-action sentinel;
- limit/fallback timing relative to the live runner;
- entry/stop ambiguity inside one OHLC bar;
- daily trend exits and execution timestamps;
- portfolio risk/gross capacity planning across simultaneous symbols;
- buy-and-hold common-start logic for late-inception assets;
- equity timestamps, initial NAV, total return, Sharpe, and max drawdown;
- session/DST anchoring of resampled equity bars;
- whether any parameter sweep/consistency fold shares a boundary observation.

Call out optimistic assumptions even if they are documented.

### 3. Research portfolio math

Re-derive the important portfolio constructions rather than trusting names:

- constrained long-only minimum variance;
- ERC/inverse-vol weights;
- leverage/vol-target turnover and financing;
- fixed-capital behavior when an asset is missing/closed;
- momentum selection bounds and dollar neutrality;
- dynamic allocation signal lags and missing reference data;
- seasonality and overnight timing/costs;
- outlier quarantine and OHLC consistency;
- explicit cash/risk-free assumptions.

Look for hidden pandas behavior such as implicit `skipna` reallocation, forward or
backward filling that changes the information set, object-dtype promotion, or a
first return being silently dropped.

### 4. Tax and FX scenario integrity

Do not treat these modules as filing software. Verify internal consistency and state
where aggregate return data cannot support a legal/tax-lot conclusion.

Review:

- calendar holding-period boundaries;
- same-year loss treatment vs any accidental carry-forward;
- final partial-year realization;
- binary strategy entry/exit timing around tax anniversaries;
- core/satellite separation;
- current-vs-proposed crypto scenario labeling;
- causal EURUSD alignment;
- exact mixed EUR-cash/USD-asset recurrence;
- transaction costs on zero-exposure entry/exit rows;
- hedging cost exposure;
- whether tax is applied before/after currency conversion consistently with the
  stated scenario.

If checking current Slovenian law, use current primary FURS/PISRS/parliamentary
sources rather than relying on repository prose.

### 5. GEX point-in-time research

Verify the real-history path, not only the synthetic demo:

- exact snapshot timestamp storage and same-date reruns;
- conservative availability of a snapshot only to later sessions;
- timezone-safe `merge_asof` behavior;
- expiry of stale levels;
- close-confirmed crossing -> next-session entry;
- gap cancellation;
- target/stop ambiguity;
- BSM risk-free/dividend inputs and snapshot reproducibility;
- dealer-sign/model caveats;
- underlying P&L vs option P&L labeling;
- removal of cross-project `sys.path` coupling.

### 6. Tests, CI, docs, and deployment

Check that tests verify the safer behavior rather than merely being edited to pass.
Look for important untested branches, especially errors and partial executions.

Also verify:

- no temporary one-shot repair workflow remains in the final branch;
- CI tests Python 3.11/3.12 for `alpaca-bot` and 3.12 for `gex-lab`;
- Actions/dependencies are bounded to intended versions;
- setup/deployment commands actually match repository layout;
- the documented Git update path contains `.git` metadata;
- live flags/adoption/custom-port instructions match code;
- README language does not present historical backtests as a proven edge;
- generated runtime files, credentials, caches, and GEX snapshots are ignored or
  handled deliberately;
- `REPRODUCIBILITY.md` is sufficient to reproduce a serious research result.

## Desired output

Return findings in severity order: **Critical / High / Medium / Low**. For each,
include:

1. exact file/function or relevant diff location;
2. concrete failure scenario;
3. why current tests do or do not catch it;
4. the smallest safe correction you would recommend.

Then provide a separate list of areas you inspected and found internally coherent.
Do not approve the PR merely because CI is green, and do not repeat
`AUDIT_FIXES.md` as your conclusion without independently verifying the code.
