# Active volatility-managed QQQ research — 2026-08-22

## Decision

Advance one **active trading** candidate to deterministic paper-target and execution-parity testing:

> **QQQ volatility-managed exposure:** once per month, estimate annualized realized
> volatility from the prior 20 daily total-return-adjusted QQQ returns, using the
> frozen one-session signal lag; set target gross QQQ exposure to `25% / realized
> volatility`, clamp the target to **0.50x–1.75x**, and trade only if target exposure
> differs from current exposure by at least **0.20 exposure points**.

This is not buy-and-hold. It changes QQQ share count and, when target exposure exceeds
1.0x, borrowing dynamically. Exposure is reduced when realized volatility rises and
increased when realized volatility falls.

The candidate is **not approved for live money**. The next stage is deterministic
paper-only parity and execution-risk testing. No merge or live deployment should be
inferred from these research results.

## Why this family was tested

The economic hypothesis is volatility management rather than directional prediction.
A large empirical literature, including Moreira & Muir's *Volatility-Managed
Portfolios*, documents that reducing exposure when realized volatility is high can
improve risk-adjusted performance when expected returns do not rise proportionally
with volatility. That literature is motivation, not validation of this exact QQQ
implementation. Other research has found materially weaker real-time/out-of-sample
results for volatility-managed portfolios, so this project treats the rule as an
empirical candidate that must survive its own holdout and robustness checks.

## Selection discipline

The search was intentionally small. Development data were 2006-2024, split into four
fixed regimes: 2006-2009, 2010-2014, 2015-2019, and 2020-2024. The recent holdout was
2025-01-01 through 2026-08-20 and was not used to rank the grid.

The volatility-management grid contained only 32 economically interpretable variants:

- underlying: SPY or QQQ;
- realized-volatility lookback: 20 or 60 trading days;
- target volatility: 20% or 25% annualized;
- maximum target exposure: 1.50x or 1.75x;
- optional 200-day trend cap.

The ranking objective prioritized the number of development regimes beating SPY,
worst-regime excess CAGR versus SPY, then the corresponding QQQ comparisons. The
winner was fixed before opening the holdout:

`QQQ / 20-day vol / 25% target vol / 1.75x cap / no trend gate`.

A neighboring 1.50x cap also beat both SPY and QQQ in all four development regimes,
which is useful evidence that the result is not unique to one leverage setting.

## Families rejected before this result

### Sector / cross-asset momentum

A compact family using 12-month momentum, 6/12-month momentum, volatility-adjusted
momentum, quarterly/semiannual switching, 1-3 holdings, one-rank hysteresis, optional
200-day absolute trend, and a fixed credit-spread stress gate was rejected. The best
variant beat SPY in only two of four development regimes, had a worst development
excess CAGR of approximately -6.18 percentage points, and materially underperformed
in the recent holdout. No additional thresholds were tuned around that failure.

### Slow trend plus modest leverage

QQQ/SPY slow-trend variants using 200/250-day moving averages, 1.25x/1.50x exposure,
cash/GLD fallbacks, and an optional 12-month confirmation were also rejected. The
best development variant only barely cleared SPY in its weakest regime and then lost
the recent holdout by roughly 10.9 CAGR points versus SPY. A neighboring setting
showed the opposite pattern. That instability was treated as overfitting evidence,
not an invitation for further parameter search.

### Macro overlays

Earlier credit-spread, yield-curve, breadth, and trend gates did not add robust
after-tax return. They increased turnover or reduced exposure at unhelpful times.
They are therefore **not** part of the winning rule. Complexity was removed rather
than retained for narrative appeal.

Point-in-time company-level micro/fundamental factors were not added because this
repository does not currently contain a survivorship-bias-safe historical fundamental
dataset. Backfilling today's constituents or fundamentals would invalidate the test.

## Strict EUR / tax / distribution validation

The final validation rebuilt accounting using raw prices and corporate actions rather
than treating adjusted-close appreciation as capital gain. The scenario assumes a
Slovenian-resident EUR investor and uses:

- initial wealth: EUR 100,000;
- raw Yahoo closes, splits, and cash distributions;
- adjusted closes only for the volatility signal;
- historical EURUSD conversion using FRED DEXUSEU as the research FX proxy;
- FIFO tax lots with acquisition and disposal values translated to EUR on their
  respective dates;
- Slovenian securities capital-gains holding-period rates from `bot/taxes.py`;
- 25% dividend tax before reinvestment/debt repayment;
- no assumed benefit from capital-loss offsets;
- active trading cost: 20 bp per transaction in the base case;
- borrowing cost: historical effective Fed Funds plus a 1.5% annual spread;
- 30% maintenance-margin stress rule.

The FRED FX series is a research proxy, not the exact Bank of Slovenia tax conversion
table. Actual tax reporting must use the legally applicable source/rules.

### Frozen 1.75x development winner

| Period | Strategy CAGR | SPY CAGR | QQQ CAGR | Excess vs SPY | Excess vs QQQ | Strategy max DD |
|---|---:|---:|---:|---:|---:|---:|
| 2006-2009 | 0.53% | -6.08% | -1.74% | +6.60 pp | +2.27 pp | -45.80% |
| 2010-2014 | 23.15% | 15.12% | 18.52% | +8.04 pp | +4.63 pp | -28.49% |
| 2015-2019 | 15.47% | 10.17% | 14.56% | +5.30 pp | +0.90 pp | -30.79% |
| 2020-2024 | 19.01% | 12.57% | 17.17% | +6.44 pp | +1.83 pp | -32.67% |

It beat both SPY and QQQ in **4/4 development regimes**. The minimum development
advantage was +5.30 CAGR points versus SPY and +0.90 points versus QQQ.

### Untouched 2025-2026 holdout

Under the same strict accounting, the frozen 1.75x candidate produced:

- CAGR: **12.89%**;
- total return: **21.84%**;
- max drawdown: **-29.06%**;
- realized CGT charged by the model: approximately EUR 12,504;
- dividend tax charged: approximately EUR 265;
- modeled USD borrowing interest: approximately USD 2,888;
- cumulative traded notional: approximately 8.1x initial capital.

Comparable passive results under the normal 20 bp implementation model were 7.58%
CAGR for SPY and 10.56% for QQQ. The strategy therefore exceeded them by +5.31 and
+2.34 CAGR points respectively.

## Hardened passive benchmark

To make the comparison intentionally unfavorable to the active strategy, a second
validation charged the active strategy its full 20 bp trading cost while giving
passive SPY and QQQ **zero transaction cost**, including zero cost on dividend
reinvestment.

The development-selected 1.75x strategy still beat both passive benchmarks in all
four development regimes:

| Period | Strategy CAGR | Zero-cost SPY | Zero-cost QQQ | Excess vs SPY | Excess vs QQQ |
|---|---:|---:|---:|---:|---:|
| 2006-2009 | 0.53% | -5.98% | -1.64% | +6.51 pp | +2.17 pp |
| 2010-2014 | 23.15% | 15.22% | 18.63% | +7.93 pp | +4.53 pp |
| 2015-2019 | 15.47% | 10.26% | 14.66% | +5.20 pp | +0.80 pp |
| 2020-2024 | 19.01% | 12.66% | 17.28% | +6.34 pp | +1.73 pp |

In the untouched holdout it returned 12.89% CAGR versus 7.85% for zero-cost SPY and
10.83% for zero-cost QQQ, an advantage of +5.04 and +2.06 CAGR points respectively.

The lower 1.50x neighboring cap also beat zero-cost SPY and QQQ in all four
development regimes, but it narrowly trailed zero-cost QQQ in the short recent
holdout (10.67% versus 10.83%). This is one reason the project should not claim that
any arbitrary volatility-managed variant dominates QQQ.

## Rolling-window robustness

For the more conservative fixed 1.50x neighboring cap under full EUR/tax accounting:

- 3-year development windows: beat SPY in 96.9% of 64 windows; median excess +6.33
  CAGR points; worst -1.40. Beat QQQ in 70.3%; median +2.62; worst -4.86.
- 5-year windows: beat SPY in 100% of 56 windows; worst excess +1.75. Beat QQQ in
  75.0%; median +1.27; worst -2.37.
- 7-year windows: beat SPY in 100% of 48 windows; worst excess +2.36. Beat QQQ in
  75.0%; median +1.02; worst -3.32.
- 10-year windows: beat SPY in 100% of 36 windows; worst excess +3.13. QQQ comparison
  was weaker: 47.2% wins with median excess -0.11 and worst -1.71.

The robust claim is therefore much stronger versus broad-market SPY than versus QQQ.
The strategy does **not** dominate QQQ over every possible rolling horizon.

## Cost and financing stress

For the fixed 1.50x neighbor under the strict EUR model:

| Trading cost | Borrow spread over Fed Funds | Dev wins vs SPY | Worst dev excess vs SPY | Dev wins vs QQQ | Holdout excess vs SPY | Holdout excess vs QQQ |
|---:|---:|---:|---:|---:|---:|---:|
| 10 bp | 1.0% | 4/4 | +5.44 pp | 4/4 | +3.50 pp | +0.52 pp |
| 20 bp | 1.5% | 4/4 | +5.09 pp | 4/4 | +3.09 pp | +0.11 pp |
| 40 bp | 2.0% | 4/4 | +4.19 pp | 3/4 | +2.39 pp | -0.57 pp |
| 80 bp | 3.0% | 4/4 | +3.10 pp | 2/4 | +1.01 pp | -1.94 pp |

Again, the SPY edge is substantially more robust than the QQQ edge under severe cost
and financing assumptions.

## Parameter-neighborhood diagnostics

After the winning rule was frozen, nearby settings were tested **only as diagnostics**
and were not used to re-select the strategy.

The 1.75x rule continued to beat SPY in all four development regimes for every tested
perturbation: 15/20/25/30-day volatility windows, 22.5%/25%/27.5% target volatility,
15/20/25-point no-trade bands, and execution delayed by 1, 2, or 4 trading days.
Worst-regime excess versus SPY remained positive in all of those tests.

QQQ was less uniform. The 20-, 25-, and 30-day windows at the frozen 25% target all
beat QQQ in 4/4 development regimes, but the 25- and 30-day variants trailed QQQ in
the recent holdout. Delaying monthly execution by several trading days also reduced
recent performance materially. The baseline first-session implementation was chosen
from development data before the holdout was inspected, but this timing sensitivity
is a real limitation and should be tested in paper execution rather than dismissed.

## Is the result merely leverage?

A separate comparison initialized QQQ at constant 1.75x leverage and held it, charging
similar borrowing/tax assumptions. Constant leverage generated higher returns than
the volatility-timed strategy in several bull regimes, but it was catastrophically
fragile in 2006-2009: approximately -28.95% CAGR, roughly -80.7% max drawdown, and a
modeled forced margin liquidation. The volatility-managed strategy returned +0.53%
CAGR in that regime.

Therefore the mechanism is not a free return increase relative to constant leverage.
It is dynamic risk allocation: the strategy gives up some bull-market upside in
exchange for cutting leverage when volatility becomes dangerous. The relevant
benchmark for the user's goal remains unlevered buy-and-hold, which it beat robustly
in the development regimes and recent holdout.

## Overfitting assessment

Reasons the result is more credible than the rejected families:

1. The family was specified before the holdout and contained only 32 variants.
2. The winner beat both SPY and QQQ in every fixed development regime rather than
   relying on one full-sample CAGR.
3. A neighboring leverage cap also passed all four development regimes.
4. The result survived a complete raw-price/EUR/FIFO/dividend-tax rebuild.
5. It survived an intentionally advantaged zero-cost passive benchmark.
6. SPY outperformance survived every tested parameter and execution perturbation.
7. Macro/trend overlays that did not help were rejected rather than tuned further.
8. No point-in-time-unsafe fundamental dataset was introduced.

Reasons not to overstate it:

1. QQQ rolling-window outperformance is not universal, especially over some 7- and
   10-year windows.
2. The recent holdout is only about 20 months and has now been observed; it must not
   be reused to tune future parameters.
3. Monthly execution timing matters, especially relative to QQQ in the recent period.
4. The strategy still suffered historical drawdowns around 29-46%; leverage makes
   implementation risk consequential.
5. Yahoo/FRED research data are not identical to broker bars, broker financing, or
   the legally required FX/tax reporting source.
6. Borrow availability, margin rules, partial fills, gaps, corporate actions, and
   forced-liquidation behavior need paper-broker validation.
7. Backtests cannot establish future profitability.

## Frozen implementation contract

The target engine must preserve the backtested rule exactly:

- symbol: QQQ;
- lookback: 20 daily adjusted total-return observations;
- annualization: 252;
- signal lag: one completed session inside the decision close;
- target volatility: 25%;
- minimum target exposure: 0.50x;
- maximum target exposure: 1.75x;
- cadence: monthly, based on the completed close immediately before the first actual
  exchange session of a new month;
- no-trade band: 0.20 exposure points;
- no trend, breadth, yield-curve, or credit-spread gate;
- no shorts;
- no target exposure above 1.75x.

`alpaca-bot/bot/vol_managed_targets.py` is the pure, broker-free target implementation.
It intentionally does not submit orders or calculate margin borrowing. Paper execution
must consume this target through a separate fail-closed planner and must use an actual
exchange calendar and explicitly adjusted corporate-action data.

## Next gates before any deployment

1. Normal unit/CI suite green with causality and target-invariant tests.
2. Broker-data parity: Alpaca adjusted-all daily QQQ signals versus the frozen Yahoo
   research signal over a sufficiently long overlap.
3. Paper-only exposure/rebalance planner with explicit buying-power and maintenance-
   margin safety constraints.
4. Paper tests covering partial fills, rejects, restarts, stale prices, corporate
   actions, month boundaries, and manual/unrelated positions.
5. Independent adversarial review of the new target/execution code.
6. A paper observation period with parameters frozen. No holdout-driven retuning.

Until those gates pass, this remains a research candidate, not a live trading system.
