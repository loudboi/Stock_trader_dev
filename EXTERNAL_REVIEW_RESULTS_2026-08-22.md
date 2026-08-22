# External-review rerun results — volatility-managed QQQ — 2026-08-22

## Controlling verdict

**REJECT.** The continuous after-tax rerun does not support an alpha claim against QQQ.
The frozen parameters remain unchanged and the 2025-2026 interval remains burned.
This branch is research-only and must not be merged into a live/paper deployment path
on the strength of these results.

The external review correctly identified that the prior segmented tax comparisons
were not a fair primary after-tax comparison to a long-term Slovenian buy-and-hold
investor, and that leverage needed to be separated from volatility timing. Two review
findings were subsequently withdrawn after direct verification: the artifacts were
present on the remote research branch, and the claimed impossible 2020 drawdown was
based on an incorrect QQQ price move. Those corrections do not change the controlling
REJECT verdict.

## Frozen rule

No parameter was changed for this rerun:

- QQQ;
- 20-day realized volatility;
- 25% annualized volatility target;
- target gross exposure clipped to 0.50x-1.75x;
- monthly decision cadence;
- 0.20 exposure-point no-trade band;
- one completed-session signal lag relative to decision close;
- execution on the next exchange session in the research simulator;
- 20 bp base trading cost;
- historical effective Fed Funds plus 1.5% base borrowing spread;
- 25% dividend-tax scenario;
- FIFO capital-gains tax approximation in EUR using the repository's Slovenian
  holding-period schedule.

The active tax model remains incomplete: it does not credit eligible same-year loss
offsets or normalized acquisition/disposal expenses. FRED DEXUSEU remains a research
FX proxy rather than the legally authoritative filing source.

## Frozen target-engine parity

The deterministic target implementation was checked against every monthly research
event in the continuous review period.

- events checked: **248**;
- maximum target-exposure error: **0.000e+00**;
- maximum realized-volatility error: **0.000e+00**.

Therefore the research event targets and `bot/vol_managed_targets.py` are in exact
parity for the tested Yahoo history. This does not establish broker-data parity.

## Continuous 2006-2026 after-tax result

The primary tax comparison is one continuous path from 2006-01-03 through 2026-08-20,
not independently restarted 4-5 year folds. Passive benchmarks receive zero
transaction cost. Dividend reinvestments become separate younger tax lots; the
original passive lot can become exempt under the modeled long-holding schedule.

| Metric | Active vol-managed QQQ | Passive QQQ | Passive SPY |
|---|---:|---:|---:|
| CAGR | **15.1989%** | **15.5754%** | 10.6713% |
| Annualized volatility | 27.4306% | 22.6604% | 19.8820% |
| Sharpe, rf=0 | 0.655 | **0.754** | 0.610 |
| Sortino, rf=0 | 0.858 | **1.009** | 0.782 |
| Max drawdown | -45.7987% | -46.4465% | -51.7375% |

Active minus QQQ CAGR is **-0.3765 percentage points**. Active minus SPY remains
+4.5276 points, but SPY is not an adequate primary alpha benchmark for a QQQ-based
strategy. The continuous result therefore reverses the prior segmented headline.

### Tax, financing and activity reconciliation

- active modeled CGT: approximately **EUR 692,344**;
- active modeled dividend tax: approximately **EUR 24,944**;
- active modeled borrowing interest: approximately **USD 143,411**;
- active realized gains: approximately **EUR 2.769 million**;
- active realized losses: approximately **EUR 61,717**;
- active cumulative traded notional / initial capital: approximately **286.38x**;
- active trade operations: **116**;
- modeled forced liquidations: **0**;
- passive QQQ modeled CGT: approximately **EUR 23,847**;
- passive QQQ exempt terminal gain: approximately **EUR 1.716 million**;
- passive QQQ taxable terminal gain: approximately **EUR 133,935**.

The 286.38x figure is cumulative traded notional divided by initial capital, not an
annualized turnover percentage. Because portfolio wealth grows substantially through
the period, this statistic should not be interpreted as 286x annual turnover.

## Leverage attribution

### Realized-exposure-matched control — primary attribution

The active strategy's actual time-weighted realized gross exposure was **1.380307x**.
A constant-target QQQ control was calibrated ex post to reproduce that realized
exposure using the same simulator mechanics. A constant target of 1.454000x produced
mean realized gross exposure of **1.380475x**, a mismatch of only 0.000169x.

| Measure | Active | Realized-exposure-matched control |
|---|---:|---:|
| Mean realized gross exposure | 1.380307x | 1.380475x |
| CAGR | **15.1989%** | **19.3367%** |

The active-minus-matched timing residual is therefore **-4.1378 percentage points of
CAGR** over the continuous sample.

This control is ex-post calibrated and is not a proposed tradable strategy or an
out-of-sample benchmark. It is an attribution device. Under that attribution, the
volatility-timing rule reduced return relative to maintaining the same average gross
exposure.

### Volatility-matched control

The older risk-matching diagnostic chose a constant target of 1.283424x, producing
27.7621% annualized volatility versus the active strategy's 27.4306%.

- volatility-matched CAGR: **16.2258%**;
- active CAGR: **15.1989%**;
- active residual: **-1.0270 percentage points**.

The direction is consistent with the realized-exposure control: the full-sample data
do not show positive return alpha from the timing rule after controlling for risk or
exposure.

## Realized gross exposure by regime

| Period | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| 2006-2009 | 1.2550x | 1.3036x | 0.4533x | 1.8295x |
| 2010-2014 | 1.5068x | 1.6053x | 0.5040x | 1.8850x |
| 2015-2019 | 1.5017x | 1.6744x | 0.7277x | 1.9129x |
| 2020-2024 | 1.2377x | 1.2258x | 0.5004x | 1.8262x |
| 2025-2026, burned | 1.3650x | 1.3749x | 0.5013x | 1.7925x |

For April-December 2020, mean realized exposure was **0.9337x**, confirming that the
strategy materially under-participated in the post-COVID rebound.

Actual exposure can drift outside the 0.50x-1.75x target range between rebalances.
The maximum pre-trade close exposure observed in the full path was **1.9129x on
2015-08-25**. The target cap must therefore not be described as a hard intramonth
leverage cap.

## Crash / rebound attribution

The following event windows are hindsight-defined diagnostics. They are not new
selection folds and must not be used to tune the strategy.

### Global Financial Crisis

| Leg | Active | Exposure-matched control | QQQ | Active minus matched |
|---|---:|---:|---:|---:|
| 2007-10-31 to 2009-03-09 crash | -42.74% | -64.52% | -46.26% | **+21.77 pp** |
| 2009-03-09 to 2009-12-31 rebound | +58.60% | +102.55% | +57.35% | **-43.95 pp** |

The additive log-return timing residual was +0.47842 during the crash and -0.24459
during the rebound, for a **+0.23383 net log residual** across this round trip. In the
GFC episode, crash protection outweighed rebound drag relative to the matched control.

### COVID crash and rebound

| Leg | Active | Exposure-matched control | QQQ | Active minus matched |
|---|---:|---:|---:|---:|
| 2020-02-19 to 2020-03-23 crash | -31.33% | -35.83% | -27.73% | **+4.50 pp** |
| 2020-03-23 to 2020-12-31 rebound | +32.38% | +100.70% | +62.36% | **-68.32 pp** |

The additive log-return timing residual was +0.06783 during the crash and -0.41610
during the rebound, for a **-0.34828 net log residual** across the COVID round trip.
In this V-shaped episode, rebound under-participation overwhelmed the crash benefit.

The evidence therefore does not support the blanket statement that the crash brake
always costs more than it saves, nor the opposite. It shows a regime-dependent
tradeoff: net-positive timing in the GFC diagnostic and strongly net-negative timing
in the COVID diagnostic. Across the full continuous sample, the net effect is
negative relative to the realized-exposure-matched control.

## 2020 trace reconciliation

The exact continuous-path trace resolves the earlier drawdown dispute:

| Date | NAV EUR | Pre-trade exposure | Post-trade exposure | Frozen target |
|---|---:|---:|---:|---:|
| 2020-02-19 | 774,444 | 1.5170x | 1.5170x | 1.7500x |
| 2020-03-02 | 630,166 | 1.5960x | 0.8838x | 0.8569x |
| 2020-03-16 | 510,044 | 0.8561x | 0.8561x | 0.8569x |
| 2020-03-23 | 531,834 | 0.8558x | 0.8558x | 0.8569x |

The 2020 continuous-path drawdown was **-34.1406%**, troughing on 2020-03-16. This is
mechanically consistent with the frozen rule and does not support the withdrawn
"impossible drawdown" finding.

The actual modeled pre-close exposure on 2020-02-28 was **1.6466x**, not the earlier
hand estimate near 1.97x. The conceptual point about exposure drift remains valid,
as demonstrated by the 1.9129x full-path maximum.

## Daily-low margin diagnostic

A stricter diagnostic reconstructed the prior-close position at each next day's raw
QQQ daily low, including debt, daily financing, splits and after-tax dividends.
It is stricter than a close-only test but is **not** a broker-specific intraday
portfolio-margin engine and cannot establish exact liquidation survival.

| Window | Minimum margin equity / market value | Date | Max gross at daily low | Days below 30% / 35% / 40% / 50% |
|---|---:|---|---:|---:|
| GFC diagnostic | 61.19% | 2007-10-04 | 1.634x | 0 / 0 / 0 / 0 |
| COVID diagnostic | 59.22% | 2020-02-28 | 1.689x | 0 / 0 / 0 / 0 |
| Full path | **44.76%** | 2015-08-24 | **2.234x** | 0 / 0 / 0 / 1 |

The generic 30% threshold was not breached at daily lows in this dataset. However,
a 50% house-margin threshold would have produced one breach day, and actual broker
house rules, intraday ordering, gap behavior, concentration add-ons and forced
liquidation logic remain unmodeled.

## Cost / financing sensitivity

Passive QQQ remains zero-cost in these comparisons:

| Active trading cost | Borrow spread | Active CAGR | Excess vs QQQ |
|---:|---:|---:|---:|
| 20 bp | Fed Funds +1.5% | 15.1989% | **-0.3765 pp** |
| 40 bp | Fed Funds +2.0% | 14.3975% | **-1.1779 pp** |
| 80 bp | Fed Funds +3.0% | 12.8012% | **-2.7742 pp** |

The QQQ-relative claim therefore fails even at the base cost assumptions and weakens
further under higher retail frictions.

## Final interpretation

The defensible description is:

> A dynamically levered QQQ strategy with a volatility-sensitive crash brake. The
> brake materially reduced exposure in crisis periods, but its delayed re-risking
> can sacrifice substantial rebound participation. Under one continuous 2006-2026
> Slovenian after-tax scenario, the strategy slightly trailed passive QQQ, had worse
> Sharpe and Sortino, and materially underperformed an ex-post control matched to its
> realized average gross exposure.

This is a legitimate risk-management research result, but it is **not demonstrated
alpha** and does not satisfy the project's objective of reliably beating QQQ
buy-and-hold after tax and costs.

## Remaining limitations / gates

- The active tax model still lacks complete same-year loss offsetting, normalized
  expenses and filing-authoritative FX conversion.
- The continuous tax result is one historical path, not an independent significance
  test.
- The effective project-wide hypothesis count exceeds the final 32-variant family;
  no clean project-wide SPA/reality-check can be reconstructed retrospectively.
- The 2025-2026 holdout is burned.
- Daily Yahoo low tests do not reproduce actual IBKR/Alpaca house-margin or intraday
  liquidation behavior.
- Broker adjusted-data parity has not been established.
- No future parameter change may use the burned holdout as validation.

**Deployment status: rejected.**
