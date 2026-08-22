# Alternative active-strategy research results — 2026-08-22

## Verdict

**No tested alternative is approved as after-tax alpha.**

This batch deliberately kept the already-burned 2025-2026 interval out of selection and strict validation. All primary results below end on 2024-12-31. The work is research-only and must not be merged into a live or paper trading path on the strength of these results.

The strongest recurring lesson is that beating plain QQQ is not a sufficient gate. Strategies that take more Nasdaq leverage can beat QQQ while still losing badly to a passive QQQ/QLD buy-and-hold portfolio matched to their realized risk. Under the modeled Slovenian holding-period tax schedule, frequent realization of active gains is a large structural disadvantage versus long-held passive lots.

## Stage 1 — preregistered family screen

Before results, 28 variants across four economically distinct families were registered. Selection data ended on 2024-12-31; 2025-2026 was not loaded.

Development regimes:

1. 2007-2009
2. 2010-2014
3. 2015-2019
4. 2020-2024

Active strategies paid 20 bp per unit of traded notional; passive QQQ paid zero trading cost. A Stage-1 variant had to beat passive QQQ CAGR in all four regimes.

| Family | Best preregistered variant | Regimes beating QQQ | Worst regime excess |
|---|---|---:|---:|
| Diversified time-series momentum | 126d momentum / 25% target vol / 1.50x cap | 1/4 | -12.2636 pp |
| Asymmetric QQQ trend | 200d exit / 20d re-entry / 1.50x risk-on | 3/4 | -2.3717 pp |
| Panic short-term reversal | -5% 5-day panic / VIX>=30 / 1.25x for 5 sessions | 4/4 | +0.1965 pp |
| QQQ/QLD rotation | 126d positive momentum + above 200d MA => QLD, else QQQ | 4/4 | **+4.9330 pp** |

The preregistered selection rule froze the QQQ/QLD rotation because it had the largest minimum regime excess. Its development excesses versus QQQ were approximately +6.418 pp, +7.183 pp, +4.933 pp and +12.289 pp across the four regimes.

The panic-reversal rule was retained only as a secondary already-preregistered survivor after the primary winner failed strict validation; it was not retuned.

## Frozen QQQ/QLD monthly rotation

Rule: on the first trading session of each month, hold QLD only if QQQ's prior 126-session return is positive and prior QQQ close is above its prior 200-session moving average; otherwise hold QQQ.

### Strict continuous EUR / tax path

A source bug was caught before accepting the first strict result: Yahoo historical `Close` is already split-adjusted, while the initial validator also multiplied QLD share quantities on split events. That double-counted multiple QLD splits and produced implausible returns. The source was corrected and the rule was rerun unchanged.

Corrected continuous 2007-01-03 through 2024-12-31 result:

| Metric | Active rotation | Passive QQQ | Passive QLD |
|---|---:|---:|---:|
| After-tax CAGR | **18.4319%** | 16.8951% | 25.2037% |
| Annualized volatility | 37.0276% | 22.8608% | 43.9490% |
| Sharpe, rf=0 | 0.646 | 0.798 | 0.733 |
| Sortino, rf=0 | 0.793 | 1.064 | 0.951 |
| Max drawdown | -57.7123% | -46.4465% | -80.9605% |

The active rule beat QQQ by +1.5368 pp CAGR at the base 20 bp switching cost, but did so with much higher risk and worse risk-adjusted metrics.

Activity/tax diagnostics:

- 24 QQQ/QLD switches;
- modeled active CGT about EUR 771,700;
- modeled active dividend tax about EUR 8,586;
- realized gains about EUR 3.087 million;
- realized losses about EUR 311,226;
- cumulative traded notional / initial capital about 135.43x (not annualized turnover).

Switch-cost sensitivity versus zero-cost passive QQQ:

- 20 bp: 18.4319% CAGR, +1.5368 pp versus QQQ;
- 40 bp: 17.7771%, +0.8820 pp;
- 80 bp: 16.4782%, -0.4169 pp.

A costless adjusted-USD diagnostic showed a positive timing residual versus a static daily QQQ/QLD exposure mix: rotation CAGR 24.2767% versus 22.2413%, a +2.0354 pp residual. This established that the gross timing signal was not purely an average-leverage artifact, but it did not resolve the tax problem.

### Hard passive risk-matched benchmark

A zero-trading passive QQQ/QLD buy-and-hold portfolio was calibrated ex post to the active strategy's realized volatility, excluding only the artificial terminal evaluation liquidation from risk measurement. It preserves passive long-holding tax treatment and has no discretionary switching cost.

| Measure | Active rotation | Risk-matched passive mix |
|---|---:|---:|
| QLD starting weight | dynamic | 60.90% |
| Risk vol, ex terminal liquidation | 36.8525% | 36.8568% |
| After-tax CAGR | **18.4319%** | **22.9638%** |
| Sharpe, ex terminal | 0.672 | 0.748 |
| Sortino, ex terminal | 0.832 | 0.972 |
| Max DD, ex terminal | -57.7123% | -68.7466% |

Active minus risk-matched passive CAGR: **-4.5319 pp**.

The active rule reduced drawdown versus the passive risk-matched blend, but did not create enough return benefit to overcome switching taxes. Therefore the QQQ/QLD rotation is **rejected as timing alpha**.

## Frozen panic-reversal overlay

Already-preregistered rule: baseline 1.00x QQQ. If the prior five-session QQQ return is at or below -5% and prior VIX is at least 30, increase to 1.25x for five sessions. New episodes begin only after the previous five-session episode ends.

The strict validator uses next-session open execution based only on prior-session information, historical Fed Funds +1.5% borrowing spread, 20 bp trading cost, FIFO EUR tax lots and the same dividend-tax assumptions. 2025-2026 is not loaded.

| Measure | Panic overlay | Passive QQQ |
|---|---:|---:|
| After-tax CAGR | **14.1923%** | **16.8622%** |
| Risk vol, ex terminal | 24.5662% | 22.8593% |
| Sharpe, ex terminal | 0.690 | 0.803 |
| Sortino, ex terminal | 0.897 | 1.070 |
| Max DD, ex terminal | -47.2789% | -46.4465% |

The active overlay trails QQQ by **-2.6698 pp CAGR**. There were 43 panic triggers and 82 trade operations. Modeled active CGT was about EUR 329,045.

A passive QQQ/QLD buy-and-hold mix with only 5.45% initial QLD matched the active strategy's 24.57% volatility and achieved **17.7091% after-tax CAGR**, leaving an active-minus-risk-match residual of **-3.5168 pp**.

The overlay also worsened under higher costs. It is **rejected**.

## Fixed low-turnover leverage regime

Because the first two strict failures showed large tax realization drag, one additional single-rule family was preregistered before its result. This is exploratory rather than an independent holdout because the broader 2007-2024 history had already been inspected elsewhere in the project.

Rule:

- start QQQ;
- monthly review;
- switch QQQ -> QLD after two consecutive monthly observations where QQQ is above its prior 200-session MA and prior 252-session return is positive;
- switch QLD -> QQQ after two consecutive monthly observations where QQQ is below its prior 200-session MA and prior 126-session return is negative;
- otherwise retain the current ETF;
- no parameter variants.

The rule generated only 13 switches through 2024.

| Measure | Low-turnover active | Passive QQQ | Passive QLD |
|---|---:|---:|---:|
| After-tax CAGR | **19.4213%** | 16.8951% | 25.2037% |
| Risk vol, ex terminal | 37.7917% | 22.8593% | 43.9520% |
| Sharpe, ex terminal | 0.683 | 0.803 | 0.735 |
| Sortino, ex terminal | 0.847 | 1.070 | 0.954 |
| Max DD, ex terminal | -60.7941% | -46.4465% | -80.9605% |

It remained above QQQ even with high switch costs:

- 20 bp: 19.4213%, +2.5262 pp versus QQQ;
- 40 bp: 19.0531%, +2.1580 pp;
- 80 bp: 18.3201%, +1.4250 pp.

However, a 66.15% QLD / 33.85% QQQ passive buy-and-hold mix matched its 37.79% volatility and produced **23.3077% after-tax CAGR**, leaving an active-minus-risk-match residual of **-3.8863 pp**. The preregistered hard gate therefore failed. This rule is **rejected**.

## Interpretation

Across these experiments, three different mechanisms repeatedly appear:

1. **Gross leverage can make an active strategy beat QQQ.** This is not alpha by itself.
2. **Some timing rules improve drawdown or pre-tax path shape.** The monthly QLD rotation had a positive costless timing residual, and the low-turnover regime materially avoided some QLD drawdowns.
3. **The modeled Slovenian tax structure strongly rewards long holding periods.** Passive risk-matched QQQ/QLD portfolios retain old tax lots while active switches repeatedly realize gains. In every strict comparison so far, that structural tax advantage has been larger than the timing benefit.

Therefore the next useful strategy family should not merely find a better switch between QQQ and a levered Nasdaq exposure. It should either:

- generate expected-return alpha at similar risk without requiring frequent liquidation of appreciated core holdings; or
- use a separate overlay whose economics can be modeled without resetting the holding period of the long-term core position.

Any such next family needs its own legal/tax treatment and historical data model before it can be compared fairly.

## Limitations

- The tax engine remains a conservative research approximation: it does not credit eligible same-year loss offsets or normalized acquisition/disposal expenses.
- FRED DEXUSEU is a research FX proxy, not the legally authoritative filing conversion source.
- The project-wide number of researched ideas is now large; no retrospective p-value can erase that multiplicity.
- QLD contains embedded leverage, financing and fund expenses in its observed return series.
- Yahoo historical data are adequate for screening/research but broker-data parity has not been established.
- The 2025-2026 interval is burned from prior project work and was intentionally not used in this alternative-strategy selection/validation batch.

## Status

**No deployment. No merge. No alpha claim.** PR #6 remains a research branch.