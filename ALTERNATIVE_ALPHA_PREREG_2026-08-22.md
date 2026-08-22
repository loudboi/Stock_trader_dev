# Alternative alpha research preregistration — 2026-08-22

## Objective

Find a genuinely active, implementable strategy that can beat passive QQQ after realistic costs and Slovenian tax treatment without reusing the burned 2025-2026 interval for selection.

The previously rejected volatility-managed QQQ family is not retuned here.

## Stage 1: development-only family screen

Data used for selection ends on 2024-12-31. The 2025-2026 interval is not loaded by the screen.

Development regimes:

1. 2007-01-01 through 2009-12-31
2. 2010-01-01 through 2014-12-31
3. 2015-01-01 through 2019-12-31
4. 2020-01-01 through 2024-12-31

Stage-1 prices are Yahoo adjusted closes in USD. Passive QQQ has zero trading cost. Active strategies pay 20 bp per unit of traded notional. Synthetic exposure above 1.0x pays prior-day effective Fed Funds plus 1.5% annualized. QLD uses its observed ETF total-return series and therefore receives no extra synthetic borrowing charge.

No Stage-1 result is an after-tax result. A survivor must be rerun using a continuous EUR tax-lot path before any alpha claim.

## Family A — diversified long-only time-series momentum

Universe fixed before results: QQQ, SPY, IWM, EFA, EEM, IEF, TLT, GLD, DBC.

At the first session of each month:

- compute each asset's own trailing return using 126 or 252 sessions, with the signal ending before the execution session;
- hold only assets with positive own momentum;
- equal-weight selected assets before portfolio scaling;
- estimate the selected basket's 63-session covariance;
- scale the basket toward 20% or 25% annualized ex-ante volatility;
- cap total gross exposure at 1.25x or 1.50x;
- otherwise hold cash.

Grid: 2 momentum lookbacks × 2 target vols × 2 leverage caps = 8 variants.

Economic thesis: trend following has evidence across asset classes and may diversify equity crash/rebound timing rather than relying on QQQ volatility alone.

## Family B — asymmetric QQQ trend exit / fast re-entry

Weekly decisions, using only information available before the execution session.

State machine:

- risk-on exposure is 1.25x or 1.50x QQQ;
- risk-off exposure is fixed at 0.50x QQQ;
- exit risk-on after QQQ closes below its 150- or 200-session moving average;
- while risk-off, re-enter when QQQ closes above its prior 20- or 50-session high.

Grid: 2 exit windows × 2 re-entry breakouts × 2 risk-on exposures = 8 variants.

Economic thesis: the prior volatility rule failed partly because it re-risked slowly after V-shaped crashes. This family deliberately separates the exit and re-entry mechanisms.

## Family C — panic short-term reversal overlay

Baseline exposure is 1.00x QQQ. A temporary leveraged state is triggered when, based only on prior-session information:

- QQQ's trailing 5-session return is at or below -5.0% or -7.5%; and
- VIX is at least 30.

The temporary exposure is 1.25x or 1.50x for 5 or 10 sessions, then returns to 1.00x unless a new trigger occurs after the prior episode ends.

Grid: 2 panic thresholds × 2 holding periods × 2 temporary exposures = 8 variants.

Economic thesis: short-term reversal/liquidity-provision returns have historically strengthened during high-volatility market stress. This is deliberately contrarian rather than trend-following.

## Family D — QQQ / QLD risk-on rotation

At the first session of each month, hold QLD only when both of the following prior-session conditions are true; otherwise hold QQQ:

- QQQ trailing 126- or 252-session return is positive;
- QQQ is above its 150- or 200-session moving average.

Grid: 2 momentum windows × 2 moving-average windows = 4 variants.

Economic thesis: obtain conditional Nasdaq leverage through the traded 2x ETF rather than broker borrowing, while remaining invested in QQQ during risk-off periods instead of moving to cash.

## Selection rule

There are 28 preregistered variants total. Rank each variant by its minimum annualized CAGR excess over passive QQQ across the four development regimes.

A variant is eligible for Stage 2 only if it beats passive QQQ CAGR in all four development regimes. If none pass, reject the entire batch without opening 2025-2026 data.

If one or more pass, freeze the single variant with the highest minimum regime excess. Ties are broken by lower full-development max drawdown, then lower turnover. No parameter may be changed after this freeze.

## Stage 2 gates for a survivor

Before viewing 2025-2026 performance:

- one continuous 2007-2024 EUR tax-lot simulation;
- passive QQQ with zero transaction cost and long-holding tax treatment;
- same-year loss-offset sensitivity and normalized-expense sensitivity if practical;
- realized exposure, turnover, financing, and tax reconciliation;
- constant-risk / constant-exposure attribution where applicable;
- cost and financing stress;
- broker-data signal parity where applicable.

Only after those gates are fixed may the already-burned 2025-2026 interval be shown as descriptive evidence. It cannot select or retune the strategy.