# Profitability research — 2026-08-21

This note records the fixed research decisions and results from the profitability pass on `agent/profitability-research`. It is not a promise of future returns, investment advice, or authorization for live trading.

## Bottom line

The repository now has credible historical evidence that a simple long-only diversified strategy family can be profitable after conservative modeled transaction costs. The strongest evidence is not the existing trend-pullback strategy. The pullback family remains historically positive but low-return and is deprioritized as the primary candidate.

The preferred fixed research candidate is an unlevered four-sleeve blend:

- 25% equal-capital 150-day trend exposure, 0% re-entry buffer;
- 25% equal-capital 200-day trend exposure, 1% re-entry buffer;
- 25% 9-month momentum, skipping the most recent month, top 3 positive assets;
- 25% 60-day long-only minimum variance, updated monthly.

Universe: `SPY`, `QQQ`, `GLD`, `TLT`, `EFA`, `EEM`, `IWM`. Any sleeve allocation not assigned to risky assets remains cash. Gross target exposure is capped at 100%; no shorting or leverage is used.

## Methodology

Parameters and blend weights were ranked using development data only:

- 2006-2009
- 2010-2014
- 2015-2019
- 2020-2024

The period from 2025-01-01 through 2026-08-20 was held out from ranking and inspected only after the development ranking was fixed.

The search compared pullback parameters and then simpler strategy families including long/cash trend exposure, momentum rotation, managed-futures-style long-only trend, inverse-volatility, minimum variance, ERC, trend/volatility targeting, and small mean-reversion controls. After the simple families dominated the pullback family, parameter hunting stopped and the fixed leaders were blended using development data only.

Stress tests included 5/10/20 bp modeled costs, a 4% annual risk-free Sharpe assumption, 64 rolling three-year development windows, 20-day block bootstrap resampling of development returns, and leave-one-asset-out holdout tests.

## Pullback result: profitable but weak

The pullback grid showed a broad low-return plateau rather than a strong edge. The best development configurations had minimum regime Sharpe around 0.29 and median Sharpe around 0.35, with median CAGR around 1%. In the untouched 2025-2026 holdout, stronger variants produced roughly 3% CAGR and Sharpe around 0.7. A 20 bp fill-cost stress still left a small positive return, but the opportunity cost versus simpler strategies was large.

Conclusion: retain pullback as a possible low-volatility niche/risk-control strategy, but do not keep tuning it to manufacture a stronger result.

## Simple families

Several simple candidates had positive Sharpe in all four development regimes. Representative development results included:

- trend exposure, MA 150 / 0% buffer: minimum Sharpe 0.65, median Sharpe 0.68, median CAGR 5.67%, worst drawdown -16.26%;
- 9-month momentum / skip 1 / top 3: minimum Sharpe 0.60, median Sharpe 0.75, median CAGR 9.96%, worst drawdown -24.31%;
- 60-day minimum variance: minimum Sharpe 0.54, median Sharpe 1.06, median CAGR 8.51%, worst drawdown -25.32%.

In the untouched 2025-2026 holdout, these remained profitable. This supported blending distinct sources of return rather than tuning the pullback family further.

## Fixed blend results

### Development ranking

The development-ranked #1 blend was the equal four-sleeve candidate described above:

- positive Sharpe in 4/4 development regimes;
- minimum regime Sharpe: 0.78;
- median regime Sharpe: 0.89;
- median regime CAGR: 7.78%;
- worst regime drawdown: about -15.58%.

### Untouched 2025-2026 holdout

At 5 bp modeled costs:

- CAGR: 18.62%;
- total return: 32.00%;
- Sharpe: 1.52;
- max drawdown: -8.70%.

At a deliberately harsh 20 bp modeled cost stress:

- CAGR: 17.78%;
- total return: 30.50%;
- Sharpe at 0% risk-free: 1.46;
- Sharpe at 4% risk-free: 1.13;
- max drawdown: -8.84%.

The stressed yearly slices were also positive:

- 2025 return: 20.78%, Sharpe 2.00;
- 2026 through 2026-08-20: 8.05%, Sharpe 0.94.

### Rolling development windows

Using 64 rolling three-year windows advanced quarterly, all inside 2006-2024 and at 20 bp modeled costs:

- 64/64 had positive CAGR;
- 64/64 had positive Sharpe;
- median CAGR: 6.78%;
- 10th-percentile CAGR: 2.82%;
- minimum observed three-year CAGR: 0.68%;
- median Sharpe: 0.76;
- minimum Sharpe: 0.12;
- worst drawdown: -16.20%.

### Block-bootstrap diagnostic

A 1,000-path 20-trading-day block bootstrap generated five-year paths from development returns only:

- 96.6% of paths had positive five-year CAGR;
- median CAGR: 6.69%;
- 5th percentile CAGR: 0.98%;
- 1st percentile CAGR: -1.77%.

This is a resampling diagnostic, not a probability forecast. It explicitly shows that multi-year losses remain possible.

### Leave-one-asset-out holdout

At 20 bp modeled costs, the fixed blend remained profitable and positive-Sharpe after removing each asset individually. Holdout CAGR ranged from 13.25% when omitting GLD to 24.65% when omitting TLT. This reduces concern that the result depends entirely on one ETF, but does not eliminate universe-selection bias.

## Benchmark comparison

The fixed blend is better supported as a drawdown/risk-adjusted strategy than as a passive-beating alpha claim.

Development 2006-2024 initial-allocation equal-weight buy-and-hold:

- CAGR: 8.95%;
- Sharpe at 4% risk-free: 0.39;
- max drawdown: -40.41%.

Holdout 2025-2026 equal-weight buy-and-hold:

- CAGR: 24.32%;
- Sharpe at 4% risk-free: 1.25;
- max drawdown: -12.37%.

Therefore the fixed blend substantially reduced historical drawdowns and improved long-development risk-adjusted performance, but it did not establish a persistent ability to beat passive holdings on raw CAGR. Recent passive performance was stronger.

## Implementation parity

`bot/blend_targets.py` now translates the fixed four sleeves into causal long-only close-decision target weights. It contains no broker/order code and no leverage. Regression tests enforce:

- weights never go short;
- gross risky target exposure never exceeds 100%;
- unallocated weight is explicit cash;
- minimum-variance targets cannot be changed by future observations;
- next-session target timing is explicit;
- with transaction costs disabled, the combined target-weight engine reproduces the four existing research sleeves to machine precision.

Normal CI passed with 263 alpaca-bot tests on Python 3.11 and also passed on Python 3.12, with the separate GEX suite green.

## Corporate-action data contract

Yahoo research history uses `auto_adjust=True`. Existing Alpaca live historical requests default to raw stock bars because no adjustment is specified. Those two histories are not signal-identical around splits/dividends.

For the blend only, `bot/blend_market_data.py` builds Alpaca daily-bar requests with `Adjustment.ALL` explicitly. The existing reviewed pullback data path is intentionally unchanged. A paper implementation must compare adjusted Alpaca signals against the Yahoo research model before any order engine is enabled.

## Remaining limitations

- Future profitability is not guaranteed; the holdout is only about 20 months.
- The ETF universe was selected by the researcher and is not free of ex-post universe-selection bias.
- Yahoo and Alpaca adjusted data can still differ by vendor/feed even under the same corporate-action policy.
- Modeled costs are not a substitute for real spread, market impact, partial-fill, and latency measurements.
- Taxes are excluded from the strategy-return comparison and depend on account/jurisdiction specifics.
- The bootstrap reuses historical regimes; it cannot represent genuinely new market structure.
- Recent holdout performance benefited materially from GLD, although leave-one-out tests remained positive.
- A target-weight implementation must define calendar/rebalance timing, rounding, minimum trade sizes, cash handling, order attribution, crash recovery, and manual-position policy before paper trading.

## Decision

Do not continue parameter-grid optimization. The current evidence is strong enough to advance the fixed equal-four blend to deterministic paper-only implementation testing. Keep the existing live pullback safety PR separate and unmerged. Do not enable real-money execution until adjusted-data parity, target-to-order parity, paper-broker reconciliation, and operational failure scenarios are tested.