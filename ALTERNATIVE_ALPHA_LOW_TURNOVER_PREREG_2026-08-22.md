# Low-turnover leverage regime test — preregistration — 2026-08-22

This is a second exploratory family after the first 28-variant preregistered batch produced no survivor under strict continuous-tax and risk-matched benchmarking. The 2025-2026 interval remains excluded and is not loaded. Because 2007-2024 data have already been inspected elsewhere in the project, this test is exploratory rather than a clean independent holdout.

## Fixed rule

Assets: QQQ and QLD only.

Review cadence: first trading session of each month, using only prior-session information.

Start in QQQ.

Risk-on observation:
- prior QQQ adjusted close is above its trailing 200-session moving average; and
- prior QQQ 252-session total return is positive.

Risk-off observation:
- prior QQQ adjusted close is below its trailing 200-session moving average; and
- prior QQQ 126-session total return is negative.

Switch QQQ -> QLD only after **two consecutive monthly risk-on observations**.
Switch QLD -> QQQ only after **two consecutive monthly risk-off observations**.
Otherwise retain the current asset. There are no additional thresholds, stop-losses, volatility filters, or parameter variants.

The two-month confirmation and asymmetric 252-session entry / 126-session exit are fixed before results. Their purpose is economic rather than data-mined: reduce taxable turnover while allowing faster de-risking than re-risking.

## Accounting

Primary test: one continuous 2007-01-03 through 2024-12-31 EUR path.

- Yahoo historical Close/Adj Close; historical Close is treated as split-adjusted and split actions are not reapplied.
- Active switches pay 20 bp per sell and buy notional.
- Passive QQQ, passive QLD, and passive QQQ/QLD buy-and-hold blends pay zero transaction cost.
- Net dividends are reinvested in the same ETF after a 25% dividend-tax scenario.
- FIFO Slovenian capital-gains holding-period schedule in EUR using the repository tax helper.
- FRED DEXUSEU is a research FX proxy.
- No same-year loss-offset credit or normalized acquisition/disposal expense credit is granted to active, so those omissions are conservative for active.

## Hard benchmark and decision gate

Calibrate a zero-trading passive QQQ/QLD buy-and-hold blend to the active strategy's realized annualized volatility (excluding only the artificial terminal evaluation liquidation from risk measurement).

The rule passes this exploratory gate only if all are true:
1. active after-tax CAGR exceeds passive QQQ;
2. active after-tax CAGR exceeds the risk-matched static buy-and-hold blend;
3. active remains above passive QQQ at 40 bp switch cost;
4. no parameter is changed after viewing the result.

A pass would justify reproducibility/broker-parity work, not an alpha claim. A failure rejects this rule. The burned 2025-2026 interval cannot rescue or retune it.