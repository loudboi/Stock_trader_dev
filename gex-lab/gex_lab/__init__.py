"""gex_lab — a free, DIY gamma-exposure (GEX) levels tool + rules backtester.

Computes approximate dealer-GEX levels from free option-chain data (yfinance),
logs nightly snapshots to build a historical dataset over time, and backtests the
"cross above pTrans -> ride to +GEX" thesis on the underlying.

HONEST LIMITATIONS (read gex_lab/README.md):
  - Levels use a NAIVE dealer-sign convention (long calls / short puts). That one
    assumption drives every number and is genuinely uncertain.
  - Only the CURRENT option chain is free, so there is no historical GEX to
    backtest on until you accumulate snapshots (or buy option history).
  - pTrans / nTrans / dealer-delta-balance / the 11 grade rules are approximated
    or omitted; the real system's proprietary definitions may differ.
"""
