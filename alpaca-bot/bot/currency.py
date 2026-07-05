"""
bot/currency.py
================
Every backtest and every tax model in this project so far has quietly assumed
a USD-based investor. The user is Slovenian (EUR-based): SPY/QQQ/GLD/TLT are
USD-denominated, so the REAL return that lands in a EUR-based investor's
pocket is the USD return adjusted for the EUR/USD exchange-rate move over the
same period, not the raw USD return reported everywhere else in this project.
This turns out to be a large enough effect to matter as much as anything else
this project has tested -- see bot/aftertax.py's --currency flag for the
combined (currency + tax) picture, since Slovenian capital gains tax is
naturally computed on the EUR-denominated gain (cost basis and sale price
both converted to EUR at their respective transaction dates), so the correct
order is: convert to EUR first, THEN apply the tax model in bot/taxes.py.

Two currency-exposure models, because it matters whether idle/uninvested cash
sits in USD or gets converted back to EUR:

  unhedged_eur_equity()  Converts a USD return series to a EUR equity curve
                         day by day using the EURUSD rate, weighted by an
                         `invested_frac` series/scalar controlling how much of
                         that day's FX move actually applies -- 1.0 means the
                         whole book is a USD-denominated asset that day (full
                         FX exposure, e.g. buy-and-hold or risk parity, which
                         are always invested), 0.0 means flat/in EUR cash
                         that day (no FX exposure, e.g. trend-exposure while
                         it's sitting out of the market). Pass a constant 1.0
                         for a strategy that's always fully invested in USD
                         assets; pass the actual exposure series for anything
                         that spends real time in cash.

  hedged_eur_equity()    Removes currency risk entirely (as if using
                         EUR-hedged ETF share classes, which are commercially
                         available in Europe, or a rolling FX forward
                         overlay), at a FIXED, pre-chosen annual cost drag
                         (default 1.5%/yr -- a rough approximation of the
                         historical USD-EUR short-rate differential via
                         covered interest rate parity). This is the same kind
                         of fixed-cost convention already used elsewhere in
                         this project (bot.combo's borrow_rate for leverage).

Both take a USD return series and produce a EUR equity curve, plugging
directly into bot.backtest_pullback.compute_metrics.
"""

import pandas as pd

import bot.trend_exposure as te

_ANNUAL = 252


def te_invested_fraction(daily_data: dict, ma_period: int = 200, buffer: float = 0.01) -> pd.Series:
    """The trend-exposure leg in bot.combo.compute_books is the MEAN of each
    symbol's own independent trend-exposure position (not one signal on a
    blended close) -- so the correct "how much of this equal-weighted sleeve
    is actually sitting in USD assets today" fraction is the mean of each
    symbol's own (no-lookahead) exposure series, with the exact same
    ma_period/buffer combo.py uses by default. Used as the invested_frac for
    unhedged_eur_equity when currency-adjusting the TE leg or a blend that
    includes it."""
    per_symbol = {name: te.exposure_series(daily["close"], ma_period, buffer).shift(1).fillna(0.0)
                 for name, daily in daily_data.items()}
    return pd.concat(per_symbol, axis=1).mean(axis=1, skipna=True)


def unhedged_eur_equity(usd_returns: pd.Series, fx_close: pd.Series,
                        invested_frac=1.0, initial: float = 100_000.0) -> pd.Series:
    """fx_close: EURUSD=X closes (USD per 1 EUR) aligned to usd_returns' dates
    (or a superset -- forward-filled and reindexed here). invested_frac: 1.0
    (always FX-exposed) or a 0..1 series (e.g. a trend-exposure position) --
    a day with invested_frac=0 has NO FX move applied that day (idle capital
    assumed converted back to EUR cash)."""
    if usd_returns.empty:
        return pd.Series(dtype=float)
    idx = usd_returns.index
    fx_aligned = fx_close.reindex(idx).ffill().bfill()
    fx_ret = fx_aligned.pct_change().fillna(0.0)
    if isinstance(invested_frac, (int, float)):
        frac = pd.Series(float(invested_frac), index=idx)
    else:
        frac = invested_frac.reindex(idx).fillna(0.0)
    effective_fx_ret = fx_ret * frac
    usd_equity = initial * (1.0 + usd_returns).cumprod()
    eur_equity = usd_equity / (1.0 + effective_fx_ret).cumprod()
    return eur_equity


def hedged_eur_equity(usd_returns: pd.Series, annual_hedge_cost: float = 0.015,
                      initial: float = 100_000.0) -> pd.Series:
    """Approximates a fully currency-hedged position: currency risk is
    removed, at a fixed annual cost drag (covered-interest-rate-parity
    approximation of the historical USD-EUR short-rate differential)."""
    if usd_returns.empty:
        return pd.Series(dtype=float)
    daily_cost = annual_hedge_cost / _ANNUAL
    hedged_returns = usd_returns - daily_cost
    return initial * (1.0 + hedged_returns).cumprod()
