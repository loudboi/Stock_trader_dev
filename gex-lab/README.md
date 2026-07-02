# gex-lab — a free, DIY GEX levels tool + rules backtester

A from-scratch attempt at the "Vol Desk" GEX / dealer-positioning idea, built on
**free** option-chain data (yfinance). Two connected pieces:

- **`gex_lab/screen.py`** — computes approximate dealer-GEX levels for a watchlist
  each evening (call wall / +GEX, gamma flip, put mass), prints a screen, and
  **logs a dated snapshot** so you accumulate a real point-in-time GEX history.
- **`gex_lab/backtest.py`** — a mechanical backtest of the "cross above pTrans →
  ride to +GEX" thesis with the full stop framework. It consumes the snapshots the
  screen produces (plus price bars).

**They're designed to connect: the screen is the data collector, the backtester
consumes what it collects.** Free data only gives *today's* chain, so you build
history forward by running the screen nightly.

## ⚠️ Read this before trusting a single number

This is an honest *approximation*, not the real system. Specifically:

1. **Naive dealer-sign assumption.** GEX is computed assuming dealers are long
   calls / short puts. That one assumption drives every level and is genuinely
   uncertain — real vendors (SpotGamma etc.) use far more sophisticated models.
2. **Levels are approximated.** `+GEX` = nearest call-gamma peak above spot,
   `pTrans` ≈ the gamma flip, `nTrans` ≈ the put wall. The real system's pTrans /
   nTrans / COTMC definitions are proprietary and almost certainly differ — which
   is why a live screen may show everything BLOCKED on R:R (my +GEX sits close
   above spot). Tune the level definitions in `gex.py` to taste.
3. **No history = no real backtest yet.** Until you've accumulated snapshots (or
   bought historical option data), `backtest.py` only runs on `--demo` synthetic
   data. It's a working *engine*, not a result.
4. **P&L is on the UNDERLYING, not options.** It tests whether price reaches +GEX
   (the core thesis). Real single-stock option P&L (theta, IV crush, 5–15%
   spreads) is very different and much harsher.
5. **Grade / db_change / dealer-delta-balance are omitted** — they need vendor
   data, so the screen uses only the free-data filters (spot vs pTrans, R:R ≥ 2,
   put-mass cushion ≥ 2%).

## Usage

```bash
pip install -r requirements.txt

# Nightly: screen a watchlist and log a snapshot (builds your dataset)
python -m gex_lab.screen --tickers NVDA AAPL TSLA AMD META MSFT AMZN

# Backtest the engine now on synthetic data (proves it runs)
python -m gex_lab.backtest --demo

# Once you've run the screen for a few weeks, backtest on your REAL history
python -m gex_lab.backtest --snapshots snapshots/
```

## The honest path this enables

1. Run the screen nightly → accumulate free GEX history.
2. Forward-observe the **thesis** (does price actually accelerate to +GEX after a
   pTrans cross?) for $0, logging every signal.
3. **Only if it holds up**, invest in historical option data (ThetaData / ORATS)
   to properly backtest the full system, and/or a vendor feed (SpotGamma, Unusual
   Whales) for real dealer positioning.

Don't skip to step 3, and don't trade real options off the 2-week "100% win rate"
claim — that's a small-sample mirage, the same trap that flatters every unvalidated
backtest.

## Tests

```bash
pip install pytest && pytest -q      # 16 offline tests, no network
```
