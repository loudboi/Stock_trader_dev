# gex-lab — point-in-time GEX research with free option-chain data

`gex-lab` is a research tool, not an option-trading engine. It has two connected
parts:

- `gex_lab.screen` fetches current yfinance option chains, computes approximate GEX
  levels, prints a watchlist screen, and saves a timestamped point-in-time snapshot.
- `gex_lab.backtest` combines accumulated snapshots with daily underlying prices and
  tests a mechanical pTrans-cross / +GEX-target thesis.

Free option-chain sources do not provide the historical point-in-time chains needed
for a clean historical GEX study. The intended workflow is therefore to accumulate
snapshots going forward or substitute a licensed historical chain dataset later.

## Important model limitations

1. **Dealer sign is assumed.** Calls are assigned positive dealer gamma and puts
   negative dealer gamma. Real dealer positioning is not observable from free OI
   alone.
2. **Transition levels are reconstructions.** `pTrans` uses the modeled gamma flip,
   `nTrans` uses the put wall, and `+GEX` uses a positive gamma concentration above
   spot. These are not proprietary vendor definitions.
3. **BSM assumptions are inputs.** The screen exposes `--risk-free` and
   `--dividend-yield`. Both are stored in snapshots. A fixed rate/dividend assumption
   can materially change gamma estimates for longer-dated options.
4. **Underlying P&L only.** The backtest does not model option delta/gamma P&L,
   implied-volatility changes, theta, bid/ask spreads, assignment/exercise,
   liquidity, or commissions.
5. **Daily bars do not reveal intraday ordering.** When a bar is compatible with
   both the locked target and a price stop, the backtest takes the adverse stop
   result rather than automatically awarding a target win.
6. **Close signals cannot fill at the same close.** A crossing confirmed from a
   daily close enters, at earliest, at the next available session open.
7. **Snapshots cannot explain their own day.** A nightly snapshot is not made
   available to the backtest until a later calendar day/session. Legacy date-only
   snapshots are treated the same conservative way.
8. **Snapshot levels expire.** The real-history loader does not carry an old GEX
   surface forward indefinitely when collection stops.

## Install and test

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest -q
```

Python 3.11/3.12 are the recommended repository versions.

## Screen and collect history

```bash
python -m gex_lab.screen --tickers NVDA AAPL TSLA AMD META MSFT AMZN
```

Optional BSM inputs:

```bash
python -m gex_lab.screen --risk-free 0.04 --dividend-yield 0.00
```

Each run creates a unique file such as:

```text
snapshots/gex_2026-08-10T201530.123456Z.csv
```

The CSV includes `asof_utc`, model rates, and the dealer-sign convention. Rerunning
on the same date does not overwrite the earlier file. `snapshots/` is ignored by Git
by default because it is runtime/research data; back it up separately if it matters.

## Backtest

Engine-only synthetic demo:

```bash
python -m gex_lab.backtest --demo
```

Point-in-time accumulated history:

```bash
python -m gex_lab.backtest --snapshots snapshots/
```

The only built-in price source is currently Yahoo. Passing an unsupported source is
an error rather than being silently ignored.

## Interpreting results

A profitable underlying-price simulation is, at most, evidence that the modeled
levels may contain useful directional/target information in the collected sample.
It is not evidence that an option implementation is profitable. Before spending
money on a vendor or trading options, accumulate enough point-in-time observations
to evaluate different regimes, inspect losing trades, and test the same hypothesis
with realistic option-chain history and execution costs.
