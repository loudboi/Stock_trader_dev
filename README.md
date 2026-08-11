# Stock_trader_dev

This repository contains two separate trading/research projects. They share a repo,
but they do **not** have the same purpose or reliability boundary.

## 1. `alpaca-bot/`

A long-only trend/pullback system plus a collection of historical research tools.
The live Strategy 4 runner supports Alpaca and an EUR-only Interactive Brokers path.
Research modules cover the pullback strategy, risk-based portfolios, trend exposure,
momentum, regime overlays, seasonality, overnight returns, tax/currency scenarios,
and parameter robustness.

Live execution has deliberately conservative invariants: broker read failures are
not interpreted as flat positions, submitted orders are not treated as fills until
broker state confirms them, protective-stop replacement must be confirmed, stale
prices are not substituted for live prices, and a state file cannot silently drop a
managed symbol.

Start here:

```bash
cd alpaca-bot
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

For setup and paper trading, read [`alpaca-bot/SETUP_GUIDE.md`](alpaca-bot/SETUP_GUIDE.md).
For IBKR, read [`alpaca-bot/IBKR_SETUP.md`](alpaca-bot/IBKR_SETUP.md). For Linux
systemd deployment, read [`alpaca-bot/deploy/README.md`](alpaca-bot/deploy/README.md).

### Live-money warning

Paper and live runtimes are deliberately separated. A live endpoint also requires an
explicit `--live` confirmation. Existing broker positions are **not** silently
adopted by a new strategy process; adoption requires `--adopt-existing` so an
operator has to make that choice explicitly.

A backtest or research score is historical evidence under stated assumptions, not
proof of a future edge. The repository models transaction costs approximately and
cannot reproduce every broker fill, market-data outage, tax lot, option spread,
liquidity constraint, or market microstructure effect.

## 2. `gex-lab/`

A separate Gamma Exposure (GEX) research tool built around free option-chain data.
It can screen a watchlist, save point-in-time chain-derived level snapshots, and
backtest the underlying-price response once enough historical snapshots have been
accumulated.

The GEX model is explicitly approximate. In particular, dealer call/put sign is an
assumption, transition levels are reconstructed rather than proprietary vendor
fields, and the backtest measures **underlying P&L rather than option P&L**.
Nightly snapshots are made available only to later sessions in the backtester; a
same-day snapshot is never used to explain that same day's price action.

Quick check:

```bash
cd gex-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest -q
python -m gex_lab.backtest --demo
```

See [`gex-lab/README.md`](gex-lab/README.md) for methodology and limitations.

## Supported Python

CI tests the trading project on Python **3.11 and 3.12**. Use one of those versions
for a reproducible deployment unless CI is expanded to another version first.

## Repository workflow

Before using a research result or deploying a live change:

1. Run the relevant offline tests.
2. Run the full CI suite.
3. Review the exact data source, date window, cost model, currency/tax assumptions,
   and whether a result is in-sample, chronological consistency, or true fit/test
   walk-forward.
4. Paper trade execution changes before using real money.

Runtime state, broker credentials, generated reports, caches, and accumulated GEX
snapshots are local artifacts and should not be committed unless a specific
reproducibility artifact is intentionally curated and scrubbed of secrets.
