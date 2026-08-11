# Setup Guide

This is the shortest supported path from a clean checkout to an Alpaca **paper**
run. Use Python **3.11 or 3.12**; those are the versions exercised by CI.

## 1. Get the repository

```bash
git clone https://github.com/loudboi/Stock_trader_dev.git
cd Stock_trader_dev/alpaca-bot
```

You should now be in the directory containing `config.py`, `bot/`, `tests/`, and
`requirements.txt`.

## 2. Create a virtual environment

macOS/Linux:

```bash
python3.12 -m venv .venv        # use python3.11 if that is your supported interpreter
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install and test

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

For the Interactive Brokers path, also install:

```bash
python -m pip install -r requirements-ibkr.txt
```

A successful test run is a prerequisite, not proof that live execution will behave
identically to the test environment.

## 4. Configure Alpaca PAPER credentials

Copy the template:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Fill in the paper credentials:

```text
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Never commit `.env`. The repository ignore rules exclude it, but you should still
check `git status` before any commit.

## 5. Backtest before running the executor

```bash
python -m bot.backtest_pullback
python -m bot.backtest_pullback --start 2025-02-04 --end 2026-06-24
```

The backtest uses completed daily signals and models later execution. It is still an
approximation: fill ordering inside OHLC bars, transaction costs, market impact,
data provenance, interest on cash, and broker behavior can differ from reality.
Use `--data-source yahoo` for long daily-history research when appropriate; Yahoo
adjusted history is not bar-identical to the live Alpaca feed.

If any requested symbol has no usable data, the command refuses to silently change
the requested universe.

## 6. Start the paper runner

```bash
python -m bot.live_pullback
python -m bot.live_pullback --symbols SPY GLD
```

The runner:

- acts on completed daily bars;
- checks current broker state instead of trusting local state;
- treats broker/API failures as errors, not as a flat position;
- does not mutate local position state merely because an order was accepted;
- waits for the broker position change to confirm the fill;
- uses a broker-resident protective stop where supported and an in-process check as
  an additional path;
- writes restart state and trade/P&L audit files;
- refuses a real-money Alpaca endpoint without `--live`.

### Existing broker positions

An untracked long is **not adopted automatically**. Startup fails with a clear
message. If you have inspected that position and intentionally want this strategy to
own/manage it, rerun with:

```bash
python -m bot.live_pullback --adopt-existing
```

The adopted position is treated conservatively as fully built; the runner cannot
reconstruct historical tranche intent from broker quantity alone.

### Changing the symbol list

Do not remove a symbol from `--symbols` while its state file still manages a live
position or pending order. The runner refuses to start rather than orphaning it.
Flatten/migrate the position deliberately first.

A process lock prevents a second process from opening the same runtime state file.
That is not a substitute for account-level operational discipline: do not assign the
same broker symbol to two independent strategies/processes.

Stop the foreground runner with `Ctrl+C`. Shutdown flushes current state/P&L and
waits briefly for queued alerts.

## 7. Live mode

Switching to a real-money endpoint is a separate operational decision. The executor
requires the `--live` confirmation in addition to the endpoint configuration.
Do not put `--live` into an unattended service until the paper deployment, account,
symbol list, stop behavior, alerting, and restart/reconciliation path have all been
reviewed.

For Linux systemd deployment, follow `deploy/README.md`; its install keeps a real
Git clone so updates are reproducible. For IBKR, follow `IBKR_SETUP.md` as well.

## Troubleshooting

**Missing required env vars** — `.env` must live beside `config.py` and contain the
Alpaca key/secret.

**`module not found: bot`** — run commands from `alpaca-bot/` using the module form,
for example `python -m bot.live_pullback`.

**Stock-data subscription errors** — the free Alpaca stock path uses IEX. A paid SIP
subscription is a separate feed/configuration decision.

**Crypto symbol errors** — Alpaca uses slash-form symbols such as `BTC/USD`.

**Existing-position refusal** — this is intentional. Inspect the broker position;
use `--adopt-existing` only if the strategy should take ownership of it.

**Order remains uncertain/pending** — the broker accepted a request but the position
change was not confirmed. The runner blocks further strategy orders for that symbol
until reconciliation rather than assuming a fill.

**Install problems** — confirm Python 3.11/3.12, activate the venv, upgrade pip, and
rerun the dependency install.

## Research and risk reminder

Historical performance, Sharpe, drawdown, tax scenarios, and parameter sweeps are
model outputs. They are not guarantees, and they are sensitive to data source,
transaction costs, execution assumptions, tax facts, and the chosen sample window.
Paper trading is the minimum operational check before real-money automation.
