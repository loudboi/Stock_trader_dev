# Setup Guide

A start-to-finish walkthrough. Follow it in order. **Do the backtest before the
live bot, and keep the live bot on paper trading until you trust it.**

Estimated time: ~15 minutes.

---

## Step 0 — What you need

- Python 3.9 or newer.
- A free Alpaca account (paper trading is free and uses fake money).
- A terminal (Terminal on macOS/Linux, PowerShell or Command Prompt on Windows).

Check your Python version:

```bash
python3 --version       # macOS/Linux
python --version        # Windows
```

If it prints 3.9+ you're set. If `python3` isn't found, install Python from
https://www.python.org/downloads/ (tick "Add Python to PATH" on Windows).

---

## Step 1 — Get the project onto your computer

Unzip `alpaca-bot.zip` somewhere you'll remember, then open a terminal **inside
that folder**. You should be in the directory that contains `config.py` and the
`bot/` folder. Verify:

```bash
cd path/to/alpaca-bot
ls          # macOS/Linux  -> should list: bot  config.py  requirements.txt  README.md ...
dir         # Windows
```

Everything below is run from this folder.

---

## Step 2 — Create a virtual environment (recommended)

This keeps the bot's packages isolated from the rest of your system.

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Your prompt should now show `(.venv)`. (To leave it later, type `deactivate`.)

---

## Step 3 — Install the dependencies

```bash
pip install -r requirements.txt
```

This installs `alpaca-py` (Alpaca's current SDK), `pandas`, `numpy`,
`python-dotenv`, and `matplotlib`. If `pip` isn't found, try `pip3` or
`python -m pip`.

> Works on modern Python, including 3.13. All Alpaca API calls live in
> `bot/portfolio.py`, so the rest of the project never imports the SDK directly.

---

## Step 4 — Get your Alpaca API keys (paper)

1. Sign up / log in at https://alpaca.markets/.
2. In the dashboard, switch to **Paper Trading** (toggle near the top — it should
   say "Paper" not "Live").
3. Find the **API Keys** panel and click **Generate** (or **Regenerate**).
4. Copy the **Key ID** and the **Secret Key**. The secret is shown only once —
   copy it now.

The paper endpoint is `https://paper-api.alpaca.markets`. Keep it on paper for now.

---

## Step 5 — Create your `.env` file

The repo ships a template called `.env.example`. Copy it to `.env` and fill in
your keys.

**macOS / Linux**
```bash
cp .env.example .env
```

**Windows (PowerShell)**
```powershell
copy .env.example .env
```

Open `.env` in any text editor and paste your keys:

```
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Save it. **Never share this file or commit it to git** — `.gitignore` already
excludes it.

---

## Step 6 — Backtest first

Before risking anything (even paper), see how the strategy would have behaved on
historical data. This also confirms your keys and data access work.

```bash
python -m bot.backtest_pullback                 # daily signal, 4h fills
```

To backtest a specific date window, pass `--start` / `--end` (it automatically
fetches ~320 days of warm-up history before the start so the 200-day MA is valid on
day one):

```bash
python -m bot.backtest_pullback --start 2025-02-04 --end 2026-06-24
python -m bot.backtest_pullback --exec-timeframe 1Hour --months 9 --ema
```

It prints a summary table (trades, win rate, profit factor, max drawdown, Sharpe,
return), flags any strategy with a negative Sharpe, and saves an equity-curve chart
(`backtest_pullback_results.png`). If Alpaca's free data doesn't reach your start
date, the run uses what's available and prints the actual range. If you see a data
error, check Step 8 (troubleshooting).

---

## Step 7 — Run the live (paper) bot

When you're ready to watch it trade with fake money:

```bash
python -m bot.live_pullback                  # paper, default symbols
python -m bot.live_pullback --symbols SPY GLD
```

It will:
- reconcile with any existing position on startup (adopting it as fully built),
- poll the latest price every 60s: act on completed daily bars, fill armed
  entries/adds on a dip, and monitor the `max(5%, 2×ATR)` stop in real time,
- write completed trades to `pullback_trades.csv` and daily P&L to
  `pullback_daily_pnl.csv`,
- keep restart-safe state in `pullback_state.json`.

Stop it any time with **Ctrl+C** — it shuts down cleanly and writes the day's P&L.
It refuses the real-money endpoint unless you pass `--live`.

> **Important — one position per symbol per account.** Don't run two copies of the
> runner on the *same* symbol against the *same* Alpaca account at the same time;
> they'll fight over the position. To change which symbols it trades, edit
> `PULLBACK_SYMBOLS` (and `PULLBACK_UNIVERSE` for new tickers) in `config.py`.

Leave it running in the terminal. To run it unattended later, look into `tmux`,
`screen`, or a `systemd` service (Linux), but get comfortable watching it live
first. (For the EUR-only Interactive Brokers path, see `IBKR_SETUP.md`.)

---

## Step 8 — Troubleshooting

**"Missing required env vars"** — your `.env` isn't being read. Make sure the file
is named exactly `.env` (not `.env.txt`) and lives in the same folder as
`config.py`.

**Stock data / subscription errors on backtest** — free Alpaca data plans use the
IEX feed. `bot/portfolio.py` already requests `feed="iex"`; if you have a paid SIP
subscription you can change that there.

**"module not found: bot"** — run commands from the project root (the folder with
`config.py`) using the `python -m bot.live_pullback` form, not `python bot/live_pullback.py`.

**Crypto symbol errors** — `alpaca-py` expects the slash form (`BTC/USD`), which is
what `config.py` uses (`api_symbol`). If you add a crypto ticker, use the same
slash form (e.g. `ETH/USD`).

**A short order was rejected** — you can't short crypto on Alpaca, and shorting
equities needs a margin account with the asset shortable. The bot logs the rejection
and stays flat; this is expected, not a crash.

**Install problems** — make sure your virtual environment is active and pip is
current: `python -m pip install --upgrade pip`, then re-run
`pip install -r requirements.txt`.

---

## A few honest reminders

- **Keep it on paper.** Switching `ALPACA_BASE_URL` to the live endpoint trades real
  money. Don't do that until you've watched the paper bot behave for a good while.
- **A good backtest is not a guarantee.** Historical results are optimistic relative
  to live trading, and tuning parameters to make a backtest look good is the easiest
  way to fool yourself. Treat the numbers as "not obviously broken," not "proven."
- **This is software, not financial advice.** Automated trading can lose money fast.
  Only ever use money you can afford to lose.

See `README.md` for how each strategy works and the design details.
