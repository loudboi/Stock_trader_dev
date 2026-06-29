# Strategy 4 — Phased Trend-Pullback Bot

A long-only trend strategy that builds positions in **30/30/40 tranches**, decides
on **completed daily bars**, and executes in real time with a volatility-adaptive
stop. It runs on Alpaca, and there's a parallel EUR-only path on Interactive
Brokers (see `IBKR_SETUP.md`).

## File structure

```
alpaca-bot/
├── config.py                       # instruments, the pullback universe, risk setting
├── .env.example                    # copy to .env and add your keys
├── requirements.txt                # Alpaca path
├── requirements-ibkr.txt           # extra dep for the IBKR path
├── requirements-dev.txt            # pytest (tests / CI)
├── bot/
│   ├── indicators.py               # SMA, EMA, ATR (pure functions)
│   ├── risk_manager.py             # quantity rounding
│   ├── portfolio.py                # the only module that talks to Alpaca
│   ├── notifier.py                 # optional outbound alerts (webhook / Telegram)
│   ├── live_pullback.py            # LIVE runner (daily signal, real-time fills/stop)
│   ├── backtest_pullback.py        # pyramiding backtester
│   ├── sweep.py                    # parameter grid + walk-forward robustness
│   ├── portfolio_ibkr.py           # IBKR adapter (same method surface)
│   ├── ibkr_universe.py            # EUR instruments + IBKR contract specs
│   ├── live_pullback_ibkr.py       # LIVE runner on IBKR (reuses the trader)
│   └── strategies/
│       └── trend_pullback.py       # the phased-entry trend method (long-only)
├── tests/                          # offline test suite (no network/broker)
├── deploy/                         # systemd units, env templates, VPS runbook
└── (created at runtime)
    ├── pullback_trades.csv         # one row per closed trade
    ├── pullback_daily_pnl.csv      # realized P&L per day + equity snapshot
    ├── pullback_state.json         # restart-safe pyramided positions & stops
    └── backtest_pullback_results.png   # equity curve chart
```

## How the strategy works (`trend_pullback.py`)

It encodes five discretionary trend-trading principles. Every threshold lives in
`PullbackParams`, so treat the defaults as a starting point, not gospel.

| Principle | Implementation |
|---|---|
| 1. Only hold in a healthy uptrend | `trend_ok()`: close > 50MA, close > 200MA, and 50MA > 200MA — a hard gate on every entry |
| 2. Buy the low-volume pullback to the 50MA, on the rebound | `_pullback_entry()`: dipped within 2% of the 50MA, volume contracted vs. baseline, current bar rebounds back above the MA |
| 3. Buy the consolidation breakout on rising volume | `_breakout_entry()`: prior bars formed a tight range, current bar closes above it with ≥1.5× average volume |
| 4. Phased 30/30/40 entries | tranche 1 on the first signal; tranches 2 and 3 added as price makes new highs (`add_step`, default +3%) while the trend gate holds |
| 5. Hold the trend; exit on MA/structure break | `trend_exit()`: exit on a daily close below the 50MA or the recent swing low |
| Volatility stop | `stop_distance() = max(5%, 2 × ATR(14) / price)` (on daily). A GTC stop is **rested at the broker** for equities (survives downtime/gaps) *and* monitored in real time by the runner. Alpaca crypto can't take a resting stop, so it relies on the in-process monitor. |

**The daily/real-time split.** The daily chart decides *whether* to trade (trend
gate, entries, tranche adds, MA/structural exit — all on completed daily bars).
Execution then only improves the *fill price*: an armed entry/add tries to fill on
a dip to `daily close × (1 − improve_pct)` (default 0.4% below), but still fills at
market on the next daily evaluation if no dip comes, so real-time price can't veto a
valid daily setup. The volatility stop is the one thing checked continuously.

Sizing keeps risk consistent: each tranche is sized so a fully-built position
stopped at `stop_distance` below the average entry loses ~1% of equity (`RISK_PER_TRADE`),
split 30/30/40.

## Running it live (Alpaca, paper by default)

```bash
python -m bot.live_pullback                      # paper, default symbols
python -m bot.live_pullback --symbols SPY GLD     # a subset
python -m bot.live_pullback --ema                 # use EMAs instead of SMAs
```

The runner polls the latest price every 60s: it acts on completed daily bars for
signals, fills armed entries/adds when price dips to the limit (falling back to a
market fill on the next daily bar), and protects each position two ways — a GTC
stop **rested at the broker** (re-priced as tranches are added) plus the in-process
`max(5%, 2×ATR)` monitor. If a resting stop fires while the bot is down or between
loops, startup/loop reconciliation detects the now-flat broker position, recovers
the real fill price, and logs the trade. Daily history is fetched once per session
and cached, so a quiet day costs one historical-bars request per symbol, not one
per minute. It writes to `pullback_trades.csv`, `pullback_daily_pnl.csv`, and
`pullback_state.json`, is restart-safe, and reconciles with the broker on startup.
Stop with Ctrl+C.

It refuses to run against the live (real-money) endpoint unless you pass `--live`.

### Alerts (optional)

Set any of these env vars to get pings on startup/shutdown, every entry/exit, a
data stall, or a processing error (leave them unset to run silently):

```bash
ALERT_WEBHOOK_URL=...        # Slack-compatible incoming webhook ({"text": ...})
TELEGRAM_BOT_TOKEN=...       # or Telegram — set both
TELEGRAM_CHAT_ID=...
```

Alerting is dependency-free, sent on a background thread, and can never crash the
trading loop.

> **One account, one position per symbol.** Don't run two copies of the runner on
> the same symbol against the same Alpaca account at once — they'll fight over the
> position. Use distinct symbols or separate accounts.

### Expanding the symbols it trades

Edit `PULLBACK_SYMBOLS` in `config.py`. If a symbol isn't one of the five already
defined in `INSTRUMENTS`, add it to `PULLBACK_UNIVERSE` first (there's a commented
example in the file), then list its name in `PULLBACK_SYMBOLS`. Both the backtester
(`--symbols`) and the live runner pick these up.

## Backtesting

Replay history through the **same** strategy module the live bot uses (no parallel
re-implementation, so the backtest reflects real logic):

```bash
python -m bot.backtest_pullback                              # daily signal, 4h fills
python -m bot.backtest_pullback --exec-timeframe 1Hour       # daily signal, 1h fills
python -m bot.backtest_pullback --exec-timeframe none        # fill at next daily bar
python -m bot.backtest_pullback --symbols SPY QQQ GLD --months 9 --ema
python -m bot.backtest_pullback --start 2025-02-04 --end 2026-06-24
```

It runs a per-instrument pass and a combined-portfolio pass (all symbols sharing one
equity), and for each reports total trades, win rate, average win/loss, profit
factor, max drawdown, Sharpe (daily-resampled, annualized, risk-free 0), and total
return. **Any strategy with a negative Sharpe is flagged.** It saves an equity-curve
chart to `backtest_pullback_results.png`.

When you pass `--start`, the fetcher pulls ~320 days of **lead** history before it
(the 200-day MA needs ~200 bars) so indicators are warm on day one of your window.

Modelling assumptions: 0.05% slippage applied adversely to every fill, $0 commission,
the daily signal filled via the chosen execution timeframe, and the volatility stop
checked against execution-candle lows with gap handling.

> **Data depth caveat:** history is bounded by your Alpaca data plan. Free IEX
> history doesn't go back indefinitely; if a fetch returns fewer bars than `--start`
> requests, the backtest uses what's available and logs the actual range it got.

### Parameter robustness (`bot/sweep.py`)

A single backtest is easy to fool yourself with. `sweep.py` sits on top of the same
backtester and answers "are the defaults robust, or a lucky cell?":

```bash
python -m bot.sweep --mode grid --months 18          # rank a parameter grid
python -m bot.sweep --mode walk --folds 4            # walk-forward (in/out-of-sample)
python -m bot.sweep --mode walk --add-step 0.02 0.03 0.05 --atr-mult 1.5 2.0 2.5
```

- **grid** evaluates every combination over the window and reports how many are
  profitable / positive-Sharpe — a broad band of winners beats one knife-edge cell.
- **walk** splits the window into folds, picks the best params on each fold's
  in-sample half, then scores *those* params on the untouched out-of-sample half.
  Consistent out-of-sample results are the honest read; in-sample-best that falls
  apart OOS is the overfit tell.

## Testing

Offline test suite (no network, no broker) covering the strategy logic, the
backtester, the sweep mechanics, and the live runner's order/stop/reconcile
machinery against fakes:

```bash
pip install -r requirements-dev.txt
pytest -q
```

CI runs the same suite on every push (`.github/workflows/ci.yml`).

## Deployment (Linux VPS)

`deploy/` has systemd units (Alpaca and IBKR), env templates, and a step-by-step
runbook (`deploy/README.md`) for running it as an auto-restarting service with logs
in the journal and graceful SIGTERM shutdown.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then edit .env with your real keys
python -m bot.backtest_pullback   # sanity-check data + see historical behavior
python -m bot.live_pullback       # paper trading
```

Keep `ALPACA_BASE_URL=https://paper-api.alpaca.markets` until you've watched it
behave for a while. Switching to the live URL trades real money. See `SETUP_GUIDE.md`
for a step-by-step walkthrough and `IBKR_SETUP.md` for the EUR-only IBKR path.

## Things to know before trusting this with money

1. **Built on `alpaca-py`** (Alpaca's current, maintained SDK), so it runs on modern
   Python. All SDK calls are isolated in `bot/portfolio.py`. Crypto symbols use the
   slash form (`BTC/USD`) that `alpaca-py` expects.

2. **You can't short crypto on Alpaca** — but Strategy 4 is long-only anyway, so this
   doesn't bite here.

3. **Stops: belt and suspenders, but not magic.** Equities get a GTC stop rested at
   the broker (covers downtime/gaps) plus a 60s in-process monitor. A fast gap can
   still fill below the stop level, and Alpaca **crypto** can't rest a stop, so BTC
   relies on the in-process monitor only — meaning crypto downtime is unmanaged risk.

4. **The 200-day MA needs ~200 daily bars of history** before the trend gate can pass,
   so expect the bot to sit idle on a fresh symbol at first.

5. **A good backtest is not a guarantee.** Signals firing correctly is not the same as
   the strategy being profitable, and tuning parameters to a backtest is the easiest
   way to fool yourself.

I'm not a financial advisor, and this is software, not investment advice. Automated
trading can lose money quickly, including more than your intended risk per trade if
stops slip on a gap. Run it on paper first and size in with money you can afford to lose.
