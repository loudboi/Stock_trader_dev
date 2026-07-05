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

Every run also prints a **VS BUY-AND-HOLD** table — strategy vs. an equal-weight
buy-and-hold of the same symbols (return, max drawdown, Sharpe, and a "beat it?"
verdict) — and overlays the buy-and-hold curve on the chart, so "did I actually
beat just holding?" is always answered, not assumed.

### Full-cycle history (`--data-source yahoo`)

Alpaca's free IEX history only reaches ~mid-2020, which is a bull market — a window
that flatters any long-only strategy. For an honest test you want a full cycle
including a crash. Pass `--data-source yahoo` to pull decades of free daily history:

```bash
pip install -r requirements-backtest.txt
python -m bot.backtest_pullback --symbols SPY QQQ GLD --start 2005-01-01 \
    --end 2026-06-01 --exec-timeframe none --data-source yahoo
```

Yahoo bars are split/dividend-adjusted (not bar-identical to live Alpaca data) and
daily-only, so treat this as a regime study, not a live proxy. It's the right tool
for "how would this have behaved through 2008?".

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

## Beating buy-and-hold (`bot/trend_exposure.py`)

The pullback strategy is defensive — over a full cycle (2005–2026) it badly trails
buy-and-hold on both return *and* Sharpe, because it sits in cash most of the time.
`trend_exposure.py` is the opposite posture: a classic 200-day-MA timing model —
**hold (optionally leveraged) while price is above its long MA, move to cash below
it** — so it captures most of the uptrend but sidesteps the big bear drawdowns.

```bash
python -m bot.trend_exposure --symbols SPY QQQ GLD --start 2005-01-01 \
    --data-source yahoo --leverage 1.5 --buffer 0.01
```

On the 2005–2026 full cycle, **unlevered** it beats buy-and-hold risk-adjusted
(Sharpe ~0.98 vs 0.88, roughly half the max drawdown) — a real, robust edge. But
the `--borrow-rate` (default 6%/yr) models the cost of leverage, and once it's
applied the "beat on raw return" result **evaporates**: 1.5× trails B&H on return
*and* Sharpe; 2× beats on raw return only by taking more risk at a worse Sharpe.
The honest takeaway: the trend filter beats buy-and-hold **risk-adjusted, unlevered**
— leverage doesn't buy you a free lunch on raw return, because financing eats the
thin edge. It's a daily allocation model and is **research/backtest only** — not
wired into a live runner. (Leveraged-ETF daily-reset decay isn't modeled either.)

## Strategy lab (`bot/lab.py`)

Nine research-backed beat-buy-and-hold approaches in one comparable harness, ranked
against an equal-weight buy-and-hold over a full cycle: `vol_target` (target a
constant volatility), `inverse_vol` (naive risk parity), `erc` (equal-risk-
contribution risk parity, using the full covariance matrix), `min_var` (long-only
global minimum-variance), `rp_voltarget` (inverse-vol weights scaled to a target
vol), `managed_futures` (diversified inverse-vol trend), `mean_reversion` (RSI
dip-buying in uptrends), `trend_vol` (vol-targeted trend), and `ensemble` (a blend).
`erc`/`min_var`/`rp_voltarget` use fixed, standard textbook parameters (60-day
trailing covariance, monthly rebalance, 10–15% vol target) chosen before looking at
any result — not swept for the best number. Costs modelled (0.05% turnover slippage
+ a borrow rate on leverage); all signals act next-day (no lookahead).

```bash
python -m bot.lab --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
```

Finding, run ONCE each on two different universes (no post-hoc tuning) over
2005/2006–2026:

| Universe | B&H Sharpe | Beat B&H on Sharpe |
|---|---|---|
| SPY/QQQ/GLD/TLT (4 assets) | 0.96 | `min_var` 1.13, `inverse_vol` 1.11, `erc` 1.02 (+return: 1162% vs 972%) |
| SPY/QQQ/IWM/EFA/EEM/TLT/IEF/GLD (8 assets) | 0.77 | `min_var` 0.88, `inverse_vol` 0.87 |

**`min_var` and `inverse_vol` beat buy-and-hold's Sharpe in BOTH universes.** `erc`
won big on universe 1 (even beating raw return) but *failed* on universe 2
(0.66 < 0.77) — testing two universes is what catches that kind of luck.
`vol_target`/`rp_voltarget`/`managed_futures`/`trend_vol`/`mean_reversion`
underperformed on both.

**`--mode walk` (walk-forward across 5 sequential time folds, both universes,
no per-fold parameter fitting — these strategies have nothing left to fit)
sharpens the finding further, and it's more modest than the full-window average
suggests:**

```bash
python -m bot.lab --mode walk --folds 5 --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
```

`min_var`/`inverse_vol` do **not** beat B&H's Sharpe in every fold (3/5 and 3/5 in
universe 1; 2/5 and 3/5 in universe 2) — so "robust winner in all conditions" is
too strong a claim. What actually holds up: **the edge is concentrated in the
high-volatility/crisis folds** (the 2008-era fold in both universes shows the
largest outperformance — e.g. min_var 0.90 vs B&H 0.51 in universe 1, 1.28 vs 0.38
in universe 2), while in calm strong-bull folds they roughly tie or slightly trail
(sensible: a risk-based book underweights the highest-beta winners in a bull run).
Mean Sharpe across folds still beats B&H's mean in both universes for both
strategies. `erc`/`vol_target`/`managed_futures` are weaker and more sporadic here
too. **Honest read: not "wins always," but "smoother ride, protection concentrated
in crises, roughly even the rest of the time" — a real, explainable pattern, not
noise, but a more modest claim than the single full-window number implies.**
Research/backtest only.

### Long/short strategies (`trend_ls`, `xsmom_ls`, `managed_futures_ls`) — tried hard, still don't work

Every strategy above is long-or-cash. Three long/short constructions were added
and tested the same disciplined way (fixed textbook parameters chosen before
looking at any result, full-window AND walk-forward):

- **`trend_ls`** — the same 200-day MA filter as `trend_vol`/`managed_futures`, but
  SHORTS below the MA instead of moving to cash, equal-weighted across active
  signals (the classic managed-futures/CTA construction).
- **`managed_futures_ls`** — the same long/short trend signal, but INVERSE-VOL
  weighted (like `managed_futures`, but allowed to short) — so a calm FX pair and
  a volatile commodity don't get equal dollar weight.
- **`xsmom_ls`** — the classic academic cross-sectional momentum factor: monthly,
  long the strongest-momentum assets and short the weakest, equal dollar amounts
  each side (market-neutral), 12-1 lookback.

All charge a `--short-borrow` cost (default 1%/yr) on short notional, separate
from the leverage `--borrow-rate` — see the caveat in the module docstring (real
single-name borrow costs can be far higher or unavailable).

**Round 1 (SPY/QQQ/GLD/TLT and the 8-asset ETF mix): both `trend_ls` and
`xsmom_ls` failed everywhere** — 0/5 walk-forward folds beaten in either universe,
`xsmom_ls` reaching Sharpe −0.13 with a −60.8% max drawdown on the 8-asset mix.
Diagnosed cause: every asset in both universes has persistent structural upward
drift over 2005–2026, so shorting any of it (via `trend_ls`'s downtrend legs)
fights that drift, and 4–8 broad, correlated ETFs give `xsmom_ls` barely a
cross-section to work with.

**Round 2 — a genuinely fairer test.** Two structural fixes, each addressing a
specific objection above (not parameter tuning):

- `xsmom_ls` re-tested on **30 diversified individual large-cap stocks** (real
  cross-sectional breadth, the setting the factor is actually validated on). Still
  failed — Sharpe −0.08, and the walk-forward failure pattern (worst fold exactly
  2009–2013) matches the well-documented real-world **"momentum crash"** of
  2009, when prior losers rallied hard off the 2008 bottom and wrecked
  momentum books industry-wide. Not a fluke of this test; a known phenomenon.
- `trend_ls`/`managed_futures_ls` re-tested on **6 major FX pairs + 7 commodity
  futures** (no persistent one-directional drift, a genuine two-sided market).

**A data-integrity problem surfaced during that second test, and was fixed before
trusting any number.** `CL=F` (crude oil) showed a "306% daily move" that is
actually the real April 2020 WTI negative-price settlement — a percentage return
is mathematically undefined across a sign change, and it poisons any 20-day
volatility/trend window it touches. `EURUSD=X`/`JPY=X` separately showed 17%+
single-day moves in Dec 2008 that reversed almost completely within two days —
implausible for the most liquid FX pairs in the world and almost certainly a free-
data glitch. **`bot/data.py:clean_price_series`/`clean_daily_data`** (a fixed,
asset-class-agnostic 15% cap, chosen before re-running anything — see
`--clean-outliers`) patches these; only carried-forward prices are changed, not
strategy logic. Re-running with `--clean-outliers` moved buy-and-hold's own Sharpe
on that universe from 0.42 to 0.50 (confirming the fix mattered) but **did not
rescue either strategy** — still 2/5 and 1/5 walk-forward folds, negative mean
Sharpe. The remaining honest story: `trend_ls` actually beat buy-and-hold in the
2005–2009 fold (crisis/high-vol trending regime), then failed hard every fold
from 2013 onward — which matches the real, well-documented industry-wide struggle
of trend-following/CTA strategies through the low-volatility, central-bank-
suppressed 2011–2019 era.

```bash
python -m bot.lab --symbols EURUSD=X GBPUSD=X JPY=X USDCAD=X USDCHF=X NZDUSD=X GC=F CL=F NG=F SI=F ZC=F ZS=F HG=F \
    --start 2005-01-01 --data-source yahoo --clean-outliers --strategies trend_ls managed_futures_ls
```

**Bottom line: after two genuinely fair attempts (real breadth for momentum, a
genuinely two-sided market for trend, and a verified data-integrity fix), neither
long/short construction produced a durable edge.** This isn't for lack of trying —
it's consistent with well-documented, real phenomena (the 2009 momentum crash, the
2011–2019 CTA drought), which is exactly what makes it trustworthy rather than a
broken test. If you want to keep pursuing long/short, the next honest step isn't
another universe swap — it's a fundamentally different signal (e.g.
volatility-managed momentum, or a shorter/adaptive trend lookback with its own
walk-forward validation), not re-tuning these same constructions until one
number looks good. Research/backtest only.

## Macro-regime-conditioned strategies (`bot/regime.py`)

Free macro indicators (VIX, the 10Y−13-week yield curve, HYG-vs-IEF credit
spread) with fixed, conventional thresholds (VIX 15/25, curve sign, credit sign
— chosen from standard industry usage *before* any backtest, not fit to one) used
to condition/combine the strategies above:

```bash
python -m bot.lab --strategies regime_leverage_rp --regime-indicator vix \
    --symbols SPY QQQ GLD TLT --data-source yahoo --start 2005-01-01
```

- **`regime_gated_rp`** (de-risk to cash when the regime signals stress) and
  **`regime_leverage_rp`** (lever up when calm, delever when stressed) — both
  layered on top of `inverse_vol` risk parity. **Result: consistently WORSE than
  plain `inverse_vol`, every time** — all 3 indicators × both universes (12/12
  negative). Adding a tactical macro overlay on top of an already well-diversified
  systematic strategy mostly adds whipsaw/turnover cost without a compensating
  benefit — a known result in the broader literature, replicated here.
- **`regime_gated_trend_ls`** — the `trend_ls` long/short signal, but shorts are
  only allowed when the regime signals stress (targeting the specific failure
  found in the long/short section above: shorting a persistently-drifting asset
  most of the time). **Result: a real, validated improvement to the construction**
  — VIX-gating beat plain `trend_ls` in **5/5 walk-forward folds** on the
  FX/commodities universe (mean Sharpe −0.32 → +0.17), but still doesn't reliably
  beat buy-and-hold on its own (2/5 folds). Gating shorts to genuine stress
  periods fixes most of the "shorting a drifting asset" problem — just not enough
  to become an outright market-beater by itself.

## Combining proven strategies (`bot/combo.py`)

The strongest, most validated result of this entire project. Two DIFFERENT
strategies have each independently been shown to beat buy-and-hold's Sharpe on
their own: `inverse_vol` risk parity and the `trend_exposure` 200-day filter.
Blending them (fixed 50/50 capital split, chosen before running anything) tests
whether their moderate (~0.7) correlation is low enough for real diversification
benefit — unlike blending in the FX/commodity trend book above, whose own weak
standalone Sharpe (0.22) meant *any* meaningful allocation to it just dragged the
combination down despite its low (0.11) correlation. The lesson: correlation
alone isn't enough for a diversification benefit — the components need standalone
quality too.

```bash
python -m bot.combo --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
python -m bot.combo --mode walk --folds 5 --symbols SPY QQQ GLD TLT --data-source yahoo --start 2005-01-01
```

**Result, tested on two universes:** the 50/50 blend's full-window Sharpe beat
**both** pure components in **both** universes —

| Universe | B&H | pure RP | pure TE | **50/50 blend** |
|---|---|---|---|---|
| SPY/QQQ/GLD/TLT | 0.96 | 1.11 | 1.01 | **1.15** |
| 8-asset broader mix | 0.77 | 0.87 | 0.78 | **0.90** |

Walk-forward is honest about the limits: the blend beat buy-and-hold in 5/5
folds on universe 1 but only 3/5 on universe 2 (the first universe's "perfect"
record was partly universe-specific, exactly what testing a second universe is
for) — so, like everything else in this project, its edge over B&H is
regime-dependent, not universal. But it robustly beat pure RP (3/5 and 3/5 folds)
and especially pure TE (4/5 and 4/5 folds) — i.e., **if choosing between holding
risk parity alone, trend-exposure alone, or the blend, the blend was the better
choice in both universes tested, both on average and most of the time regime-by-
regime.** That's a genuine, validated diversification benefit, not a magic
"beats B&H everywhere" strategy — none of those exist in this project.
Research/backtest only; `--rp-weight` lets you try other fixed splits.

## Momentum rotation (`bot/momentum_rotation.py`)

Dual-momentum rotation: each month, hold the top-`k` assets by trailing return,
but only those with *positive* momentum (else cash). The other research-backed
beat-the-market idea.

```bash
python -m bot.momentum_rotation --symbols SPY QQQ GLD TLT EFA EEM IWM \
    --start 2005-01-01 --data-source yahoo --lookback-months 12 --top-k 2
```

Honest finding on 2005–2026: it's **highly parameter-sensitive** (Sharpe ranged
0.42→0.76 across lookback/top-k), and the canonical 12-1 spec *underperformed*
buy-and-hold. A few shorter-lookback configs beat it — but picking those after
seeing the results is textbook overfitting. Treat any single "win" with suspicion
and walk-forward it before believing it.

## Testing

Offline test suite (no network, no broker) covering the strategy logic, the
backtesters, the sweep mechanics, the trend-exposure and momentum-rotation models,
the strategy lab (including long/short and macro-regime-conditioned strategies),
the RP+TE combo tool, the benchmark, and the live runner's order/stop/reconcile
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
