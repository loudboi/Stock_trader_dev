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

### Dynamic allocation between the two good strategies (`bot/dynamic_combo.py`)

A genuinely different mechanism from the macro overlays above: instead of
gating/levering ONE strategy on a macro signal (which failed 12/12), use a
real-time, causal signal to shift the CAPITAL SPLIT between the RP+TE combo's
two already-good ingredients. Two signals, fixed thresholds chosen before any
result — **importantly, neither uses hindsight; both are the kind of
information a real trader would have had in real time, with no hardcoded
historical event dates**:

- **`breadth`** — the fraction of a *reference* universe (9 SPDR sector ETFs,
  independent of the traded assets, avoiding a self-referential signal) trading
  above its own 200-day MA. Below 40% → lean risk-based (0.7/0.3); above 60% →
  lean trend-following (0.3/0.7); else the validated 50/50.
- **`vix_term`** — the VIX/VIX3M term-structure ratio (backwardation vs.
  contango — a different dimension from the VIX *level* check above).

```bash
python -m bot.dynamic_combo --signal breadth --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
python -m bot.dynamic_combo --mode walk --signal breadth --symbols SPY QQQ GLD TLT --data-source yahoo --start 2005-01-01
```

**Result: another clean negative — dynamic allocation never beat the static
50/50 blend, in any test.** Full-window: breadth-based allocation trailed on
both universes (1.11 vs 1.18 on universe 1; 0.86 vs 0.99 on universe 2), and
VIX-term-based allocation also trailed (1.06 vs 1.18). Walk-forward: **0/5
folds** beaten. This is actually informative, not just another dead end: it
means the static 50/50 split is already close to the efficient allocation
between these two ingredients given their relative quality and correlation —
timing *when* to favor one over the other adds turnover cost without a
compensating edge, consistent with the well-documented difficulty of tactical
asset allocation relative to static diversification (e.g. Ibbotson & Kaplan).
**The practical takeaway: don't try to time the RP/TE mix — the fixed 50/50
blend is the strategy.** Research/backtest only.

## Combining proven strategies (`bot/combo.py`)

The strongest, most validated result of this entire project. Two DIFFERENT
strategies have each independently been shown to beat buy-and-hold's Sharpe on
their own: a risk-based portfolio (`--rp-strategy {min_var, inverse_vol}`, default
`min_var`) and the `trend_exposure` 200-day filter. Blending them (fixed 50/50
capital split, chosen before running anything) tests whether their moderate
(~0.7) correlation is low enough for real diversification benefit — unlike
blending in the FX/commodity trend book above, whose own weak standalone Sharpe
(0.22) meant *any* meaningful allocation to it just dragged the combination down
despite its low (0.11) correlation. The lesson: correlation alone isn't enough
for a diversification benefit — the components need standalone quality too.

```bash
python -m bot.combo --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
python -m bot.combo --rp-strategy inverse_vol --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
python -m bot.combo --mode walk --folds 5 --symbols SPY QQQ GLD TLT --data-source yahoo --start 2005-01-01
```

**Result, tested on two universes, both risk-based legs:** the 50/50 blend's
full-window Sharpe beat **both** pure components in **both** universes, for
**either** choice of risk-based leg —

| Universe | B&H | pure min_var | pure inverse_vol | pure TE | blend (min_var+TE) | blend (inverse_vol+TE) |
|---|---|---|---|---|---|---|
| SPY/QQQ/GLD/TLT | 0.96 | 1.13 | 1.11 | 1.01 | **1.18** | 1.15 |
| 8-asset broader mix | 0.77 | 0.88 | 0.87 | 0.78 | **0.99** | 0.90 |

`min_var` is the marginally better risk-based leg (higher full-window Sharpe in
both universes) and is the default; `inverse_vol` remains available and fully
validated as the original variant. **A third option, `erc`** (equal-risk-
contribution — the one that looked great on universe 1 early in the project but
FAILED universe 2 as a *standalone* strategy, Sharpe 0.66 < B&H's 0.77): blended
with TE, it beats B&H on **both** universes (1.14 vs 0.96; 0.80 vs 0.77) — the
blend *rescues* `erc`'s earlier standalone failure, and walk-forward confirms
it's real (4/5 folds beat pure `erc`, 3/5 beat B&H on universe 2, not just a
full-window average). `erc`+TE is weaker than `min_var`+TE or `inverse_vol`+TE,
so it isn't the recommended default — but **this closes out the "which risk leg"
question: all three risk-based constructions, when blended with trend-exposure,
beat buy-and-hold on both universes.** The diversification mechanism itself is
robust to which specific risk-based strategy anchors it, not a fluke of one
particular implementation.

Walk-forward is honest about the limits: the `min_var`+TE blend beat buy-and-hold
in 5/5 folds on universe 1 but only 3/5 on universe 2 (the same partly-universe-
specific pattern found with `inverse_vol`+TE — exactly what testing a second
universe is for) — so, like everything else in this project, its edge over B&H is
regime-dependent, not universal. It beat pure TE consistently (4/5 both
universes) but was less consistent fold-by-fold against its own `min_var` leg
(2/5 and 3/5) than the `inverse_vol` variant was against its own leg (3/5 and
3/5) — because `min_var` alone is simply a strong performer in several individual
folds, so blending in TE doesn't always help THAT fold even though it improves
the full-window average. **If choosing between holding a risk-based portfolio
alone, trend-exposure alone, or the blend, the blend was the better choice on
average and most of the time in both universes tested** — a genuine, validated
diversification benefit, not a magic "beats B&H everywhere" strategy (none of
those exist in this project). Research/backtest only; `--rp-weight` lets you try
other fixed splits.

**Parameter robustness — checked, and it holds.** Varying the underlying
parameters (RP lookback 10/20/40 days, TE MA period 150/200/250, TE buffer
1%/2%) on universe 1: the blend beat **both** pure components in **all 5**
configurations tested. Not fragile to the exact defaults.

**A third universe — and a real limit found.** Tested the SAME construction on
the FX + commodity futures universe (`--clean-outliers`, 6 FX pairs + 7
commodities). Here it does **not** work: both components are individually weak
on this asset class (pure RP Sharpe 0.13, pure TE 0.22 — far below their
equity-universe values of ~1.1 and ~1.0), and the blend (0.20) sits *between*
them rather than beating both, trailing buy-and-hold's 0.40. Walk-forward: only
1/5 vs pure RP, 2/5 vs pure TE, 3/5 vs B&H — much less consistent than on either
equity-like universe. **Honest scope correction: the RP+TE diversification
benefit is validated for equity-like universes (broad index/bond/gold ETFs),
where both ingredients have real standalone edges — it is NOT a universal
combination technique.** This matches the mechanistic lesson above:
diversification needs component quality, and on FX/commodities neither
component has much.

**A fourth universe — refining "equity-like" to an asset-class distinction, not
a geography one.** Tested the SAME construction on 11 major country ETFs
spanning developed and emerging markets across four continents (Japan, Germany,
UK, Canada, Australia, Brazil, China, South Korea, Taiwan, Hong Kong, Singapore
— no US exposure at all). Full-window: min_var alone is weak here (0.43, similar
to buy-and-hold's own 0.44 — country-level risk parity isn't as strong a
standalone edge as it is on the SPY/QQQ/GLD/TLT-style universes), so the blend
(0.53) ties/slightly trails pure TE (0.54) rather than clearly beating both.
**But walk-forward tells a better story: the blend beat buy-and-hold in 4/5
folds** (matching the best record found on any universe) **and pure TE in 4/5
folds**, only trailing pure min_var (2/5, since min_var's own edge here is
weak). **This generalizes the finding correctly: it's not "US equities/bonds/
gold only" — it's equities broadly (US or international), vs. FX/commodities
specifically**, which lack the structural risk premia both components rely on.
Also parameter-robust on this universe: varying `min_var`'s covariance lookback
(30/60/90 days) and the trend filter's MA/buffer, the blend beat buy-and-hold in
**all 5** configurations tested (Sharpe 0.50–0.56 vs B&H's 0.44).

**A fifth universe — combining everything, and the SECOND perfect walk-forward
record found in this project.** Merged US and international into one 15-asset
mega-universe (SPY/QQQ/GLD/TLT + all 11 country ETFs). Here `min_var` alone is
strong (Sharpe 0.87 — a bigger, broader book gives risk parity more to diversify
across), so the blend (0.86) essentially ties pure `min_var` full-window while
clearly beating pure TE (0.67) and buy-and-hold (0.56). **Walk-forward is where
this shines: the blend beat buy-and-hold in 5/5 folds** — matching the best
record in the whole project — **and beat pure TE in 4/5 folds.** The pattern is
consistent across every equity-like universe tested: whichever component is
individually stronger on a given universe, the blend tracks close to it while
still meaningfully beating the weaker component and buy-and-hold — and broader
diversification (more assets, more countries) only strengthens this further.

**Leverage on the blend (`--leverage`) — tested, and it doesn't help, again.**
The RP+TE blend starts from the highest Sharpe found in this project (1.15/0.90
with `inverse_vol`, 1.18/0.99 with `min_var`), so it's the best candidate
leverage has had — but the same pattern that killed every other leverage attempt
here repeats: Sharpe degrades monotonically with leverage (1.15→0.98→0.90→0.78
at 1×/1.3×/1.5×/2× on universe 1; 0.90→0.65→0.52 at 1×/1.5×/2× on universe 2,
both with the standard 6%/yr borrow cost). At 2× on universe 1, raw return
finally exceeds buy-and-hold (998% vs 972%) — but only by taking on more risk
(worse Sharpe, −33% max drawdown), not by adding edge. **Best Sharpe in both
universes is unlevered.** This is now the fifth base strategy in this project
(after `trend_exposure`, `vol_target`, `rp_voltarget`, `regime_leverage_rp`)
where leverage was tried and failed to improve risk-adjusted return — about as
solid a conclusion as anything here: **you beat
buy-and-hold by managing risk (diversifying across weakly-correlated, individually
decent strategies), not by levering up a good one.**

## Overnight vs. intraday returns (`bot/overnight.py`)

A fundamentally different idea from everything else here — not a directional
signal, but a question of *when* during the day returns happen. The well-
documented "overnight effect": historically, most of the stock market's gain has
come from the close-to-open (overnight) session, with open-to-close (intraday)
flat or negative. Two mechanical books, each requiring a full round-trip trade
**every single day** (unlike everything else in this project, which only trades
on a signal change):

```bash
python -m bot.overnight --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
python -m bot.overnight --cost-per-side 0.0001 --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
```

**Finding: the raw effect is real and strong, but EXTREMELY cost-sensitive —
more than anything else in this project, because daily round-trip trading means
a per-trade cost gets compounded roughly 5,000+ times over the backtest.**

| Cost/side | overnight_only Sharpe | intraday_only Sharpe |
|---|---|---|
| $0 (raw signal) | **1.18** (highest pre-cost Sharpe in this project) | 0.39 |
| 1bp (0.01%, optimistic even for a top-tier liquid ETF) | 0.43 | −0.20 |
| 5bp (0.05%, this project's standard "infrequent trade" convention) | −2.55 (catastrophic — cost compounds daily for 20 years) | −2.54 |

The project-wide 0.05% slippage convention was calibrated for strategies that
trade a handful of times a year; applying it to a strategy that round-trips
*every day* for two decades is a different regime entirely — `(1-0.001)^5469 ≈
0.4%`, i.e. compounding alone erases the account regardless of any real edge.
Even a much more optimistic 1bp/side cost — already hard to achieve in practice
once the bid-ask spread is counted — cuts the raw Sharpe by more than half and
flips the intraday book negative. On the broader 8-asset universe the effect is
weaker even before cost (overnight Sharpe 0.73 vs B&H's own 0.77 — barely
positive pre-cost, then destroyed the same way by cost).

**Honest conclusion: this is a genuine, well-documented market-structure
anomaly, but it isn't retail-tradeable** — it requires the kind of near-zero,
often rebate-earning execution costs available to market makers and HFT firms,
not a normal broker account. It's the single strategy in this project where
the gap between "paper edge" and "tradeable edge" is starkest — a useful,
concrete illustration of why "beats B&H in a spreadsheet" and "beats B&H after
real costs" are very different bars. `--cost-per-side` lets you explore this
sensitivity directly. Research/backtest only.

## Turn-of-month effect (`bot/seasonality.py`)

Another fresh idea, and a fully calendar-driven one: the classic "turn-of-month"
effect (Ariel 1987; Lakonishok & Smidt 1988) — historically, equity returns
cluster disproportionately around month boundaries, attributed to systematic
flows (payroll/401k contributions, pension rebalancing). Fixed, literature-
standard window (chosen before backtesting): invested on the last 1 trading day
of a month through the first 3 of the next; cash otherwise (turnover-costed like
everything in `bot/lab.py`, not a daily round-trip like `bot/overnight.py`).

```bash
python -m bot.seasonality --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
```

**Result: a clean negative finding at two levels.** As a strategy, it badly
trails buy-and-hold (Sharpe 0.23 vs 0.96 on universe 1; 0.12 vs 0.77 on universe
2) — invested only 19% of the time, it simply forgoes most of a persistently
rising market's gains. But the deeper, more important check is whether the
underlying anomaly still exists at all: comparing average daily returns on
turn-of-month days vs. all other days directly, there's a small elevation
(1.12× on universe 1, 1.41× on universe 2) in the historically-expected
direction — but a Welch's t-test puts both nowhere near significance (t≈0.23
and t≈0.53; ~2.0 would be needed). **The classic effect, once dramatic enough in
older studies to explain nearly all of the market's historical gain, shows no
statistically distinguishable signal in this modern (2005–2026), liquid,
closely-arbitraged ETF data.** This is consistent with "anomaly decay" — a
well-documented phenomenon where publishing an anomaly lets systematic funds
arbitrage it away (see McLean & Pontiff 2016). Research/backtest only.

**Day-of-week ("Monday effect") — same pattern, checked directly (`--mode weekday`):**

```bash
python -m bot.seasonality --mode weekday --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
```

A purely descriptive comparison (not a backtested strategy — holding only
specific weekdays hits the same daily-round-trip cost wall documented under
Overnight vs. Intraday above). Monday is the weakest day in both universes —
directionally consistent with the classic "weekend effect" (French 1980) — but
again nowhere near significant (t≈−0.84 and −1.22 vs. the rest of the week;
~2.0 needed). Same conclusion as turn-of-month: a historically-documented
calendar effect, present in direction but not in statistical substance, in this
modern data. Two-for-two on "anomaly decay" — don't expect a different result
from testing other classical calendar effects (e.g. "Sell in May") on these same
liquid, heavily-arbitraged ETF universes.

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

## Slovenian capital-gains tax and EUR currency risk (`bot/taxes.py`, `bot/currency.py`, `bot/aftertax.py`)

Every backtest above this section is **pre-tax**. That's a bad basis for a real
decision if you're a Slovenian taxpayer. Verified against the Financial
Administration of the Republic of Slovenia (fu.gov.si): capital gains on
securities are **not a flat rate** — it's a graduated CLIFF based on the total
holding period of the lot at sale (the whole gain gets one rate, not a
marginal/bracket split):

| Holding period | Rate |
|---|---|
| 0–5 years | 25% |
| 5–10 years | 20% |
| 10–15 years | 15% |
| more than 15 years | 0% (fully exempt) |

That's not a minor tax-drag footnote — it's a structural moat around literal,
untouched buy-and-hold that no actively-traded strategy in this project can
cross, because every one of them (including the best pre-tax result, the
`min_var`+trend-exposure combo) realizes gains at the top 25% short-term rate
almost every year. Losses **do carry forward** to future tax years (confirmed
via the official Doh-KDVP filing instructions), and no Slovenian wash-sale
rule (a restriction on claiming a loss if you immediately rebuy the same
security) was found in the sources checked.

`bot/taxes.py` models this with mechanics that differ by how a position is
actually held and traded:

- `after_tax_active()` — an annual realize-and-tax simulation (with loss
  carryforward) for anything that rebalances or flips exposure regularly. Uses
  the flat 25% top rate, correctly, since an annual realization is always held
  under 5 years.
- `after_tax_buy_hold()` — tax deferred to a single final sale, at which point
  the real graduated schedule above applies (0% past 15y, but also the
  correct 15%/20% discount for a 10–15y/5–10y hold rather than a flat 25%).
- `after_tax_exposure_based()` — a trade-level version of the schedule for
  binary in/out strategies like trend-exposure: each holding run is taxed, at
  exit, using the graduated rate for *that run's* actual holding period.
- `after_tax_core_satellite()` — splits capital between an untouched
  buy-and-hold **core** (tax-deferred) and an actively-traded **satellite**
  (taxed annually), as two separate tax lots by default, or with
  `cross_offset_losses=True` letting a leftover satellite loss shelter part of
  the core's eventual sale gain (see the tax-loss-harvesting section below).

```bash
python -m bot.aftertax --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
python -m bot.aftertax --mode checkpoints --core-weight 0.7 \
    --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
```

**Headline finding: taxed, the project's best pre-tax strategy loses to plain
buy-and-hold.** On the 4-asset universe (2005–2026, ~21.5y), the min_var+TE
blend's Sharpe drops 1.17→0.86 after tax while buy-and-hold's holding period
clears the 15-year exemption and stays at 0.94 untouched — same story on the
8-asset universe (0.98→0.72 vs. buy-and-hold's untaxed 0.74). Slowing
trend-exposure's turnover (MA periods up to 800 days) narrows the gap but never
closes it.

**But a core-satellite split beats after-tax buy-and-hold, on 3 of 4 universes
tested.** Putting 50–90% of capital in a real, untouched buy-and-hold core and
the rest in the active min_var+TE blend as a satellite comes out ahead of pure
buy-and-hold's after-tax Sharpe — 0.947 vs. 0.938 (4-asset), 0.799 vs. 0.744
(8-asset), and 0.613 vs. 0.544 (15-asset US+international mega-universe, where
it keeps improving all the way to the widest satellite weight tested). This is
the **first construction in this project's history to beat buy-and-hold after
realistic tax treatment.** The intuition: the core's gains stay fully
tax-deferred exactly like pure buy-and-hold, while the taxed satellite still
contributes enough diversification/edge to lift the blend's risk-adjusted
return above what the untouched core alone delivers.

**A real scope limit found on the 4th universe (11 international-country ETFs):
core-satellite does NOT beat pure buy-and-hold there** (every core/satellite
split tested came out slightly below pure B&H, 0.42–0.43 vs. 0.430). This
universe's pre-tax active edge over B&H is thin to begin with (blend 0.52 vs.
B&H 0.43 pre-tax — a much smaller premium than the ~25–60% relative edge seen on
the other 3 universes), so 25% annual tax on the satellite's gains eats the
entire premium even at a small allocation. **Refined rule: core-satellite only
helps when the underlying active strategy's PRE-TAX edge over buy-and-hold is
large enough to survive the tax drag at minority weight — a thin edge isn't
worth the tax cost of any satellite allocation at all**, in which case pure,
literal buy-and-hold is simply the better answer.

**Checked for robustness with `--mode checkpoints`**, since a core-satellite
structure can't be walk-forward-folded in the usual sense (the core's tax
treatment depends on one continuous multi-decade hold, not independent
periods). Instead this measures the SAME continuous hold's after-tax Sharpe at
several different end dates (10, 12, 15, 18, 21 years in, each correctly taxed
at the graduated rate for THAT holding period — e.g. buy-and-hold sold at the
10-year mark owes 20%, not a flat 25%). A 70% core / 30% satellite split beat
pure buy-and-hold at **every checkpoint tested, on both primary (4-asset and
8-asset) universes (10/10)** — including well before the core itself reaches
the 15-year exemption, where naive intuition might expect the tax drag to look
worse.

**Tax-loss harvesting on the satellite — tried, and the obvious version is a
proven no-op; the real lever is narrower than it sounds.** The first idea —
harvest a loss the moment it occurs during the year (sell and immediately
rebuy a similar instrument, unrestricted since no wash-sale rule was found)
instead of only netting at year-end — was implemented and tested, then
DISPROVEN by algebra and a direct test: since `after_tax_active` already taxes
the whole year's NET gain with full loss carryforward, moving the bookkeeping
of a mid-year dip earlier changes nothing about the year-end number (taxable
gain is always `nav_end − year_start_basis`, regardless of how many times the
cost basis got reset in between). The one place intra-year loss timing
genuinely matters — Slovenian tax nets gains/losses across a taxpayer's
holdings within the same filing, not per security lot — is letting a
**leftover satellite loss** (banked via carryforward but never absorbed by a
later satellite gain) shelter part of the **core's** gain when the core is
finally sold (`cross_offset_losses=True`, shown as the `+crossoffst` column in
`--mode checkpoints`). Even this real, legally-grounded lever turned out to
provide **zero measurable benefit for the actual min_var+TE satellite** on
both primary universes at every checkpoint tested (10/10 identical with and
without it) — the satellite is simply too consistently profitable to carry a
persistent unused loss into a later year. A weaker or more volatile satellite
strategy might see a real effect here; this one doesn't.

**A second, comparably large factor: currency risk (`bot/currency.py`).**
Every number in this project — including everything above this paragraph —
has quietly assumed a USD-based investor. SPY/QQQ/GLD/TLT are USD-denominated;
the user is Slovenian (EUR-based). If those ETFs are actually held through a
USD brokerage account, the REAL return that lands in a EUR-based investor's
pocket is the USD return adjusted for the EUR/USD exchange-rate move, not the
raw USD return everywhere else in this project reports — and Slovenian
capital-gains tax is naturally computed on that EUR-denominated gain (cost
basis and sale price both converted to EUR at their transaction dates), so the
honest order of operations is: convert to EUR first, THEN apply the tax model.

```bash
python -m bot.aftertax --currency eur_naive --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
python -m bot.aftertax --currency eur_hedged --symbols SPY QQQ GLD TLT --start 2005-01-01 --data-source yahoo
```

**Unhedged currency risk alone compresses every Sharpe by roughly as much as
tax does** — on the 4-asset universe, buy-and-hold's Sharpe drops from 0.94
(USD) to 0.83 (EUR, unhedged) purely from daily EUR/USD fluctuation, even
though the multi-decade USD-strengthening trend over this window actually
*boosted* EUR-denominated total return (927%→1116%). Two exposure models were
tested: `eur_naive` (FX risk applies at all times, as if idle cash sits in a
USD brokerage account) and `eur_smart` (FX risk only while actually invested
in USD assets — trend-exposure's cash periods are assumed converted back to
EUR, removing FX risk while flat). The smarter model gives the active blend a
real boost (its own Sharpe rises from 0.80 to 0.87 pre-tax, since dodging
market downturns via trend-exposure also happens to dodge some USD downside
days) — but it isn't enough:

**Combined with tax, unhedged currency risk REVERSES the core-satellite
finding on the two moderate-edge universes — plain buy-and-hold becomes the
best choice there, beating every core-satellite split, under BOTH exposure
models.** 4-asset universe (`eur_naive`): buy-and-hold's after-tax-after-FX
Sharpe is 0.826, and the core-satellite Sharpe falls MONOTONICALLY as
satellite weight increases (0.818 at 90/10, all the way down to 0.615 at
0/100) — every single split underperforms pure buy-and-hold. Same shape holds
for `eur_smart` (0.826 vs. a monotonic decline to 0.659) and on the 8-asset
universe under both models. The core-satellite edge found in the tax-only
analysis above was real but was implicitly relying on a USD-based investor;
once real currency risk is priced in for the user's actual EUR base currency,
it evaporates on these two universes.

**But this isn't universal — it tracks the same "edge size" rule already
found for tax alone.** Re-checked on the other 2 universes under `eur_naive`:
on the 15-asset mega-universe (the LARGEST pre-tax edge found anywhere in this
project), core-satellite still narrowly beats buy-and-hold even with unhedged
currency risk (0.579 vs. 0.571 at 50% core) — its edge is simply too large for
tax + currency combined to fully erase. On the international-11 universe
(already the thinnest edge, and already a loser under tax alone), it remains a
loser under currency too (0.463 best vs. 0.466 for pure B&H). **Refined rule:
whether core-satellite survives BOTH tax and unhedged currency risk depends on
the same edge-size threshold as the tax-only analysis — thin/moderate edges
(4-asset, 8-asset, international) don't survive, but a large enough edge
(mega-universe) still does.**

**Currency-hedging (`--currency eur_hedged`, a fixed 1.5%/yr cost drag
approximating the historical USD-EUR short-rate differential via covered
interest rate parity) partially restores the core-satellite edge on the two
universes where unhedged currency reversed it, but much thinner than the
tax-only picture suggested.** 4-asset universe: core-satellite peaks at 0.820
(60–70% core) vs. pure buy-and-hold's 0.817 — a real but tiny margin, not the
0.948-vs-0.938 gap found pre-currency. 8-asset universe: peaks at 0.649 (50%
core) vs. 0.634 — similarly thin. **Practical conclusion: for this user's
actual situation (Slovenian, EUR-based) trading the 4-asset or 8-asset
universe, core-satellite is only worth pursuing at all if currency-hedged
(e.g. EUR-hedged ETF share classes, a real, commercially available product in
Europe) — left unhedged, plain, literal buy-and-hold is simply the better,
simpler answer there. On a large-enough-edge universe (the mega-universe),
core-satellite is worth it even unhedged, though a real implementation would
still be safer hedged given how much currency volatility eats into the
margin.**

**Honest caveats on the currency model:** the 1.5%/yr hedging cost is a fixed,
illustrative approximation (matching this project's existing convention for
`bot.combo`'s leverage borrow-cost), not a fitted or historically-varying
USD-EUR rate differential — a real hedge's cost moved a lot over 2005–2026 (US
rates were near zero for much of 2009–2015, then well above EUR rates
2022–2026), so actual results from a real EUR-hedged product would vary by
period. The `eur_smart` exposure model assumes idle capital is fully converted
back to EUR the moment a position is closed, which requires actually managing
currency separately from the USD brokerage account — not automatic with a
typical US broker.

**Widened the grid (0–100% core in 10% steps) — there is no single universal
split, it depends on how large the active edge is.** The optimum shifts by
universe: 4-asset peaks near 60% core (Sharpe 0.948), 8-asset peaks near
20–30% core (0.799, a much more aggressive satellite weight), and on the
mega-universe the peak is 0% core — i.e. **going fully active (no B&H core at
all) is the best after-tax choice there**, because that universe's active edge
is so large (pre-tax blend 0.85 vs. B&H's pre-tax 0.54) that paying 25% tax on
it every year still beats any blend with buy-and-hold. Practical rule: the
bigger the strategy's pre-tax edge over buy-and-hold, the MORE weight the
satellite deserves — up to and including 100% on a strong enough edge; on a
thin edge (the international universe), the correct satellite weight is 0%.
Don't treat any single ratio (e.g. 70/30) as a universal recommendation —
`--mode score` prints the full curve so you can see where your own universe
falls, and picking a round default (50%) before looking is safer than chasing
the exact per-universe peak.

**Also tested: a slower, less-taxed satellite doesn't help.** Swapping the
satellite from the daily-rebalanced min_var+TE blend to pure trend-exposure
alone (taxed only at actual exits via `after_tax_exposure_based`, not
blanket-annual) sounded like it should reduce tax drag further — but it
underperforms the blend satellite on both primary universes anyway (e.g. 0.943
vs. 0.947 on the 4-asset universe at 70% core). The blend's stronger standalone
edge matters more than its less-favorable tax timing; don't swap to a
"tax-efficient but weaker" satellite hoping the timing advantage wins out.

**Honest caveats:** the annual-realization model approximates real per-trade
tax-lot accounting (exact holding periods and cost basis per individual trade
aren't tracked) — the economically dominant effect, that active strategies pay
tax almost every year and buy-and-hold doesn't, holds regardless. The
graduated schedule is applied as a cliff on the WHOLE gain based on total
holding period, matching the official description ("reduced after every five
years") rather than a marginal-bracket split — if that's ever clarified
otherwise, `slovenia_rate_for_holding`/`SLOVENIA_SCHEDULE` is the one place to
fix it. No Slovenian wash-sale rule was found in the sources checked, so
loss-harvest-and-immediately-rebuy is treated as unrestricted — if one exists
and is later confirmed, it would remove `cross_offset_losses`' (already
empirically negligible) benefit entirely. This is a personal-finance model,
not tax advice — verify current rates and rules before acting on them.

Sources: [Financial Administration of the Republic of Slovenia — disposal of
securities](https://www.fu.gov.si/en/life_events_individuals/disposal_of_securities_other_holdings_or_investment_coupons),
[Tax Foundation Europe — capital gains tax rates](https://taxfoundation.org/data/all/eu/capital-gains-tax-rates-europe/).

## Volatility risk premium (tested, negative)

A different, well-documented risk premium not tried anywhere else in this
project: systematically selling volatility (short-VIX-futures products earn a
persistent premium historically, since implied vol tends to run above realized
vol). Tested with `SVXY` (ProShares Short VIX Short-Term Futures, real
tradeable history since Oct 2011), blended into the min_var+TE combo at small
weights (5–20%).

**Standalone, SVXY looks decent on Sharpe alone (0.57) but has catastrophic
tail risk** — a -95% max drawdown and a single day (2018-02-06, the
"Volpocalypse") that lost 83% of its value. **Blended into the existing
min_var+TE book, it makes things WORSE almost everywhere**: Sharpe declines
monotonically as SVXY weight increases on the 4-asset universe (1.168 → 1.124
→ 1.044 → 0.965 → 0.898 at 0/5/10/15/20%), and max drawdown roughly doubles by
20% weight (-16%→-32%). The 8-asset universe shows a marginal improvement at
5% (0.906→0.924) but the same clear degradation by 10–15% — inconsistent
enough between universes to distrust the small win, and dominated anyway by
the standalone tail risk.

**Why it fails the project's established diversification rule** (a good
diversifier needs BOTH low correlation AND standalone quality/safety — see the
RP+TE combo section): short-vol strategies crash exactly when equity markets
crash, which is precisely when risk-parity/trend-following are earning their
keep by NOT crashing — the correlation is bad in exactly the way that matters
most. **Conclusion: don't pursue volatility-risk-premium harvesting as a
diversifier for this project's existing books — the tail-risk correlation
structure works against it, not with it.**

## Testing

Offline test suite (no network, no broker) covering the strategy logic, the
backtesters, the sweep mechanics, the trend-exposure and momentum-rotation models,
the strategy lab (including long/short and macro-regime-conditioned strategies),
the RP+TE combo tool, the overnight/intraday decomposition, the turn-of-month
effect, the benchmark, the Slovenian after-tax comparison, the EUR currency-risk
model, and the live runner's order/stop/reconcile machinery against fakes:

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
