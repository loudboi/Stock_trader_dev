# alpaca-bot — live pullback executor and historical research tools

This directory contains one live-capable long-only strategy plus several separate
research modules. Historical results are intentionally **not** hard-coded into this
README: rerun a study against a defined data source, window, dependency environment
and cost model instead of treating old output as a permanent fact.

Use Python **3.11 or 3.12**, the versions covered by CI.

## Live-capable strategy: phased trend-pullback

The strategy in `bot/strategies/trend_pullback.py` uses completed daily bars:

- healthy-uptrend gate: close above fast/slow MAs and fast MA above slow MA;
- pullback entry: recent low inside a two-sided band around the fast MA, lighter
  pullback volume, then a rebound above the MA;
- alternate consolidation-breakout entry;
- 30/30/40 pyramiding while the trend extends;
- trend/structure daily exits;
- volatility stop distance `max(min_stop, ATR multiple / price)` fixed when a
  position is initiated so later tranche sizing matches the stop that actually
  protects the position.

Default risk sizing targets 1% account risk for a fully built symbol, subject to
portfolio-wide risk and gross-exposure caps in `config.py`.

### Execution invariants

`bot/live_pullback.py` is designed around broker confirmation rather than optimistic
local assumptions:

- broker/API read errors are errors, not “flat” positions;
- a submitted/accepted market order is not a fill until broker position state
  confirms the change;
- uncertain orders block further strategy orders for that symbol;
- protective-stop cancellation must be confirmed before replacement/close actions;
- live price failure is a data stall, not permission to substitute yesterday's
  close for a real-time stop check;
- untracked shorts are refused by the long-only strategy;
- untracked existing longs require explicit `--adopt-existing`;
- a symbol managed by the state file cannot be silently omitted from a later run;
- paper/live state and output files are separate;
- a process lock prevents two processes from using the same runtime state file.

Alpaca equities and IBKR positions use broker-resident protective stops where the
adapter supports them. Alpaca crypto does not use the same resting-stop path, so
in-process monitoring remains an important limitation for crypto.

## Install and test

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

For IBKR:

```bash
python -m pip install -r requirements-ibkr.txt
pytest -q tests/test_ibkr_adapter.py tests/test_live_pullback.py
```

See `SETUP_GUIDE.md` and `IBKR_SETUP.md` before running either broker path.

## Paper execution

```bash
python -m bot.live_pullback
python -m bot.live_pullback --symbols SPY GLD
```

A real-money Alpaca endpoint additionally requires `--live`. For IBKR:

```bash
python -m bot.live_pullback_ibkr
```

Standard IBKR live/paper ports are detected. A custom port requires
`IBKR_MODE=paper|live`; live mode additionally requires `--live`.

Do not assign the same broker symbol/account position to multiple independent
strategy processes.

## Pullback backtest

```bash
python -m bot.backtest_pullback
python -m bot.backtest_pullback --exec-timeframe none --data-source yahoo \
  --start 2005-01-01 --end 2026-06-01 --symbols SPY QQQ GLD TLT
```

Important model details:

- same-close multi-symbol decisions are planned before any later execution window is
  processed, preventing cross-symbol future-fill leakage;
- a same execution candle that can contain both entry and stop is handled
  conservatively;
- a no-dip intraday entry falls back on the later daily evaluation timing used by
  the live runner rather than an earlier fictional fill;
- a requested universe is not silently changed when data is missing;
- buy-and-hold comparisons start when all requested sleeves are actually investable,
  preventing late-inception capital from appearing out of nowhere;
- `--risk-free` makes the Sharpe risk-free assumption explicit;
- transaction costs remain a simplified model and should be stress-tested.

Yahoo adjusted daily data is useful for long-horizon regime studies but is not
bar-identical to Alpaca live data.

## True parameter walk-forward

`bot/sweep.py` is the module that actually performs parameter selection in sample
and scoring out of sample:

```bash
python -m bot.sweep --mode grid --months 18
python -m bot.sweep --mode walk --folds 4 --data-source yahoo --start 2005-01-01
```

In walk mode, the in-sample half is end-exclusive and the out-of-sample half begins
at that boundary, so the same timestamp is not scored in both samples.

## Other research modules

These are research tools, not live strategies:

- `bot/trend_exposure.py` — long/cash moving-average exposure;
- `bot/momentum_rotation.py` — cross-asset momentum rotation;
- `bot/lab.py` — risk/vol/trend/long-short strategy comparison;
- `bot/combo.py` — fixed-capital risk-book + trend-exposure blend;
- `bot/regime.py`, `bot/dynamic_combo.py` — regime-conditioned experiments;
- `bot/seasonality.py` — calendar effects including turn-of-month;
- `bot/overnight.py` — overnight/intraday return decomposition;
- `bot/crypto_sleeve.py` — small trend-filtered BTC-sleeve scenarios;
- `bot/taxes.py`, `bot/aftertax.py`, `bot/currency.py` — tax/currency scenario tools.

### Research allocation rules

Multi-asset tools use fixed capital allocations when that is the stated portfolio
construction. If an asset/market has no observation on a date, its capital stays
idle; it is not implicitly reallocated to the other assets through pandas'
`mean(skipna=True)` behavior.

`bot/data.py` does not silently rewrite every >15% move. Large moves are flagged;
a row is automatically quarantined only when it is unusable (for example a
non-positive price) or has the isolated spike/reversal pattern expected from a bad
vendor tick. Quarantines remove the whole OHLCV row and record timestamps in frame
metadata.

## Tax and currency scenarios

Tax modules are approximations, not tax advice or filing software. They do not have
complete FIFO tax lots, normalized acquisition/disposal expenses, all legal loss-
offset facts, or enough information to decide whether activity constitutes a
business.

The Slovenian assumptions were rechecked in August 2026. The code models the
securities holding-period schedule using calendar anniversaries and does **not**
assume a generic future-year carry-forward for ordinary securities losses. It also
documents the 30-day replacement-capital restriction that aggregate return series
cannot enforce.

For virtual currencies, `CRYPTO_TAX_RATE` reflects the current FURS non-business-
individual scenario in the research code. The separately proposed 25% crypto regime
is kept only as `CRYPTO_PROPOSED_TAX_RATE` for a hypothetical scenario; the project
does not describe that proposal as enacted law or invent a grandfathering rule.
Recheck current law before relying on any tax output.

EUR conversion is causal: FX data may be forward-filled from a known quote but a
future first quote is never backfilled into earlier portfolio dates. Partial USD
exposure is converted as an exact EUR-cash/USD-asset wealth recurrence rather than
an approximate FX-return multiplier.

## Reproducibility checklist

Before treating a result as meaningful, record:

1. Git commit SHA.
2. Python version and installed package versions (`python -m pip freeze`).
3. Data vendor/source and download date.
4. Exact symbol universe and date range.
5. Transaction-cost, borrow, hedge, tax and risk-free assumptions.
6. Whether the result is full-sample, chronological consistency, or true fit/test
   walk-forward.
7. Any quarantined/suspect vendor-data timestamps.

The current numeric output is more important than any prose written months earlier.
