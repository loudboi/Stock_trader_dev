# Strategy 4 on Interactive Brokers (EUR-only)

This is a **parallel** path to the Alpaca setup. It runs Strategy 4 (trend-pullback)
against IBKR using EUR-denominated instruments, so there's no currency conversion
and no FX exposure. None of the Alpaca code is touched — you can keep the Alpaca
bot running while you build and paper-test this.

New files (all separate):
- `bot/portfolio_ibkr.py` — IBKR adapter (same method surface as the Alpaca Portfolio)
- `bot/ibkr_universe.py` — the EUR instruments + their IBKR contract specs
- `bot/live_pullback_ibkr.py` — the runner (reuses the existing trader unchanged)
- `requirements-ibkr.txt` — adds `ib_async`
- `tests/test_ibkr_adapter.py` — offline tests (no Gateway needed)

Outputs are separate too: `pullback_ibkr_trades.csv`, `pullback_ibkr_daily_pnl.csv`,
`pullback_ibkr_state.json`, and a `ibkr_cache/` folder for cached daily bars.

---

## 1. The big architectural difference vs Alpaca

Alpaca is a cloud REST endpoint. **IBKR is not** — your bot talks to a local
**IB Gateway** (or TWS) that must be running and logged in, and *that* talks to
IBKR. Everything below follows from this.

## 2. Prerequisites

1. **IB Gateway** (lighter) or **TWS**, logged into your IBKR account.
2. In Gateway/TWS: **Configure → Settings → API → Settings**:
   - Enable "ActiveX and Socket Clients"
   - Add the bot's IP (e.g. `127.0.0.1`) to Trusted IPs
   - Note the port. Defaults: **4002** = Gateway paper, 4001 = Gateway live,
     7497 = TWS paper, 7496 = TWS live.
   - Increase Memory Allocation to ~4096 MB (avoids crashes on bulk data).
3. **Market-data subscriptions** for the European exchanges you'll trade
   (Xetra/Germany, Euronext). Without them, live prices and historical requests
   come back delayed or empty.
4. Install deps: `pip install -r requirements.txt -r requirements-ibkr.txt`

## 3. Configure the connection (env vars)

```bash
export IBKR_HOST=127.0.0.1
export IBKR_PORT=4002          # paper Gateway. Live ports (4001/7496) need --live
export IBKR_CLIENT_ID=17       # any unique integer per API client
```

## 4. The EUR universe — verify before trading

`bot/ibkr_universe.py` defines the instruments. **The contract codes are sensible
defaults, not gospel** — IBKR symbols/exchanges are exact, and a wrong one fails
the order or resolves the wrong instrument. Confirm each in TWS (right-click →
Contract Details) before trusting it.

Indexes (UCITS ETFs, Xetra, EUR):
- `STOXX600` — iShares STOXX Europe 600 (broad Europe)
- `ESTX50`   — iShares EURO STOXX 50 (eurozone blue-chips)
- `DAX`      — iShares Core DAX (Germany)

Stocks (EUR):
- `SAP` (Xetra), `ASML` (Amsterdam), `SIE` (Xetra), `MC` (LVMH, Paris), `TTE` (Paris)

To add/remove, edit `IBKR_EUR_UNIVERSE` in that file, or pass `--symbols`.

## 5. Run it (paper)

```bash
python -m bot.live_pullback_ibkr                       # default EUR shortlist
python -m bot.live_pullback_ibkr --symbols STOXX600 DAX SAP
```

It refuses a live port unless you also pass `--live`. Expect long quiet stretches —
same daily trend-pullback logic as the Alpaca version, just EUR instruments.

## 6. Verify offline first

```bash
python tests/test_ibkr_adapter.py      # logic tests, no Gateway required
```

## 7. Running on the VPS — what changes from your tmux setup

The bot now needs **IB Gateway running on the VPS** alongside it. Two real
caveats versus the clean Alpaca deployment:

- **Memory.** Gateway is a Java app wanting ~4 GB. Your 4 GB VPS goes from
  comfortable to tight; consider a size bump.
- **The reboot story breaks.** Gateway needs **2FA approval on login**, so an
  unattended reboot won't silently bring it back the way your Alpaca tmux+cron
  does. Either keep Gateway running for long stretches (avoid restarts) or add a
  tool like **IBC** to automate Gateway login. The *bot's* auto-restart loop
  still works, but it depends on Gateway being up underneath it.

The bot is self-healing about short drops: every adapter call reconnects if the
socket dropped, and the run loop sleeps in 1-second steps so the ib_async event
loop is never blocked for long.

## 8. Honest limitations

- **Tested offline only.** The adapter's translation/caching/order logic is unit-
  tested against a fake IB, but it has **not** been run against a real Gateway from
  here. Paper-test it yourself before trusting it, and watch the first orders fill
  the way you expect.
- **Market-hours check is best-effort.** It parses IBKR trading hours and, if it
  can't tell, *allows* the trade (IBKR rejects/queues if truly closed) rather than
  silently blocking. Verify behavior against your exchanges' hours.
- **The in-process-stop caveat still applies** — stops are monitored while the bot
  runs, not resting at the broker. Downtime = unmanaged risk.
- This isn't financial advice. EUR-only removes FX risk but also removes FX upside
  and concentrates you in European markets. Keep it on paper until it behaves.
