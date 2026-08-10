# Strategy 4 on Interactive Brokers (EUR-only)

This is a parallel broker path for the same long-only pullback executor. Use Python
3.11/3.12 and paper-test it against IB Gateway/TWS before considering live money.

Key files:

- `bot/portfolio_ibkr.py` — broker/data adapter
- `bot/ibkr_universe.py` — EUR instruments and contract metadata
- `bot/live_pullback_ibkr.py` — IBKR entry point
- `requirements-ibkr.txt` — `ib_async`/timezone dependencies
- `tests/test_ibkr_adapter.py` — offline adapter tests

Paper and live IBKR runs use separate state/output files from Alpaca and from each
other.

## 1. Gateway/TWS prerequisites

IBKR API trading requires a locally reachable IB Gateway or TWS session.

1. Log in to IB Gateway/TWS.
2. Enable socket/API clients and configure trusted IPs.
3. Record the port. Common defaults are:
   - 4002 Gateway paper
   - 4001 Gateway live
   - 7497 TWS paper
   - 7496 TWS live
4. Make sure the account has the required market-data permissions/subscriptions for
   every venue/instrument you intend to use.
5. Install:

```bash
python -m pip install -r requirements.txt -r requirements-ibkr.txt -r requirements-dev.txt
pytest -q tests/test_ibkr_adapter.py tests/test_live_pullback.py
```

## 2. Connection configuration

Paper Gateway example:

```bash
export IBKR_HOST=127.0.0.1
export IBKR_PORT=4002
export IBKR_CLIENT_ID=17
```

Standard ports are classified automatically. For a **non-standard** port, set the
mode explicitly; the runner refuses to guess:

```bash
export IBKR_PORT=12345
export IBKR_MODE=paper        # or live
```

If mode resolves to live, `--live` is also required. Supplying `--live` while the
runtime is explicitly paper is treated as an error rather than an ambiguous hint.

## 3. Verify the EUR contracts

`bot/ibkr_universe.py` contains contract symbols, primary exchanges and currencies.
Treat those as configuration that must be verified in TWS/Contract Details before
live use. A syntactically valid symbol can still resolve to the wrong security if
contract metadata is wrong.

Default categories include European UCITS index ETFs and selected EUR-listed
stocks. Pass `--symbols` to run a subset.

## 4. Paper run

```bash
python -m bot.live_pullback_ibkr
python -m bot.live_pullback_ibkr --symbols STOXX600 DAX SAP
```

The runner uses the same broker-confirmed order/state semantics as the Alpaca path.
An accepted order is not recorded as a fill until the position actually changes.
An existing untracked long requires explicit adoption:

```bash
python -m bot.live_pullback_ibkr --adopt-existing
```

Do that only after inspecting the broker position and deciding that this strategy
should own it.

## 5. Protective stops

The IBKR adapter now places broker-resident **GTC sell stop orders** for managed
long positions and tracks/cancels/replaces those orders when position size changes.
If replacement cancellation cannot be confirmed, the runner does not create a
second stop. If a close cannot safely cancel the protective stop first, the close is
blocked rather than risking a double sell.

A broker-resident stop materially improves downtime protection but does not make the
system fail-proof. Gaps can fill below the stop price, broker/exchange order handling
can fail, and strategy reconciliation still needs Gateway connectivity.

## 6. Market hours and cached data

The adapter reads the exchange timezone and IBKR liquid/trading-hours metadata.
If the timezone/hours state cannot be determined, it now **fails closed** for new
trading actions rather than guessing that the market is open.

Historical data is cached to reduce IBKR pacing pressure. When a refresh fails, a
cache is used only if it is still within the adapter's freshness bound; stale cache
is not silently treated as current history.

Account equity is required in the configured base currency (EUR by default). A USD
`NetLiquidation` row is not silently reinterpreted as EUR for risk sizing.

## 7. VPS operation

IB Gateway/TWS must remain logged in alongside the service. Account 2FA and Gateway
restarts are operational dependencies; a systemd restart of the Python process
cannot recreate an authenticated Gateway session by itself.

Use `deploy/README.md` for the supported Git clone/systemd layout. Monitor both the
Python service and Gateway/TWS. A broker stop is an additional protection layer, not
an excuse to operate without monitoring.

## 8. Live mode

Standard live ports or `IBKR_MODE=live` require:

```bash
python -m bot.live_pullback_ibkr --live
```

Paper-test contract resolution, market data, order quantities, fills, GTC stop
placement/replacement, restart reconciliation, and alerting before adding `--live`
to an unattended service.
