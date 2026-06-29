# Deploying Strategy 4 on a Linux VPS (systemd)

Runs the live runner as an auto-restarting systemd service, logging to the
journal. Two units are provided: the Alpaca runner and the IBKR (EUR) runner.
Run **paper** until you trust it.

Paths below assume the app at `/opt/alpaca-bot`, a venv at `/opt/alpaca-bot/.venv`,
and a dedicated `alpaca` user. Adjust to taste (and update the `.service` files
if you change them).

## 1. User + code

```bash
sudo useradd --system --create-home --home-dir /opt/alpaca-bot --shell /usr/sbin/nologin alpaca
sudo -u alpaca git clone https://github.com/loudboi/Stock_trader_dev.git /tmp/st && \
  sudo mv /tmp/st/alpaca-bot/* /opt/alpaca-bot/ && sudo chown -R alpaca:alpaca /opt/alpaca-bot
# (or rsync your working copy of alpaca-bot/ into /opt/alpaca-bot)
```

## 2. Virtualenv + dependencies

```bash
sudo -u alpaca python3 -m venv /opt/alpaca-bot/.venv
sudo -u alpaca /opt/alpaca-bot/.venv/bin/pip install --upgrade pip
sudo -u alpaca /opt/alpaca-bot/.venv/bin/pip install -r /opt/alpaca-bot/requirements.txt
# IBKR runner only:
sudo -u alpaca /opt/alpaca-bot/.venv/bin/pip install -r /opt/alpaca-bot/requirements-ibkr.txt
```

## 3. Secrets / config

```bash
sudo mkdir -p /etc/alpaca-bot
sudo cp /opt/alpaca-bot/deploy/pullback.env.example /etc/alpaca-bot/pullback.env
sudo nano /etc/alpaca-bot/pullback.env          # fill in keys (+ optional alerts)
sudo chown alpaca:alpaca /etc/alpaca-bot/pullback.env
sudo chmod 600 /etc/alpaca-bot/pullback.env
# IBKR runner: same with pullback-ibkr.env.example -> /etc/alpaca-bot/pullback-ibkr.env
```

Alerts are optional. Set `ALERT_WEBHOOK_URL` (Slack-compatible) and/or
`TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` to get pings on start/stop, every
entry/exit, a data stall, or a processing error. Leave blank to run silently.

## 4. Install + start the service

```bash
sudo cp /opt/alpaca-bot/deploy/alpaca-pullback.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now alpaca-pullback.service
```

To pass flags (e.g. a symbol subset or `--live`), edit `ExecStart` in the unit,
e.g. `... -m bot.live_pullback --symbols SPY GLD`. After editing a unit, run
`sudo systemctl daemon-reload && sudo systemctl restart alpaca-pullback`.

## 5. Watch it

```bash
systemctl status alpaca-pullback
journalctl -u alpaca-pullback -f          # live logs
journalctl -u alpaca-pullback --since today
```

Runtime CSVs/state (`pullback_trades.csv`, `pullback_daily_pnl.csv`,
`pullback_state.json`) are written under `WorkingDirectory` (`/opt/alpaca-bot`).
State is restart-safe and reconciled against the broker on startup.

## 6. Updating

```bash
sudo -u alpaca git -C /opt/alpaca-bot pull        # or rsync a new copy
sudo -u alpaca /opt/alpaca-bot/.venv/bin/pip install -r /opt/alpaca-bot/requirements.txt
sudo systemctl restart alpaca-pullback
```

Stops are graceful: systemd sends SIGTERM, the runner finishes the cycle and
writes the day's P&L (TimeoutStopSec=90 covers the ≤60s poll loop).

## 7. The IBKR runner — extra caveats

`alpaca-pullback-ibkr.service` is the same idea, but it needs **IB Gateway/TWS
running and logged in** underneath it. Two things differ from the clean Alpaca
deploy (see `../IBKR_SETUP.md`):

- **2FA on Gateway login** means an unattended reboot won't silently restore it.
  Keep Gateway up for long stretches, or automate login with **IBC**. The bot's
  `Restart=always` still applies, but it depends on Gateway being up.
- **Memory.** Gateway is a ~4 GB Java app; size the VPS accordingly.

```bash
sudo cp /opt/alpaca-bot/deploy/alpaca-pullback-ibkr.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now alpaca-pullback-ibkr.service
```

> One position per symbol per account. Don't point two runners at the same
> symbol on the same account.
