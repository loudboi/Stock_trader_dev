# Deploying the pullback runner on Linux (systemd)

Use Python **3.11 or 3.12** and start with a paper account. The service files assume
`/opt/alpaca-bot` as the application path. The install below keeps a real Git clone
at `/opt/stock-trader-dev` and makes `/opt/alpaca-bot` a symlink to its
`alpaca-bot/` directory, so later `git pull` operations actually work.

## 1. Create the service user and clone the repository

```bash
sudo useradd --system --create-home --home-dir /var/lib/alpaca --shell /usr/sbin/nologin alpaca
sudo mkdir -p /opt/stock-trader-dev
sudo chown alpaca:alpaca /opt/stock-trader-dev
sudo -u alpaca git clone https://github.com/loudboi/Stock_trader_dev.git /opt/stock-trader-dev
sudo ln -s /opt/stock-trader-dev/alpaca-bot /opt/alpaca-bot
```

Do not copy only the contents of `alpaca-bot/` and then expect `git -C
/opt/alpaca-bot pull` to work; that directory would no longer contain the repository
metadata.

## 2. Create the virtual environment

```bash
sudo -u alpaca python3.12 -m venv /opt/alpaca-bot/.venv
sudo -u alpaca /opt/alpaca-bot/.venv/bin/python -m pip install --upgrade pip
sudo -u alpaca /opt/alpaca-bot/.venv/bin/pip install -r /opt/alpaca-bot/requirements.txt
# IBKR runner only:
sudo -u alpaca /opt/alpaca-bot/.venv/bin/pip install -r /opt/alpaca-bot/requirements-ibkr.txt
```

If your host uses Python 3.11, substitute `python3.11`. CI currently covers 3.11 and
3.12; do not assume a newer interpreter is deployment-tested until CI includes it.

## 3. Install secrets/config outside the repository

```bash
sudo mkdir -p /etc/alpaca-bot
sudo cp /opt/alpaca-bot/deploy/pullback.env.example /etc/alpaca-bot/pullback.env
sudo editor /etc/alpaca-bot/pullback.env
sudo chown alpaca:alpaca /etc/alpaca-bot/pullback.env
sudo chmod 600 /etc/alpaca-bot/pullback.env
```

For IBKR, copy `pullback-ibkr.env.example` to
`/etc/alpaca-bot/pullback-ibkr.env` instead. Keep credentials and alert tokens out of
the Git working tree.

Alerts are optional. `ALERT_WEBHOOK_URL` and/or `TELEGRAM_BOT_TOKEN` plus
`TELEGRAM_CHAT_ID` enable entry/exit, stall and error notifications. Alert delivery
is best-effort and must not be treated as the only operational monitoring channel.

## 4. Install and start the PAPER service

```bash
sudo cp /opt/alpaca-bot/deploy/alpaca-pullback.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now alpaca-pullback.service
```

Inspect it immediately:

```bash
systemctl status alpaca-pullback
journalctl -u alpaca-pullback -f
```

The process writes paper runtime state under `/opt/alpaca-bot`. Live Alpaca and
IBKR modes use separate state/output names. A process lock prevents two instances
from sharing the same runtime state file, but you should still enforce **one strategy
owner per symbol/account** operationally.

Existing broker longs are not adopted automatically. If startup finds an untracked
long, the runner refuses to proceed unless you deliberately use `--adopt-existing`.
Review that broker position before enabling this flag.

## 5. Real-money confirmation

Do not add `--live` until you have deliberately reviewed the service unit, endpoint,
account and symbol list. The runner refuses an Alpaca live endpoint without
`--live`.

For IBKR, standard live/paper ports are recognized. A non-standard port requires
an explicit `IBKR_MODE=paper` or `IBKR_MODE=live`; the process refuses to guess.
The live mode also requires `--live`.

After editing a unit:

```bash
sudo systemctl daemon-reload
sudo systemctl restart alpaca-pullback
```

## 6. Updating safely

Do not update a running strategy by pulling code underneath it. Stop the service,
update the real clone, install dependency changes, run tests, then restart.

```bash
sudo systemctl stop alpaca-pullback
sudo -u alpaca git -C /opt/stock-trader-dev pull --ff-only
sudo -u alpaca /opt/alpaca-bot/.venv/bin/pip install -r /opt/alpaca-bot/requirements.txt
sudo -u alpaca /opt/alpaca-bot/.venv/bin/pip install -r /opt/alpaca-bot/requirements-dev.txt
sudo -u alpaca /opt/alpaca-bot/.venv/bin/pytest -q /opt/alpaca-bot/tests
sudo systemctl start alpaca-pullback
systemctl status alpaca-pullback
```

For IBKR, also reinstall `requirements-ibkr.txt` and restart the IBKR unit. For a
real-money deployment, review the Git diff/commit you are deploying before the
restart rather than automatically tracking arbitrary new `main` changes.

## 7. IBKR-specific operational caveats

`alpaca-pullback-ibkr.service` requires IB Gateway/TWS to be running and logged in.
2FA can prevent unattended login restoration after a reboot. The strategy now uses
broker-resident GTC stop orders, but Gateway/account connectivity is still required
for reconciliation, new entries, stop replacement, and monitoring.

```bash
sudo cp /opt/alpaca-bot/deploy/alpaca-pullback-ibkr.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now alpaca-pullback-ibkr.service
```

See `../IBKR_SETUP.md` before using this path.
