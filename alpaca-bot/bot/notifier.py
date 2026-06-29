"""
bot/notifier.py
===============
Best-effort outbound alerts for the live runner. Dependency-free (stdlib only)
and fail-safe: if it isn't configured, or a send fails, it never raises into the
trading loop. Sends happen on a daemon thread so a slow endpoint can't stall a
cycle.

Configure via env (any/all):
  ALERT_WEBHOOK_URL   POST {"text": "<msg>"} to this URL (Slack-compatible).
  TELEGRAM_BOT_TOKEN  + TELEGRAM_CHAT_ID  -> Telegram sendMessage.

With none set, Notifier is a silent no-op (it still logs locally).
"""

import json
import logging
import os
import threading
import urllib.request

log = logging.getLogger("notifier")


class Notifier:
    def __init__(self, webhook_url=None, telegram_token=None, telegram_chat_id=None,
                 timeout=5):
        self.webhook_url = webhook_url or os.getenv("ALERT_WEBHOOK_URL")
        self.telegram_token = telegram_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.timeout = timeout
        self.enabled = bool(self.webhook_url or
                            (self.telegram_token and self.telegram_chat_id))
        if self.enabled:
            log.info("Alerts enabled (%s).", ", ".join(self._channels()))
        else:
            log.info("Alerts not configured; running silently.")

    def _channels(self):
        ch = []
        if self.webhook_url:
            ch.append("webhook")
        if self.telegram_token and self.telegram_chat_id:
            ch.append("telegram")
        return ch

    def notify(self, message: str) -> None:
        """Fire-and-forget. Returns immediately; sends on a background thread."""
        if not self.enabled:
            return
        threading.Thread(target=self._send_all, args=(message,), daemon=True).start()

    # ------------------------------------------------------------------ #
    def _send_all(self, message: str) -> None:
        if self.webhook_url:
            self._post(self.webhook_url, {"text": message})
        if self.telegram_token and self.telegram_chat_id:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            self._post(url, {"chat_id": self.telegram_chat_id, "text": message})

    def _post(self, url: str, payload: dict) -> None:
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=self.timeout).read()
        except Exception as e:  # noqa: BLE001 - alerting must never crash trading
            log.debug("Alert send failed (%s): %s", url.split("/")[2], e)
