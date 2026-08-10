"""
bot/notifier.py
===============
Best-effort outbound alerts for the live runner. Alert delivery is serialized by a
single worker queue so bursts do not spawn unbounded threads. `close()` drains the
queue on graceful shutdown; transient HTTP failures get one retry. Alert failures
still never raise into the trading loop.
"""

import json
import logging
import os
import queue
import threading
import time
import urllib.request

log = logging.getLogger("notifier")


class Notifier:
    def __init__(self, webhook_url=None, telegram_token=None, telegram_chat_id=None,
                 timeout=5, retries=1):
        self.webhook_url = webhook_url or os.getenv("ALERT_WEBHOOK_URL")
        self.telegram_token = telegram_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.timeout = timeout
        self.retries = max(0, int(retries))
        self.enabled = bool(self.webhook_url or
                            (self.telegram_token and self.telegram_chat_id))
        self._queue = queue.Queue()
        self._closed = False
        self._worker = None
        if self.enabled:
            self._worker = threading.Thread(target=self._run, name="trade-alerts", daemon=False)
            self._worker.start()
            log.info("Alerts enabled (%s).", ", ".join(self._channels()))
        else:
            log.info("Alerts not configured; running silently.")

    def _channels(self):
        out = []
        if self.webhook_url:
            out.append("webhook")
        if self.telegram_token and self.telegram_chat_id:
            out.append("telegram")
        return out

    def notify(self, message: str) -> None:
        if self.enabled and not self._closed:
            self._queue.put(str(message))

    def close(self) -> None:
        """Drain queued alerts and stop the worker on a normal process shutdown."""
        if not self.enabled or self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._queue.join()
        if self._worker:
            self._worker.join(timeout=self.timeout * (self.retries + 1) * 2 + 1)

    def _run(self):
        while True:
            message = self._queue.get()
            try:
                if message is None:
                    return
                self._send_all(message)
            finally:
                self._queue.task_done()

    def _send_all(self, message: str) -> None:
        if self.webhook_url:
            self._post(self.webhook_url, {"text": message})
        if self.telegram_token and self.telegram_chat_id:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            self._post(url, {"chat_id": self.telegram_chat_id, "text": message})

    def _post(self, url: str, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    response.read()
                return
            except Exception as e:  # noqa: BLE001
                if attempt >= self.retries:
                    log.warning("Alert delivery failed (%s): %s", url.split("/")[2], e)
                    return
                time.sleep(0.25 * (attempt + 1))
