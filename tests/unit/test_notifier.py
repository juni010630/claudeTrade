import json
import threading
import urllib.error

import pandas as pd

from execution.notifier import TelegramNotifier


def test_disabled_notifier_does_not_start_worker():
    notifier = TelegramNotifier("", "", enabled=True)

    assert not notifier.enabled
    assert notifier._worker_thread is None
    assert notifier.flush()


def test_long_message_is_split_under_telegram_limit():
    text = "\n".join(["x" * 200 for _ in range(50)])

    chunks = TelegramNotifier._split_message(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= TelegramNotifier.MAX_MESSAGE_LEN for chunk in chunks)
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")


def test_dynamic_entry_fields_are_html_escaped(monkeypatch):
    notifier = TelegramNotifier("", "", enabled=False)
    sent = []
    monkeypatch.setattr(notifier, "send", sent.append)

    notifier.notify_entry(
        symbol="A&B<USDT>", direction="long", fill_price=10, size_usd=100,
        leverage=2, tp_price=11, sl_price=9, strategy="x<y & z", tier="S&S",
        score=4, equity=1000,
    )

    assert "A&amp;B&lt;USDT&gt;" in sent[0]
    assert "x&lt;y &amp; z" in sent[0]
    assert "S&amp;S" in sent[0]


def test_send_and_wait_reports_delivery_result(monkeypatch):
    monkeypatch.setattr(TelegramNotifier, "_send_sync", lambda self, text: text == "ok")
    notifier = TelegramNotifier("token", "chat")
    try:
        assert notifier.send_and_wait("ok", timeout=1)
        assert not notifier.send_and_wait("bad", timeout=1)
    finally:
        assert notifier.close(timeout=1)


def test_html_400_falls_back_to_plain_text(monkeypatch):
    notifier = TelegramNotifier("", "", enabled=False)
    calls = []

    def request(text, *, html_mode):
        calls.append((text, html_mode))
        if html_mode:
            raise urllib.error.HTTPError("url", 400, "bad html", {}, None)
        return True

    monkeypatch.setattr(notifier, "_request", request)

    assert notifier._send_chunk("<b>A &amp; B</b>")
    assert calls == [("<b>A &amp; B</b>", True), ("A & B", False)]


def test_exit_message_contains_realized_and_cumulative_pnl(monkeypatch):
    notifier = TelegramNotifier("", "", enabled=False)
    sent = []
    monkeypatch.setattr(notifier, "send", sent.append)
    now = pd.Timestamp("2026-01-01", tz="UTC")

    notifier.notify_exit(
        symbol="BTC/USDT", direction="long", entry_price=100, exit_price=110,
        size_usd=1000, leverage=2, pnl=95, exit_reason="early_exit",
        entry_time=now, exit_time=now + pd.Timedelta(hours=2), equity=1095,
        cum_trades=3, cum_wr=66.7, cum_pnl=123.45,
    )

    assert "$+95.00" in sent[0]
    assert "PnL $+123.45" in sent[0]
    assert "조기청산" in sent[0]
