"""Telegram 알림 전송기.

진입/청산 이벤트를 텔레그램으로 전송한다. 전송 실패는 거래 플로우를
막지 않도록 조용히 로깅만 한다.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class TelegramNotifier:
    API = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, token: str, chat_id: str, enabled: bool = True) -> None:
        self.token = token
        self.chat_id = str(chat_id)
        self.enabled = enabled and bool(token) and bool(chat_id)

    @classmethod
    def from_env(cls) -> "TelegramNotifier":
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat  = os.environ.get("TELEGRAM_CHAT_ID", "")
        return cls(token, chat, enabled=bool(token and chat))

    def send(self, text: str) -> None:
        if not self.enabled:
            return
        try:
            url = self.API.format(token=self.token)
            data = urllib.parse.urlencode({
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": "true",
            }).encode()
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read())
                if not body.get("ok"):
                    logger.warning("텔레그램 응답 실패: %s", body)
        except Exception as e:
            logger.warning("텔레그램 전송 실패: %s", e)

    # ── 이벤트 ──
    def notify_entry(
        self,
        *,
        symbol: str,
        direction: str,
        fill_price: float,
        size_usd: float,
        leverage: int,
        tp_price: float,
        sl_price: float,
        strategy: str,
        tier: str,
        score: int,
        equity: float,
    ) -> None:
        arrow = "🟢 LONG 진입" if direction == "long" else "🔴 SHORT 진입"
        notional = size_usd * leverage
        equity_pct = (size_usd / equity * 100) if equity > 0 else 0.0
        text = (
            f"<b>{arrow}</b>\n"
            f"종목: <code>{symbol}</code>\n"
            f"가격: <code>{fill_price:,.4f}</code>\n"
            f"마진: <code>${size_usd:,.2f}</code> (자본의 {equity_pct:.1f}%)\n"
            f"노셔널: <code>${notional:,.2f}</code>  레버리지: <b>x{leverage}</b>\n"
            f"TP: <code>{tp_price:,.4f}</code> / SL: <code>{sl_price:,.4f}</code>\n"
            f"전략: {strategy} | Tier <b>{tier}</b> (점수 {score})\n"
            f"잔고: ${equity:,.2f}"
        )
        self.send(text)

    def notify_exit(
        self,
        *,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        size_usd: float,
        leverage: int,
        pnl: float,
        exit_reason: str,
        entry_time: pd.Timestamp,
        exit_time: pd.Timestamp,
        equity: float,
    ) -> None:
        reason_icon = {"tp": "🎯", "sl": "🛑", "timeout": "⏰", "forced_stop": "⚠️"}.get(exit_reason, "📤")
        pct = (exit_price - entry_price) / entry_price * 100
        if direction == "short":
            pct = -pct
        lev_pct = pct * leverage
        pnl_pct_equity = (pnl / equity * 100) if equity > 0 else 0.0
        hold_sec = (exit_time - entry_time).total_seconds()
        h = int(hold_sec // 3600); m = int((hold_sec % 3600) // 60)
        arrow = "📈" if direction == "long" else "📉"
        text = (
            f"<b>{reason_icon} {direction.upper()} 청산 [{exit_reason}]</b>\n"
            f"종목: <code>{symbol}</code> {arrow}\n"
            f"진입: <code>{entry_price:,.4f}</code> → 청산: <code>{exit_price:,.4f}</code> ({pct:+.2f}%)\n"
            f"PnL: <b>{pnl:+,.2f} USDT</b> (레버리지 {lev_pct:+.1f}% / 자본 {pnl_pct_equity:+.2f}%)\n"
            f"보유: {h}h {m}m\n"
            f"잔고: ${equity:,.2f}"
        )
        self.send(text)

    def notify_info(self, text: str) -> None:
        self.send(text)
