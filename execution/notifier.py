"""Telegram 알림 전송기.

진입/청산 이벤트를 텔레그램으로 전송한다. 전송 실패는 거래 플로우를
막지 않도록 조용히 로깅만 한다. send()는 백그라운드 스레드로 비동기 처리.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
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
        self._q: queue.Queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    @classmethod
    def from_env(cls) -> "TelegramNotifier":
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat  = os.environ.get("TELEGRAM_CHAT_ID", "")
        return cls(token, chat, enabled=bool(token and chat))

    def _worker(self) -> None:
        while True:
            text = self._q.get()
            if text is None:
                break
            self._send_sync(text)

    def _send_sync(self, text: str) -> None:
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

    def send(self, text: str) -> None:
        if not self.enabled:
            return
        self._q.put(text)

    # ── 이벤트 ──────────────────────────────────────────────────────────────

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
        margin = size_usd / leverage if leverage > 0 else size_usd
        equity_pct = size_usd / equity * 100 if equity > 0 else 0.0
        margin_pct = margin / equity * 100 if equity > 0 else 0.0

        # TP/SL 거리 및 R:R
        if direction == "long":
            tp_dist = (tp_price - fill_price) / fill_price * 100
            sl_dist = (fill_price - sl_price) / fill_price * 100
        else:
            tp_dist = (fill_price - tp_price) / fill_price * 100
            sl_dist = (sl_price - fill_price) / fill_price * 100
        rr = tp_dist / sl_dist if sl_dist > 0 else 0.0

        text = (
            f"<b>{arrow}</b>\n"
            f"종목: <code>{symbol}</code>\n"
            f"가격: <code>{fill_price:,.4f}</code>\n"
            f"노셔널: 자본의 <b>{equity_pct:.1f}%</b>  "
            f"레버리지: <b>x{leverage}</b>  마진: 자본의 {margin_pct:.1f}%\n"
            f"TP: <code>{tp_price:,.4f}</code> (+{tp_dist:.1f}%) / "
            f"SL: <code>{sl_price:,.4f}</code> (-{sl_dist:.1f}%)  "
            f"R:R <b>{rr:.2f}</b>\n"
            f"전략: {strategy} | Tier <b>{tier}</b> (점수 {score})"
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
        strategy: str = "",
        tier: str = "",
        score: int = 0,
        cum_trades: int = 0,
        cum_wr: float = 0.0,
        cum_pnl: float = 0.0,
        entry_slip_pct: float | None = None,
        commission: float = 0.0,
    ) -> None:
        reason_map = {
            "tp":          ("🎯", "익절"),
            "sl":          ("🛑", "손절"),
            "timeout":     ("⏰", "타임아웃"),
            "forced_stop": ("⚠️", "강제청산"),
        }
        icon, reason_kr = reason_map.get(exit_reason, ("📤", exit_reason))

        pct = (exit_price - entry_price) / entry_price * 100
        if direction == "short":
            pct = -pct
        lev_pct = pct * leverage
        pnl_pct_equity = pnl / equity * 100 if equity > 0 else 0.0
        hold_sec = (exit_time - entry_time).total_seconds()
        h = int(hold_sec // 3600)
        m = int((hold_sec % 3600) // 60)
        arrow = "📈" if direction == "long" else "📉"

        strat_info = f"  <i>{strategy} {tier}({score})</i>" if strategy else ""
        slip_info = ""
        if entry_slip_pct is not None:
            fee_pct = commission / size_usd * 100 if size_usd > 0 else 0.0
            slip_info = f"\n슬리피지(진입): {entry_slip_pct:+.3f}% · 수수료: {fee_pct:.3f}%"
        cum_info = ""
        if cum_trades > 0:
            cum_info = f"\n누적: {cum_trades}건 | WR {cum_wr:.1f}%"

        text = (
            f"<b>{icon} {direction.upper()} {reason_kr} [{exit_reason}]</b>\n"
            f"종목: <code>{symbol}</code> {arrow}{strat_info}\n"
            f"진입: <code>{entry_price:,.4f}</code> → "
            f"청산: <code>{exit_price:,.4f}</code> ({pct:+.2f}%)\n"
            f"PnL: <b>자본 {pnl_pct_equity:+.2f}%</b>  (레버리지 {lev_pct:+.1f}%)\n"
            f"보유: {h}h {m}m{slip_info}{cum_info}"
        )
        self.send(text)

    def notify_heartbeat(
        self,
        *,
        interval_h: int,
        bar_count: int,
        equity: float,
        initial_capital: float,
        positions: int,
        trades: int,
        wr_pct: float,
        pf: float,
        cum_pnl: float,
    ) -> None:
        total_pct = (equity / initial_capital - 1) * 100 if initial_capital > 0 else 0.0
        pf_str = f"{pf:.2f}" if pf < 999 else "∞"
        text = (
            f"💓 <b>Heartbeat</b> (봉 #{bar_count})\n"
            f"자산: 시작 대비 <b>{total_pct:+.1f}%</b>\n"
            f"누적: {trades}건 | WR {wr_pct:.1f}% | PF {pf_str}\n"
            f"포지션: {positions}개 오픈"
        )
        self.send(text)

    def notify_info(self, text: str) -> None:
        self.send(text)
