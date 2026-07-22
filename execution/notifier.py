"""Telegram 알림 전송기.

진입/청산 이벤트를 텔레그램으로 전송한다. 전송 실패는 거래 플로우를
막지 않도록 조용히 로깅만 한다. send()는 백그라운드 스레드로 비동기 처리.
"""
from __future__ import annotations

import json
import html
import logging
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class _Delivery:
    text: str
    done: threading.Event = field(default_factory=threading.Event)
    ok: bool = False


@dataclass
class _Flush:
    done: threading.Event = field(default_factory=threading.Event)


class TelegramNotifier:
    API = "https://api.telegram.org/bot{token}/sendMessage"
    # Telegram 한도는 4096자. 머리말/HTML 처리 여유를 둔다.
    MAX_MESSAGE_LEN = 3900

    def __init__(self, token: str, chat_id: str, enabled: bool = True) -> None:
        self.token = token
        self.chat_id = str(chat_id)
        self.enabled = enabled and bool(token) and bool(chat_id)
        self._q: queue.Queue[_Delivery | _Flush | None] = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        if self.enabled:
            self._worker_thread = threading.Thread(target=self._worker, daemon=True)
            self._worker_thread.start()

    @classmethod
    def from_env(cls) -> "TelegramNotifier":
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat  = os.environ.get("TELEGRAM_CHAT_ID", "")
        return cls(token, chat, enabled=bool(token and chat))

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                if isinstance(item, _Flush):
                    item.done.set()
                    continue
                item.ok = self._send_sync(item.text)
                item.done.set()
            except Exception:
                # 워커 자체가 죽으면 뒤의 모든 알림이 영구 유실된다. 개별 실패로 격리한다.
                logger.exception("텔레그램 알림 워커 오류")
                if isinstance(item, _Delivery):
                    item.done.set()
            finally:
                self._q.task_done()

    @staticmethod
    def _plain_text(text: str) -> str:
        return html.unescape(re.sub(r"<[^>]*>", "", text))

    @classmethod
    def _split_message(cls, text: str) -> list[str]:
        """가능하면 줄 경계로 나눠 Telegram 4096자 제한을 피한다."""
        if len(text) <= cls.MAX_MESSAGE_LEN:
            return [text]
        chunks: list[str] = []
        current = ""
        for line in text.splitlines(keepends=True):
            while len(line) > cls.MAX_MESSAGE_LEN:
                if current:
                    chunks.append(current.rstrip("\n"))
                    current = ""
                chunks.append(line[:cls.MAX_MESSAGE_LEN])
                line = line[cls.MAX_MESSAGE_LEN:]
            if current and len(current) + len(line) > cls.MAX_MESSAGE_LEN:
                chunks.append(current.rstrip("\n"))
                current = ""
            current += line
        if current:
            chunks.append(current.rstrip("\n"))
        return chunks

    def _request(self, text: str, *, html_mode: bool) -> bool:
        url = self.API.format(token=self.token)
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": "true",
        }
        if html_mode:
            payload["parse_mode"] = "HTML"
        data = urllib.parse.urlencode(payload).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=8) as resp:
            body = json.loads(resp.read())
        if not body.get("ok"):
            raise RuntimeError(f"Telegram API 응답 실패: {body}")
        return True

    def _send_chunk(self, text: str) -> bool:
        for attempt in range(3):
            try:
                return self._request(text, html_mode=True)
            except urllib.error.HTTPError as e:
                # 동적 문자열의 잘못된 태그 등 HTML 파싱 오류여도 알림 자체는 살린다.
                if e.code == 400:
                    try:
                        return self._request(self._plain_text(text), html_mode=False)
                    except Exception as fallback_error:
                        logger.warning("텔레그램 일반문자 폴백 실패: %s", fallback_error)
                        return False
                if e.code != 429 and e.code < 500:
                    logger.warning("텔레그램 HTTP 실패(%s): %s", e.code, e)
                    return False
                retry_after = e.headers.get("Retry-After") if e.headers else None
                delay = float(retry_after) if retry_after else 0.5 * (2 ** attempt)
                logger.warning("텔레그램 일시 오류(%s), %.1f초 후 재시도", e.code, delay)
                time.sleep(min(delay, 5.0))
            except (urllib.error.URLError, TimeoutError, OSError, RuntimeError,
                    json.JSONDecodeError) as e:
                if attempt == 2:
                    logger.warning("텔레그램 전송 실패(3회): %s", e)
                    return False
                time.sleep(0.5 * (2 ** attempt))
        return False

    def _send_sync(self, text: str) -> bool:
        ok = True
        for chunk in self._split_message(text):
            # 앞 조각 실패가 뒤 조각 전송까지 막지 않게 모두 시도한다.
            ok = self._send_chunk(chunk) and ok
        return ok

    def send(self, text: str) -> None:
        if not self.enabled:
            return
        self._q.put(_Delivery(str(text)))

    def send_and_wait(self, text: str, timeout: float = 20.0) -> bool:
        """전송 완료와 성공 여부가 필요한 단발성 스크립트용."""
        if not self.enabled:
            return False
        item = _Delivery(str(text))
        self._q.put(item)
        return item.done.wait(timeout) and item.ok

    def flush(self, timeout: float = 20.0) -> bool:
        """호출 시점까지 큐에 들어온 알림이 모두 처리될 때까지 기다린다."""
        if not self.enabled:
            return True
        marker = _Flush()
        self._q.put(marker)
        return marker.done.wait(timeout)

    def close(self, timeout: float = 20.0) -> bool:
        if not self.enabled or self._worker_thread is None:
            return True
        flushed = self.flush(timeout)
        self._q.put(None)
        self._worker_thread.join(timeout)
        closed = not self._worker_thread.is_alive()
        if closed:
            self.enabled = False
        return flushed and closed

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
        symbol_text = html.escape(str(symbol))
        strategy_text = html.escape(str(strategy))
        tier_text = html.escape(str(tier))
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
            f"종목: <code>{symbol_text}</code>\n"
            f"가격: <code>{fill_price:,.4f}</code>\n"
            f"노셔널: 자본의 <b>{equity_pct:.1f}%</b>  "
            f"레버리지: <b>x{leverage}</b>  마진: 자본의 {margin_pct:.1f}%\n"
            f"TP: <code>{tp_price:,.4f}</code> (+{tp_dist:.1f}%) / "
            f"SL: <code>{sl_price:,.4f}</code> (-{sl_dist:.1f}%)  "
            f"R:R <b>{rr:.2f}</b>\n"
            f"전략: {strategy_text} | Tier <b>{tier_text}</b> (점수 {score})"
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
            "forced":      ("⚠️", "강제청산"),
            "early_exit":  ("↩️", "조기청산"),
            "external_close": ("🔄", "거래소 청산 동기화"),
            "deep_floor":  ("⛔", "딥플로어 청산"),
            "liquidated":  ("💥", "강제청산 감지"),
        }
        icon, reason_kr = reason_map.get(exit_reason, ("📤", exit_reason))
        symbol_text = html.escape(str(symbol))
        strategy_text = html.escape(str(strategy))
        tier_text = html.escape(str(tier))
        reason_text = html.escape(str(exit_reason))
        reason_kr_text = html.escape(str(reason_kr))

        pct = (exit_price - entry_price) / entry_price * 100
        if direction == "short":
            pct = -pct
        lev_pct = pct * leverage
        pnl_pct_equity = pnl / equity * 100 if equity > 0 else 0.0
        hold_sec = (exit_time - entry_time).total_seconds()
        h = int(hold_sec // 3600)
        m = int((hold_sec % 3600) // 60)
        arrow = "📈" if direction == "long" else "📉"

        strat_info = f"  <i>{strategy_text} {tier_text}({score})</i>" if strategy else ""
        slip_info = ""
        if entry_slip_pct is not None:
            fee_pct = commission / size_usd * 100 if size_usd > 0 else 0.0
            slip_info = f"\n슬리피지(진입): {entry_slip_pct:+.3f}% · 수수료: {fee_pct:.3f}%"
        cum_info = ""
        if cum_trades > 0:
            cum_info = (f"\n누적: {cum_trades}건 | WR {cum_wr:.1f}% | "
                        f"PnL ${cum_pnl:+,.2f}")

        text = (
            f"<b>{icon} {direction.upper()} {reason_kr_text} [{reason_text}]</b>\n"
            f"종목: <code>{symbol_text}</code> {arrow}{strat_info}\n"
            f"진입: <code>{entry_price:,.4f}</code> → "
            f"청산: <code>{exit_price:,.4f}</code> ({pct:+.2f}%)\n"
            f"PnL: <b>${pnl:+,.2f} · 자본 {pnl_pct_equity:+.2f}%</b>  "
            f"(레버리지 {lev_pct:+.1f}%)\n"
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
            f"💓 <b>Heartbeat</b> ({interval_h}h · 봉 #{bar_count})\n"
            f"자산: 시작 대비 <b>{total_pct:+.1f}%</b>\n"
            f"누적: {trades}건 | WR {wr_pct:.1f}% | PF {pf_str} | PnL ${cum_pnl:+,.2f}\n"
            f"포지션: {positions}개 오픈"
        )
        self.send(text)

    def notify_info(self, text: str) -> None:
        self.send(text)
