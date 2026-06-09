"""SL STOP_MARKET 등록 실패 처리: 메인넷=재시도+텔레그램 경보 / testnet=단발 로그.

메인넷에서 거래소 SL이 안 걸리면 그 포지션은 엔진 1h 백업만으로 보호되므로(무방비 위험),
재시도 후에도 실패하면 즉시 경보해 수동 확인을 유도한다.
"""
from __future__ import annotations

import pandas as pd

from execution.live_broker import LiveBroker
from execution.models import Order, OrderSide, OrderType


class _RecNotifier:
    enabled = True

    def __init__(self):
        self.info_msgs = []

    def notify_info(self, text):
        self.info_msgs.append(text)

    def notify_entry(self, **kwargs):
        pass


class _SLFailExchange:
    """진입 시장가·TP limit은 성공, SL STOP_MARKET만 항상 실패."""

    def __init__(self):
        self.stop_calls = 0
        self.limit_calls = 0

    def market(self, symbol):
        return {"limits": {"amount": {"min": 0.0}}}

    def amount_to_precision(self, symbol, qty):
        return f"{float(qty):.3f}"

    def set_leverage(self, lev, symbol):
        pass

    def create_order(self, symbol, type_, side, amount, price=None, params=None):
        if type_ == "STOP_MARKET":
            self.stop_calls += 1
            raise Exception("-4120 STOP_MARKET not supported")
        if type_ == "limit":
            self.limit_calls += 1
            return {}
        if type_ == "market":
            return {"average": 100.0, "fee": {"cost": 0.05}}
        return {}


def _order():
    return Order(
        symbol="BTCUSDT", side=OrderSide.BUY, size_usd=1000.0, price=100.0,
        order_type=OrderType.MARKET, leverage=10, strategy="ema_cross",
        signal_score=None, timestamp=pd.Timestamp("2026-06-09T00:00:00Z"),
        direction="long", tp_price=110.0, sl_price=95.0,
    )


def test_mainnet_sl_failure_retries_and_alerts(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)  # 재시도 대기 스킵
    ex = _SLFailExchange()
    notif = _RecNotifier()
    broker = LiveBroker(ex, dry_run=False, notifier=notif, demo=False)  # 메인넷
    broker.submit(_order())
    assert ex.stop_calls == 3                  # 3회 재시도
    assert len(notif.info_msgs) == 1           # 최종 실패 경보 1회
    assert "SL 등록 실패" in notif.info_msgs[0]


def test_testnet_sl_failure_single_attempt_no_alert():
    ex = _SLFailExchange()
    notif = _RecNotifier()
    broker = LiveBroker(ex, dry_run=False, notifier=notif, demo=True)  # testnet
    broker.submit(_order())
    assert ex.stop_calls == 1                  # 단발 (sl_poller가 5m로 대체)
    assert notif.info_msgs == []               # 경보 없음 (-4120은 정상)
