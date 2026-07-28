"""maker-first 진입의 이중진입 가드 검증 (2026-07-04 2차 리뷰 HIGH).

핵심 불변식: 지정가가 살아있을 가능성이 있는 한 시장가 폴백 금지 —
살아있는 지정가 + 시장가 = 최대 2배 포지션(SL/TP는 1배 수량만 커버).
①취소 불가(계속 open) → 진입 스킵, ②시장가 추격 응답 유실 → 맹목 재주문
금지 + 실포지션 조회로 확정, ③정상 타임아웃 추격은 변경 전과 동일.
"""
from __future__ import annotations

import ccxt
import pandas as pd
import pytest

from execution.live_broker import LiveBroker
from execution.models import Order, OrderSide, OrderType


class _FakeExchange:
    def __init__(self, *, cancel_fails=False, market_raises=False,
                 position_after=None, market_fill_after=None):
        self.cancel_fails = cancel_fails      # cancel_order/cancel_all 전부 실패
        self.market_raises = market_raises    # 시장가 create_order 응답 유실
        # create_order 응답에는 평단이 없고 fetch_order에만 (qty, avg)가 있는 경우
        self.market_fill_after = market_fill_after
        # fetch_positions가 보고할 (contracts, entryPrice) — None이면 무포지션
        self.position_after = position_after
        self.orders = []                      # 접수 성공한 (type, side, qty)
        self.market_attempts = 0
        self._canceled = False

    def price_to_precision(self, s, p):
        return f"{float(p):.4f}"

    def amount_to_precision(self, s, q):
        return f"{float(q):.3f}"

    def market(self, symbol):
        return {"limits": {"amount": {"min": 0.001}, "cost": {"min": 1.0}}}

    def set_leverage(self, lev, symbol):
        pass

    def create_order(self, symbol, type_, side, qty, price=None, params=None):
        if type_ == "market":
            self.market_attempts += 1
            if self.market_raises:
                raise ccxt.NetworkError("response lost")
            if self.market_fill_after is not None:
                self.orders.append((type_, side, float(qty)))
                return {"id": "mkt1", "average": None, "price": None,
                        "filled": float(qty)}
        self.orders.append((type_, side, float(qty)))
        return {"id": "oid1", "average": 100.0, "filled": float(qty)}

    def fetch_order(self, oid, symbol):
        if oid == "mkt1":
            qty, avg = self.market_fill_after
            return {"status": "closed", "filled": qty, "average": avg}
        status = "canceled" if self._canceled else "open"
        return {"status": status, "filled": 0.0, "average": None}

    def cancel_order(self, oid, symbol):
        if self.cancel_fails:
            raise ccxt.NetworkError("cancel down")
        self._canceled = True

    def cancel_all_orders(self, symbol):
        if self.cancel_fails:
            raise ccxt.NetworkError("cancel down")
        self._canceled = True

    def fetch_open_orders(self, symbol):
        return [] if self._canceled else [{"id": "oid1"}]

    def fetch_positions(self, symbols=None):
        if self.position_after is None:
            return []
        contracts, ep = self.position_after
        return [{"symbol": "ETH/USDT:USDT", "contracts": contracts,
                 "entryPrice": ep}]


def _order():
    return Order(
        symbol="ETHUSDT", side=OrderSide.BUY, size_usd=500.0, price=100.0,
        order_type=OrderType.MARKET, leverage=5, strategy="ema_cross",
        signal_score=None, timestamp=pd.Timestamp("2026-07-04", tz="UTC"),
        direction="long", tp_price=110.0, sl_price=95.0,
    )


def _broker(ex):
    return LiveBroker(exchange=ex, dry_run=False,
                      maker_timeout_sec=0.03, maker_poll_sec=0.01)


def _markets(ex):
    return [o for o in ex.orders if o[0] == "market"]


def test_cancel_impossible_skips_entry_no_market_chase():
    """지정가 취소 불가(계속 open) → 시장가 추격 절대 금지 + 진입 스킵.
    회귀 대상: 가드 RuntimeError가 except Exception에 삼켜져 None(시장가 폴백)."""
    ex = _FakeExchange(cancel_fails=True)  # 무포지션 → 스킵 판정
    with pytest.raises(ccxt.NetworkError):
        _broker(ex).submit(_order())
    assert ex.market_attempts == 0  # 시장가 시도 자체가 없어야 함
    assert _markets(ex) == []


def test_chase_response_lost_no_blind_reorder_adopts_position():
    """시장가 추격 응답 유실 → 맹목 재주문 금지, 실포지션 조회로 채택."""
    ex = _FakeExchange(market_raises=True, position_after=(5.0, 100.0))
    fill = _broker(ex).submit(_order())
    assert ex.market_attempts == 1          # 두 번째 시장가(=2배 진입)가 없어야 함
    assert fill.fill_price == pytest.approx(100.0)
    assert fill.order.size_usd == pytest.approx(500.0)  # 실수량 5.0 × 100
    # 채택 후 TP가 실수량으로 등록됨 (limit: [0]=진입 지정가, [-1]=TP reduceOnly)
    tp = [o for o in ex.orders if o[0] == "limit"][-1]
    assert tp[1] == "sell" and tp[2] == pytest.approx(5.0)


def test_normal_timeout_chase_still_works():
    """정상 경로 회귀 — 타임아웃 → 취소 성공 → 잔량 시장가 추격."""
    ex = _FakeExchange()
    fill = _broker(ex).submit(_order())
    assert ex.market_attempts == 1
    (mkt,) = _markets(ex)
    assert mkt[2] == pytest.approx(5.0)
    assert fill.fill_price == pytest.approx(100.0)


def test_timeout_chase_requeries_missing_market_average():
    """Binance 시장가 응답의 평단이 비어도 주문 재조회 실체결가를 기록한다."""
    ex = _FakeExchange(market_fill_after=(5.0, 101.0))
    fill = _broker(ex).submit(_order())

    assert fill.fill_price == pytest.approx(101.0)
    assert fill.order.size_usd == pytest.approx(505.0)
    # 신호가 대비 +1 체결 슬리피지를 보호가격에도 동일하게 반영한다.
    assert fill.order.tp_price == pytest.approx(111.0)
    assert fill.order.sl_price == pytest.approx(96.0)


def test_direct_market_requeries_missing_market_average():
    """maker 비활성 순수 시장가 경로에도 같은 체결가 확정을 적용한다."""
    ex = _FakeExchange(market_fill_after=(5.0, 101.0))
    broker = LiveBroker(exchange=ex, dry_run=False, maker_timeout_sec=0.0)
    fill = broker.submit(_order())

    assert ex.market_attempts == 1
    assert fill.fill_price == pytest.approx(101.0)
    assert fill.order.size_usd == pytest.approx(505.0)
