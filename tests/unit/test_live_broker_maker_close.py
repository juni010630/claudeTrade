"""LiveBroker.market_close maker-first (타임아웃 청산 한정) 시나리오 검증.

핵심 불변식: 어떤 분기에서도 청산이 보장되거나 예외가 전파(고아 방지)되고,
reduceOnly라 과청산은 구조적으로 불가. allow_maker=False(기존 SL/긴급 경로)는
변경 전과 동일하게 즉시 시장가여야 함.
"""
from __future__ import annotations

import pytest

from execution.live_broker import LiveBroker


class _FakeExchange:
    """maker 청산 시나리오 제어용 — 주문/취소/조회 전부 기록."""

    def __init__(self, *, positions_seq=None, maker_report=("open", 0.0),
                 reject_post_only=False, fetch_order_error=False):
        # fetch_positions 호출마다 순서대로 반환할 잔량 (마지막 값 유지)
        self._pos_seq = list(positions_seq if positions_seq is not None else [1.0])
        self.maker_report = maker_report  # fetch_order가 보고할 (status, filled)
        self.reject_post_only = reject_post_only
        self.fetch_order_error = fetch_order_error
        self.orders = []      # (type, side, qty, params)
        self.cancelled = []
        self.ticker_calls = 0

    def cancel_all_orders(self, symbol):
        self.cancelled.append(("all", symbol))

    def cancel_order(self, oid, symbol):
        self.cancelled.append((oid, symbol))

    def fetch_ticker(self, symbol):
        self.ticker_calls += 1
        return {"last": 100.0}

    def price_to_precision(self, s, p):
        return f"{float(p):.4f}"

    def amount_to_precision(self, s, q):
        return f"{float(q):.3f}"

    def fetch_positions(self, symbols=None):
        qty = self._pos_seq.pop(0) if len(self._pos_seq) > 1 else self._pos_seq[0]
        return [{"contracts": qty}] if qty else []

    def create_order(self, symbol, type_, side, qty, price=None, params=None):
        params = params or {}
        if type_ == "limit" and params.get("postOnly") and self.reject_post_only:
            raise Exception("Order would immediately match and take")
        self.orders.append((type_, side, float(qty), params))
        return {"id": "oid1", "average": price, "filled": 0.0}

    def fetch_order(self, oid, symbol):
        if self.fetch_order_error:
            raise Exception("network down")
        status, filled = self.maker_report
        return {"status": status, "filled": filled, "average": 100.0}


def _broker(ex):
    return LiveBroker(exchange=ex, dry_run=False,
                      maker_timeout_sec=0.03, maker_poll_sec=0.01)


def _limits(ex):
    return [o for o in ex.orders if o[0] == "limit"]


def _markets(ex):
    return [o for o in ex.orders if o[0] == "market"]


def test_maker_full_fill_no_market_order():
    # maker 전량 체결 → 재조회 0 확인 → 시장가 없음
    ex = _FakeExchange(positions_seq=[1.0, 0.0], maker_report=("closed", 1.0))
    _broker(ex).market_close("ETHUSDT", "long", 1.0, allow_maker=True)
    (lim,) = _limits(ex)
    assert lim[3].get("postOnly") and lim[3].get("reduceOnly")
    assert lim[1] == "sell"  # long 청산 = sell
    assert _markets(ex) == []  # 전량 maker — 시장가 없음


def test_partial_999_fill_is_not_full_fill():
    """리뷰 major 회귀: 99.9% 부분체결(status open)을 전량으로 오판하면 더스트 고아.
    타임아웃 취소 → 잔량 0.001을 시장가로 정리해야 함."""
    ex = _FakeExchange(positions_seq=[1.0, 0.001], maker_report=("open", 0.999))
    _broker(ex).market_close("DOGEUSDT", "long", 1.0, allow_maker=True)
    assert any(c[0] == "oid1" for c in ex.cancelled)  # 지정가 취소됨 (방치 금지)
    (mkt,) = _markets(ex)
    assert mkt[2] == pytest.approx(0.001)


def test_full_fill_report_but_dust_on_refetch_gets_marketed():
    """maker가 전량 보고해도 재조회가 진실 — 더스트 발견 시 시장가 정리."""
    ex = _FakeExchange(positions_seq=[1.0, 0.002], maker_report=("closed", 1.0))
    _broker(ex).market_close("ADAUSDT", "long", 1.0, allow_maker=True)
    (mkt,) = _markets(ex)
    assert mkt[2] == pytest.approx(0.002)


def test_maker_partial_then_market_remainder():
    # 1차 조회 1.0 → maker 0.4 체결 → 재조회 0.6 → 시장가 0.6
    ex = _FakeExchange(positions_seq=[1.0, 0.6], maker_report=("open", 0.4))
    _broker(ex).market_close("ETHUSDT", "long", 1.0, allow_maker=True)
    assert len(_limits(ex)) == 1
    (mkt,) = _markets(ex)
    assert mkt[2] == pytest.approx(0.6)
    assert mkt[3].get("reduceOnly")
    assert any(c[0] == "oid1" for c in ex.cancelled)  # 타임아웃 취소 발생


def test_post_only_rejected_falls_back_to_market():
    ex = _FakeExchange(reject_post_only=True)
    _broker(ex).market_close("ETHUSDT", "short", 1.0, allow_maker=True)
    assert _limits(ex) == []
    (mkt,) = _markets(ex)
    assert mkt[1] == "buy" and mkt[2] == pytest.approx(1.0)


def test_timeout_zero_fill_market_full():
    ex = _FakeExchange(positions_seq=[1.0, 1.0], maker_report=("open", 0.0))
    _broker(ex).market_close("ETHUSDT", "long", 1.0, allow_maker=True)
    (mkt,) = _markets(ex)
    assert mkt[2] == pytest.approx(1.0)


def test_fetch_order_error_cleans_up_and_markets_full():
    ex = _FakeExchange(positions_seq=[1.0, 1.0], fetch_order_error=True)
    _broker(ex).market_close("ETHUSDT", "long", 1.0, allow_maker=True)
    (mkt,) = _markets(ex)
    assert mkt[2] == pytest.approx(1.0)
    # step1 cancel_all + maker 오류경로 cancel_all = 2회 (1회면 오류경로 정리 누락)
    assert ex.cancelled.count(("all", "ETHUSDT")) == 2


def test_allow_maker_false_is_pure_market():
    """기존 SL/긴급 경로 회귀 — 지정가·티커 조회 없이 즉시 시장가."""
    ex = _FakeExchange()
    _broker(ex).market_close("ETHUSDT", "long", 1.0)  # allow_maker 기본 False
    assert _limits(ex) == []
    assert ex.ticker_calls == 0
    (mkt,) = _markets(ex)
    assert mkt[2] == pytest.approx(1.0) and mkt[3].get("reduceOnly")


def test_maker_disabled_config_is_pure_market():
    """maker_entry 비활성(config) 시 allow_maker=True여도 시장가."""
    ex = _FakeExchange()
    broker = LiveBroker(exchange=ex, dry_run=False, maker_timeout_sec=0.0)
    broker.market_close("ETHUSDT", "long", 1.0, allow_maker=True)
    assert _limits(ex) == [] and ex.ticker_calls == 0
    assert len(_markets(ex)) == 1
