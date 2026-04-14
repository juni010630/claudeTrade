"""Binance USDM Futures 라이브/데모 브로커.

BacktestBroker와 동일한 인터페이스(submit)를 구현하여
BacktestEngine에 그대로 주입 가능.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import ccxt
import pandas as pd

from execution.commission import CommissionModel
from execution.models import Fill, Order, OrderSide, OrderType
from execution.notifier import TelegramNotifier
from execution.slippage import SlippageModel

logger = logging.getLogger(__name__)


class LiveBroker:
    """
    ccxt 기반 Binance 선물 라이브 주문 실행기.

    BacktestBroker와 동일한 .commission / .slippage 속성을 노출하여
    BacktestEngine에 그대로 주입 가능.
    """

    def __init__(
        self,
        exchange: ccxt.Exchange,
        dry_run: bool = False,
        notifier: Optional[TelegramNotifier] = None,
        equity_provider=None,
    ) -> None:
        self.exchange = exchange
        self.dry_run = dry_run
        self._leverage_cache: dict[str, int] = {}
        # BacktestEngine 이 참조하는 속성 (수수료 계산용)
        self.commission = CommissionModel(maker_rate=0.0002, taker_rate=0.0005)
        self.slippage   = SlippageModel(default_bps=2.0)  # 라이브는 슬리피지 낮음
        self.notifier = notifier
        self.equity_provider = equity_provider  # callable -> float (현재 자본)

    # ── 레버리지 설정 (중복 호출 방지) ──────────────────────────────
    def _ensure_leverage(self, symbol: str, leverage: int) -> None:
        if self._leverage_cache.get(symbol) == leverage:
            return
        try:
            self.exchange.set_leverage(leverage, symbol)
            self._leverage_cache[symbol] = leverage
            logger.info("레버리지 설정: %s x%d", symbol, leverage)
        except Exception as e:
            logger.warning("레버리지 설정 실패 %s x%d: %s", symbol, leverage, e)

    # ── TP/SL 주문 ──────────────────────────────────────────────────
    #
    # Binance Testnet 은 STOP_MARKET / TAKE_PROFIT_MARKET / TRAILING_STOP_MARKET
    # 전부 -4120 으로 막혀 있어서:
    #   - TP: LIMIT sell(long) / LIMIT buy(short) + reduceOnly  → 거래소에 걸림
    #   - SL: 거래소 주문 불가 → 엔진이 매 봉 close 에서 폴링하여 market_close() 호출
    # 라이브 계정에서는 STOP_MARKET 이 정상 동작하므로 그때만 폴백으로 시도.
    def _place_tp_sl(self, order: Order, entry_price: float, qty: float) -> None:
        """진입 후 TP 지정가 주문 등록. SL 은 엔진 폴링에 위임."""
        close_side = "sell" if order.direction == "long" else "buy"

        # ── TP: LIMIT reduceOnly ─────────────────────────────────────
        if order.tp_price > 0:
            try:
                if not self.dry_run:
                    self.exchange.create_order(
                        order.symbol, "limit", close_side, float(qty), float(order.tp_price),
                        {"reduceOnly": "true"},
                    )
                logger.info("TP(LIMIT reduceOnly) 등록: %s @%.4f", order.symbol, order.tp_price)
            except Exception as e:
                logger.warning("TP 등록 실패 %s: %s", order.symbol, e)

        # ── SL: 거래소 STOP_MARKET 시도, 실패 시 엔진 폴링 ──────────────
        if order.sl_price > 0:
            try:
                if not self.dry_run:
                    self.exchange.create_order(
                        order.symbol, "STOP_MARKET", close_side, float(qty),
                        None, {"stopPrice": float(order.sl_price), "reduceOnly": "true"},
                    )
                    logger.info("SL(STOP_MARKET) 등록: %s @%.4f", order.symbol, order.sl_price)
            except Exception as e:
                logger.info("SL 거래소 등록 불가 (엔진 폴링 위임): %s @%.4f  (%s)",
                            order.symbol, order.sl_price, type(e).__name__)

    # ── 메인: 주문 실행 ──────────────────────────────────────────────
    def submit(self, order: Order, current_bar: Optional[pd.Series] = None) -> Fill:
        """
        주문을 Binance 선물에 제출하고 Fill 반환.
        BacktestBroker.submit() 과 동일 시그니처.
        """
        symbol    = order.symbol
        side      = "buy" if order.side == OrderSide.BUY else "sell"
        price     = order.price
        size_usd  = order.size_usd

        self._ensure_leverage(symbol, order.leverage)

        # 수량 계산 (USD → 코인 수량)
        qty = size_usd / price

        # 최소 수량/스텝 정밀도 맞추기
        try:
            market = self.exchange.market(symbol)
            qty = self.exchange.amount_to_precision(symbol, qty)
        except Exception:
            qty = round(qty, 3)

        logger.info(
            "[%s] %s %s %.4f @ %.4f  (TP=%.4f SL=%.4f)  dry=%s",
            order.strategy, symbol, side, float(qty), price,
            order.tp_price, order.sl_price, self.dry_run,
        )

        if self.dry_run:
            fill = Fill(
                order=order,
                fill_price=price,
                commission=size_usd * 0.0005,
                slippage_cost=size_usd * 0.0005,
                timestamp=pd.Timestamp.now(tz="UTC"),
            )
            self._notify_entry(order, price)
            return fill

        # 실제 시장가 주문
        try:
            result = self.exchange.create_order(symbol, "market", side, qty)
        except ccxt.InsufficientFunds as e:
            logger.error("잔고 부족: %s", e)
            raise
        except ccxt.ExchangeError as e:
            logger.error("거래소 오류: %s", e)
            raise

        fill_price = float(result.get("average") or result.get("price") or price)
        fee_info   = result.get("fee") or {}
        commission = float(fee_info.get("cost") or size_usd * 0.0005)
        ts         = pd.Timestamp.now(tz="UTC")

        fill = Fill(
            order=order,
            fill_price=fill_price,
            commission=commission,
            slippage_cost=0.0,
            timestamp=ts,
        )

        # TP/SL 등록
        self._place_tp_sl(order, fill_price, float(qty))

        self._notify_entry(order, fill_price)
        return fill

    def _notify_entry(self, order: Order, fill_price: float) -> None:
        if self.notifier is None or not self.notifier.enabled:
            return
        try:
            equity = float(self.equity_provider()) if self.equity_provider else 0.0
            tier_name = getattr(order.signal_score, "tier", None)
            tier_str = tier_name.name if hasattr(tier_name, "name") else str(tier_name)
            score_val = int(getattr(order.signal_score, "score", 0) or 0)
            self.notifier.notify_entry(
                symbol=order.symbol,
                direction=order.direction,
                fill_price=float(fill_price),
                size_usd=float(order.size_usd),
                leverage=int(order.leverage),
                tp_price=float(order.tp_price),
                sl_price=float(order.sl_price),
                strategy=order.strategy,
                tier=tier_str,
                score=score_val,
                equity=equity,
            )
        except Exception as e:
            logger.warning("entry 알림 실패: %s", e)

    def fetch_open_symbols(self) -> set[str]:
        """거래소에서 현재 열려있는 포지션 심볼 집합 반환 (sync 용)."""
        try:
            positions = self.exchange.fetch_positions()
            return {
                p["symbol"].replace("/USDT:USDT", "USDT")
                for p in positions
                if float(p.get("contracts") or 0) != 0
            }
        except Exception as e:
            logger.warning("fetch_positions 실패: %s", e)
            return set()

    def market_close(self, symbol: str, direction: str, qty: float) -> None:
        """엔진 TP/SL 히트 시 호출. 열린 TP LIMIT 주문 취소 + 시장가 청산."""
        if self.dry_run:
            logger.info("[DRY] market_close: %s %s qty=%s", symbol, direction, qty)
            return
        # 1) 남은 TP 주문 취소
        try:
            self.exchange.cancel_all_orders(symbol)
        except Exception as e:
            logger.warning("주문 취소 실패 %s: %s", symbol, e)
        # 2) 시장가 reduceOnly 청산
        close_side = "sell" if direction == "long" else "buy"
        try:
            q = self.exchange.amount_to_precision(symbol, qty)
            self.exchange.create_order(symbol, "market", close_side, q, None,
                                       {"reduceOnly": "true"})
            logger.info("market_close 완료: %s %s qty=%s", symbol, close_side, q)
        except Exception as e:
            logger.error("market_close 실패 %s: %s", symbol, e)

    def cancel_all_orders(self, symbol: str) -> None:
        """특정 심볼의 열린 주문 전체 취소 (긴급 청산 시 사용)."""
        if self.dry_run:
            return
        try:
            self.exchange.cancel_all_orders(symbol)
        except Exception as e:
            logger.warning("주문 취소 실패 %s: %s", symbol, e)

    def close_position(self, symbol: str, direction: str) -> None:
        """시장가 포지션 강제 청산."""
        if self.dry_run:
            logger.info("[DRY] 강제청산: %s %s", symbol, direction)
            return
        close_side = "sell" if direction == "long" else "buy"
        try:
            self.exchange.create_order(
                symbol, "market", close_side, 0,
                None, {"reduceOnly": True, "closePosition": True},
            )
            logger.info("강제청산: %s", symbol)
        except Exception as e:
            logger.error("강제청산 실패 %s: %s", symbol, e)
