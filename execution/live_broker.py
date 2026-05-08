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
        """진입 후 TP/SL 주문 등록. 실제 fill_price 기준으로 가격 보정."""
        close_side = "sell" if order.direction == "long" else "buy"
        # fill_price와 order.price 차이만큼 TP/SL 보정 (슬리피지 반영)
        shift = entry_price - order.price
        tp = order.tp_price + shift if order.tp_price > 0 else 0
        sl = order.sl_price + shift if order.sl_price > 0 else 0

        # ── TP: LIMIT reduceOnly ─────────────────────────────────────
        if tp > 0:
            try:
                if not self.dry_run:
                    self.exchange.create_order(
                        order.symbol, "limit", close_side, float(qty), float(tp),
                        {"reduceOnly": True},
                    )
                logger.info("TP(LIMIT reduceOnly) 등록: %s @%.4f", order.symbol, tp)
            except Exception as e:
                logger.warning("TP 등록 실패 %s: %s", order.symbol, e)

        # ── SL: 거래소 STOP_MARKET 시도, 실패 시 엔진 폴링 ──────────────
        if sl > 0:
            try:
                if not self.dry_run:
                    self.exchange.create_order(
                        order.symbol, "STOP_MARKET", close_side, float(qty),
                        None, {"stopPrice": float(sl), "reduceOnly": "true"},
                    )
                    logger.info("SL(STOP_MARKET) 등록: %s @%.4f", order.symbol, sl)
            except Exception as e:
                logger.error("SL 거래소 등록 실패: %s @%.4f — %s: %s",
                             order.symbol, sl, type(e).__name__, e)

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
            min_qty = market.get("limits", {}).get("amount", {}).get("min")
            if min_qty is not None and float(qty) < float(min_qty):
                logger.warning("qty %s < 최소 %s (%s), 최소값 사용", qty, min_qty, symbol)
                qty = min_qty
        except Exception:
            qty = round(qty, 6)

        logger.info(
            "[%s] %s %s %.4f @ %.4f  (TP=%.4f SL=%.4f)  dry=%s",
            order.strategy, symbol, side, float(qty), price,
            order.tp_price, order.sl_price, self.dry_run,
        )

        if self.dry_run:
            fill = Fill(
                order=order,
                fill_price=price,
                commission=self.commission.calculate(size_usd, OrderType.MARKET),
                slippage_cost=self.slippage.cost(size_usd, OrderType.MARKET),
                timestamp=pd.Timestamp.now(tz="UTC"),
            )
            self._notify_entry(order, price)
            return fill

        # 실제 시장가 주문 (재시도 포함, 이중 주문 방지)
        result = None
        for _attempt in range(3):
            try:
                result = self.exchange.create_order(symbol, "market", side, qty)
                break
            except ccxt.InsufficientFunds as e:
                logger.error("잔고 부족: %s", e)
                raise
            except ccxt.RateLimitExceeded as e:
                logger.warning("Rate limit, 재시도 %d/3: %s", _attempt + 1, e)
                time.sleep(2 ** _attempt)
            except ccxt.NetworkError as e:
                logger.warning("네트워크 오류, 재시도 %d/3: %s", _attempt + 1, e)
                # 이미 체결됐을 수 있으므로 포지션 확인
                try:
                    if symbol in self.fetch_open_symbols():
                        logger.info("네트워크 오류지만 포지션 존재 확인 — 체결된 것으로 간주: %s", symbol)
                        trades = self.exchange.fetch_my_trades(symbol, limit=3)
                        if trades:
                            result = {"average": float(trades[-1]["price"]), "fee": trades[-1].get("fee")}
                        break
                except Exception:
                    pass
                time.sleep(1)
            except ccxt.ExchangeError as e:
                logger.error("거래소 오류: %s", e)
                raise
        if result is None:
            raise ccxt.NetworkError("3회 재시도 후에도 주문 실패")

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
            score_val = int(getattr(order.signal_score, "total", 0) or 0)
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

    def fetch_recent_fill_price(self, symbol: str) -> float | None:
        """최근 체결 내역에서 해당 심볼의 마지막 체결가 조회.
        TP LIMIT 등 거래소 측 체결을 tracker에 정확히 반영하기 위함."""
        try:
            trades = self.exchange.fetch_my_trades(symbol, limit=5)
            if trades:
                return float(trades[-1]["price"])
        except Exception as e:
            logger.warning("fetch_my_trades 실패 %s: %s", symbol, e)
        return None

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
        """엔진 TP/SL 히트 시 호출. 열린 TP LIMIT 주문 취소 + 시장가 청산.

        Raises: 청산 실패 시 Exception을 raise (고아 포지션 방지).
        """
        if self.dry_run:
            logger.info("[DRY] market_close: %s %s qty=%s", symbol, direction, qty)
            return
        # 1) 남은 TP 주문 취소
        try:
            self.exchange.cancel_all_orders(symbol)
        except Exception as e:
            logger.warning("주문 취소 실패 %s: %s", symbol, e)
        # 2) 거래소 실제 수량 조회 (더스트 방지)
        close_side = "sell" if direction == "long" else "buy"
        actual_qty = qty
        try:
            positions = self.exchange.fetch_positions([symbol])
            for p in positions:
                if float(p.get("contracts") or 0) != 0:
                    actual_qty = abs(float(p["contracts"]))
                    break
        except Exception:
            pass  # 조회 실패 시 전달받은 qty 사용
        # 3) 시장가 reduceOnly 청산
        try:
            q = self.exchange.amount_to_precision(symbol, actual_qty)
            if float(q) <= 0:
                logger.error("market_close: qty가 0으로 반올림됨 (원래 %f) %s", actual_qty, symbol)
                raise ValueError(f"qty=0 for {symbol}")
            self.exchange.create_order(symbol, "market", close_side, q, None,
                                       {"reduceOnly": True})
            logger.info("market_close 완료: %s %s qty=%s", symbol, close_side, q)
        except Exception as e:
            logger.error("market_close 실패 %s: %s", symbol, e)
            raise  # 호출자에게 전파 (고아 포지션 방지)

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
