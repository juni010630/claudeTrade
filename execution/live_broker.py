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
        commission_maker: float = 0.0002,
        commission_taker: float = 0.0005,
        slippage_bps: float = 5.0,
        demo: bool = False,
        maker_timeout_sec: float = 0.0,  # >0 = maker-first 진입 (지정가 → timeout 후 시장가 추격)
        maker_poll_sec: float = 3.0,
    ) -> None:
        self.exchange = exchange
        self.dry_run = dry_run
        # demo(testnet): STOP_MARKET이 -4120으로 막혀 SL 등록 실패가 정상(sl_poller가 대체)
        # → 단발 로그만. 메인넷(demo=False): STOP_MARKET이 유일한 거래소측 SL → 재시도+경보.
        self.demo = demo
        self._leverage_cache: dict[str, int] = {}
        # BacktestEngine 이 참조하는 속성 (수수료/슬리피지 계산용).
        # 백테스트와 동일한 모델을 써야 청산 비용 계산이 일치한다.
        # (진입 슬리피지는 실제 체결가에 이미 반영되어 submit()에서 slippage_cost=0 처리.
        #  청산은 엔진이 broker.slippage.cost로 계산하므로 config 값과 맞춰야 한다.)
        self.commission = CommissionModel(maker_rate=commission_maker, taker_rate=commission_taker)
        self.slippage   = SlippageModel(default_bps=slippage_bps)
        self.notifier = notifier
        self.equity_provider = equity_provider  # callable -> float (현재 자본)
        self._maker_timeout = float(maker_timeout_sec)
        self._maker_poll = float(maker_poll_sec)

    # ── 주문 전체 취소 (일반 + Algo 트리거 주문부) ─────────────────────
    def _cancel_all_orders_full(self, symbol: str) -> None:
        """일반 주문부(TP LIMIT 등) + Algo 주문부(STOP_MARKET 등 트리거) 전체 취소.

        바이낸스가 트리거 주문을 별도 Algo 주문부로 분리해 fapi allOpenOrders가
        못 지운다 (2026-07-04 실증: 청산 후 구 SL이 좀비로 남아 재진입 포지션을
        구 가격에 오청산 가능). 양쪽을 각각 시도하고 실패는 경고만.
        """
        try:
            self.exchange.cancel_all_orders(symbol)
        except Exception as e:
            logger.warning("주문 취소 실패(일반) %s: %s", symbol, e)
        try:
            self.exchange.cancel_all_orders(symbol, params={"trigger": True})
        except Exception as e:
            logger.warning("주문 취소 실패(트리거) %s: %s", symbol, e)

    # ── 레버리지 설정 (중복 호출 방지) ──────────────────────────────
    def _ensure_leverage(self, symbol: str, leverage: int) -> None:
        if self._leverage_cache.get(symbol) == leverage:
            return
        try:
            self.exchange.set_leverage(leverage, symbol)
            self._leverage_cache[symbol] = leverage
            logger.info("레버리지 설정: %s x%d", symbol, leverage)
        except Exception as e:
            if self.dry_run:
                logger.warning("[DRY] 레버리지 설정 생략 %s x%d: %s", symbol, leverage, e)
                return
            raise RuntimeError(f"레버리지 설정 실패 {symbol} x{leverage}: {e}") from e

    # ── TP/SL 주문 ──────────────────────────────────────────────────
    #
    # Binance Testnet 은 STOP_MARKET / TAKE_PROFIT_MARKET / TRAILING_STOP_MARKET
    # 전부 -4120 으로 막혀 있어서:
    #   - TP: LIMIT sell(long) / LIMIT buy(short) + reduceOnly  → 거래소에 걸림
    #   - SL: 거래소 주문 불가 → 엔진이 매 봉 close 에서 폴링하여 market_close() 호출
    # 라이브 계정에서는 STOP_MARKET 이 정상 동작하므로 그때만 폴백으로 시도.
    @staticmethod
    def _adjust_protection_prices(order: Order, entry_price: float) -> None:
        """체결 슬리피지를 주문 객체에도 반영해 거래소·tracker를 일치시킨다."""
        shift = float(entry_price) - float(order.price)
        if order.tp_price > 0:
            order.tp_price = float(order.tp_price) + shift
        if order.sl_price > 0:
            order.sl_price = float(order.sl_price) + shift

    def _place_tp_sl(self, order: Order, entry_price: float, qty: float) -> bool:
        """진입 후 TP/SL 주문 등록. 반환값은 SL 보호 성공 여부."""
        close_side = "sell" if order.direction == "long" else "buy"
        tp = order.tp_price
        sl = order.sl_price

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

        # ── SL: 거래소 STOP_MARKET ──────────────────────────────────────
        if sl <= 0 or self.dry_run:
            return True
        return self._place_sl_stop(order, close_side, float(qty), float(sl))

    def _place_sl_stop(self, order: Order, close_side: str, qty: float, sl: float) -> bool:
        """SL STOP_MARKET 등록. 메인넷은 재시도 후 실패 시 경보(거래소 SL 부재=무방비 위험),
        testnet은 -4120 차단이 정상(sl_poller가 5m로 대체)이라 단발 로그만."""
        def _create() -> None:
            self.exchange.create_order(
                order.symbol, "STOP_MARKET", close_side, qty,
                None, {"stopPrice": sl, "reduceOnly": "true"},
            )

        if self.demo:
            try:
                _create()
                logger.info("SL(STOP_MARKET) 등록: %s @%.4f", order.symbol, sl)
            except Exception as e:
                logger.warning("SL 등록 실패(testnet 예상, sl_poller 대체): %s @%.4f — %s",
                               order.symbol, sl, e)
            return True

        # 메인넷: STOP_MARKET이 유일한 거래소측 SL → 재시도 후 실패 시 경보.
        # reduceOnly라 혹시 중복 등록돼도 한 쪽만 청산·나머지는 무해(다음 봉 sync가 정리).
        last_err = None
        for attempt in range(3):
            try:
                _create()
                logger.info("SL(STOP_MARKET) 등록: %s @%.4f", order.symbol, sl)
                return True
            except Exception as e:
                last_err = e
                logger.warning("SL 등록 실패 재시도 %d/3 %s @%.4f — %s: %s",
                               attempt + 1, order.symbol, sl, type(e).__name__, e)
                if attempt < 2:  # 마지막 시도 후엔 즉시 경보 (불필요한 대기 제거)
                    time.sleep(1.0 * (attempt + 1))
        logger.error("SL 거래소 등록 최종 실패: %s @%.4f — %s: %s (거래소 SL 없음, 엔진 1h 백업만)",
                     order.symbol, sl, type(last_err).__name__, last_err)
        self._notify_sl_failure(order, sl, last_err)
        return False

    def _notify_sl_failure(self, order: Order, sl: float, err) -> None:
        if self.notifier is None or not getattr(self.notifier, "enabled", False):
            return
        try:
            self.notifier.notify_info(
                f"🚨 <b>SL 등록 실패 — 거래소 측 손절 없음!</b>\n"
                f"{order.symbol} {order.direction} | SL @{sl:.4f}\n"
                f"사유: {type(err).__name__}: {str(err)[:150]}\n"
                f"⚠️ 엔진 1h 백업 청산만 작동 — 수동 확인 권장"
            )
        except Exception as e:
            logger.warning("SL 실패 경보 전송 실패: %s", e)

    # ── maker-first 진입 (post-only 지정가 + 타임아웃 + 시장가 추격) ────
    def _try_maker_entry(self, symbol: str, side: str, qty: float, limit_price: float):
        """시그널가 post-only 지정가 → timeout 내 미체결 잔량 시장가 추격.

        경로 시뮬 검증(MAKER_ENTRY_STUDY.md): 폴백 없이 놓치면 러너 9% 유실 = 참사.
        → 모든 분기에서 '체결 보장 or 명시적 스킵'. 어떤 오류든 최악 = 기존 시장가 경로.

        반환: {"average", "qty">0} = 체결 완료(부분 포함, 가중평균가)
              {"average", "qty": 0} = 상태 확인 불가 → 호출자가 이번 진입 스킵
              None = 0체결 확인 → 호출자가 기존 시장가 경로 실행
        """
        try:
            px = self.exchange.price_to_precision(symbol, limit_price)
            o = self.exchange.create_order(symbol, "limit", side, qty, px, {"postOnly": True})
            oid = o["id"]
            logger.info("maker 지정가 등록: %s %s qty=%s @%s (timeout %.0fs)",
                        symbol, side, qty, px, self._maker_timeout)
        except Exception as e:
            # postOnly 즉시체결 거부(가격이 유리한 쪽으로 이미 관통) 포함 — 시장가가 정답
            logger.info("maker 등록 불가 → 시장가 폴백: %s — %s: %s", symbol, type(e).__name__, e)
            return None

        def _status():
            o2 = self.exchange.fetch_order(oid, symbol)
            return (o2.get("status"), float(o2.get("filled") or 0.0),
                    float(o2.get("average") or limit_price))

        deadline = time.time() + self._maker_timeout
        chased = False  # 시장가 추격 발사 여부 — 예외 시 추격 체결 미상 판정에 사용
        try:
            while time.time() < deadline:
                time.sleep(self._maker_poll)
                status, filled, avg = _status()
                # 전량 판정은 status=="closed"만 인정 — 99.9% 체결·미종결로 조기 반환하면
                # 잔여 ~0.1% 지정가(reduceOnly 아님)가 고아로 남고, sync는 닫힌 포지션만
                # 정리하므로 영구 표류(_try_maker_close와 동일 규칙). 미종결이면 계속 대기 →
                # 타임아웃 취소 경로가 잔량을 시장가로 정리한다.
                if status == "closed":
                    logger.info("maker 전량 체결: %s @%.6g", symbol, avg)
                    return {"average": avg, "qty": filled if filled > 0 else qty}
                if status in ("canceled", "expired", "rejected"):
                    return {"average": avg, "qty": filled} if filled > 0 else None

            # 타임아웃 → 취소. 취소-체결 레이스는 취소 후 재조회가 진실.
            try:
                self.exchange.cancel_order(oid, symbol)
            except Exception as e:
                logger.info("maker 취소 응답(이미 체결/취소 가능): %s", e)
            status, filled, avg = _status()
            if status == "closed" or filled >= qty * 0.999:
                logger.info("maker 취소 직전 전량 체결: %s @%.6g", symbol, avg)
                return {"average": avg, "qty": filled if filled > 0 else qty}
            # 취소가 실패(네트워크 등)해 지정가가 여전히 open이면 시장가 추격 금지 —
            # 살아있는 비-reduceOnly 지정가 + 시장가 = 최대 2배 포지션. 재취소로 종결 확인.
            if status == "open":
                for _retry in range(2):
                    try:
                        self.exchange.cancel_order(oid, symbol)
                    except Exception:
                        pass
                    time.sleep(1)
                    status, filled, avg = _status()
                    if status != "open":
                        break
                if status == "open":
                    # raise 금지 — 아래 except Exception이 삼켜 시장가 폴백(None)으로
                    # 흐르면 2배 포지션. 스킵 판정을 직접 반환해야 한다.
                    logger.error("지정가 취소 불가 — 이중 진입 방지 위해 추격 중단(진입 스킵): %s", symbol)
                    return {"average": limit_price, "qty": 0.0}
                if status == "closed" or filled >= qty * 0.999:
                    logger.info("maker 재취소 중 전량 체결: %s @%.6g", symbol, avg)
                    return {"average": avg, "qty": filled if filled > 0 else qty}

            remain = qty - filled
            try:
                remain_p = float(self.exchange.amount_to_precision(symbol, remain))
            except Exception:
                remain_p = round(remain, 6)
            if remain_p <= 0:
                return {"average": avg, "qty": filled} if filled > 0 else None
            chased = True
            res = self.exchange.create_order(symbol, "market", side, remain_p)
            mkt_px = float(res.get("average") or res.get("price") or limit_price)
            mkt_qty = float(res.get("filled") or remain_p)
            total = filled + mkt_qty
            wavg = (avg * filled + mkt_px * mkt_qty) / total if total > 0 else mkt_px
            logger.info("maker 타임아웃 → 시장가 추격: %s maker %.6f@%.6g + market %.6f@%.6g",
                        symbol, filled, avg, mkt_qty, mkt_px)
            return {"average": wavg, "qty": total}
        except Exception as e:
            # 미지 상태 — 이중 주문 방지 최우선: 잔여 지정가 정리 후 상태 기반 판정.
            # (진입 시점엔 이 심볼의 열린 주문 = 방금 그 지정가뿐이라 cancel_all 안전)
            logger.error("maker 경로 오류 %s — 정리 시도: %s: %s", symbol, type(e).__name__, e)
            try:
                self._cancel_all_orders_full(symbol)
            except Exception:
                pass
            for _ in range(3):
                try:
                    status, filled, avg = _status()
                    if status == "open":
                        # 지정가 생존(취소 실패) — 시장가 폴백하면 지정가 추후 체결분과
                        # 합쳐 2배 포지션 → 스킵 판정(호출자가 실포지션 조회로 채택/스킵)
                        logger.error("maker 지정가 취소 불가 — 진입 스킵: %s", symbol)
                        return {"average": limit_price, "qty": 0.0}
                    if chased:
                        # 시장가 추격 발사 후 오류 — 추격 체결 여부는 지정가 filled로
                        # 알 수 없음 → 스킵 판정(호출자가 실포지션으로 확정)
                        return {"average": avg if filled > 0 else limit_price, "qty": 0.0}
                    if filled > 0:
                        return {"average": avg, "qty": filled}
                    return None  # 주문 종결·0체결·추격 전 확인 → 시장가 폴백 안전
                except Exception:
                    time.sleep(1)
            return {"average": limit_price, "qty": 0.0}  # 확인 불가 → 진입 스킵

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

        if price <= 0 or size_usd <= 0 or order.tp_price <= 0 or order.sl_price <= 0:
            raise ValueError(
                f"유효하지 않은 진입/보호 가격: price={price}, "
                f"size={size_usd}, tp={order.tp_price}, sl={order.sl_price}"
            )
        valid_geometry = (
            order.sl_price < price < order.tp_price
            if order.direction == "long"
            else order.tp_price < price < order.sl_price
        )
        if not valid_geometry:
            raise ValueError(
                f"TP/SL 방향 불일치: {order.symbol} {order.direction} "
                f"tp={order.tp_price}, entry={price}, sl={order.sl_price}"
            )

        self._ensure_leverage(symbol, order.leverage)

        # 수량 계산 (USD → 코인 수량)
        qty = size_usd / price

        # 최소 수량/스텝 정밀도 맞추기. 의도한 노셔널을 초과하는
        # 최소수량 bump는 절대 하지 않고 주문을 거부한다.
        try:
            market = self.exchange.market(symbol)
            qty = self.exchange.amount_to_precision(symbol, qty)
        except Exception as e:
            if not self.dry_run:
                raise RuntimeError(f"주문 시장정보/정밀도 확인 실패 {symbol}: {e}") from e
            market = {}
            qty = round(qty, 6)

        final_qty = float(qty)
        limits = market.get("limits", {}) if isinstance(market, dict) else {}
        min_qty = limits.get("amount", {}).get("min")
        min_cost = limits.get("cost", {}).get("min")
        if final_qty <= 0:
            raise ValueError(f"정밀도 반영 후 주문 수량이 0입니다: {symbol}")
        if min_qty is not None and final_qty < float(min_qty):
            raise ValueError(
                f"주문 수량 {final_qty} < 최소 {float(min_qty)} ({symbol}); "
                "리스크 초과 방지를 위해 진입 거부"
            )
        if min_cost is not None and final_qty * price < float(min_cost):
            raise ValueError(
                f"주문 금액 {final_qty * price:.8g} < 최소 {float(min_cost)} ({symbol}); "
                "진입 거부"
            )

        logger.info(
            "[%s] %s %s %.4f @ %.4f  (TP=%.4f SL=%.4f)  dry=%s",
            order.strategy, symbol, side, final_qty, price,
            order.tp_price, order.sl_price, self.dry_run,
        )

        if self.dry_run:
            order.size_usd = final_qty * price
            fill = Fill(
                order=order,
                fill_price=price,
                commission=self.commission.calculate(order.size_usd, OrderType.MARKET),
                slippage_cost=self.slippage.cost(order.size_usd, OrderType.MARKET),
                timestamp=order.timestamp if order.timestamp is not None
                else pd.Timestamp.now(tz="UTC"),
            )
            self._notify_entry(order, price)
            return fill

        # maker-first 진입 (활성 시): 어떤 결과든 안전 — None이면 아래 기존 시장가 경로
        result = None
        if self._maker_timeout > 0:
            mk = self._try_maker_entry(symbol, side, float(qty), float(price))
            if mk is not None:
                if mk["qty"] <= 0:
                    # 상태 확인 불가. 지정가가 실제 체결됐을 수 있으므로 거래소 포지션을 직접
                    # 질의 — 실재하면 SL 없는 고아 방지 위해 채택(평단·실수량)하고 아래서 TP/SL
                    # 등록, 미실재면 안전하게 진입 스킵. (시장가 경로 311-336과 동일 패턴)
                    # 그 전에 잔존 지정가 최종 정리 — 살아있는 채 채택/스킵하면 추후 체결분이
                    # SL/TP 커버 밖 수량(2배/나체)이 된다. 종결 확인 불가면 강경보.
                    _residual = None
                    try:
                        self._cancel_all_orders_full(symbol)
                        _residual = bool(self.exchange.fetch_open_orders(symbol))
                    except Exception:
                        pass
                    if _residual is not False and self.notifier is not None \
                            and getattr(self.notifier, "enabled", False):
                        try:
                            self.notifier.notify_info(
                                f"🚨 <b>{symbol}: 진입 지정가 잔존 가능 — 수동 확인 필요</b>\n"
                                f"취소/종결 확인 실패. 추후 체결 시 SL/TP 커버 밖 수량이 됩니다."
                            )
                        except Exception:
                            pass
                    _open_syms = self._fetch_open_symbols_retry()
                    if _open_syms is None:
                        # 포지션 조회까지 실패 = 체결 여부 완전 미상 — 스킵하되 강경보
                        # (지정가가 체결됐다면 SL 없는 나체 포지션일 수 있음 → 수동 확인)
                        if self.notifier is not None and getattr(self.notifier, "enabled", False):
                            try:
                                self.notifier.notify_info(
                                    f"🚨 <b>진입 상태 미상 — 수동 확인 필요</b>\n"
                                    f"{symbol}: maker 주문 상태·포지션 조회 모두 실패.\n"
                                    f"체결됐다면 SL 없는 포지션일 수 있음."
                                )
                            except Exception:
                                pass
                        raise ccxt.NetworkError(
                            f"maker 진입·포지션 조회 모두 실패 — 진입 스킵(수동 확인 요망): {symbol}"
                        )
                    if symbol in _open_syms:
                        ep, contracts = None, None
                        try:
                            for pp in self.exchange.fetch_positions([symbol]):
                                c = abs(float(pp.get("contracts") or 0))
                                if c > 0:
                                    contracts = c
                                    ep = float(pp.get("entryPrice") or 0) or None
                                    break
                        except Exception:
                            pass
                        if contracts:
                            final_qty = contracts
                        result = {"average": ep or float(price)}
                        logger.warning(
                            "maker 상태불확실이나 포지션 실재 — 채택+SL등록: %s qty=%.6f @%.6g",
                            symbol, final_qty, ep or float(price),
                        )
                    else:
                        raise ccxt.NetworkError(
                            f"maker 진입 상태 불확실(포지션 미실재 확인) — 진입 스킵: {symbol}"
                        )
                else:
                    result = {"average": mk["average"]}
                    final_qty = float(mk["qty"])

        # 시장가 주문 (재시도 포함, 이중 주문 방지)
        if result is None:
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
                    # 이미 체결됐을 수 있으므로 포지션 확인.
                    # 조회까지 실패(None)면 체결 여부 미상 — 맹목 재주문은 이중 진입
                    # 위험이라 이번 진입을 중단한다 (아래 flag, inner except에 안 삼켜지게).
                    _abort_unknown = False
                    try:
                        _open_syms = self._fetch_open_symbols_retry()
                        if _open_syms is None:
                            _abort_unknown = True
                        elif symbol in _open_syms:
                            logger.info("네트워크 오류지만 포지션 존재 확인 — 체결된 것으로 간주: %s", symbol)
                            # 포지션이 실재하면 result를 절대 None으로 두지 않는다 — None이면
                            # 아래 raise→호출자 스킵→SL/TP 없는 고아 포지션이 남는다.
                            # 체결가: 체결내역 → 포지션 평단 → 의도가 순으로 폴백.
                            avg_px = None
                            try:
                                trades = self.exchange.fetch_my_trades(symbol, limit=3)
                                if trades:
                                    avg_px = float(trades[-1]["price"])
                                    result = {"average": avg_px, "fee": trades[-1].get("fee")}
                            except Exception:
                                pass
                            if avg_px is None:
                                ep = None
                                try:
                                    for pp in self.exchange.fetch_positions([symbol]):
                                        if float(pp.get("contracts") or 0) != 0:
                                            ep = float(pp.get("entryPrice") or 0) or None
                                            break
                                except Exception:
                                    pass
                                result = {"average": ep if ep else price}
                            break
                    except Exception:
                        pass
                    if _abort_unknown:
                        raise ccxt.NetworkError(
                            f"주문 상태·포지션 조회 모두 실패 — 진입 중단(이중주문 방지): {symbol}"
                        )
                    time.sleep(1)
                except ccxt.ExchangeError as e:
                    logger.error("거래소 오류: %s", e)
                    raise
        if result is None:
            raise ccxt.NetworkError("3회 재시도 후에도 주문 실패")

        fill_price = float(result.get("average") or result.get("price") or price)
        result_qty = float(result.get("filled") or 0)
        if result_qty > 0:
            final_qty = result_qty
        # 실제 체결 수량×체결가 = 실제 노셔널 → tracker가 거래소와 동일 size 기록
        order.size_usd = final_qty * fill_price
        self._adjust_protection_prices(order, fill_price)
        fee_info   = result.get("fee") or {}
        # 수수료는 USDT 표기일 때만 실측 사용 — feeBurn ON 계정은 currency='BNB'/cost=BNB수량이라
        # 그대로 쓰면 수백 배 과소 기록. 비USDT면 모델값 폴백.
        _fee_cost = fee_info.get("cost")
        _fee_cur = (fee_info.get("currency") or "USDT").upper()
        commission = (float(_fee_cost) if _fee_cost and _fee_cur == "USDT"
                      else self.commission.calculate(order.size_usd, OrderType.MARKET))
        # 백테 패리티: opened_at = 신호봉 close 시각 (벽시계면 max_hold 타임아웃이 1봉 밀림)
        ts         = order.timestamp if order.timestamp is not None else pd.Timestamp.now(tz="UTC")

        fill = Fill(
            order=order,
            fill_price=fill_price,
            commission=commission,
            slippage_cost=0.0,
            timestamp=ts,
        )

        # TP/SL 등록. 메인넷에서 SL이 없는 포지션은 즉시 원상복구한다.
        sl_ok = self._place_tp_sl(order, fill_price, final_qty)
        if not sl_ok:
            try:
                self.market_close(order.symbol, order.direction, final_qty)
            except Exception as close_err:
                logger.critical(
                    "SL 등록 실패 후 긴급청산도 실패 %s: %s", order.symbol, close_err
                )
                if self.notifier is not None and getattr(self.notifier, "enabled", False):
                    try:
                        self.notifier.notify_info(
                            f"🚨 <b>{order.symbol}: SL 등록·긴급청산 모두 실패</b>\n"
                            f"즉시 수동 청산 필요: {type(close_err).__name__}: {str(close_err)[:150]}"
                        )
                    except Exception:
                        pass
                raise RuntimeError(
                    f"SL 등록 실패 후 긴급청산 실패: {order.symbol}"
                ) from close_err
            raise RuntimeError(f"SL 등록 실패로 진입 즉시 청산: {order.symbol}")

        self._notify_entry(order, fill_price)
        return fill

    # ── 피라미딩 증액 주문 ────────────────────────────────────────────
    def place_pyramid_add(
        self, symbol: str, direction: str, trigger_price: float,
        add_size_usd: float, leverage: int,
    ) -> None:
        """진입 직후 호출 — 트리거가에 STOP_MARKET(비-reduceOnly) 증액 주문 등록.

        백테스트의 intrabar stop-market 체결과 동일 의미론.
        실패 시 에러 로그만 (포지션 자체는 유효 — 백테와 증액분만 괴리).
        """
        side = "buy" if direction == "long" else "sell"
        qty = add_size_usd / trigger_price
        try:
            qty = self.exchange.amount_to_precision(symbol, qty)
        except Exception:
            qty = round(qty, 6)
        if self.dry_run:
            logger.info("[DRY] PYRAMID 주문: %s %s qty=%s stop@%.4f", symbol, side, qty, trigger_price)
            return
        try:
            self.exchange.create_order(
                symbol, "STOP_MARKET", side, float(qty),
                None, {"stopPrice": float(trigger_price)},
            )
            logger.info("PYRAMID(STOP_MARKET) 등록: %s %s qty=%s @%.4f",
                        symbol, side, qty, trigger_price)
        except Exception as e:
            logger.error("PYRAMID 주문 등록 실패 %s @%.4f — %s: %s (백테 대비 증액 누락 주의)",
                         symbol, trigger_price, type(e).__name__, e)

    def refresh_tp_sl_after_add(
        self, symbol: str, direction: str, qty_total: float,
        tp_price: float, sl_price: float,
    ) -> None:
        """증액 체결 후 호출 — 기존 TP/SL(원 수량)을 취소하고 총 수량으로 재등록."""
        if self.dry_run:
            logger.info("[DRY] TP/SL 재등록: %s qty=%.6f tp=%.4f sl=%.4f",
                        symbol, qty_total, tp_price, sl_price)
            return
        close_side = "sell" if direction == "long" else "buy"
        self._cancel_all_orders_full(symbol)  # 기존 TP/SL(트리거 포함) 제거 후 재등록
        try:
            qty = self.exchange.amount_to_precision(symbol, qty_total)
        except Exception:
            qty = round(qty_total, 6)
        if tp_price > 0:
            try:
                self.exchange.create_order(
                    symbol, "limit", close_side, float(qty), float(tp_price),
                    {"reduceOnly": True},
                )
                logger.info("TP 재등록(증액 반영): %s qty=%s @%.4f", symbol, qty, tp_price)
            except Exception as e:
                logger.error("TP 재등록 실패 %s: %s", symbol, e)
        if sl_price > 0:
            try:
                self.exchange.create_order(
                    symbol, "STOP_MARKET", close_side, float(qty),
                    None, {"stopPrice": float(sl_price), "reduceOnly": "true"},
                )
                logger.info("SL 재등록(증액 반영): %s qty=%s @%.4f", symbol, qty, sl_price)
            except Exception as e:
                logger.error("SL 재등록 실패 %s: %s", symbol, e)

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

    def fetch_open_symbols(self) -> set[str] | None:
        """거래소에서 현재 열려있는 포지션 심볼 집합 반환 (sync 용).

        조회 실패 시 None — 빈 집합("포지션 없음")과 반드시 구분할 것.
        실패를 빈 집합으로 반환하면 호출부가 전 포지션을 stale로 오판해
        가짜 청산·SL 취소·중복 진입 연쇄가 발생한다 (2026-07-04 감사 CRITICAL)."""
        details = self.fetch_open_positions()
        return None if details is None else set(details)

    def fetch_open_positions(self) -> dict[str, dict] | None:
        """거래소 실포지션을 방향·수량·평단가까지 정규화해 반환."""
        try:
            positions = self.exchange.fetch_positions()
            result: dict[str, dict] = {}
            for p in positions:
                raw_contracts = float(p.get("contracts") or 0)
                if raw_contracts == 0:
                    continue
                symbol = p["symbol"].replace("/USDT:USDT", "USDT")
                raw_side = str(p.get("side") or "").lower()
                direction = raw_side if raw_side in {"long", "short"} else (
                    "long" if raw_contracts > 0 else "short"
                )
                result[symbol] = {
                    "symbol": symbol,
                    "direction": direction,
                    "contracts": abs(raw_contracts),
                    "entry_price": float(p.get("entryPrice") or 0),
                    "unrealized_pnl": float(p.get("unrealizedPnl") or 0),
                }
            return result
        except Exception as e:
            logger.warning("fetch_positions 실패 (판단 보류): %s", e)
            return None

    def _fetch_open_symbols_retry(self, retries: int = 3, wait: float = 1.0) -> set[str] | None:
        """fetch_open_symbols 재시도 래퍼 — 전부 실패 시 None(불확실)."""
        for attempt in range(retries):
            result = self.fetch_open_symbols()
            if result is not None:
                return result
            if attempt < retries - 1:
                time.sleep(wait * (attempt + 1))
        return None

    def _try_maker_close(self, symbol: str, close_side: str, qty: float,
                         ref_price: float) -> float:
        """타임아웃 청산용 post-only 지정가 시도. 반환 = 미체결 잔량 추정(0=전량 체결).

        reduceOnly라 어떤 레이스에서도 과청산/방향 반전이 불가 — 실패 시 최악은
        기존 시장가 경로(호출자가 잔량을 거래소 포지션 재조회로 확정 후 시장가).
        """
        try:
            px = self.exchange.price_to_precision(symbol, ref_price)
            o = self.exchange.create_order(
                symbol, "limit", close_side, qty, px,
                {"postOnly": True, "reduceOnly": True},
            )
            oid = o["id"]
            logger.info("maker 청산 등록: %s %s qty=%s @%s (timeout %.0fs)",
                        symbol, close_side, qty, px, self._maker_timeout)
        except Exception as e:
            # postOnly 즉시체결 거부 포함 — 시장가가 정답
            logger.info("maker 청산 등록 불가 → 시장가 폴백: %s — %s: %s",
                        symbol, type(e).__name__, e)
            return qty

        def _status():
            o2 = self.exchange.fetch_order(oid, symbol)
            return o2.get("status"), float(o2.get("filled") or 0.0)

        deadline = time.time() + self._maker_timeout
        try:
            while time.time() < deadline:
                time.sleep(self._maker_poll)
                status, filled = _status()
                # 전량 판정은 status=="closed"만 인정 — filled 비율 허용치(예: 99.9%)로
                # 판정하면 잔여 0.1% 더스트 + 살아있는 지정가가 고아로 남음 (sync는
                # tracker쪽 고아만 정리하므로 영구 표류). 99.9% 체결·미종결이면 계속
                # 대기 → 타임아웃 취소 경로가 잔량을 시장가로 정리.
                if status == "closed":
                    logger.info("maker 청산 전량 체결: %s", symbol)
                    return 0.0
                if status in ("canceled", "expired", "rejected"):
                    return max(qty - filled, 0.0)
            # 타임아웃 → 취소. 취소-체결 레이스는 취소 후 재조회가 진실.
            try:
                self.exchange.cancel_order(oid, symbol)
            except Exception as e:
                logger.info("maker 청산 취소 응답(이미 체결/취소 가능): %s", e)
            status, filled = _status()
            if status == "closed":
                logger.info("maker 청산 취소 직전 전량 체결: %s", symbol)
                return 0.0
            logger.info("maker 청산 타임아웃: %s 체결 %.6f/%.6f → 잔량 시장가",
                        symbol, filled, qty)
            return max(qty - filled, 0.0)
        except Exception as e:
            # 미지 상태 — 잔여 지정가 정리 후 전량을 잔량으로 보고
            # (시장가 reduceOnly는 이미 체결된 만큼 자동 캡 → 안전)
            logger.error("maker 청산 경로 오류 %s — 시장가 폴백: %s: %s",
                         symbol, type(e).__name__, e)
            try:
                self.exchange.cancel_all_orders(symbol)
            except Exception:
                pass
            return qty

    def market_close(self, symbol: str, direction: str, qty: float,
                     allow_maker: bool = False) -> None:
        """엔진 TP/SL/timeout 히트 시 호출. 열린 TP LIMIT 주문 취소 + 청산.

        allow_maker=True(타임아웃 청산 한정): post-only 지정가 먼저 시도, 미체결
        잔량은 시장가 폴백. SL/긴급 청산은 기본값 False = 기존 시장가 즉시.

        Raises: 청산 실패 시 Exception을 raise (고아 포지션 방지).
        """
        if self.dry_run:
            logger.info("[DRY] market_close: %s %s qty=%s", symbol, direction, qty)
            return
        # 1) 남은 TP/SL 주문 취소 (일반 + 트리거 주문부)
        self._cancel_all_orders_full(symbol)
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
        # 2.5) maker-first 청산 (타임아웃 한정) — 현재가 post-only, 잔량은 아래 시장가
        if allow_maker and self._maker_timeout > 0:
            remain = actual_qty
            try:
                ref = float(self.exchange.fetch_ticker(symbol)["last"])
                remain = self._try_maker_close(symbol, close_side, actual_qty, ref)
            except Exception as e:
                logger.warning("maker 청산 시도 불가 %s: %s — 시장가 진행", symbol, e)
            # maker 보고값과 무관하게 항상 거래소 포지션 재조회로 잔량 확정
            # (체결 진실 = 거래소. "전량 체결" 보고를 그대로 믿고 반환하면 더스트 고아 위험)
            try:
                positions = self.exchange.fetch_positions([symbol])
                actual_qty = 0.0
                for p in positions:
                    if float(p.get("contracts") or 0) != 0:
                        actual_qty = abs(float(p["contracts"]))
                        break
                if actual_qty <= 0:
                    logger.info("market_close 완료(maker, 잔량 0 확인): %s", symbol)
                    return
            except Exception:
                if remain <= 0:
                    # maker가 status=closed로 전량 보고 + 재조회 실패 → 완료 간주
                    logger.info("market_close 완료(maker 전량, 재조회 실패): %s", symbol)
                    return
                actual_qty = remain  # 조회 실패 → 추정 잔량 (reduceOnly가 과청산 차단)
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
        """특정 심볼의 열린 주문 전체 취소 — 일반 + 트리거 주문부 (긴급 청산·sync 정리 사용)."""
        if self.dry_run:
            return
        self._cancel_all_orders_full(symbol)

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
