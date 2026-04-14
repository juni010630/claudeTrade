"""백테스트 이벤트 루프 — 모든 레이어를 조합."""
from __future__ import annotations

from typing import Iterator

import pandas as pd

from data.schemas import MarketSnapshot
from execution.broker import BacktestBroker
from execution.funding import FundingRateSimulator
from execution.models import Order, OrderSide, OrderType
from metrics.report import MetricsReport
from portfolio.equity_curve import EquityCurve
from portfolio.ledger import Ledger
from portfolio.tracker import PortfolioTracker
from regime.detector import RegimeDetector
from regime.filters import is_strategy_eligible
from risk.circuit_breaker import BreakerStatus, CircuitBreaker
from risk.correlation import CorrelationFilter
from risk.guards import DrawdownAction, RiskGuards
from risk.position_sizer import PositionSizer
from signals.models import LeverageTier
from signals.scorer import ConfluenceScorer
from signals.validators import validate
from strategies.base import BaseStrategy


class BacktestEngine:
    def __init__(
        self,
        initial_capital: float,
        strategies: list[BaseStrategy],
        regime_detector: RegimeDetector | None = None,
        confluence_scorer: ConfluenceScorer | None = None,
        risk_guards: RiskGuards | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        correlation_filter: CorrelationFilter | None = None,
        position_sizer: PositionSizer | None = None,
        broker: BacktestBroker | None = None,
        funding_simulator: FundingRateSimulator | None = None,
        price_tf: str = "1h",           # MTM/TP-SL 가격 참조 타임프레임
        max_hold_hours: float | None = None,  # 최대 보유 시간 (초과 시 강제 청산)
        notifier=None,
    ) -> None:
        self.strategies = strategies
        self.regime_detector = regime_detector or RegimeDetector()
        self.scorer = confluence_scorer or ConfluenceScorer()
        self.guards = risk_guards or RiskGuards()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.corr_filter = correlation_filter or CorrelationFilter()
        self.sizer = position_sizer or PositionSizer()
        self.broker = broker or BacktestBroker()
        self.funding_sim = funding_simulator or FundingRateSimulator()
        self._commission = self.broker.commission
        self._slippage = self.broker.slippage
        self.price_tf = price_tf
        self.max_hold_hours = max_hold_hours

        self.ledger = Ledger()
        self.equity_curve = EquityCurve()
        self.tracker = PortfolioTracker(initial_capital, self.ledger, notifier=notifier)

        self._last_day: pd.Timestamp | None = None

    def _get_prices(self, snapshot: MarketSnapshot) -> dict[str, float]:
        """설정된 price_tf 기준으로 현재 가격 추출. 없으면 가용한 TF로 폴백."""
        prices: dict[str, float] = {}
        fallback_order = [self.price_tf, "5m", "15m", "1h", "4h", "1d"]
        for sym in snapshot.bars:
            for tf in fallback_order:
                bars = snapshot.bars[sym].get(tf)
                if bars is not None and not bars.empty:
                    prices[sym] = float(bars["close"].iloc[-1])
                    break
        return prices

    def run(self, snapshots: Iterator[MarketSnapshot]) -> MetricsReport:
        for snapshot in snapshots:
            self._process_bar(snapshot)

        return MetricsReport.from_run(self.equity_curve, self.ledger)

    def _process_bar(self, snapshot: MarketSnapshot) -> None:
        now = snapshot.timestamp

        # 0. 거래소 포지션 sync (LiveBroker 전용)
        #    - tracker에 있는데 거래소에 없는 포지션은 수동 청산/외부 체결로 간주 → tracker에서 제거
        if hasattr(self.broker, "fetch_open_symbols"):
            try:
                open_syms = self.broker.fetch_open_symbols()
                tracker_state = self.tracker.snapshot()
                stale = [s for s in tracker_state.positions.keys() if s not in open_syms]
                for sym in stale:
                    pos = tracker_state.positions[sym]
                    last_price = self._get_prices(snapshot).get(sym, pos.entry_price)
                    self.tracker.close_position(
                        symbol=sym, exit_price=last_price, exit_time=now,
                        exit_reason="external_close",
                        regime=self.regime_detector.classify(snapshot).regime,
                        confluence_score=0,
                        commission=self._commission.calculate(pos.size_usd, OrderType.MARKET),
                        slippage_cost=self._slippage.cost(pos.size_usd, OrderType.MARKET),
                    )
                    import logging as _l
                    _l.getLogger(__name__).warning("sync: tracker에만 있던 %s 제거 (외부 청산)", sym)
            except Exception as e:
                import logging as _l
                _l.getLogger(__name__).warning("거래소 sync 실패: %s", e)

        state = self.tracker.snapshot()

        # 일일 리셋
        if self._last_day is None or now.date() != self._last_day.date():
            self.tracker.reset_daily()
            self._last_day = now

        # 1. 펀딩비 적용
        accruals = self.funding_sim.accrue(state, now, snapshot.funding_rates)
        if accruals:
            self.tracker.apply_funding(accruals)

        # 2. MTM 업데이트 + TP/SL 체크
        prices = self._get_prices(snapshot)
        self.tracker.mark_to_market(prices)
        self._check_tp_sl(snapshot, prices)

        state = self.tracker.snapshot()

        # 3. 드로다운 체크
        dd_action = self.guards.check_daily_drawdown(state)
        if dd_action == DrawdownAction.STOP:
            # 강제 청산
            for sym in list(state.positions.keys()):
                self._force_close(sym, prices.get(sym, 0.0), now, "forced_stop")
            return
        if dd_action == DrawdownAction.PAUSE:
            return

        # 4. 시장 국면 분류
        regime = self.regime_detector.classify(snapshot)

        # 5. 전략별 신호 생성
        for strategy in self.strategies:
            cb_status = self.circuit_breaker.get_status(strategy.name, now)
            if cb_status != BreakerStatus.ACTIVE:
                continue

            if not is_strategy_eligible(regime.regime, strategy.name):
                continue

            signals = strategy.generate_signals(snapshot, regime)

            for signal in signals:
                if not validate(signal):
                    continue

                scored = self.scorer.score(signal, snapshot, regime)
                if scored.tier == LeverageTier.NO_TRADE:
                    continue

                if not self.guards.is_entry_allowed(state, signal):
                    continue

                if self.corr_filter.is_blocked(signal, state):
                    continue

                size_usd, leverage = self.sizer.calculate(
                    scored.tier, state.equity, signal.entry_price, signal.sl_price
                )
                if size_usd <= 0:
                    continue

                side = OrderSide.BUY if signal.direction == "long" else OrderSide.SELL
                order = Order(
                    symbol=signal.symbol,
                    side=side,
                    size_usd=size_usd,
                    price=signal.entry_price,
                    order_type=OrderType.LIMIT,
                    leverage=leverage,
                    strategy=strategy.name,
                    signal_score=scored,
                    timestamp=now,
                    direction=signal.direction,
                    tp_price=signal.tp_price,
                    sl_price=signal.sl_price,
                )
                current_bar = (
                    snapshot.bars[signal.symbol]["1h"].iloc[-1]
                    if "1h" in snapshot.bars.get(signal.symbol, {})
                    else None
                )
                fill = self.broker.submit(order, current_bar)
                self.tracker.apply_fill(fill)
                state = self.tracker.snapshot()

        # 6. 자산곡선 기록
        self.equity_curve.append(now, state.equity, state.open_position_count)

    def _check_tp_sl(self, snapshot: MarketSnapshot, prices: dict[str, float]) -> None:
        state = self.tracker.snapshot()
        regime = self.regime_detector.classify(snapshot)

        for sym, pos in list(state.positions.items()):
            price = prices.get(sym)
            if price is None:
                continue

            hit_tp = (pos.direction == "long" and price >= pos.tp_price) or (
                pos.direction == "short" and price <= pos.tp_price
            )
            hit_sl = (pos.direction == "long" and price <= pos.sl_price) or (
                pos.direction == "short" and price >= pos.sl_price
            )

            # 최대 보유 시간 초과 시 시장가 강제 청산
            hit_time = False
            if self.max_hold_hours is not None:
                elapsed_h = (snapshot.timestamp - pos.opened_at).total_seconds() / 3600
                hit_time = elapsed_h >= self.max_hold_hours

            if hit_tp or hit_sl or hit_time:
                if hit_tp:
                    exit_price = pos.tp_price
                    exit_reason = "tp"
                elif hit_sl:
                    exit_price = pos.sl_price
                    exit_reason = "sl"
                else:
                    exit_price = price   # 시장가
                    exit_reason = "timeout"

                commission = self._commission.calculate(pos.size_usd, OrderType.MARKET)
                slippage = self._slippage.cost(pos.size_usd, OrderType.MARKET)

                # 라이브 브로커면 실제 거래소 포지션도 시장가 청산
                if hasattr(self.broker, "market_close"):
                    qty = pos.size_usd / pos.entry_price
                    try:
                        self.broker.market_close(sym, pos.direction, qty)
                    except Exception as e:
                        import logging as _l
                        _l.getLogger(__name__).error("broker.market_close 실패 %s: %s", sym, e)

                self.tracker.close_position(
                    symbol=sym,
                    exit_price=exit_price,
                    exit_time=snapshot.timestamp,
                    exit_reason=exit_reason,
                    regime=regime.regime,
                    confluence_score=0,
                    commission=commission,
                    slippage_cost=slippage,
                )

                is_win = exit_reason == "tp"
                self.circuit_breaker.record_result(pos.strategy, is_win)

    def _force_close(self, symbol: str, price: float, now: pd.Timestamp, reason: str) -> None:
        state = self.tracker.snapshot()
        if symbol not in state.positions:
            return
        pos = state.positions[symbol]
        regime = None
        # 강제 청산 시 국면 정보 없으면 RANGING으로 기본값
        from regime.models import MarketRegime
        self.tracker.close_position(
            symbol=symbol,
            exit_price=price,
            exit_time=now,
            exit_reason=reason,
            regime=MarketRegime.RANGING,
            confluence_score=0,
            commission=self._commission.calculate(pos.size_usd, OrderType.MARKET),
            slippage_cost=self._slippage.cost(pos.size_usd, OrderType.MARKET),
        )
