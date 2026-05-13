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
from risk.margin_tiers import MarginTierTable
from risk.position_sizer import PositionSizer
from signals.models import LeverageTier, Signal, SignalScore
from signals.scorer import ConfluenceScorer
from signals.validators import validate
from strategies.base import BaseStrategy


def _cand_proxy(cand: dict) -> Signal:
    """후보 dict → Signal 객체 (risk guards / corr filter용 최소 프록시)."""
    return Signal(
        symbol=cand["symbol"],
        strategy=cand["strategy"],
        direction=cand["direction"],
        entry_price=cand["entry_price"],
        tp_price=cand["tp_price"],
        sl_price=cand["sl_price"],
        timestamp=cand["timestamp"] if isinstance(cand["timestamp"], pd.Timestamp)
                  else pd.Timestamp(cand["timestamp"]),
    )


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
        breakeven_trigger_r: float | None = None,  # +N×R 도달 시 SL을 진입가로 이동
        trailing_r_mult: float | None = None,      # 1R 초과 이익 후 peak에서 N×R 만큼 trailing
        maintenance_margin_rate: float = 0.005,    # 폴백 MMR (margin_tier_table 미지정 시 사용)
        margin_tier_table: MarginTierTable | None = None,  # Binance 티어 MMR (권장)
        daily_reset_tz: str = "UTC",               # 일일 DD 리셋 기준 시간대 (예: "Asia/Seoul")
        cross_worst_case_check: bool = True,       # 다중 포지션 동시 역행 worst-case 청산 체크
        strategy_min_score: dict[str, int] | None = None,  # 전략별 최소 score 게이트 (scorer 위 덮어씀)
        strategy_block_hours: dict[str, list[int]] | None = None,  # 전략별 차단 UTC 시간대
        strategy_block_symbols: dict[str, list[str]] | None = None,  # 전략별 차단 심볼
        tier_block_symbols: dict[str, list[str]] | None = None,  # 티어별 차단 심볼 (e.g. {"S": ["ADAUSDT"]})
        abort_mdd_threshold: float | None = None,  # e.g. -0.35 → MDD 35% 초과 시 조기 중단 (None=끝까지)
        subbar_tpsl: bool = False,  # True → trailing 없어도 sub-bar(5m)로 TP/SL 체크
        isolated_margin: bool = False,  # True → 포지션별 isolated margin 청산 (cross 대신)
        equity_curve_trading: int = 0,   # >0 → equity SMA(N) 미만 시 사이징 50% 축소
        adx_scaling: bool = False,       # True → ADX 20-25 구간에서 사이징 60% 축소
        sl_reversal: bool = False,       # True → SL 히트 시 반대 방향 절반 사이즈 진입
        sl_reversal_leverage: int = 10,  # reversal 포지션 레버리지 (기본 10x)
        notifier=None,
        trade_log_path: str | None = None,  # CSV 거래 로그 경로 (라이브용)
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
        self.breakeven_trigger_r = breakeven_trigger_r
        self.trailing_r_mult = trailing_r_mult
        self._mm_rate = maintenance_margin_rate
        self._tier_table = margin_tier_table
        self._daily_reset_tz = daily_reset_tz
        self._cross_worst_case_check = cross_worst_case_check
        self._strategy_min_score = strategy_min_score or {}
        self._strategy_block_hours = {
            k: set(v) for k, v in (strategy_block_hours or {}).items()
        }
        self._strategy_block_symbols = {
            k: set(v) for k, v in (strategy_block_symbols or {}).items()
        }
        self._tier_block_symbols = {
            k: set(v) for k, v in (tier_block_symbols or {}).items()
        }
        self._block_directions: set[str] = set()  # {"short"} → 숏 차단, {"long"} → 롱 차단
        self._bankrupt = False

        self._subbar_tpsl = subbar_tpsl
        self._isolated_margin = isolated_margin
        self._equity_curve_trading = equity_curve_trading
        self._adx_scaling = adx_scaling
        self._sl_reversal = sl_reversal
        self._sl_reversal_leverage = sl_reversal_leverage
        self._equity_history: list[float] = []

        # 시그널 덤프/리플레이 모드
        self._dump_mode = False
        self._signal_log: list[dict] = []
        self._replay_index: dict[int, list[dict]] | None = None

        # Early-stop (옵셔널): running peak 대비 equity가 threshold 초과 하락 시 중단.
        # wf_score의 MDD 필터와 등가 (중간 위반 시 최종 MDD도 위반 확정).
        self._abort_mdd = abort_mdd_threshold
        self._peak_equity = initial_capital
        self._aborted = False

        self.ledger = Ledger(csv_path=trade_log_path)
        self.equity_curve = EquityCurve()
        self.tracker = PortfolioTracker(initial_capital, self.ledger, notifier=notifier)

        self._last_day: pd.Timestamp | None = None

    def _localize(self, ts: pd.Timestamp) -> pd.Timestamp:
        """daily_reset_tz 기준 로컬 시간으로 변환."""
        if self._daily_reset_tz == "UTC":
            return ts if ts.tz is None else ts.tz_convert("UTC")
        if ts.tz is None:
            return ts.tz_localize("UTC").tz_convert(self._daily_reset_tz)
        return ts.tz_convert(self._daily_reset_tz)

    def _mm_rate_for(self, symbol: str, notional: float) -> float:
        if self._tier_table is not None:
            return self._tier_table.mm_rate(symbol, notional)
        return self._mm_rate

    def _get_bars(self, snapshot: MarketSnapshot) -> dict[str, pd.Series]:
        """설정된 price_tf 기준으로 현재 OHLC bar 추출. 없으면 폴백."""
        bars: dict[str, pd.Series] = {}
        fallback_order = [self.price_tf, "5m", "15m", "1h", "4h", "1d"]
        for sym in snapshot.bars:
            for tf in fallback_order:
                df = snapshot.bars[sym].get(tf)
                if df is not None and not df.empty:
                    bars[sym] = df.iloc[-1]
                    break
        return bars

    def _get_prices(self, snapshot: MarketSnapshot) -> dict[str, float]:
        """MTM/일반 가격 참조용 close 추출."""
        return {sym: float(bar["close"]) for sym, bar in self._get_bars(snapshot).items()}

    # ── 시그널 덤프/리플레이 ─────────────────────────────────────
    def _generate_all_candidates(
        self, snapshot: MarketSnapshot, regime, prices: dict[str, float],
    ) -> list[dict]:
        """전략별 시그널 생성 + 스코링 + 정적 필터 → 후보 시그널 목록.

        CB 체크는 여기서 하지 않는다 (동적 상태 의존 → 실행 루프에서 처리).
        """
        candidates: list[dict] = []
        for strategy in self.strategies:
            if not is_strategy_eligible(regime.regime, strategy.name):
                continue
            signals = strategy.generate_signals(snapshot, regime)
            for signal in signals:
                if not validate(signal):
                    continue
                if signal.direction in self._block_directions:
                    continue
                scored = self.scorer.score(signal, snapshot, regime)
                if scored.tier == LeverageTier.NO_TRADE:
                    continue
                min_req = self._strategy_min_score.get(strategy.name)
                if min_req is not None and scored.total < min_req:
                    continue
                blocked = self._strategy_block_hours.get(strategy.name)
                if blocked and signal.timestamp.hour in blocked:
                    continue
                blocked_syms = self._strategy_block_symbols.get(strategy.name)
                if blocked_syms and signal.symbol in blocked_syms:
                    continue
                candidates.append({
                    "timestamp": signal.timestamp,
                    "symbol": signal.symbol,
                    "strategy": strategy.name,
                    "direction": signal.direction,
                    "entry_price": signal.entry_price,
                    "tp_price": signal.tp_price,
                    "sl_price": signal.sl_price,
                    "score": scored.total,
                    "tier": scored.tier.value,
                })
        return candidates

    def run(self, snapshots: Iterator[MarketSnapshot]) -> MetricsReport:
        for snapshot in snapshots:
            self._process_bar(snapshot)
            if self._bankrupt or self._aborted:
                break  # 파산 or MDD early-stop → 루프 즉시 종료

        report = MetricsReport.from_run(self.equity_curve, self.ledger)
        report.bankrupt = self._bankrupt
        report.aborted = self._aborted
        return report

    def run_dump(
        self, snapshots: Iterator[MarketSnapshot],
    ) -> tuple[MetricsReport, pd.DataFrame]:
        """풀 백테스트 실행 + 후보 시그널 수집 → (report, signals_df)."""
        self._dump_mode = True
        self._signal_log = []
        report = self.run(snapshots)
        self._dump_mode = False
        df = pd.DataFrame(self._signal_log) if self._signal_log else pd.DataFrame()
        return report, df

    def run_replay(
        self, signals_df: pd.DataFrame, snapshots: Iterator[MarketSnapshot],
    ) -> MetricsReport:
        """리플레이 모드: 사전 수집된 시그널로 사이징/리스크만 재계산."""
        self._replay_index = {}
        for _, row in signals_df.iterrows():
            ts_key = pd.Timestamp(row["timestamp"]).value
            if ts_key not in self._replay_index:
                self._replay_index[ts_key] = []
            self._replay_index[ts_key].append(row.to_dict())
        report = self.run(snapshots)
        self._replay_index = None
        return report

    def _process_bar(self, snapshot: MarketSnapshot) -> None:
        if self._bankrupt:
            return  # 파산 상태 — 더 이상 처리하지 않음
        now = snapshot.timestamp

        # 0. 거래소 포지션 sync (LiveBroker 전용)
        #    - tracker에 있는데 거래소에 없는 포지션은 TP 체결/수동 청산/외부 체결로 간주
        #    - 실제 체결가를 fetch_recent_fill_price로 조회 (없으면 close price 폴백)
        if hasattr(self.broker, "fetch_open_symbols"):
            try:
                open_syms = self.broker.fetch_open_symbols()
                tracker_state = self.tracker.snapshot()
                stale = [s for s in tracker_state.positions.keys() if s not in open_syms]
                for sym in stale:
                    pos = tracker_state.positions[sym]
                    # 실제 체결가 조회 시도 (TP LIMIT 체결 등)
                    exit_price = None
                    exit_reason = "external_close"
                    if hasattr(self.broker, "fetch_recent_fill_price"):
                        exit_price = self.broker.fetch_recent_fill_price(sym)
                    if exit_price is not None:
                        # TP 체결가와 비교하여 exit_reason 판별
                        if pos.tp_price > 0 and abs(exit_price - pos.tp_price) / pos.tp_price < 0.005:
                            exit_reason = "tp"
                            exit_type = OrderType.LIMIT  # TP는 maker
                        else:
                            exit_type = OrderType.MARKET
                    else:
                        exit_price = self._get_prices(snapshot).get(sym, pos.entry_price)
                        exit_type = OrderType.MARKET
                    exit_notional = pos.size_usd / pos.entry_price * exit_price
                    self.tracker.close_position(
                        symbol=sym, exit_price=exit_price, exit_time=now,
                        exit_reason=exit_reason,
                        regime=self.regime_detector.classify(snapshot).regime,
                        confluence_score=pos.confluence_score,
                        commission=self._commission.calculate(exit_notional, exit_type),
                        slippage_cost=self._slippage.cost(exit_notional, exit_type),
                    )
                    import logging
                    logging.getLogger(__name__).warning(
                        "sync: %s 제거 (reason=%s, exit_price=%.4f)", sym, exit_reason, exit_price
                    )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("거래소 sync 실패: %s", e)

        state = self.tracker.snapshot()

        # 일일 리셋 (설정된 시간대 기준 자정)
        local_now = self._localize(now)
        local_last = self._localize(self._last_day) if self._last_day is not None else None
        if local_last is None or local_now.date() != local_last.date():
            self.tracker.reset_daily()
            self._last_day = now

        # 가격 추출 (펀딩/MTM/TP-SL 공용)
        bars = self._get_bars(snapshot)
        prices = {sym: float(bar["close"]) for sym, bar in bars.items()}

        # 1. 펀딩비 적용 (mark price 기준 notional)
        accruals = self.funding_sim.accrue(state, now, snapshot.funding_rates, prices=prices)
        if accruals:
            self.tracker.apply_funding(accruals)

        # 2. MTM 업데이트 + 동적 SL + TP/SL 체크
        self.tracker.mark_to_market(prices)

        # 상관계수 필터용 close 시리즈 업데이트
        for sym in snapshot.bars:
            tf_order = [self.price_tf, "1h", "4h", "1d", "15m", "5m"]
            for tf in tf_order:
                df = snapshot.bars[sym].get(tf)
                if df is not None and not df.empty and "close" in df.columns:
                    self.corr_filter.update(sym, df["close"])
                    break
        # TP/SL + trailing SL을 sub-bar 순회로 통합 처리.
        # dynamic SL 활성 + sub-bar 데이터 존재 시 → 1h sub-bar 순회로 intrabar 경로 재현
        # → trailing SL 이동 후 터치되는 케이스를 정확히 포착.
        # sub-bar 없거나 dynamic SL 비활성 시 → 기존 bar-level 로직 폴백.
        self._process_tp_sl_trailing(snapshot, bars)

        # 청산 체크 (TP/SL 이후 남은 포지션 대상)
        self.tracker.mark_to_market(prices)  # TP/SL 체결 후 equity 갱신
        if self._isolated_margin:
            if self._check_isolated_liquidation(bars, prices, now):
                self._record_equity(now, prices)
                return
        else:
            if self._check_cross_liquidation(bars, prices, now):
                self._record_equity(now, prices)
                return  # 파산 → 신규 진입 없이 종료

        state = self.tracker.snapshot()

        # 3. 드로다운 체크
        dd_action = self.guards.check_daily_drawdown(state)
        if dd_action == DrawdownAction.STOP:
            # 강제 청산
            for sym in list(state.positions.keys()):
                self._force_close(sym, prices.get(sym, 0.0), now, "forced_stop")
            self._record_equity(now, prices)
            return
        if dd_action == DrawdownAction.PAUSE:
            self._record_equity(now, prices)
            return

        # 4. 시장 국면 분류
        regime = self.regime_detector.classify(snapshot)

        # 5. 후보 시그널 수집 (전략 생성 or 리플레이 캐시)
        if self._replay_index is not None:
            candidates = self._replay_index.get(now.value, [])
        else:
            candidates = self._generate_all_candidates(snapshot, regime, prices)
            if self._dump_mode:
                self._signal_log.extend(candidates)

        # 6. 동적 필터 + 체결
        for cand in candidates:
            # 티어별 심볼 차단
            tier_blocked = self._tier_block_symbols.get(cand["tier"])
            if tier_blocked and cand["symbol"] in tier_blocked:
                continue

            cb_status = self.circuit_breaker.get_status(cand["strategy"], now)
            if cb_status != BreakerStatus.ACTIVE:
                continue

            if not self.guards.is_entry_allowed(state, _cand_proxy(cand)):
                continue

            if self.corr_filter.is_blocked(_cand_proxy(cand), state):
                continue

            tier = LeverageTier(cand["tier"])
            size_usd, leverage = self.sizer.calculate(
                tier, state.equity, cand["entry_price"], cand["sl_price"]
            )
            if size_usd <= 0:
                continue

            # Equity Curve Trading: equity < SMA(N) → 사이즈 50% 축소
            if self._equity_curve_trading > 0 and len(self._equity_history) >= self._equity_curve_trading:
                sma = sum(self._equity_history[-self._equity_curve_trading:]) / self._equity_curve_trading
                if state.equity < sma:
                    size_usd *= 0.5

            # ADX Scaling: 레짐 애매 구간 (ADX 20-25) → 사이즈 60%
            if self._adx_scaling and hasattr(regime, 'adx'):
                if 20 < regime.adx < 25:
                    size_usd *= 0.6

            side = OrderSide.BUY if cand["direction"] == "long" else OrderSide.SELL
            mkt_price = prices.get(cand["symbol"], cand["entry_price"])
            shift = mkt_price - cand["entry_price"]

            scored_proxy = SignalScore(
                total=cand["score"],
                tier=tier,
                signal=_cand_proxy(cand),
            )

            order = Order(
                symbol=cand["symbol"],
                side=side,
                size_usd=size_usd,
                price=mkt_price,
                order_type=OrderType.MARKET,
                leverage=leverage,
                strategy=cand["strategy"],
                signal_score=scored_proxy,
                timestamp=now,
                direction=cand["direction"],
                tp_price=cand["tp_price"] + shift,
                sl_price=cand["sl_price"] + shift,
            )
            _1h_df = snapshot.bars.get(cand["symbol"], {}).get("1h")
            current_bar = (
                _1h_df.iloc[-1]
                if _1h_df is not None and len(_1h_df) > 0
                else None
            )
            try:
                fill = self.broker.submit(order, current_bar)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(
                    "주문 실패 %s %s: %s — 건너뜀", cand["symbol"], cand["direction"], e)
                continue
            self.tracker.apply_fill(fill)
            state = self.tracker.snapshot()

        # 6. 최종 MTM 재계산 후 자산곡선 기록 + early-stop 체크
        self._record_equity(now, prices)

    def _record_equity(self, now: pd.Timestamp, prices: dict[str, float]) -> None:
        """equity curve + early-stop 기록 (모든 return 경로에서 호출)."""
        self.tracker.mark_to_market(prices)
        state = self.tracker.snapshot()
        self.equity_curve.append(now, state.equity, state.open_position_count)
        self._equity_history.append(state.equity)
        if self._abort_mdd is not None:
            if state.equity > self._peak_equity:
                self._peak_equity = state.equity
            if self._peak_equity > 0:
                dd = (state.equity - self._peak_equity) / self._peak_equity
                if dd <= self._abort_mdd:
                    self._aborted = True

    # ── sub-bar 통합 TP/SL + trailing ─────────────────────────────
    _TF_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
    _TF_ORDER = ["5m", "15m", "1h", "4h", "1d"]

    def _get_sub_tf(self, snapshot: MarketSnapshot | None = None) -> str | None:
        """price_tf보다 낮으면서 snapshot에 실제 데이터가 있는 TF 반환."""
        try:
            idx = self._TF_ORDER.index(self.price_tf)
        except ValueError:
            return None
        # price_tf보다 낮은 TF를 높은 순서대로 탐색
        for i in range(idx - 1, -1, -1):
            candidate = self._TF_ORDER[i]
            if snapshot is not None:
                # 실제 데이터 존재 여부 확인 (첫 심볼로)
                for sym_bars in snapshot.bars.values():
                    df = sym_bars.get(candidate)
                    if df is not None and not df.empty:
                        return candidate
                    break  # 첫 심볼만 확인
            else:
                return candidate
        return None

    def _process_tp_sl_trailing(
        self, snapshot: MarketSnapshot, bars: dict[str, pd.Series]
    ) -> None:
        """TP/SL 체크 + trailing SL 업데이트를 sub-bar 순회로 통합.

        Dynamic SL 활성 + sub-bar 데이터 존재 시:
          1h sub-bar를 시간순 순회하며 매 sub-bar에서
          ① 현재 SL/TP 체크 → ② trailing 업데이트 → 다음 sub-bar.
          이를 통해 "SL 이동 후 같은 primary bar 내에서 히트" 케이스를 정확히 포착.

        Sub-bar 없거나 dynamic SL 비활성 시:
          bar-level 로직으로 폴백 (인라인 구현).
        """
        has_dynamic_sl = (self.breakeven_trigger_r is not None or
                          self.trailing_r_mult is not None)
        use_subbars = has_dynamic_sl or self._subbar_tpsl
        sub_tf = self._get_sub_tf(snapshot) if use_subbars else None

        state = self.tracker.snapshot()
        regime = self.regime_detector.classify(snapshot)

        # primary TF 1봉 내 sub-bar 개수
        primary_mins = self._TF_MINUTES.get(self.price_tf, 60)
        sub_mins = self._TF_MINUTES.get(sub_tf, 0) if sub_tf else 0
        n_sub = primary_mins // sub_mins if sub_mins > 0 else 0

        for sym, pos in list(state.positions.items()):
            bar = bars.get(sym)
            if bar is None:
                continue

            close = float(bar["close"])

            # timeout 체크 (max_hold_hours)
            hit_time = False
            if self.max_hold_hours is not None:
                elapsed_h = (snapshot.timestamp - pos.opened_at).total_seconds() / 3600
                hit_time = elapsed_h >= self.max_hold_hours

            # sub-bar 데이터 확보 시도
            sub_df = snapshot.bars.get(sym, {}).get(sub_tf) if sub_tf else None
            use_subbars = ((has_dynamic_sl or self._subbar_tpsl) and sub_df is not None
                           and len(sub_df) >= n_sub and n_sub >= 2)

            if use_subbars:
                result = self._iterate_subbars(
                    sym, pos, sub_df.tail(n_sub), hit_time, close
                )
                if result is not None:
                    exit_price, exit_reason = result
                    self._close_with_reason(
                        sym, pos, exit_price, exit_reason, snapshot, regime
                    )
                # sub-bar 순회 완료 (청산 여부 무관) → trailing 상태 이미 업데이트됨
                continue

            # ── 폴백: bar-level 체크 (기존 로직) ────────────────
            high = float(bar["high"])
            low = float(bar["low"])

            if pos.direction == "long":
                hit_sl = low <= pos.sl_price
                hit_tp = high >= pos.tp_price
            else:
                hit_sl = high >= pos.sl_price
                hit_tp = low <= pos.tp_price

            if hit_tp or hit_sl or hit_time:
                if hit_tp and hit_sl:
                    exit_price, exit_reason = self._resolve_ambiguous_tp_sl(
                        snapshot, sym, pos
                    )
                elif hit_sl:
                    exit_price, exit_reason = pos.sl_price, "sl"
                elif hit_tp:
                    exit_price, exit_reason = pos.tp_price, "tp"
                else:
                    exit_price, exit_reason = close, "timeout"
                self._close_with_reason(
                    sym, pos, exit_price, exit_reason, snapshot, regime
                )
            else:
                # bar-level trailing 업데이트 (dynamic SL이 있지만 sub-bar 없을 때)
                self._update_trailing_for_pos(pos, high, low)

    def _iterate_subbars(
        self, sym: str, pos, sub_bars: pd.DataFrame, hit_time: bool, bar_close: float
    ) -> tuple[float, str] | None:
        """Sub-bar 시간순 순회. 매 sub-bar에서 SL/TP 체크 → trailing 업데이트.
        Returns (exit_price, exit_reason) if closed, None if survived.
        """
        for _, sb in sub_bars.iterrows():
            sb_high = float(sb["high"])
            sb_low = float(sb["low"])

            # ① 현재 SL/TP 체크
            if pos.direction == "long":
                sub_hit_sl = sb_low <= pos.sl_price
                sub_hit_tp = sb_high >= pos.tp_price
            else:
                sub_hit_sl = sb_high >= pos.sl_price
                sub_hit_tp = sb_low <= pos.tp_price

            if sub_hit_sl and sub_hit_tp:
                return pos.sl_price, "sl"  # 동시 → 보수적(SL)
            if sub_hit_sl:
                return pos.sl_price, "sl"
            if sub_hit_tp:
                return pos.tp_price, "tp"

            # ② trailing 업데이트 (이 sub-bar 기준)
            self._update_trailing_for_pos(pos, sb_high, sb_low)

        # 모든 sub-bar 순회 완료, timeout 체크
        if hit_time:
            return bar_close, "timeout"
        return None

    def _update_trailing_for_pos(self, pos, high: float, low: float) -> None:
        """단일 봉(또는 sub-bar)의 high/low로 peak + breakeven + trailing 업데이트."""
        if pos.direction == "long":
            if high > pos.peak_price:
                pos.peak_price = high
        else:
            if low < pos.peak_price:
                pos.peak_price = low

        initial_risk = abs(pos.entry_price - pos.initial_sl_price)
        if initial_risk <= 0:
            return

        if self.breakeven_trigger_r is not None:
            if pos.direction == "long":
                trigger = pos.entry_price + self.breakeven_trigger_r * initial_risk
                if pos.peak_price >= trigger and pos.sl_price < pos.entry_price:
                    pos.sl_price = pos.entry_price
            else:
                trigger = pos.entry_price - self.breakeven_trigger_r * initial_risk
                if pos.peak_price <= trigger and pos.sl_price > pos.entry_price:
                    pos.sl_price = pos.entry_price

        if self.trailing_r_mult is not None:
            if pos.direction == "long":
                profit = pos.peak_price - pos.entry_price
                if profit > initial_risk:
                    new_sl = pos.peak_price - initial_risk * self.trailing_r_mult
                    if new_sl > pos.sl_price:
                        pos.sl_price = new_sl
            else:
                profit = pos.entry_price - pos.peak_price
                if profit > initial_risk:
                    new_sl = pos.peak_price + initial_risk * self.trailing_r_mult
                    if new_sl < pos.sl_price:
                        pos.sl_price = new_sl

    def _close_with_reason(
        self, sym: str, pos, exit_price: float, exit_reason: str,
        snapshot: MarketSnapshot, regime
    ) -> None:
        """TP/SL/timeout 공통 청산 처리."""
        exit_type = OrderType.LIMIT if exit_reason == "tp" else OrderType.MARKET
        # exit 시점 notional 기준 수수료 (바이낸스: qty × mark_price)
        exit_notional = pos.size_usd / pos.entry_price * exit_price
        commission = self._commission.calculate(exit_notional, exit_type)
        slippage = self._slippage.cost(exit_notional, exit_type)

        if hasattr(self.broker, "market_close"):
            # TP LIMIT이 거래소 측에서 이미 체결된 경우 market_close 불필요
            already_closed = False
            if hasattr(self.broker, "fetch_open_symbols"):
                try:
                    if sym not in self.broker.fetch_open_symbols():
                        already_closed = True
                        import logging
                        logging.getLogger(__name__).info(
                            "거래소 포지션 이미 청산됨 (TP/SL 체결): %s — tracker만 업데이트", sym)
                        # 실제 체결가 조회 시도
                        if hasattr(self.broker, "fetch_recent_fill_price"):
                            real_price = self.broker.fetch_recent_fill_price(sym)
                            if real_price is not None:
                                exit_price = real_price
                except Exception:
                    pass
            if not already_closed:
                qty = pos.size_usd / pos.entry_price
                try:
                    self.broker.market_close(sym, pos.direction, qty)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(
                        "broker.market_close 실패 %s: %s — tracker 청산 건너뜀 (고아 방지)", sym, e
                    )
                    return

        self.tracker.close_position(
            symbol=sym, exit_price=exit_price, exit_time=snapshot.timestamp,
            exit_reason=exit_reason, regime=regime.regime,
            confluence_score=pos.confluence_score,
            commission=commission, slippage_cost=slippage,
        )
        last_trade = self.ledger._records[-1] if self.ledger._records else None
        is_win = last_trade is not None and last_trade.pnl > 0
        self.circuit_breaker.record_result(pos.strategy, is_win)

        # SL Reversal: SL 히트 시 반대 방향으로 절반 사이즈 진입
        if (self._sl_reversal and exit_reason == "sl"
                and sym not in self.tracker.snapshot().positions):
            rev_dir = "short" if pos.direction == "long" else "long"
            rev_side = OrderSide.SELL if pos.direction == "long" else OrderSide.BUY
            prices = self._get_prices(snapshot)
            rev_price = prices.get(sym, exit_price)
            # 반대 방향 ATR 기반 TP/SL (원래 거리의 절반 leverage로)
            atr_dist = abs(pos.tp_price - pos.entry_price)  # 원래 TP 거리 참고
            sl_dist = abs(pos.entry_price - pos.sl_price)
            if rev_dir == "short":
                rev_tp = rev_price - atr_dist * 0.7
                rev_sl = rev_price + sl_dist
            else:
                rev_tp = rev_price + atr_dist * 0.7
                rev_sl = rev_price - sl_dist
            rev_size = pos.size_usd * 0.5 * (self._sl_reversal_leverage / pos.leverage)
            if rev_size > 0 and self.guards.is_entry_allowed(
                self.tracker.snapshot(),
                type('S', (), {'direction': rev_dir, 'symbol': sym})()
            ):
                rev_order = Order(
                    symbol=sym, side=rev_side, size_usd=rev_size,
                    price=rev_price, order_type=OrderType.MARKET,
                    leverage=self._sl_reversal_leverage,
                    strategy=pos.strategy + "_rev",
                    signal_score=None, timestamp=snapshot.timestamp,
                    direction=rev_dir, tp_price=rev_tp, sl_price=rev_sl,
                )
                current_bar = (
                    snapshot.bars[sym]["1h"].iloc[-1]
                    if "1h" in snapshot.bars.get(sym, {}) else None
                )
                fill = self.broker.submit(rev_order, current_bar)
                self.tracker.apply_fill(fill)

    def _resolve_ambiguous_tp_sl(
        self, snapshot: MarketSnapshot, sym: str, pos
    ) -> tuple[float, str]:
        """TP+SL 동시 터치 봉 → 5m 서브바 순회로 실제 선후 판별.

        5m 데이터가 없거나 5m 레벨에서도 동시 터치면 SL 우선(보수적 폴백).
        서브바 순회 중 trailing SL도 실시간 업데이트(intrabar 경로 재현).
        """
        sub_df = snapshot.bars.get(sym, {}).get("5m")
        if sub_df is None or len(sub_df) < 2:
            return pos.sl_price, "sl"  # 데이터 없음 → 보수적 폴백

        # 현재 primary TF 봉을 구성하는 5m 서브바 개수 동적 계산
        primary_mins = self._TF_MINUTES.get(self.price_tf, 60)
        sub_mins = self._TF_MINUTES.get("5m", 5)
        n_sub_expected = primary_mins // sub_mins
        sub_bars = sub_df.tail(n_sub_expected)

        # 가상 trailing 상태 (pos 객체를 직접 변경하지 않음)
        v_sl = pos.sl_price
        v_peak = pos.peak_price
        initial_risk = abs(pos.entry_price - pos.initial_sl_price)

        for _, sb in sub_bars.iterrows():
            sb_high = float(sb["high"])
            sb_low = float(sb["low"])

            # 현재 v_sl 기준으로 SL/TP 체크
            if pos.direction == "long":
                sub_hit_sl = sb_low <= v_sl
                sub_hit_tp = sb_high >= pos.tp_price
            else:
                sub_hit_sl = sb_high >= v_sl
                sub_hit_tp = sb_low <= pos.tp_price

            if sub_hit_sl and sub_hit_tp:
                return v_sl, "sl"   # 5m에서도 동시 → 보수적
            if sub_hit_sl:
                return v_sl, "sl"
            if sub_hit_tp:
                return pos.tp_price, "tp"

            # 미체결 → 이 5m 봉 기준 가상 trailing SL 전진
            if initial_risk > 0:
                # peak 업데이트
                if pos.direction == "long":
                    v_peak = max(v_peak, sb_high)
                else:
                    v_peak = min(v_peak, sb_low) if v_peak > 0 else sb_low

                # Breakeven
                if self.breakeven_trigger_r is not None:
                    if pos.direction == "long":
                        trigger = pos.entry_price + self.breakeven_trigger_r * initial_risk
                        if v_peak >= trigger and v_sl < pos.entry_price:
                            v_sl = pos.entry_price
                    else:
                        trigger = pos.entry_price - self.breakeven_trigger_r * initial_risk
                        if v_peak <= trigger and v_sl > pos.entry_price:
                            v_sl = pos.entry_price

                # Trailing
                if self.trailing_r_mult is not None:
                    if pos.direction == "long":
                        profit = v_peak - pos.entry_price
                        if profit > initial_risk:
                            new_sl = v_peak - initial_risk * self.trailing_r_mult
                            if new_sl > v_sl:
                                v_sl = new_sl
                    else:
                        profit = pos.entry_price - v_peak
                        if profit > initial_risk:
                            new_sl = v_peak + initial_risk * self.trailing_r_mult
                            if new_sl < v_sl:
                                v_sl = new_sl

        # 서브바 전부 순회했는데 체결 없음 → 데이터 불일치.
        # bar-level에서 TP+SL 동시 히트 확인됐으나 sub-bar에서 미확인:
        # open→close 방향으로 추정 (close가 TP 쪽이면 TP, 아니면 SL)
        last_close = float(sub_bars.iloc[-1]["close"]) if len(sub_bars) > 0 else pos.entry_price
        if pos.direction == "long":
            if last_close >= pos.tp_price:
                return pos.tp_price, "tp"
        else:
            if last_close <= pos.tp_price:
                return pos.tp_price, "tp"
        return v_sl, "sl"

    def _check_isolated_liquidation(
        self, bars: dict[str, pd.Series], prices: dict[str, float], now: pd.Timestamp
    ) -> bool:
        """Isolated margin 청산 체크.

        각 포지션이 독립적 마진을 보유. 손실이 초기마진(notional/leverage)을
        초과하면 해당 포지션만 청산. 다른 포지션에 영향 없음.
        """
        state = self.tracker.snapshot()
        if not state.positions:
            return False

        for sym in list(state.positions.keys()):
            state = self.tracker.snapshot()
            pos = state.positions.get(sym)
            if pos is None:
                continue
            bar = bars.get(sym)
            if bar is None:
                continue

            qty = pos.size_usd / pos.entry_price
            notional = pos.size_usd
            initial_margin = notional / pos.leverage
            mmr = self._mm_rate_for(sym, notional)
            mm = notional * mmr

            # isolated 청산가: 마진 소진 시점
            # long: liq = entry - (initial_margin - mm) / qty
            # short: liq = entry + (initial_margin - mm) / qty
            margin_buffer = initial_margin - mm
            if margin_buffer <= 0 or qty <= 0:
                continue
            if pos.direction == "long":
                liq_price = pos.entry_price - margin_buffer / qty
                if liq_price <= 0:
                    continue
                hit = float(bar["low"]) <= liq_price
            else:
                liq_price = pos.entry_price + margin_buffer / qty
                hit = float(bar["high"]) >= liq_price

            if not hit:
                continue

            # 청산: 해당 포지션만 강제청산, 손실 = initial_margin (마진 전액 손실)
            self._force_close(sym, liq_price, now, "liquidated")

        return self._bankrupt

    def _check_cross_liquidation(
        self, bars: dict[str, pd.Series], prices: dict[str, float], now: pd.Timestamp
    ) -> bool:
        """Cross margin 청산 체크.

        1) Close price 기준: equity ≤ total_maintenance_margin → 전 포지션 강제청산 + 파산
        2) Intrabar 기준: 각 포지션의 청산가(liq_price)를 계산하고
           해당 봉의 high/low가 청산가를 터치하면 청산가에 강제청산.
           이후 equity ≤ 0이면 파산.

        Returns True if bankruptcy occurred.
        """
        state = self.tracker.snapshot()
        if not state.positions:
            return False

        def _mark_notional(p, px_map: dict[str, float]) -> float:
            """현재 mark price 기준 notional = qty × mark_price.
            바이낸스 MM은 mark 기준이므로 entry_price 고정 notional은 근사치."""
            mark = px_map.get(p.symbol, p.entry_price)
            qty = p.size_usd / p.entry_price
            return qty * mark

        def _total_mm_at(px_map: dict[str, float]) -> float:
            st = self.tracker.snapshot()
            total = 0.0
            for s, p in st.positions.items():
                n = _mark_notional(p, px_map)
                total += n * self._mm_rate_for(s, n)
            return total

        total_mm = _total_mm_at(prices)

        # ── 1. Close price 기준 ──────────────────────────────────────────────
        # MM 미달 → 전 포지션 강제청산. 청산 후 잔존 cash가 있을 수 있으므로
        # 청산 완료 후 equity ≤ 0일 때만 실제 파산 처리.
        if state.equity <= total_mm:
            for sym in list(state.positions.keys()):
                self._force_close(sym, prices.get(sym, 0.0), now, "liquidated")
            self.tracker.mark_to_market(prices)
            if self.tracker.snapshot().equity <= 0:
                self._bankrupt = True
                return True
            return False

        # ── 1b. Worst-case (다중 포지션 동시 역행) intrabar 체크 ─────────────
        # 각 포지션이 봉 내 최악의 방향(LONG→low, SHORT→high)으로 동시에 움직였을 때
        # equity ≤ total_mm(worst) 이면 가장 손실 큰 포지션부터 그 worst price에서 강제 청산.
        if self._cross_worst_case_check:
            worst_unreal = 0.0
            worst_prices: dict[str, float] = {}
            for s, p in state.positions.items():
                bar = bars.get(s)
                if bar is None:
                    worst_unreal += p.unrealized_pnl
                    worst_prices[s] = prices.get(s, p.entry_price)
                    continue
                if p.direction == "long":
                    wp = float(bar["low"])
                else:
                    wp = float(bar["high"])
                worst_prices[s] = wp
                pnl = p.size_usd * (wp - p.entry_price) / p.entry_price
                if p.direction == "short":
                    pnl = -pnl
                worst_unreal += pnl
            worst_equity = state.cash + worst_unreal
            worst_total_mm = _total_mm_at(worst_prices)
            if worst_equity <= worst_total_mm:
                losers = sorted(
                    state.positions.keys(),
                    key=lambda s: (
                        (worst_prices.get(s, prices.get(s, 0.0)) - state.positions[s].entry_price)
                        / state.positions[s].entry_price
                        * state.positions[s].size_usd
                        * (1 if state.positions[s].direction == "long" else -1)
                    ),
                )
                for sym in losers:
                    cur = self.tracker.snapshot()
                    if not cur.positions:
                        break
                    self._force_close(
                        sym, worst_prices.get(sym, prices.get(sym, 0.0)), now, "liquidated"
                    )
                    self.tracker.mark_to_market(prices)
                    cur = self.tracker.snapshot()
                    if cur.equity <= 0:
                        for rem in list(cur.positions.keys()):
                            self._force_close(rem, prices.get(rem, 0.0), now, "liquidated")
                        self._bankrupt = True
                        return True
                    if cur.equity > _total_mm_at(prices):
                        break
                state = self.tracker.snapshot()
                if not state.positions:
                    return self._bankrupt

        # ── 2. Intrabar 청산가 터치 체크 (self-reference MM) ─────────────────
        # 청산가 L에서 이 포지션의 MM = qty*L*mmr(qty*L)  → L이 L에 의존 (자기참조).
        # 방정식: other_equity + qty*(L-entry)[long] = other_mm(mark) + qty*L*mmr
        # mmr이 티어에 따라 step이라 고정점 반복(최대 3회)으로 수렴.
        for sym in list(state.positions.keys()):
            state = self.tracker.snapshot()
            pos = state.positions.get(sym)
            if pos is None:
                continue

            bar = bars.get(sym)
            if bar is None:
                continue

            qty = pos.size_usd / pos.entry_price
            # 다른 포지션들의 mark 기준 MM 및 equity 기여
            other_mm = 0.0
            for s2, p2 in state.positions.items():
                if s2 == sym:
                    continue
                n = _mark_notional(p2, prices)
                other_mm += n * self._mm_rate_for(s2, n)
            other_equity = state.equity - pos.unrealized_pnl

            # 고정점 반복으로 L 산출 (mmr 티어 자기참조)
            mmr = self._mm_rate_for(sym, pos.size_usd)
            liq_price = 0.0
            for _ in range(3):
                if pos.direction == "long":
                    denom = qty * (1 - mmr)
                    if denom <= 0:
                        liq_price = 0.0
                        break
                    liq_price = (other_mm - other_equity + qty * pos.entry_price) / denom
                else:
                    denom = qty * (1 + mmr)
                    liq_price = (qty * pos.entry_price + other_equity - other_mm) / denom
                if liq_price <= 0:
                    break
                new_mmr = self._mm_rate_for(sym, qty * liq_price)
                if abs(new_mmr - mmr) < 1e-6:
                    break
                mmr = new_mmr

            # 청산가가 엔트리를 넘어 "이익 구간"이면 유효 리스크 아님 (L>=entry for long 등)
            if pos.direction == "long":
                if liq_price <= 0 or liq_price >= pos.entry_price:
                    continue
                hit = float(bar["low"]) <= liq_price
            else:
                if liq_price <= 0 or liq_price <= pos.entry_price:
                    continue
                hit = float(bar["high"]) >= liq_price

            if not hit:
                continue

            # 청산가에 강제청산 (수수료+슬리피지 포함)
            self._force_close(sym, liq_price, now, "liquidated")

            # 청산 후 equity 재확인
            self.tracker.mark_to_market(prices)
            state = self.tracker.snapshot()

            if state.equity <= 0:
                # equity 음수 → 나머지 포지션도 현재가에 청산
                for rem in list(state.positions.keys()):
                    self._force_close(rem, prices.get(rem, 0.0), now, "liquidated")
                self._bankrupt = True
                return True

        return self._bankrupt

    def _force_close(self, symbol: str, price: float, now: pd.Timestamp, reason: str) -> None:
        state = self.tracker.snapshot()
        if symbol not in state.positions:
            return
        pos = state.positions[symbol]
        # 가격 누락(0 이하) → entry_price 폴백 (손익 0). 0.0 폴백은 가짜 대손실 발생.
        if price is None or price <= 0:
            price = pos.entry_price

        # 라이브 브로커가 있으면 거래소 청산 먼저 시도
        if hasattr(self.broker, "market_close"):
            qty = pos.size_usd / pos.entry_price
            try:
                self.broker.market_close(symbol, pos.direction, qty)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(
                    "force_close broker.market_close 실패 %s: %s — tracker 청산 건너뜀", symbol, e
                )
                return  # 거래소 청산 실패 → tracker도 건너뜀 (고아 방지)

        # 강제 청산 시 국면 정보 없으면 RANGING으로 기본값
        from regime.models import MarketRegime
        self.tracker.close_position(
            symbol=symbol,
            exit_price=price,
            exit_time=now,
            exit_reason=reason,
            regime=MarketRegime.RANGING,
            confluence_score=pos.confluence_score,
            commission=self._commission.calculate(pos.size_usd / pos.entry_price * price, OrderType.MARKET),
            slippage_cost=self._slippage.cost(pos.size_usd / pos.entry_price * price, OrderType.MARKET),
        )

        # forced_stop(일일 DD)은 전략 손실로 circuit_breaker에 기록.
        # liquidation은 계정 전반 이벤트라 특정 전략 귀속 부적절 → 미기록.
        if reason == "forced_stop":
            last = self.ledger._records[-1] if self.ledger._records else None
            if last is not None:
                self.circuit_breaker.record_result(pos.strategy, last.pnl > 0)
