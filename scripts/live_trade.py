"""Binance USDM Futures 데모 라이브 트레이딩 러너.

Group A 최적 파라미터(agg-W8 #10 또는 basic-W1 #6)를 기반으로
1h 봉마다 신호를 생성하고 Binance 데모 계정에 주문을 전송한다.

Usage:
    python scripts/live_trade.py                    # 데모 모드 (기본)
    python scripts/live_trade.py --dry-run          # 주문 전송 없이 로그만
    python scripts/live_trade.py --snap-now         # 즉시 한 번 실행 후 종료 (테스트)
    python scripts/live_trade.py --params config/params_best.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

import ccxt
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.live_feed import LiveFeed
from engine.backtest import BacktestEngine
from portfolio import state_store
from execution.commission import CommissionModel
from execution.funding import FundingRateSimulator
from execution.live_broker import LiveBroker
from execution.notifier import TelegramNotifier
from execution.sl_poller import SLPoller
from execution.slippage import SlippageModel
from metrics.report import MetricsReport
from regime.detector import RegimeDetector
from risk.circuit_breaker import CircuitBreaker
from risk.correlation import CorrelationFilter
from risk.guards import RiskGuards
from risk.position_sizer import PositionSizer
from signals.scorer import ConfluenceScorer
from strategies.ema_cross import EMACrossStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.multi_tf_breakout import MultiTFBreakoutStrategy

# ── 로깅 ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("live_trade.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("live_trade")

# ── 기본 파라미터 파일 ────────────────────────────────────────────
DEFAULT_PARAMS = "config/merged_noblock_sleeve.yaml"  # 채택 2026-06-09: 무차단+슬리브 50:50


# ── ccxt Exchange 생성 ────────────────────────────────────────────
def build_exchange(demo: bool) -> ccxt.Exchange:
    api_key = os.environ.get("BINANCE_API_KEY", "")
    secret  = os.environ.get("BINANCE_SECRET", "")

    if not api_key or not secret:
        raise ValueError(
            "BINANCE_API_KEY / BINANCE_SECRET 환경변수가 없습니다. "
            ".env 파일을 확인하세요."
        )

    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",
            "disableFuturesSandboxWarning": True,
        },
    })

    if demo:
        exchange.set_sandbox_mode(True)
        logger.info("데모(Testnet) 모드로 실행")
    else:
        logger.warning("실제 계정으로 실행 중! 주의하세요.")

    return exchange


# ── 엔진 빌드 ─────────────────────────────────────────────────────
def build_engine(p: dict, broker: LiveBroker, notifier: TelegramNotifier | None = None, initial_capital: float | None = None) -> BacktestEngine:
    symbols    = p["symbols"]
    r          = p.get("risk", {})
    e          = p.get("execution", {})
    rg         = p.get("regime", {})
    sc         = p.get("scorer", {})

    strategy_map = {
        "ema_cross":          EMACrossStrategy,
        "ema_cross_slow":     EMACrossStrategy,   # 다중 속도 변형 (strategy_name으로 구분)
        "multi_tf_breakout":  MultiTFBreakoutStrategy,
        "mean_reversion":     MeanReversionStrategy,
    }
    strategies = []
    for key, cls in strategy_map.items():
        cfg = p.get("strategies", {}).get(key)
        if cfg is None:
            continue  # config 미정의 전략은 비활성 (run_backtest와 동일 규칙)
        if not cfg.get("enabled", True):
            continue
        cfg.setdefault("symbols", symbols)  # config에 전략별 symbols 있으면 존중(run_backtest와 동일)
        strategies.append(cls(cfg))

    cap = initial_capital if initial_capital is not None else p.get("backtest", {}).get("initial_capital", 10_000)

    # 피라미딩 라이브: 진입 시 STOP_MARKET 증액 주문 등록 → 체결 시 엔진이
    # 봉 high/low 트리거로 tracker 동기화 + TP/SL 총수량 재등록 (engine/backtest.py)
    # ⚠️ testnet은 STOP_MARKET -4120 차단 → 증액 주문 등록 실패 로그 발생 (포지션은 정상)

    # ML 소프트 스코링 (선택적) — run_backtest.py와 동일 로직
    _ml_filter_live = None
    _ml_cfg_live = sc.get("ml_soft_scoring", {})
    _ml_mode_live = "bonus"
    _ml_cut_live = 0.45
    if _ml_cfg_live.get("enabled", False):
        try:
            from strategies.ml_filter import MLModels, MLSignalFilter
            _models_live = MLModels.load(_ml_cfg_live.get("model_path", "models/ml_filter.pkl"))
            _ml_mode_live = _ml_cfg_live.get("mode", "bonus")
            _ml_cut_live = _ml_cfg_live.get("cut_threshold", 0.45)
            _ml_filter_live = MLSignalFilter(
                models=_models_live,
                clf_threshold=_ml_cut_live if _ml_mode_live == "hardcut" else 0.0,
            )
        except Exception as _e:
            import logging as _lg
            _lg.getLogger(__name__).warning("ML 모델 로드 실패: %s — ML 비활성", _e)

    return BacktestEngine(
        initial_capital=cap,
        strategies=strategies,
        regime_detector=RegimeDetector(
            primary_symbol=symbols[0],
            adx_period=rg.get("adx_period", 14),
            adx_trending_threshold=rg.get("adx_trending_threshold", 25.0),
            adx_ranging_threshold=rg.get("adx_ranging_threshold", 20.0),
            bb_period=rg.get("bb_period", 20),
            bb_std=rg.get("bb_std", 2.0),
            bb_width_lookback=rg.get("bb_width_lookback", 50),
            bb_width_squeeze_pct=rg.get("bb_width_squeeze_pct", 0.2),
            primary_tf=rg.get("primary_tf", "1h"),
        ),
        confluence_scorer=ConfluenceScorer(
            volume_ratio_threshold=sc.get("volume_ratio_threshold", 1.5),
            rsi_long_max=sc.get("rsi_long_max", 65.0),
            rsi_short_min=sc.get("rsi_short_min", 35.0),
            funding_long_max=sc.get("funding_long_max", 0.0003),
            funding_short_min=sc.get("funding_short_min", -0.0003),
            daily_ema_period=sc.get("daily_ema_period", 200),
            tier_sss_min_score=sc.get("tier_sss_min_score", 99),
            tier_ss_min_score=sc.get("tier_ss_min_score", 7),
            tier_s_min_score=sc.get("tier_s_min_score", 5),
            tier_a_min_score=sc.get("tier_a_min_score", 3),
            tier_b_min_score=sc.get("tier_b_min_score", 2),
            tier_c_min_score=sc.get("tier_c_min_score", 1),
            regime_strong_adx=sc.get("regime_strong_adx"),
            regime_high_adx_cutoff=sc.get("regime_high_adx_cutoff"),
            ml_filter=_ml_filter_live,
            ml_bonus_threshold_1=_ml_cfg_live.get("bonus_threshold_1", 0.6),
            ml_bonus_threshold_2=_ml_cfg_live.get("bonus_threshold_2", 0.75),
            ml_mode=_ml_mode_live,
            ml_cut_threshold=_ml_cut_live,
            rsi_neutral_penalty=tuple(sc["rsi_neutral_penalty"]) if sc.get("rsi_neutral_penalty") else None,
        ),
        risk_guards=RiskGuards(
            max_positions=r.get("max_positions", 4),
            max_same_direction=r.get("max_same_direction", 3),
            daily_pause_threshold=r.get("daily_drawdown_pause", -0.05),
            daily_stop_threshold=r.get("daily_drawdown_stop", -0.08),
            tp_cooldown_hours=r.get("tp_cooldown_hours", 0.0),
        ),
        circuit_breaker=CircuitBreaker(
            strategy_pause_losses=r.get("circuit_breaker_pause_losses", 5),
            global_stop_losses=r.get("circuit_breaker_stop_losses", 10),
            pause_duration_hours=r.get("circuit_breaker_pause_hours", 48),
        ),
        correlation_filter=CorrelationFilter(
            block_threshold=r.get("correlation_block_threshold", 0.9),
            lookback=r.get("correlation_lookback", 100),
        ),
        position_sizer=PositionSizer(
            risk_per_trade=r.get("risk_per_trade", 0.01),
            tier_config=p.get("leverage_tiers"),
            max_notional_usd=r.get("max_notional_usd"),
            max_notional_equity_mult=r.get("max_notional_equity_mult", 3.0),
        ),
        strategy_leverage_tiers=p.get("strategy_leverage_tiers"),
        strategy_capital_fraction=p.get("strategy_capital_fraction"),
        broker=broker,
        funding_simulator=FundingRateSimulator(
            interval_hours=e.get("funding_interval_hours", 8)
        ),
        price_tf=p.get("primary_timeframe", "1h"),  # 1d 슬리브용 — 기본 1h(v16 무영향)
        max_hold_hours=p.get("engine", {}).get("max_hold_hours"),
        breakeven_trigger_r=p.get("engine", {}).get("breakeven_trigger_r"),
        trailing_r_mult=p.get("engine", {}).get("trailing_r_mult"),
        strategy_min_score=p.get("strategy_min_score"),
        strategy_block_hours=p.get("strategy_block_hours"),
        strategy_block_symbols=p.get("strategy_block_symbols"),
        tier_block_symbols=p.get("tier_block_symbols"),
        symbol_block_directions=p.get("symbol_block_directions"),
        strategy_block_tiers=p.get("strategy_block_tiers"),
        block_weekdays=p.get("block_weekdays"),
        direction_size_mult=p.get("direction_size_mult"),
        strategy_size_penalty=p.get("strategy_size_penalty"),
        strategy_size_bonus=p.get("strategy_size_bonus"),
        strategy_size_bonus_mult=p.get("strategy_size_bonus_mult", 1.5),
        pyramid_trigger_r=(p.get("pyramid", {}).get("trigger_r")
                           if p.get("pyramid", {}).get("enabled") else None),
        pyramid_add_fraction=p.get("pyramid", {}).get("add_fraction", 0.5),
        pyramid_max_adds=p.get("pyramid", {}).get("max_adds", 1),
        pyramid_strategies=p.get("pyramid", {}).get("strategies"),
        pyramid_min_score=p.get("pyramid", {}).get("min_score"),
        rsi_momentum_gate=p.get("rsi_momentum", {}).get("gate"),
        rsi_momentum_weight=p.get("rsi_momentum", {}).get("weight"),
        rsi_momentum_period=p.get("rsi_momentum", {}).get("period", 14),
        vol_target_ann=p.get("vol_target", {}).get("target_ann"),
        vol_scale_min=p.get("vol_target", {}).get("scale_min", 0.3),
        vol_scale_max=p.get("vol_target", {}).get("scale_max", 2.0),
        vol_lookback=p.get("vol_target", {}).get("lookback", 30),
        btc_mom_gate=p.get("btc_regime", {}).get("mom_gate", False),
        btc_mom_opposite_weight=p.get("btc_regime", {}).get("mom_opposite_weight"),
        btc_mom_lookback=p.get("btc_regime", {}).get("mom_lookback", 20),
        equity_curve_trading=p.get("equity_curve_trading", 0),
        adx_scaling=p.get("adx_scaling", False),
        notifier=notifier,
        trade_log_path="trades.csv",
    )


# ── 메인 ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Binance 데모 라이브 트레이딩")
    parser.add_argument("--params",    default=DEFAULT_PARAMS)
    parser.add_argument("--dry-run",   action="store_true",
                        help="주문 전송 없이 신호만 로깅")
    parser.add_argument("--snap-now",  action="store_true",
                        help="한 번만 실행하고 종료 (연결 테스트용)")
    parser.add_argument("--no-demo",   action="store_true",
                        help="실계정 사용 (주의)")
    parser.add_argument("--sl-poll-sec", type=int, default=None,
                        help="SL 폴러 주기(초). 기본: demo=300(5분), live=0(비활성). "
                             "0이면 수동 비활성. 실계정은 STOP_MARKET이 거래소에서 동작하므로 폴러 불필요.")
    args = parser.parse_args()

    # .env 로드 (python-dotenv 없어도 직접 파싱)
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    # 파라미터 로드
    params_path = Path(args.params)
    if not params_path.exists():
        # 최적 파라미터가 아직 없으면 기본 v2 사용
        logger.warning("%s 없음 → config/final_v13_eth.yaml 사용", args.params)
        params_path = Path("config/final_v13_eth.yaml")

    with open(params_path) as f:
        params = yaml.safe_load(f)
    logger.info("파라미터 로드: %s", params_path)

    demo    = not args.no_demo
    dry_run = args.dry_run

    # Telegram notifier
    notifier = TelegramNotifier.from_env()
    if notifier.enabled:
        logger.info("텔레그램 알림 활성화 (chat_id=%s)", notifier.chat_id)
    else:
        logger.info("텔레그램 알림 비활성 (TELEGRAM_BOT_TOKEN/CHAT_ID 미설정)")

    # Exchange + Broker 생성 (수수료/슬리피지는 백테스트와 동일하게 config 값 사용)
    exec_cfg = params.get("execution", {})
    exchange = build_exchange(demo=demo)
    broker   = LiveBroker(
        exchange, dry_run=dry_run, notifier=notifier,
        commission_maker=exec_cfg.get("commission_maker", 0.0002),
        commission_taker=exec_cfg.get("commission_taker", 0.0005),
        slippage_bps=exec_cfg.get("default_slippage_bps", 5.0),
        demo=demo,  # testnet=SL -4120 정상(sl_poller 대체)/메인넷=SL 재시도+경보 구분
    )

    # 잔고 확인 (엔진 초기 자본으로 사용)
    usdt = None
    try:
        balance = exchange.fetch_balance()
        usdt    = float(balance.get("USDT", {}).get("total", 0) or 0)
        logger.info("잔고: USDT %.2f (total, 엔진 초기 자본으로 사용)", usdt)
    except Exception as e:
        logger.error("잔고 조회 실패: %s", e)
        if not dry_run:
            sys.exit(1)

    # 엔진 생성 — 실제 잔고를 초기 자본으로 주입 (백테스트 값 무시)
    engine = build_engine(params, broker, notifier=notifier, initial_capital=usdt)
    broker.equity_provider = lambda: engine.tracker.snapshot().equity

    # state 복원 (이전 크래시 시 포지션/equity 유지)
    saved = state_store.load()
    if saved is not None and saved.positions:
        # 거래소에 실제로 열린 포지션과 교차 검증
        exchange_syms = broker.fetch_open_symbols()
        restored = 0
        skipped_unreal = 0.0
        for sym, pos in list(saved.positions.items()):
            if sym in exchange_syms:
                engine.tracker.state.positions[sym] = pos
                restored += 1
            else:
                logger.warning("state 복원 skip: %s (거래소에 포지션 없음)", sym)
                skipped_unreal += pos.unrealized_pnl
        if restored:
            # cash = total_balance - unrealized_pnl (이중 계산 방지)
            # 거래소 실시간 unrealized PnL 조회 (stale state 방지)
            if usdt is not None and usdt > 0:
                live_unrealized = 0.0
                try:
                    for pos_data in exchange.fetch_positions():
                        if float(pos_data.get("contracts") or 0) != 0:
                            live_unrealized += float(pos_data.get("unrealizedPnl") or 0)
                except Exception:
                    live_unrealized = sum(
                        p.unrealized_pnl for p in engine.tracker.state.positions.values()
                    )
                engine.tracker.state.cash = usdt - live_unrealized
            else:
                engine.tracker.state.cash = saved.cash
            engine.tracker.state.equity = usdt if usdt else saved.equity
            engine.tracker.state.daily_start_equity = saved.daily_start_equity
            # 같은 날 재시작 시 reset_daily() 건너뛰도록 _last_day 설정
            engine._last_day = pd.Timestamp.now(tz="UTC")
            # cash가 실잔고로 재앵커링됨(실잔고엔 이미 정산 펀딩 반영) → 현재 펀딩 버킷을
            # '정산됨'으로 표시해 첫 봉에서 같은 버킷 펀딩 중복부과 방지
            engine.funding_sim.sync_to(pd.Timestamp.now(tz="UTC"))
            logger.info("state 복원 완료: %d 포지션, cash=%.2f (거래소 잔고), daily_start=%.2f",
                        restored, engine.tracker.state.cash, saved.daily_start_equity)
        else:
            logger.info("복원할 포지션 없음 (거래소와 불일치) → 신규 시작")
    else:
        logger.info("저장된 state 없음 → 신규 시작")

    # CB 연속손절/정지·TP 쿨다운 복원 (포지션 유무 무관 — flat이어도 STOP/PAUSE 유지).
    # systemd Restart=always 환경에서 재기동마다 손실 방어 가드가 0으로 리셋되는 것을 방지.
    state_store.restore_runtime(engine)

    logger.info("엔진 초기화 완료 (전략 %d개)", len(engine.strategies))

    # LiveFeed 생성
    symbols    = params["symbols"]
    # 백테(run_backtest)는 p["timeframes"]를 필수 키로 사용 → 라이브도 동일하게 필수.
    # 기본값을 두면 config 누락 시 백테와 다른 TF로 조용히 동작(패리티 깨짐).
    timeframes = params["timeframes"]
    feed = LiveFeed(
        symbols=symbols,
        timeframes=timeframes,
        primary_tf=params.get("primary_timeframe", "1h"),
        lookback=params.get("data", {}).get("lookback_bars", 300),
        demo=demo,
        notifier=notifier,
    )

    logger.info("=" * 60)
    logger.info("라이브 트레이딩 시작")
    logger.info("심볼: %s", symbols)
    logger.info("모드: %s%s", "DEMO" if demo else "LIVE", " | DRY-RUN" if dry_run else "")
    logger.info("=" * 60)

    # 시작 알림
    if notifier and notifier.enabled:
        mode = "DEMO" if demo else "LIVE"
        if dry_run:
            mode += " | DRY-RUN"
        strats = [s.name for s in engine.strategies]
        balance_str = f"${usdt:,.2f}" if usdt is not None else "N/A"
        notifier.notify_info(
            f"🚀 <b>봇 시작</b>\n"
            f"모드: {mode}\n"
            f"심볼: {', '.join(symbols)}\n"
            f"전략: {', '.join(strats)}\n"
            f"잔고: {balance_str}"
        )

    # ── SL 폴러 (testnet 전용, 5분 간격) + engine 접근 직렬화용 lock ──
    # 실계정(Live)은 STOP_MARKET이 거래소 측에서 동작하므로 폴링 불필요.
    # Testnet은 STOP_MARKET 불가(-4120) → 폴러 필수.
    engine_lock = threading.Lock()
    sl_poller = None
    if args.sl_poll_sec is None:
        poll_sec = 300 if demo else 0  # demo 기본 5분, live 기본 비활성
    else:
        poll_sec = args.sl_poll_sec
    if poll_sec > 0 and not dry_run:
        sl_poller = SLPoller(
            engine=engine, broker=broker, exchange=exchange,
            interval_sec=poll_sec, tf="5m", lock=engine_lock,
        )
        sl_poller.start()
        logger.info("SL poller: %ds 간격 (testnet STOP_MARKET 대체)", poll_sec)
    elif not demo:
        logger.info("SL poller: 비활성 (실계정은 STOP_MARKET을 거래소가 처리)")

    # ── 실행 루프 ──
    bar_count = 0
    try:
        if args.snap_now:
            # 즉시 한 번만 실행 (연결 테스트)
            logger.info("[snap-now] 즉시 스냅샷 실행...")
            snap = feed.snapshot_now()
            with engine_lock:
                engine._process_bar(snap)
                state = engine.tracker.snapshot()
            logger.info(
                "snap-now 완료 | 자산 %.2f | 포지션 %d개",
                state.equity, state.open_position_count,
            )
        else:
            # 무한 루프: 매 1h 봉마다 실행
            for snapshot in feed.stream():
                bar_count += 1
                logger.info("─── 봉 #%d | %s ───", bar_count, snapshot.timestamp)

                # stale 데이터 방어: snapshot이 2봉 이상 지연이면 이 봉 전체를 건너뜀
                # (진입·청산·MTM 모두 미처리 — stale 가격에 행동하지 않음). 메인넷은
                # 거래소 STOP_MARKET이 SL을 커버하고, 다음 정상 봉의 거래소 sync가
                # 그 사이 체결을 tracker/CB에 정산함.
                staleness = (pd.Timestamp.now(tz="UTC") - snapshot.timestamp).total_seconds()
                if staleness > 7200:  # 2시간 이상 지연
                    logger.warning("stale 데이터 감지 (%.0f초 지연) — 이 봉 건너뜀", staleness)
                    continue

                with engine_lock:
                    engine._process_bar(snapshot)
                    state = engine.tracker.snapshot()

                logger.info(
                    "자산: %.2f USDT | 포지션: %d개 | 일일DD: %.2f%%",
                    state.equity,
                    state.open_position_count,
                    state.daily_pnl_pct * 100,
                )

                # state 디스크 저장 (매 봉 — 크래시 복구용)
                # engine 전달 → CB 연속손절/정지·TP 쿨다운도 함께 영속화
                state_store.save(state, engine=engine)

                # 매 봉 텔레그램 알림 (포지션 있을 때만)
                if notifier and notifier.enabled and state.open_position_count > 0:
                    pos_lines = []
                    for sym, pos in state.positions.items():
                        pos_lines.append(
                            f"  {sym} {pos.direction}  진입 {pos.entry_price:,.4f} | "
                            f"PnL ${pos.unrealized_pnl:+,.2f} x{pos.leverage}"
                        )
                    notifier.notify_info(
                        f"#{bar_count} | {snapshot.timestamp.strftime('%m-%d %H:%M')} UTC\n"
                        f"잔고: ${state.equity:,.2f} | DD: {state.daily_pnl_pct*100:+.1f}%\n"
                        f"포지션: {state.open_position_count}개\n" + "\n".join(pos_lines)
                    )

                # 잔고 대조 (10봉마다) — 예측 장부 vs 실제 잔고 괴리 측정·원인분해·경보·안전동기화
                if bar_count % 10 == 0:
                    try:
                        bal = exchange.fetch_balance()
                        real_usdt = float(bal.get("USDT", {}).get("total", 0) or 0)
                        expected = state.equity
                        drift_abs = real_usdt - expected          # +면 예측 과소, -면 슬리피지 손실
                        drift_pct = drift_abs / expected * 100 if expected > 0 else 0.0

                        # 원인 분해 — ledger 누적 비용 (전부 실현치)
                        trcs = engine.ledger.records
                        cum_comm = sum(t.commission for t in trcs)
                        cum_slip = sum(t.slippage_cost for t in trcs)
                        cum_fund = sum(t.funding_paid for t in trcs)
                        logger.info(
                            "괴리: 실제=%.2f 예측=%.2f (%+.2f%%, %+.2f USDT) | "
                            "누적 수수료 %.2f 슬리피지 %.2f 펀딩 %.2f",
                            real_usdt, expected, drift_pct, drift_abs,
                            cum_comm, cum_slip, cum_fund,
                        )

                        # 2단계 경보 (텔레그램): 경고 ±2% / 위험 ±5%
                        if notifier and notifier.enabled and abs(drift_pct) >= 2:
                            level = "🚨 위험" if abs(drift_pct) >= 5 else "⚠️ 경고"
                            notifier.notify_info(
                                f"{level} 잔고 괴리 {drift_pct:+.2f}% ({drift_abs:+.2f} USDT)\n"
                                f"실제: ${real_usdt:,.2f} / 예측: ${expected:,.2f}\n"
                                f"누적 수수료 ${cum_comm:,.2f} · 슬리피지 ${cum_slip:,.2f} · 펀딩 ${cum_fund:,.2f}"
                            )

                        # 안전 동기화 — 포지션 0일 때만 (unrealized 오염 방지)
                        # daily_start_equity는 건드리지 않음 — 덮어쓰면 그날 누적 손실이
                        # 0으로 리셋돼 일일 DD 가드(-4%/-10%)가 느슨해짐. 자정 reset_daily가 관리.
                        if state.open_position_count == 0 and abs(drift_pct) > 0.5 and real_usdt > 0:
                            engine.tracker.state.cash = real_usdt
                            engine.tracker.state.equity = real_usdt
                            logger.info(
                                "포지션 0 — tracker를 실제 잔고로 동기화: %.2f → %.2f (델타 %+.2f 기록됨)",
                                expected, real_usdt, drift_abs,
                            )
                    except Exception as e:
                        logger.warning("잔고 대조 실패: %s", e)

                    trades = engine.ledger.records
                    closed = len(trades)
                    if closed:
                        wins = sum(1 for t in trades if t.pnl > 0)
                        total_pnl = sum(t.pnl for t in trades)
                        logger.info(
                            "누적 거래: %d건 | 승률: %.0f%% | PnL: %.2f USDT",
                            closed, 100 * wins / closed, total_pnl,
                        )

                # heartbeat (12봉 = 12시간마다)
                if bar_count % 12 == 0 and notifier and notifier.enabled:
                    trades = engine.ledger.records
                    n = len(trades)
                    if n > 0:
                        wins = sum(1 for t in trades if t.pnl > 0)
                        wr_pct = wins / n * 100
                        gross_win  = sum(t.pnl for t in trades if t.pnl > 0)
                        gross_loss = abs(sum(t.pnl for t in trades if t.pnl <= 0))
                        pf = gross_win / gross_loss if gross_loss else float("inf")
                        cum_pnl = sum(t.pnl for t in trades)
                    else:
                        wr_pct, pf, cum_pnl = 0.0, 0.0, 0.0
                    try:
                        notifier.notify_heartbeat(
                            interval_h=12,
                            bar_count=bar_count,
                            equity=state.equity,
                            initial_capital=usdt or state.equity,
                            positions=state.open_position_count,
                            trades=n,
                            wr_pct=wr_pct,
                            pf=pf,
                            cum_pnl=cum_pnl,
                        )
                    except Exception:
                        pass

    except KeyboardInterrupt:
        shutdown_reason = "사용자 중단 (Ctrl+C)"
        shutdown_emoji = "⏹"
    except Exception as e:
        import traceback
        tb = traceback.format_exc()[-500:]
        shutdown_reason = f"에러: {type(e).__name__}: {str(e)[:200]}"
        shutdown_emoji = "🔥"
        logger.exception("예상치 못한 오류: %s", e)
    else:
        shutdown_reason = "정상 종료"
        shutdown_emoji = "⏹"
    finally:
        if sl_poller is not None:
            sl_poller.stop()
        # 종료 전 state 저장 (CB/쿨다운 포함)
        state_store.save(engine.tracker.snapshot(), engine=engine)
        logger.info("종료 전 state 저장 완료")
        # 최종 성과 출력
        report = MetricsReport.from_run(engine.equity_curve, engine.ledger)
        state = engine.tracker.snapshot()
        trades = engine.ledger.records
        logger.info("=" * 60)
        logger.info("종료 사유: %s", shutdown_reason)
        logger.info("최종 결과 | 총수익 %.1f%% | Sharpe %.3f | MDD %.1f%%",
                    report.total_return_pct, report.sharpe or 0, report.max_drawdown or 0)
        logger.info("=" * 60)
        # 텔레그램 종료 알림 (모든 종료 사유)
        if notifier and notifier.enabled:
            msg = (
                f"{shutdown_emoji} <b>봇 종료</b>\n"
                f"사유: {shutdown_reason}\n"
                f"잔고: ${state.equity:,.2f}\n"
                f"포지션: {state.open_position_count}개\n"
                f"누적 거래: {len(trades)}건\n"
                f"총수익: {report.total_return_pct:+.1f}%"
            )
            if shutdown_emoji == "🔥":
                msg += f"\n<pre>{tb}</pre>"
            try:
                notifier.notify_info(msg)
                time.sleep(1)  # 텔레그램 전송 완료 대기
            except Exception:
                logger.warning("종료 알림 전송 실패")


if __name__ == "__main__":
    main()
