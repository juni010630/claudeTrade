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
import time
from pathlib import Path

import ccxt
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.live_feed import LiveFeed
from engine.backtest import BacktestEngine
from execution.commission import CommissionModel
from execution.funding import FundingRateSimulator
from execution.live_broker import LiveBroker
from execution.notifier import TelegramNotifier
from execution.slippage import SlippageModel
from metrics.report import MetricsReport
from regime.detector import RegimeDetector
from risk.circuit_breaker import CircuitBreaker
from risk.correlation import CorrelationFilter
from risk.guards import RiskGuards
from risk.position_sizer import PositionSizer
from signals.scorer import ConfluenceScorer
from strategies.donchian_breakout import DonchianBreakoutStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum_breakout import MomentumBreakoutStrategy
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
DEFAULT_PARAMS = "config/params_best.yaml"


# ── ccxt Exchange 생성 ────────────────────────────────────────────
def build_exchange(demo: bool) -> ccxt.Exchange:
    api_key = os.environ.get("BINANCE_API_KEY", "")
    secret  = os.environ.get("BINANCE_SECRET", "")

    if not api_key or not secret:
        raise ValueError(
            "BINANCE_API_KEY / BINANCE_SECRET 환경변수가 없습니다. "
            ".env 파일을 확인하세요."
        )

    exchange = ccxt.binanceusdm({
        "apiKey": api_key,
        "secret": secret,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
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

    strategies = []
    for key, cls in [
        ("momentum_breakout", MomentumBreakoutStrategy),
        ("multi_tf_breakout", MultiTFBreakoutStrategy),
        ("mean_reversion",    MeanReversionStrategy),
    ]:
        cfg = p.get("strategies", {}).get(key, {})
        if cfg.get("enabled", True):
            cfg["symbols"] = symbols
            strategies.append(cls(cfg))

    don_cfg = p.get("strategies", {}).get("donchian_breakout", {})
    if don_cfg.get("enabled", True):
        don_cfg["symbols"] = symbols
        strategies.append(DonchianBreakoutStrategy(don_cfg))

    cap = initial_capital if initial_capital is not None else p.get("backtest", {}).get("initial_capital", 10_000)
    return BacktestEngine(
        initial_capital=cap,
        strategies=strategies,
        regime_detector=RegimeDetector(
            primary_symbol=symbols[0],
            adx_period=rg.get("adx_period", 14),
            adx_trending_threshold=rg.get("adx_trending_threshold", 25.0),
            adx_ranging_threshold=rg.get("adx_ranging_threshold", 15.0),
            bb_period=rg.get("bb_period", 20),
            bb_std=rg.get("bb_std", 2.0),
            bb_width_lookback=rg.get("bb_width_lookback", 50),
            bb_width_squeeze_pct=rg.get("bb_width_squeeze_pct", 0.2),
            primary_tf=rg.get("primary_tf", "1h"),
        ),
        confluence_scorer=ConfluenceScorer(
            volume_ratio_threshold=sc.get("volume_ratio_threshold", 1.5),
            rsi_long_max=sc.get("rsi_long_max", 70.0),
            rsi_short_min=sc.get("rsi_short_min", 35.0),
            funding_long_max=sc.get("funding_long_max", 0.0003),
            funding_short_min=sc.get("funding_short_min", -0.0003),
            daily_ema_period=sc.get("daily_ema_period", 200),
            tier_ss_min_score=sc.get("tier_ss_min_score", 7),
            tier_s_min_score=sc.get("tier_s_min_score", 5),
            tier_a_min_score=sc.get("tier_a_min_score", 4),
            tier_b_min_score=sc.get("tier_b_min_score", 2),
            tier_c_min_score=sc.get("tier_c_min_score", 1),
        ),
        risk_guards=RiskGuards(
            max_positions=r.get("max_positions", 6),
            max_same_direction=r.get("max_same_direction", 4),
            daily_pause_threshold=r.get("daily_drawdown_pause", -0.04),
            daily_stop_threshold=r.get("daily_drawdown_stop", -0.12),
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
            risk_per_trade=r.get("risk_per_trade", 0.010),
            tier_config=p.get("leverage_tiers"),
        ),
        broker=broker,
        funding_simulator=FundingRateSimulator(
            interval_hours=e.get("funding_interval_hours", 8)
        ),
        notifier=notifier,
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
        logger.warning("%s 없음 → config/params_v2.yaml 사용", args.params)
        params_path = Path("config/params_v2.yaml")

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

    # Exchange + Broker 생성
    exchange = build_exchange(demo=demo)
    broker   = LiveBroker(exchange, dry_run=dry_run, notifier=notifier)

    # 잔고 확인 (엔진 초기 자본으로 사용)
    usdt = None
    try:
        balance = exchange.fetch_balance()
        usdt    = float(balance.get("USDT", {}).get("free", 0) or 0)
        logger.info("잔고: USDT %.2f (엔진 초기 자본으로 사용)", usdt)
    except Exception as e:
        logger.error("잔고 조회 실패: %s", e)
        if not dry_run:
            sys.exit(1)

    # 엔진 생성 — 실제 잔고를 초기 자본으로 주입 (백테스트 값 무시)
    engine = build_engine(params, broker, notifier=notifier, initial_capital=usdt)
    broker.equity_provider = lambda: engine.tracker.snapshot().equity
    logger.info("엔진 초기화 완료 (전략 %d개)", len(engine.strategies))

    # LiveFeed 생성
    symbols    = params["symbols"]
    timeframes = params.get("timeframes", ["1h", "4h", "1d"])
    feed = LiveFeed(
        symbols=symbols,
        timeframes=timeframes,
        primary_tf=params.get("primary_timeframe", "1h"),
        lookback=params.get("data", {}).get("lookback_bars", 300),
        demo=demo,
    )

    logger.info("=" * 60)
    logger.info("라이브 트레이딩 시작")
    logger.info("심볼: %s", symbols)
    logger.info("모드: %s%s", "DEMO" if demo else "LIVE", " | DRY-RUN" if dry_run else "")
    logger.info("=" * 60)

    # ── 실행 루프 ──
    bar_count = 0
    try:
        if args.snap_now:
            # 즉시 한 번만 실행 (연결 테스트)
            logger.info("[snap-now] 즉시 스냅샷 실행...")
            snap = feed.snapshot_now()
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

                engine._process_bar(snapshot)
                state = engine.tracker.snapshot()

                logger.info(
                    "자산: %.2f USDT | 포지션: %d개 | 일일DD: %.2f%%",
                    state.equity,
                    state.open_position_count,
                    (state.equity / state.daily_start_equity - 1) * 100,
                )

                # 거래 내역 주기적 출력 (10봉마다)
                if bar_count % 10 == 0:
                    trades = engine.ledger.records
                    closed = len(trades)
                    if closed:
                        wins = sum(1 for t in trades if t.pnl > 0)
                        total_pnl = sum(t.pnl for t in trades)
                        logger.info(
                            "누적 거래: %d건 | 승률: %.0f%% | PnL: %.2f USDT",
                            closed, 100 * wins / closed, total_pnl,
                        )

    except KeyboardInterrupt:
        logger.info("사용자 중단 (Ctrl+C)")
    except Exception as e:
        logger.exception("예상치 못한 오류: %s", e)
        raise
    finally:
        # 최종 성과 출력
        report = MetricsReport.from_run(engine.equity_curve, engine.ledger)
        logger.info("=" * 60)
        logger.info("최종 결과 | 총수익 %.1f%% | Sharpe %.3f | MDD %.1f%%",
                    report.total_return_pct, report.sharpe or 0, report.max_drawdown or 0)
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
