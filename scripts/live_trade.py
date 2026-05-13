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
DEFAULT_PARAMS = "config/final_v13_eth.yaml"


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

    strategy_map = {
        "ema_cross":          EMACrossStrategy,
        "multi_tf_breakout":  MultiTFBreakoutStrategy,
    }
    strategies = []
    for key, cls in strategy_map.items():
        cfg = p.get("strategies", {}).get(key, {})
        if cfg.get("enabled", True):
            cfg["symbols"] = symbols
            strategies.append(cls(cfg))

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
            max_notional_usd=r.get("max_notional_usd"),
        ),
        broker=broker,
        funding_simulator=FundingRateSimulator(
            interval_hours=e.get("funding_interval_hours", 8)
        ),
        max_hold_hours=p.get("engine", {}).get("max_hold_hours"),
        breakeven_trigger_r=p.get("engine", {}).get("breakeven_trigger_r"),
        trailing_r_mult=p.get("engine", {}).get("trailing_r_mult"),
        strategy_min_score=p.get("strategy_min_score"),
        strategy_block_hours=p.get("strategy_block_hours"),
        strategy_block_symbols=p.get("strategy_block_symbols"),
        tier_block_symbols=p.get("tier_block_symbols"),
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

    # Exchange + Broker 생성
    exchange = build_exchange(demo=demo)
    broker   = LiveBroker(exchange, dry_run=dry_run, notifier=notifier)

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
            logger.info("state 복원 완료: %d 포지션, cash=%.2f (거래소 잔고), daily_start=%.2f",
                        restored, engine.tracker.state.cash, saved.daily_start_equity)
        else:
            logger.info("복원할 포지션 없음 (거래소와 불일치) → 신규 시작")
    else:
        logger.info("저장된 state 없음 → 신규 시작")

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
        notifier.notify_info(
            f"🚀 <b>봇 시작</b>\n"
            f"모드: {mode}\n"
            f"심볼: {', '.join(symbols)}\n"
            f"전략: {', '.join(strats)}\n"
            f"잔고: ${usdt:,.2f}"
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

                # stale 데이터 방어: snapshot이 2봉 이상 지연이면 시그널 무시 (청산만 처리)
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
                state_store.save(state)

                # 매 봉 텔레그램 알림
                if notifier and notifier.enabled:
                    pos_info = ""
                    for sym, pos in state.positions.items():
                        pnl_pct = pos.unrealized_pnl / state.equity * 100 if state.equity > 0 else 0
                        pos_info += f"\n  {sym} {pos.direction} {pnl_pct:+.1f}%"
                    notifier.notify_info(
                        f"#{bar_count} | {snapshot.timestamp.strftime('%m-%d %H:%M')} UTC\n"
                        f"잔고: ${state.equity:,.2f} | DD: {state.daily_pnl_pct*100:+.1f}%\n"
                        f"포지션: {state.open_position_count}개{pos_info}"
                    )

                # 잔고 대조 (10봉마다)
                if bar_count % 10 == 0:
                    try:
                        bal = exchange.fetch_balance()
                        real_usdt = float(bal.get("USDT", {}).get("total", 0) or 0)
                        drift_pct = abs(real_usdt - state.equity) / state.equity * 100 if state.equity > 0 else 0
                        logger.info("잔고 대조: 거래소=%.2f tracker=%.2f (괴리 %.1f%%)", real_usdt, state.equity, drift_pct)
                        if drift_pct > 5:
                            msg = f"⚠️ 잔고 괴리 {drift_pct:.1f}%\n거래소: {real_usdt:.2f}\ntracker: {state.equity:.2f}"
                            logger.warning(msg)
                            if notifier and notifier.enabled:
                                notifier.notify_info(msg)
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

                # heartbeat (24봉 = 24시간마다 텔레그램 알림)
                if bar_count % 24 == 0 and notifier and notifier.enabled:
                    trades = engine.ledger.records
                    hb = (f"💓 Heartbeat #{bar_count // 24}d\n"
                          f"자산: {state.equity:.2f} USDT\n"
                          f"포지션: {state.open_position_count}개\n"
                          f"누적 거래: {len(trades)}건")
                    try:
                        notifier.notify_info(hb)
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
        # 종료 전 state 저장
        state_store.save(engine.tracker.snapshot())
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
