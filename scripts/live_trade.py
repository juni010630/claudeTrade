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
import fcntl
import json
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
from execution.live_broker import LiveBroker
from execution.notifier import TelegramNotifier
from execution.sl_poller import SLPoller
from metrics.report import MetricsReport

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
DEFAULT_PARAMS = "config/final_v21d_eexit.yaml"

# ── 딥플로어 정지 플래그 ──────────────────────────────────────────
# 발동 시 생성 → systemd Restart=always가 재기동해도 이 파일이 있으면 거래 재개 안 함.
# 해제(수동): rm data/deep_floor_halt.json && sudo systemctl restart trade-bot
DEEP_FLOOR_HALT = Path("data/deep_floor_halt.json")
DRY_STATE_PATH = Path("data/state_dryrun.json")


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
def build_engine(
    p: dict,
    broker: LiveBroker,
    notifier: TelegramNotifier | None = None,
    initial_capital: float | None = None,
) -> BacktestEngine:
    """백테스트와 동일한 단일 빌더를 사용해 설정 배선 차이를 원천 차단한다."""
    from scripts.run_backtest import build_engine as build_common_engine

    capital = (
        initial_capital
        if initial_capital is not None
        else p.get("backtest", {}).get("initial_capital", 10_000)
    )
    return build_common_engine(
        p,
        initial_capital=capital,
        broker_override=broker,
        notifier=notifier,
        trade_log_path="trades_dryrun.csv" if broker.dry_run else "trades.csv",
    )


def acquire_instance_lock(dry_run: bool):
    """동일 모드 프로세스가 둘 이상 주문하지 못하게 비차단 잠금을 유지한다."""
    lock_path = Path("data/live_trade.dryrun.lock" if dry_run else "data/live_trade.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as e:
        handle.close()
        raise RuntimeError(f"이미 실행 중인 동일 모드 봇이 있습니다: {lock_path}") from e
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def fetch_live_positions_or_raise(
    broker: LiveBroker, retries: int = 4, wait_sec: float = 5.0
) -> dict[str, dict]:
    """실포지션 조회 실패를 flat으로 오판하지 않는 기동 가드."""
    for attempt in range(retries):
        positions = broker.fetch_open_positions()
        if positions is not None:
            return positions
        if attempt < retries - 1:
            time.sleep(wait_sec)
    raise RuntimeError("거래소 포지션 조회 반복 실패")


def reconcile_saved_positions(saved, live_positions: dict[str, dict]) -> dict:
    """저장 state와 거래소의 심볼·방향·수량·평단가를 교차 검증한다."""
    saved_positions = saved.positions if saved is not None else {}
    orphans = set(live_positions) - set(saved_positions)
    if orphans:
        raise RuntimeError(f"저장 state에 없는 거래소 포지션: {', '.join(sorted(orphans))}")

    restored = {}
    for symbol, pos in saved_positions.items():
        actual = live_positions.get(symbol)
        if actual is None:
            logger.warning("state 복원 skip: %s (거래소에 포지션 없음)", symbol)
            continue
        expected_qty = float(pos.size_usd) / float(pos.entry_price)
        actual_qty = float(actual["contracts"])
        actual_entry = float(actual["entry_price"])
        qty_error = abs(actual_qty - expected_qty) / max(expected_qty, 1e-12)
        entry_error = abs(actual_entry - float(pos.entry_price)) / max(float(pos.entry_price), 1e-12)
        mismatches = []
        if actual["direction"] != pos.direction:
            mismatches.append(f"방향 state={pos.direction} exchange={actual['direction']}")
        if qty_error > 0.005:
            mismatches.append(
                f"수량 state={expected_qty:.12g} exchange={actual_qty:.12g} ({qty_error:.2%})"
            )
        if actual_entry <= 0 or entry_error > 0.005:
            mismatches.append(
                f"평단 state={pos.entry_price:.12g} exchange={actual_entry:.12g} ({entry_error:.2%})"
            )
        if mismatches:
            raise RuntimeError(f"{symbol} 포지션 불일치: " + "; ".join(mismatches))
        restored[symbol] = pos
    return restored


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
        parser.error(f"파라미터 파일이 없습니다: {params_path}")

    with open(params_path) as f:
        params = yaml.safe_load(f)
    logger.info("파라미터 로드: %s", params_path)

    demo    = not args.no_demo
    dry_run = args.dry_run
    instance_lock = acquire_instance_lock(dry_run)
    runtime_state_path = DRY_STATE_PATH if dry_run else state_store.DEFAULT_PATH

    # Telegram notifier
    notifier = TelegramNotifier.from_env()
    if notifier.enabled:
        logger.info("텔레그램 알림 활성화 (chat_id=%s)", notifier.chat_id)
    else:
        logger.info("텔레그램 알림 비활성 (TELEGRAM_BOT_TOKEN/CHAT_ID 미설정)")

    # 딥플로어 정지 상태 — 거래 없이 대기 (systemd 재기동 루프 방지용으로 exit 대신 sleep)
    if DEEP_FLOOR_HALT.exists():
        try:
            halt_info = json.loads(DEEP_FLOOR_HALT.read_text())
        except Exception:
            halt_info = {}
        logger.error("⛔ 딥플로어 정지 상태 (%s) — 거래하지 않음. 해제: rm %s 후 재시작",
                     halt_info.get("triggered_at", "?"), DEEP_FLOOR_HALT)
        if notifier and notifier.enabled:
            notifier.notify_info(
                f"⛔ <b>딥플로어 정지 상태</b> — 거래 재개 안 함\n"
                f"발동: {halt_info.get('triggered_at', '?')} | "
                f"equity ${halt_info.get('equity', 0):,.2f} / peak ${halt_info.get('peak', 0):,.2f}\n"
                f"해제: <code>rm {DEEP_FLOOR_HALT}</code> 후 재시작"
            )
        while True:
            time.sleep(21600)  # 6시간마다 리마인더
            if notifier and notifier.enabled:
                notifier.notify_info("⛔ 딥플로어 정지 유지 중 (거래 없음)")

    # Exchange + Broker 생성 (수수료/슬리피지는 백테스트와 동일하게 config 값 사용)
    exec_cfg = params.get("execution", {})
    exchange = build_exchange(demo=demo)
    mk_cfg = exec_cfg.get("maker_entry", {}) or {}
    broker   = LiveBroker(
        exchange, dry_run=dry_run, notifier=notifier,
        commission_maker=exec_cfg.get("commission_maker", 0.0002),
        commission_taker=exec_cfg.get("commission_taker", 0.0005),
        slippage_bps=exec_cfg.get("default_slippage_bps", 5.0),
        demo=demo,  # testnet=SL -4120 정상(sl_poller 대체)/메인넷=SL 재시도+경보 구분
        maker_timeout_sec=(float(mk_cfg.get("timeout_sec", 300)) if mk_cfg.get("enabled") else 0.0),
        maker_poll_sec=float(mk_cfg.get("poll_sec", 3)),
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

    # 시계오차 가드: 로컬 시계가 거래소보다 >30s 앞서면 forming(미마감) 봉을
    # 완성봉으로 오인할 수 있음(live_feed forming-bar drop이 로컬 now 기준 → 약한
    # look-ahead). 시작 시 1회 점검만; NTP 동기 서버면 통과.
    try:
        _srv_ms = exchange.fetch_time()
        _skew = (time.time() * 1000) - float(_srv_ms)
        if abs(_skew) > 30_000:
            msg = f"⚠️ 시계오차 {_skew/1000:+.1f}s (로컬 vs 거래소) — NTP 동기 필요. forming 봉 오인 위험."
            logger.critical(msg)
            if notifier and notifier.enabled:
                try:
                    notifier.notify_info(msg)
                except Exception:
                    pass
            if not dry_run:
                raise RuntimeError(msg)
        else:
            logger.info("시계오차 점검 OK (%.1fs)", _skew / 1000)
    except Exception as e:
        if not dry_run:
            raise RuntimeError(f"시계오차 점검 실패: {e}") from e
        logger.warning("[DRY] 시계오차 점검 실패: %s", e)

    # 엔진 생성 — 실제 잔고를 초기 자본으로 주입 (백테스트 값 무시)
    engine = build_engine(params, broker, notifier=notifier, initial_capital=usdt)
    broker.equity_provider = lambda: engine.tracker.snapshot().equity

    # state 복원. 실계정은 저장 포지션과 거래소의 방향·수량·평단까지 일치해야 시작한다.
    saved = None if dry_run else state_store.load(runtime_state_path)
    if dry_run:
        logger.info("DRY-RUN: 운영 state를 읽지 않고 별도 상태 파일을 사용")
    else:
        try:
            live_positions = fetch_live_positions_or_raise(broker)
            restored_positions = reconcile_saved_positions(saved, live_positions)
        except RuntimeError as e:
            logger.critical("기동 포지션 검증 실패: %s", e)
            if notifier and notifier.enabled:
                try:
                    notifier.notify_info(
                        f"🚨 <b>기동 중단 — 포지션 검증 실패</b>\n{str(e)[:300]}\n"
                        "저장 state와 거래소를 확인한 뒤 재시작하세요."
                    )
                except Exception:
                    pass
            raise SystemExit(1) from e

        if restored_positions:
            for symbol, pos in restored_positions.items():
                engine.tracker.state.positions[symbol] = pos
            if usdt is not None and usdt > 0:
                live_unrealized = sum(
                    float(live_positions[symbol]["unrealized_pnl"])
                    for symbol in restored_positions
                )
                engine.tracker.state.cash = usdt - live_unrealized
            else:
                engine.tracker.state.cash = saved.cash
            engine.tracker.state.equity = usdt if usdt else saved.equity
            engine.tracker.state.daily_start_equity = saved.daily_start_equity
            try:
                saved_at = pd.Timestamp(
                    runtime_state_path.stat().st_mtime, unit="s", tz="UTC"
                )
            except Exception:
                saved_at = None
            if saved_at is not None and saved_at.date() == pd.Timestamp.now(tz="UTC").date():
                engine._last_day = pd.Timestamp.now(tz="UTC")
            engine.funding_sim.sync_to(pd.Timestamp.now(tz="UTC"))
            if engine.tracker._pool_fractions:
                engine.tracker.state.pool_cash = saved.pool_cash
            logger.info(
                "state 복원 완료: %d 포지션, cash=%.2f, daily_start=%.2f",
                len(restored_positions),
                engine.tracker.state.cash,
                saved.daily_start_equity,
            )
        elif saved is not None:
            logger.info("거래소 실포지션 없음 — 저장 state의 stale 포지션은 복원하지 않음")
        else:
            logger.info("저장된 state와 거래소 실포지션 없음 — 신규 시작")

    # CB 연속손절/정지·TP 쿨다운 복원 (포지션 유무 무관 — flat이어도 STOP/PAUSE 유지).
    # systemd Restart=always 환경에서 재기동마다 손실 방어 가드가 0으로 리셋되는 것을 방지.
    state_store.restore_runtime(engine, path=runtime_state_path)

    # 딥플로어 해제 후 거래 재개: 여기 도달 = halt 파일 없음(위 357행 통과). restore_runtime이
    # peak를 max()로 복원하므로, 딥플로어(-55%) 발동 후 운영자가 halt 파일만 지우고 재시작하면
    # 복원된 peak 대비 현재 잔고가 여전히 임계 이하 → 첫 봉에서 _aborted 재발동 → 영구 재정지된다.
    # flat(딥플로어가 전량청산)이고 복원 peak 대비 DD가 이미 deep_floor 임계 이하 = 해제 의사로
    # 보고 peak를 현재 잔고로 재앵커해 -55% 예산을 리셋한다. (flat 조건 = 청산 미완료 시 재앵커
    # 금지로 안전; 정상 가동 중 -55% 미달이면 조건 불성립이라 정상 재기동엔 영향 없음.)
    if engine._abort_mdd is not None and engine._peak_equity > 0 \
            and not engine.tracker.state.positions:
        _cur_eq = engine.tracker.snapshot().equity
        _dd = (_cur_eq - engine._peak_equity) / engine._peak_equity
        if _dd <= engine._abort_mdd:
            logger.error(
                "딥플로어 해제 감지 — peak 재앵커 %.2f → %.2f (DD %.1f%% ≤ 임계 %.1f%%, 거래 재개)",
                engine._peak_equity, _cur_eq, _dd * 100, engine._abort_mdd * 100,
            )
            engine._peak_equity = _cur_eq
            state_store.save(engine.tracker.snapshot(), path=runtime_state_path, engine=engine)
            if notifier and notifier.enabled:
                try:
                    notifier.notify_info(
                        f"♻️ <b>딥플로어 해제 — 거래 재개</b>\n"
                        f"peak 재앵커: 현재 잔고 ${_cur_eq:,.2f} 기준 "
                        f"-{abs(engine._abort_mdd) * 100:.0f}% 예산 리셋"
                    )
                except Exception:
                    pass

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
        notifier.notify_info(
            f"🚀 <b>봇 시작</b>\n"
            f"모드: {mode}\n"
            f"심볼: {', '.join(symbols)}\n"
            f"전략: {', '.join(strats)}"
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

    # ── FNG 레짐 틸트: 매 봉 갱신 점검 (일 1회 실 갱신). 봇 연속실행 시 startup
    #    스케줄이 stale되는 것 방지. enabled 아니면 완전 비활성(v18 무영향). ──
    _rt_cfg = params.get("regime_tilt", {})
    _fng_last_day = None
    _dv_cfg = params.get("dvol_scale", {})
    _dvol_last_day = None
    _dvp_cfg = params.get("dvol_perbook", {})
    _dvp_last_day = None

    # ── 실행 루프 ──
    bar_count = 0
    deep_floor_fired = False
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

                # FNG 레짐 틸트: UTC 일 경계마다 FNG 증분갱신 + 스케줄 재빌드
                # (1일 래그로 look-ahead 차단. 백테 빌더와 동일 함수 → 패리티)
                if _rt_cfg.get("enabled"):
                    _today = snapshot.timestamp.strftime("%Y-%m-%d")
                    if _today != _fng_last_day:
                        from regime.fng_tilt import build_fng_tilt_schedule, refresh_fng_csv
                        _fcsv = _rt_cfg.get("fng_csv", "data/regime/fng_daily.csv")
                        _ok = refresh_fng_csv(_fcsv)
                        with engine_lock:
                            engine.set_capital_fraction_schedule(build_fng_tilt_schedule(
                                base_fractions=params.get("strategy_capital_fraction") or {},
                                fng_csv=_fcsv, delta=_rt_cfg.get("delta", 0.10),
                                direction=_rt_cfg.get("direction", 1),
                                momentum_strategies=_rt_cfg.get("momentum_strategies", []),
                                meanrev_strategies=_rt_cfg.get("meanrev_strategies", []),
                                lag_days=_rt_cfg.get("lag_days", 1)))
                        _fng_last_day = _today
                        logger.info("FNG 스케줄 갱신 (%s, fetch=%s, 1일 래그)", _today, _ok)

                # DVOL 변동성타게팅: UTC 일 경계마다 DVOL 증분갱신 + 사이즈배수 재빌드 (1일 래그)
                if _dv_cfg.get("enabled"):
                    _today2 = snapshot.timestamp.strftime("%Y-%m-%d")
                    if _today2 != _dvol_last_day:
                        from regime.dvol_scale import build_dvol_schedule, refresh_dvol_parquet
                        _dpath = _dv_cfg.get("dvol_path", "data/regime/dvol_btc_full.parquet")
                        _ok2 = refresh_dvol_parquet(_dpath)
                        with engine_lock:
                            engine.set_size_scale_schedule(build_dvol_schedule(
                                dvol_path=_dpath, target=_dv_cfg.get("target", 45.0),
                                clip_lo=_dv_cfg.get("clip_lo", 0.3), clip_hi=_dv_cfg.get("clip_hi", 2.0),
                                lag_days=_dv_cfg.get("lag_days", 1)))
                        _dvol_last_day = _today2
                        logger.info("DVOL 스케줄 갱신 (%s, fetch=%s, 1일 래그)", _today2, _ok2)

                # DVOL per-book: 일 경계마다 DVOL 갱신 + 책별 capital_fraction_schedule 재빌드
                if _dvp_cfg.get("enabled"):
                    _today3 = snapshot.timestamp.strftime("%Y-%m-%d")
                    if _today3 != _dvp_last_day:
                        from regime.dvol_scale import build_dvol_perbook_schedule, refresh_dvol_parquet
                        _dpath3 = _dvp_cfg.get("dvol_path", "data/regime/dvol_btc_full.parquet")
                        _ok3 = refresh_dvol_parquet(_dpath3)
                        with engine_lock:
                            engine.set_capital_fraction_schedule(build_dvol_perbook_schedule(
                                base_fractions=params.get("strategy_capital_fraction") or {},
                                dvol_path=_dpath3, targets=_dvp_cfg.get("targets", {}),
                                clip_lo=_dvp_cfg.get("clip_lo", 0.3), clip_hi=_dvp_cfg.get("clip_hi", 2.0),
                                lag_days=_dvp_cfg.get("lag_days", 1)))
                        _dvp_last_day = _today3
                        logger.info("DVOL per-book 스케줄 갱신 (%s, fetch=%s, 1일 래그)", _today3, _ok3)

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

                # 딥플로어: running peak 대비 DD가 deep_floor_dd 초과 → 전량 청산 + 정지.
                # 플래그를 청산보다 먼저 기록 — 청산 중 크래시해도 재기동 시 거래 재개 안 함.
                if engine._aborted:
                    peak = engine._peak_equity
                    dd = (state.equity - peak) / peak * 100 if peak > 0 else 0.0
                    logger.error("⛔ 딥플로어 발동: equity %.2f / peak %.2f (DD %+.1f%%)",
                                 state.equity, peak, dd)
                    # 쓰기 실패(디스크 풀/권한)가 청산·정지를 막으면 안 됨 — 예외를
                    # 삼키고 청산+break는 반드시 진행(이 프로세스는 확실히 멈춤). 재기동
                    # 시 파일 부재로 거래 재개될 극단 케이스 방지용으로 3회 재시도.
                    _halt_payload = json.dumps({
                        "triggered_at": snapshot.timestamp.isoformat(),
                        "equity": state.equity, "peak": peak, "dd_pct": dd,
                    }, indent=2)
                    for _w in range(3):
                        try:
                            DEEP_FLOOR_HALT.write_text(_halt_payload)
                            break
                        except Exception as _we:
                            logger.critical("딥플로어 halt 파일 쓰기 실패 (%d/3): %s", _w + 1, _we)
                            if notifier and notifier.enabled and _w == 2:
                                try:
                                    notifier.notify_info(
                                        "🚨 딥플로어 halt 파일 쓰기 실패 — 청산은 진행하나 "
                                        "재기동 시 거래 재개 위험. 수동 확인 필요."
                                    )
                                except Exception:
                                    pass
                    with engine_lock:
                        prices = engine._get_prices(snapshot)
                        for sym in list(state.positions.keys()):
                            # 청산 실패 시 재시도 — market_close가 TP/SL 취소 후 청산이라
                            # 실패 시 SL 없는 나체 포지션이 폭락장에 방치됨. 3회 재시도 후
                            # 잔존분은 기존 텔레그램 경보(emergency_stop 안내)로 위임.
                            for _attempt in range(3):
                                engine._force_close(sym, prices.get(sym, 0.0),
                                                    snapshot.timestamp, "deep_floor")
                                if sym not in engine.tracker.snapshot().positions:
                                    break
                                time.sleep(2)
                        state = engine.tracker.snapshot()
                    state_store.save(state, path=runtime_state_path, engine=engine)
                    if notifier and notifier.enabled:
                        notifier.notify_info(
                            f"⛔ <b>딥플로어 발동 — 전량 청산·거래 정지</b>\n"
                            f"equity ${state.equity:,.2f} / peak ${peak:,.2f} (DD {dd:+.1f}%)\n"
                            f"남은 포지션: {state.open_position_count}개"
                            f"{' — emergency_stop.py로 정리 필요' if state.open_position_count else ''}\n"
                            f"재개: <code>rm {DEEP_FLOOR_HALT}</code> 후 재시작"
                        )
                    deep_floor_fired = True
                    break

                logger.info(
                    "자산: %.2f USDT | 포지션: %d개 | 일일DD: %.2f%%",
                    state.equity,
                    state.open_position_count,
                    state.daily_pnl_pct * 100,
                )

                # state 디스크 저장 (매 봉 — 크래시 복구용)
                # engine 전달 → CB 연속손절/정지·TP 쿨다운도 함께 영속화
                state_store.save(state, path=runtime_state_path, engine=engine)

                # 매 봉(매시간) 텔레그램 알림 — 항상 발송. 포지션 유무로 내용만 구분:
                #   포지션 0 → "잘 돌아감", 보유 시 → 심볼/진입가/현재 PnL
                if notifier and notifier.enabled:
                    header = f"#{bar_count} | {snapshot.timestamp.strftime('%m-%d %H:%M')} UTC"
                    if state.open_position_count > 0:
                        pos_lines = []
                        for sym, pos in state.positions.items():
                            _upnl_pct = (pos.unrealized_pnl / state.equity * 100
                                         if state.equity > 0 else 0.0)
                            pos_lines.append(
                                f"  {sym} {pos.direction} 진입 {pos.entry_price:,.4f} | "
                                f"PnL 자본 {_upnl_pct:+.2f}% x{pos.leverage}"
                            )
                        notifier.notify_info(
                            f"📊 {header}\n"
                            f"일중 DD: {state.daily_pnl_pct*100:+.1f}%\n"
                            f"포지션: {state.open_position_count}개\n" + "\n".join(pos_lines)
                        )
                    else:
                        notifier.notify_info(
                            f"✅ 잘 돌아감 | {header}\n"
                            f"포지션 0 | 일중 DD: {state.daily_pnl_pct*100:+.1f}%"
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

                        # BNB 수수료잔고 모니터 — feeBurn(BNB 결제 시 수수료 10% 할인) 사용 중.
                        # 소진되면 경고 없이 USDT 정가 결제로 복귀하므로 저잔고 경보가 필수.
                        try:
                            bnb_total = float(bal.get("BNB", {}).get("total", 0) or 0)
                            bnb_px = float(exchange.fetch_ticker("BNB/USDT:USDT")["last"])
                            bnb_usd = bnb_total * bnb_px
                            # 소진 추산: trades.csv 최근 30일 수수료(모델 추정치) × 0.9(BNB 결제분) → 일소비율
                            bnb_days = None
                            tl = Path("trades.csv")
                            if tl.exists():
                                _tdf = pd.read_csv(tl, usecols=["exit_time", "commission"])
                                _cut = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)
                                _exit_ts = pd.to_datetime(_tdf["exit_time"], utc=True, errors="coerce")
                                _burn_day = float(_tdf.loc[_exit_ts >= _cut, "commission"].sum()) * 0.9 / 30
                                if _burn_day > 0 and bnb_usd > 0:
                                    bnb_days = bnb_usd / _burn_day
                            logger.info(
                                "BNB 수수료잔고: %.4f BNB ($%.2f)%s",
                                bnb_total, bnb_usd,
                                f" | 소진 예상 ~{bnb_days:.0f}일" if bnb_days else "",
                            )
                            if notifier and notifier.enabled and 0 < bnb_usd < 30:
                                notifier.notify_info(
                                    f"⚠️ <b>BNB 수수료잔고 부족</b>: {bnb_total:.4f} BNB (${bnb_usd:.2f})\n"
                                    f"소진 시 할인 없이 USDT 정가 결제로 복귀 — 현물 매수 후 선물지갑 이체 필요"
                                )
                        except Exception as e:
                            logger.warning("BNB 잔고 모니터 실패: %s", e)
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
        if deep_floor_fired:
            shutdown_reason = "딥플로어 발동 (전량 청산·거래 정지)"
            shutdown_emoji = "⛔"
        else:
            shutdown_reason = "정상 종료"
            shutdown_emoji = "⏹"
    finally:
        if sl_poller is not None:
            sl_poller.stop()
        # 종료 전 state 저장 (CB/쿨다운 포함)
        state_store.save(
            engine.tracker.snapshot(), path=runtime_state_path, engine=engine
        )
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
