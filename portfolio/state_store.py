"""Tracker 상태 디스크 저장/복원.

매 봉 처리 후 save(), 시작 시 load()로 크래시 복구.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from risk.models import PortfolioState, Position

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/state.json")


def save(state: PortfolioState, path: Path = DEFAULT_PATH, engine=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = {}
    for sym, pos in state.positions.items():
        positions[sym] = {
            "symbol": pos.symbol,
            "strategy": pos.strategy,
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "size_usd": pos.size_usd,
            "leverage": pos.leverage,
            "tp_price": pos.tp_price,
            "sl_price": pos.sl_price,
            "opened_at": pos.opened_at.isoformat(),
            "unrealized_pnl": pos.unrealized_pnl,
            "funding_paid": pos.funding_paid,
            "peak_price": pos.peak_price,
            "initial_sl_price": pos.initial_sl_price,
            "entry_commission": pos.entry_commission,
            "entry_slippage": pos.entry_slippage,
            "confluence_score": pos.confluence_score,
            "adds_done": pos.adds_done,        # 피라미딩 증액 횟수 (재기동 시 max_adds 보존)
            "order_price": pos.order_price,    # 진입 의도가 (청산알림 슬리피지% 계산용)
        }
    data = {
        "equity": state.equity,
        "cash": state.cash,
        "daily_start_equity": state.daily_start_equity,
        "positions": positions,
        "saved_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    # 엔진 런타임 상태(서킷브레이커 연속손절/정지, TP 쿨다운) — 재기동 시 손실 방어 가드 유지.
    # 포지션과 달리 인메모리라 미저장 시 매 재기동마다 0으로 리셋됨.
    if engine is not None:
        cb = getattr(engine, "circuit_breaker", None)
        if cb is not None and hasattr(cb, "to_state"):
            data["circuit_breaker"] = cb.to_state()
        guards = getattr(engine, "guards", None)
        if guards is not None and hasattr(guards, "to_state"):
            data["guards"] = guards.to_state()
        # 딥플로어 피크 — 재기동 시 피크가 현재 잔고로 리셋되면 플로어가 느슨해짐
        data["peak_equity"] = getattr(engine, "_peak_equity", None)
        # 사이징 풀 — 풀별 가상 cash + 마지막 리밸 월 (재기동 시 풀 스케일/리밸 시점 보존)
        if state.pool_cash is not None:
            data["pool_cash"] = state.pool_cash
        lrm = getattr(engine, "_last_rebal_month", None)
        if lrm is not None:
            data["last_rebal_month"] = list(lrm)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)
    logger.debug("state 저장: equity=%.2f, positions=%d", state.equity, len(positions))


def load(path: Path = DEFAULT_PATH) -> PortfolioState | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        logger.error("state 로드 실패: %s", e)
        return None

    positions = {}
    for sym, p in data.get("positions", {}).items():
        positions[sym] = Position(
            symbol=p["symbol"],
            strategy=p["strategy"],
            direction=p["direction"],
            entry_price=p["entry_price"],
            size_usd=p["size_usd"],
            leverage=p["leverage"],
            tp_price=p["tp_price"],
            sl_price=p["sl_price"],
            opened_at=pd.Timestamp(p["opened_at"]),
            unrealized_pnl=p.get("unrealized_pnl", 0.0),
            funding_paid=p.get("funding_paid", 0.0),
            peak_price=p.get("peak_price", 0.0),
            initial_sl_price=p.get("initial_sl_price", 0.0),
            entry_commission=p.get("entry_commission", 0.0),
            entry_slippage=p.get("entry_slippage", 0.0),
            confluence_score=p.get("confluence_score", 0),
            adds_done=p.get("adds_done", 0),
            order_price=p.get("order_price", 0.0),
        )

    state = PortfolioState(
        equity=data["equity"],
        cash=data["cash"],
        daily_start_equity=data.get("daily_start_equity", data["equity"]),
        positions=positions,
    )
    state.pool_cash = data.get("pool_cash")
    saved_at = data.get("saved_at", "?")
    logger.info("state 복원: equity=%.2f, positions=%d, saved=%s",
                state.equity, len(positions), saved_at)
    return state


def restore_runtime(engine, path: Path = DEFAULT_PATH) -> None:
    """CircuitBreaker / RiskGuards 런타임 상태 복원 (포지션 유무와 무관).

    포지션이 0(flat)이어도 STOP/PAUSE·연속손절·TP 쿨다운은 유지돼야 하므로
    save()의 positions 복원과 별개로 항상 호출한다. 저장 파일이 없거나 해당 키가
    없으면 no-op (신규 시작·구버전 state.json 호환).
    """
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        logger.error("runtime state 로드 실패: %s", e)
        return
    cb = getattr(engine, "circuit_breaker", None)
    cb_data = data.get("circuit_breaker")
    if cb_data and cb is not None and hasattr(cb, "load_state"):
        cb.load_state(cb_data)
        logger.info(
            "CircuitBreaker 복원: global_losses=%d, strategy_losses=%s",
            cb._global_losses, dict(cb._strategy_losses),
        )
    guards = getattr(engine, "guards", None)
    g_data = data.get("guards")
    if g_data and guards is not None and hasattr(guards, "load_state"):
        guards.load_state(g_data)
        logger.info("TP 쿨다운 복원: %d건", len(guards._last_tp_times))
    # 딥플로어 피크 복원 — max()로 재기동이 피크를 낮추지 못하게 보장
    pk = data.get("peak_equity")
    if pk is not None and hasattr(engine, "_peak_equity"):
        engine._peak_equity = max(engine._peak_equity, float(pk))
        logger.info("딥플로어 피크 복원: %.2f", engine._peak_equity)
    # 사이징 풀 마지막 리밸 월 복원 (재기동이 월간 리밸를 중복/누락시키지 않게)
    lrm = data.get("last_rebal_month")
    if lrm and hasattr(engine, "_last_rebal_month"):
        engine._last_rebal_month = (int(lrm[0]), int(lrm[1]))
        logger.info("사이징 풀 리밸 월 복원: %s", engine._last_rebal_month)
