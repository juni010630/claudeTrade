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


def save(state: PortfolioState, path: Path = DEFAULT_PATH) -> None:
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
        }
    data = {
        "equity": state.equity,
        "cash": state.cash,
        "daily_start_equity": state.daily_start_equity,
        "positions": positions,
        "saved_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
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
        )

    state = PortfolioState(
        equity=data["equity"],
        cash=data["cash"],
        daily_start_equity=data.get("daily_start_equity", data["equity"]),
        positions=positions,
    )
    saved_at = data.get("saved_at", "?")
    logger.info("state 복원: equity=%.2f, positions=%d, saved=%s",
                state.equity, len(positions), saved_at)
    return state
