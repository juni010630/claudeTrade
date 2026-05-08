from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


@dataclass
class Position:
    symbol: str
    strategy: str
    direction: Literal["long", "short"]
    entry_price: float
    size_usd: float          # 명목 포지션 크기 (레버리지 포함)
    leverage: int
    tp_price: float
    sl_price: float
    opened_at: pd.Timestamp
    unrealized_pnl: float = 0.0
    funding_paid: float = 0.0
    # 동적 SL 관리용 (breakeven / trailing)
    peak_price: float = 0.0        # long: 진입 이후 max high, short: min low
    initial_sl_price: float = 0.0  # 진입 시점 SL (R 계산 기준, 불변)
    # 진입 비용 (cash에서 즉시 차감됨 — close_position 시 pnl에 반영)
    entry_commission: float = 0.0
    entry_slippage: float = 0.0
    # 진입 시 confluence score (ledger 기록용)
    confluence_score: int = 0


@dataclass
class PortfolioState:
    equity: float
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)  # key: symbol
    daily_start_equity: float = 0.0
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)

    @property
    def daily_pnl_pct(self) -> float:
        if self.daily_start_equity <= 0:
            return 0.0
        return (self.equity - self.daily_start_equity) / self.daily_start_equity

    @property
    def open_position_count(self) -> int:
        return len(self.positions)

    def long_count(self) -> int:
        return sum(1 for p in self.positions.values() if p.direction == "long")

    def short_count(self) -> int:
        return sum(1 for p in self.positions.values() if p.direction == "short")
