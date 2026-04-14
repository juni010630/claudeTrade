"""포지션 한도 + 일일 Drawdown 체크."""
from __future__ import annotations

from enum import Enum

from risk.models import PortfolioState
from signals.models import Signal


class DrawdownAction(Enum):
    OK = "ok"
    PAUSE = "pause"   # -5% → 신규 진입 중단
    STOP = "stop"     # -8% → 전 포지션 강제 청산


class RiskGuards:
    def __init__(
        self,
        max_positions: int = 4,
        max_same_direction: int = 3,
        daily_pause_threshold: float = -0.05,
        daily_stop_threshold: float = -0.08,
    ) -> None:
        self.max_positions = max_positions
        self.max_same_direction = max_same_direction
        self.daily_pause = daily_pause_threshold
        self.daily_stop = daily_stop_threshold

    def check_daily_drawdown(self, state: PortfolioState) -> DrawdownAction:
        pnl = state.daily_pnl_pct
        if pnl <= self.daily_stop:
            return DrawdownAction.STOP
        if pnl <= self.daily_pause:
            return DrawdownAction.PAUSE
        return DrawdownAction.OK

    def check_max_positions(self, state: PortfolioState) -> bool:
        """True = 추가 진입 가능."""
        return state.open_position_count < self.max_positions

    def check_direction_limit(self, state: PortfolioState, direction: str) -> bool:
        """같은 방향 포지션이 한도 미만이면 True."""
        count = state.long_count() if direction == "long" else state.short_count()
        return count < self.max_same_direction

    def check_symbol_free(self, state: PortfolioState, symbol: str) -> bool:
        """이미 해당 심볼 포지션이 없어야 진입 가능."""
        return symbol not in state.positions

    def is_entry_allowed(self, state: PortfolioState, signal: Signal) -> bool:
        dd = self.check_daily_drawdown(state)
        if dd != DrawdownAction.OK:
            return False
        if not self.check_max_positions(state):
            return False
        if not self.check_direction_limit(state, signal.direction):
            return False
        if not self.check_symbol_free(state, signal.symbol):
            return False
        return True
