"""전략별 연속 손절 카운터 + 전체 중단 상태 머신."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class BreakerStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"    # 특정 전략 일시 중단
    STOPPED = "stopped"  # 전체 중단


@dataclass
class CircuitBreaker:
    strategy_pause_losses: int = 5    # 전략별 연속 손절 → 48시간 정지
    global_stop_losses: int = 10      # 전체 연속 손절 → 전면 중단
    pause_duration_hours: int = 48

    _strategy_losses: dict[str, int] = field(default_factory=dict)
    _strategy_paused_until: dict[str, pd.Timestamp] = field(default_factory=dict)
    _global_losses: int = 0
    _global_stopped_until: pd.Timestamp | None = None

    def record_result(self, strategy: str, is_win: bool) -> None:
        if is_win:
            self._strategy_losses[strategy] = 0
            self._global_losses = 0
        else:
            self._strategy_losses[strategy] = self._strategy_losses.get(strategy, 0) + 1
            self._global_losses += 1

    def get_status(self, strategy: str, now: pd.Timestamp) -> BreakerStatus:
        # 전체 중단: 자동 해제 (pause_duration_hours 경과 시)
        if self._global_stopped_until is not None:
            if now < self._global_stopped_until:
                return BreakerStatus.STOPPED
            # 해제 후 카운터 전부 리셋 (전략별 포함)
            self._global_stopped_until = None
            self._global_losses = 0
            self._strategy_losses.clear()
            self._strategy_paused_until.clear()

        if self._global_losses >= self.global_stop_losses:
            self._global_stopped_until = now + pd.Timedelta(hours=self.pause_duration_hours)
            return BreakerStatus.STOPPED

        paused_until = self._strategy_paused_until.get(strategy)
        if paused_until is not None and now < paused_until:
            return BreakerStatus.PAUSED

        losses = self._strategy_losses.get(strategy, 0)
        if losses >= self.strategy_pause_losses:
            pause_end = now + pd.Timedelta(hours=self.pause_duration_hours)
            self._strategy_paused_until[strategy] = pause_end
            self._strategy_losses[strategy] = 0  # 카운터 리셋
            return BreakerStatus.PAUSED

        return BreakerStatus.ACTIVE

    def reset_global(self) -> None:
        """수동 리셋 (검토 후)."""
        self._global_stopped_until = None
        self._global_losses = 0
