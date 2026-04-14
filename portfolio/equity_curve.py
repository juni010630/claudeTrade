"""시간별 자산 곡선 + 드로다운."""
from __future__ import annotations

import pandas as pd


class EquityCurve:
    def __init__(self) -> None:
        self._timestamps: list[pd.Timestamp] = []
        self._equity: list[float] = []
        self._open_positions: list[int] = []

    def append(self, timestamp: pd.Timestamp, equity: float, open_positions: int = 0) -> None:
        self._timestamps.append(timestamp)
        self._equity.append(equity)
        self._open_positions.append(open_positions)

    def to_series(self) -> pd.Series:
        return pd.Series(self._equity, index=self._timestamps, name="equity")

    def drawdown_series(self) -> pd.Series:
        eq = self.to_series()
        rolling_max = eq.cummax()
        return (eq - rolling_max) / rolling_max.replace(0, float("nan"))

    def max_drawdown(self) -> float:
        dd = self.drawdown_series()
        return float(dd.min()) if not dd.empty else 0.0

    def __len__(self) -> int:
        return len(self._equity)
