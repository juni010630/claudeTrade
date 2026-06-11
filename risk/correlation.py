"""상관계수 0.9 이상 쌍 동방향 진입 차단."""
from __future__ import annotations

import pandas as pd

from risk.models import PortfolioState
from signals.models import Signal


class CorrelationFilter:
    def __init__(
        self,
        block_threshold: float = 0.9,
        lookback: int = 100,
    ) -> None:
        self.block_threshold = block_threshold
        self.lookback = lookback
        self._returns: dict[str, pd.Series] = {}
        self._corr_matrix: pd.DataFrame = pd.DataFrame()
        self._dirty: bool = False

    def update(self, symbol: str, close_series: pd.Series) -> None:
        """최신 종가 시리즈로 수익률을 업데이트. 행렬은 lazy 재계산 (is_blocked 읽기 시점)."""
        self._returns[symbol] = close_series.pct_change().dropna().iloc[-self.lookback :]
        self._dirty = True

    def _ensure_matrix(self) -> None:
        # 진입 체크는 해당 봉의 모든 update 완료 후에만 호출되므로 결과는 매 호출 재계산과 비트동일.
        # 다심볼 유니버스에서 봉당 심볼수만큼 풀 행렬 재계산하던 핫스팟 제거.
        if getattr(self, "_dirty", False) and len(self._returns) >= 2:
            self._corr_matrix = pd.DataFrame(self._returns).corr()
            self._dirty = False

    def is_blocked(self, signal: Signal, state: PortfolioState) -> bool:
        """
        신규 진입 심볼과 열린 포지션 심볼 중
        상관계수 >= threshold이면서 같은 방향이 있으면 차단.
        """
        if not state.positions:
            return False
        self._ensure_matrix()
        if self._corr_matrix.empty:
            return False
        new_sym = signal.symbol
        for sym, pos in state.positions.items():
            if sym == new_sym:
                continue
            if pos.direction != signal.direction:
                continue
            if new_sym not in self._corr_matrix.columns or sym not in self._corr_matrix.columns:
                continue
            corr = self._corr_matrix.loc[new_sym, sym]
            if abs(corr) >= self.block_threshold:
                return True
        return False
