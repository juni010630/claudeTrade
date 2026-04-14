"""펀딩비 시뮬레이터 (8시간마다 오픈 포지션에 적용)."""
from __future__ import annotations

import pandas as pd

from risk.models import PortfolioState


class FundingRateSimulator:
    def __init__(self, interval_hours: int = 8) -> None:
        self.interval = pd.Timedelta(hours=interval_hours)
        self._last_accrual: pd.Timestamp | None = None

    def accrue(
        self,
        state: PortfolioState,
        now: pd.Timestamp,
        funding_rates: dict[str, float],
    ) -> dict[str, float]:
        """
        8시간 간격으로 각 포지션에 펀딩비 적용.
        Returns: 심볼별 펀딩 비용 (양수 = 비용 발생, 음수 = 수취)
        """
        if self._last_accrual is None:
            self._last_accrual = now
            return {}

        if now - self._last_accrual < self.interval:
            return {}

        self._last_accrual = now
        accruals: dict[str, float] = {}

        for sym, pos in state.positions.items():
            rate = funding_rates.get(sym, 0.0)
            notional = pos.size_usd
            # 롱 포지션: 펀딩비 양수 → 지불, 음수 → 수취
            # 숏 포지션: 반대
            if pos.direction == "long":
                cost = notional * rate
            else:
                cost = -notional * rate
            accruals[sym] = cost

        return accruals
