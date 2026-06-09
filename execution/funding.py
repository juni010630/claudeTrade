"""펀딩비 시뮬레이터 (8시간마다 오픈 포지션에 적용)."""
from __future__ import annotations

import pandas as pd

from risk.models import PortfolioState


class FundingRateSimulator:
    def __init__(self, interval_hours: int = 8) -> None:
        self.interval_hours = interval_hours
        self._bucket_freq = f"{interval_hours}h"
        self._last_bucket: pd.Timestamp | None = None

    def sync_to(self, now: pd.Timestamp) -> None:
        """라이브 재기동 시 호출 — 현재 펀딩 버킷을 '이미 정산됨'으로 표시.

        live_trade의 state 복원이 cash를 실잔고로 재앵커링하면 해당 버킷 펀딩이
        이미 반영되므로, 첫 봉에서 같은 버킷을 중복 부과하지 않도록 한다.
        다음 정산 경계(예: 다음 00/08/16 UTC)부터 정상 부과 — 펀딩 누락 없음.
        """
        self._last_bucket = now.floor(self._bucket_freq)

    def accrue(
        self,
        state: PortfolioState,
        now: pd.Timestamp,
        funding_rates: dict[str, float],
        prices: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        바이낸스 절대 펀딩 시각(UTC 00/08/16) 기준으로 적용.
        Returns: 심볼별 펀딩 비용 (양수 = 비용 발생, 음수 = 수취)
        """
        # 현재 시간을 8h 버킷으로 내림 — 절대 스케줄 기준
        bucket = now.floor(self._bucket_freq)

        if self._last_bucket is None:
            # 첫 호출: 현재 버킷의 이전 버킷으로 초기화하여
            # 정확히 펀딩 시각에 시작해도 첫 적용이 누락되지 않도록 함
            self._last_bucket = bucket - pd.Timedelta(hours=self.interval_hours)

        if bucket <= self._last_bucket:
            return {}  # 같은 버킷 — 미적용

        self._last_bucket = bucket
        accruals: dict[str, float] = {}

        for sym, pos in state.positions.items():
            rate = funding_rates.get(sym, 0.0)
            # mark price 기준 notional (바이낸스 공식: qty × mark_price × rate)
            mark = prices.get(sym, pos.entry_price) if prices else pos.entry_price
            notional = pos.size_usd / pos.entry_price * mark
            # 롱 포지션: 펀딩비 양수 → 지불, 음수 → 수취
            # 숏 포지션: 반대
            if pos.direction == "long":
                cost = notional * rate
            else:
                cost = -notional * rate
            accruals[sym] = cost

        return accruals
