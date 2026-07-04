"""펀딩비 시뮬레이터 — 정산 타임스탬프 전진 감지 (심볼별 4h/8h 주기 자동 대응)."""
from __future__ import annotations

import pandas as pd

from risk.models import PortfolioState


class FundingRateSimulator:
    def __init__(self, interval_hours: int = 8) -> None:
        self.interval_hours = interval_hours
        self._bucket_freq = f"{interval_hours}h"
        self._last_bucket: pd.Timestamp | None = None
        # 심볼별 마지막 관측 정산 시각 — funding_ts 전진 = 새 정산 발생.
        # 심볼별 정산 주기(8h 기본·4h 전환 심볼)와 과거 전환 이력을 데이터로 자동 추적.
        self._last_settle_ts: dict[str, pd.Timestamp] = {}
        self._sync_ts: pd.Timestamp | None = None

    def sync_to(self, now: pd.Timestamp) -> None:
        """라이브 재기동 시 호출 — 재기동 이전 정산을 '이미 정산됨'으로 표시.

        live_trade의 state 복원이 cash를 실잔고로 재앵커링하면 직전 정산까지의
        펀딩이 이미 반영되므로, 재기동 후 처음 관측되는 과거 정산을 중복 부과하지
        않는다. 재기동 이후 발생하는 정산부터 정상 부과 — 펀딩 누락 없음.
        """
        self._last_bucket = now.floor(self._bucket_freq)
        self._sync_ts = now

    def accrue(
        self,
        state: PortfolioState,
        now: pd.Timestamp,
        funding_rates: dict[str, float],
        funding_ts: dict[str, pd.Timestamp] | None = None,
        prices: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        funding_ts(직전 정산 시각) 전진 = 새 정산 발생 → 그 정산의 rate를 1회 부과.
        funding_ts가 없는 심볼(라이브 history API 실패 폴백)은 기존 글로벌
        interval_hours 버킷(UTC 00/08/16) 방식으로 부과.
        Returns: 심볼별 펀딩 비용 (양수 = 비용 발생, 음수 = 수취)
        """
        funding_ts = funding_ts or {}
        # 글로벌 버킷 (ts 없는 심볼 폴백용)
        bucket = now.floor(self._bucket_freq)
        if self._last_bucket is None:
            # 첫 호출: 이전 버킷으로 초기화 — 정확히 정산 시각에 시작해도 첫 적용 누락 방지
            self._last_bucket = bucket - pd.Timedelta(hours=self.interval_hours)
        bucket_crossed = bucket > self._last_bucket
        if bucket_crossed:
            self._last_bucket = bucket

        accruals: dict[str, float] = {}
        for sym, pos in state.positions.items():
            ts = funding_ts.get(sym)
            if ts is not None:
                last = self._last_settle_ts.get(sym)
                self._last_settle_ts[sym] = ts
                if last is not None and ts <= last:
                    continue  # 새 정산 없음 (이미 부과했거나 관측한 정산)
                # 정산 실시각은 정각 + ms 지터 — 진입/재기동 시각과의 비교는 정각 기준
                ts_hour = ts.floor("h")
                if ts_hour <= pos.opened_at:
                    continue  # 진입 이전(또는 진입 당봉) 정산 — 미보유 시점, 부과 없음
                if self._sync_ts is not None and ts_hour <= self._sync_ts:
                    continue  # 재기동 이전 정산 — 실잔고 재앵커에 이미 반영됨
            elif not bucket_crossed:
                continue
            rate = funding_rates.get(sym, 0.0)
            # mark price 기준 notional (바이낸스 공식: qty × mark_price × rate)
            mark = prices.get(sym, pos.entry_price) if prices else pos.entry_price
            notional = pos.size_usd / pos.entry_price * mark
            # 롱 포지션: 펀딩비 양수 → 지불, 음수 → 수취 / 숏: 반대
            if pos.direction == "long":
                cost = notional * rate
            else:
                cost = -notional * rate
            accruals[sym] = cost

        return accruals
