from __future__ import annotations

from abc import ABC, abstractmethod

from data.schemas import MarketSnapshot
from regime.models import RegimeState
from signals.models import Signal


class BaseStrategy(ABC):
    def __init__(self, config: dict) -> None:
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate_signals(
        self,
        snapshot: MarketSnapshot,
        regime: RegimeState,
    ) -> list[Signal]: ...

    def check_early_exit(self, position, snapshot: MarketSnapshot) -> bool:
        """포지션 조기 청산 조건(반대 신호 등) 만족 시 True 반환."""
        return False

