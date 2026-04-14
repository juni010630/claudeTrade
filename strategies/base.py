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
