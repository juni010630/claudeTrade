"""불변 거래 로그."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from regime.models import MarketRegime


@dataclass
class TradeRecord:
    trade_id: int
    symbol: str
    strategy: str
    direction: Literal["long", "short"]
    entry_price: float
    exit_price: float
    size_usd: float
    leverage: int
    pnl: float               # 수수료·슬리피지·펀딩 포함 실현 손익
    commission: float
    slippage_cost: float
    funding_paid: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    exit_reason: str          # "tp" | "sl" | "forced"
    regime_at_entry: MarketRegime
    confluence_score: int


class Ledger:
    def __init__(self) -> None:
        self._records: list[TradeRecord] = []
        self._counter = 0

    def append(self, record: TradeRecord) -> None:
        self._counter += 1
        self._records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame([vars(r) for r in self._records])

    def __len__(self) -> int:
        return len(self._records)

    @property
    def next_id(self) -> int:
        return self._counter + 1
