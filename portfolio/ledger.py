"""불변 거래 로그."""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from regime.models import MarketRegime

logger = logging.getLogger(__name__)


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


_CSV_FIELDS = [
    "trade_id", "entry_time", "exit_time", "symbol", "strategy", "direction",
    "entry_price", "exit_price", "size_usd", "leverage", "pnl",
    "exit_reason", "confluence_score", "commission", "slippage_cost", "funding_paid",
]


class Ledger:
    def __init__(self, csv_path: str | Path | None = None) -> None:
        self._records: list[TradeRecord] = []
        self._counter = 0
        self._csv_path = Path(csv_path) if csv_path else None

    def append(self, record: TradeRecord) -> None:
        self._counter += 1
        self._records.append(record)
        if self._csv_path is not None:
            self._append_csv(record)

    def _append_csv(self, record: TradeRecord) -> None:
        """거래 기록을 CSV에 한 줄 추가."""
        try:
            write_header = not self._csv_path.exists() or self._csv_path.stat().st_size == 0
            with open(self._csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
                if write_header:
                    writer.writeheader()
                row = {k: getattr(record, k) for k in _CSV_FIELDS}
                writer.writerow(row)
        except Exception as e:
            logger.warning("trades.csv 기록 실패: %s", e)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame([vars(r) for r in self._records])

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[TradeRecord]:
        return self._records

    @property
    def next_id(self) -> int:
        return self._counter + 1
