"""Binance USDⓈ-M Futures 유지증거금(MMR) 티어 테이블.

실제 Binance는 심볼별·명목가(notional)별로 MMR과 최대 레버리지가 차등 적용된다.
기본값은 BTCUSDT 브래킷(공개 기준, 2024). 알트는 일반적으로 더 보수적인 시작 MMR을 가지므로
`alt_brackets`를 기본값으로 사용하고, `symbol_brackets`로 개별 오버라이드.

라이브 운용 시에는 `fapi/v1/leverageBracket` API로 최신 값을 싱크하는 것을 권장.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Bracket:
    max_notional: float  # 이 브래킷의 상한 명목가(USDT)
    mm_rate: float       # 유지증거금률
    max_leverage: int


# BTCUSDT 공개 브래킷 (근사치)
BTC_BRACKETS: list[Bracket] = [
    Bracket(50_000,       0.004, 125),
    Bracket(250_000,      0.005, 100),
    Bracket(3_000_000,    0.010, 50),
    Bracket(15_000_000,   0.025, 20),
    Bracket(30_000_000,   0.050, 10),
    Bracket(80_000_000,   0.100, 5),
    Bracket(float("inf"), 0.125, 4),
]

# ETHUSDT (BTC와 유사)
ETH_BRACKETS: list[Bracket] = [
    Bracket(50_000,       0.004, 125),
    Bracket(250_000,      0.005, 100),
    Bracket(1_000_000,    0.010, 50),
    Bracket(5_000_000,    0.025, 20),
    Bracket(20_000_000,   0.050, 10),
    Bracket(float("inf"), 0.100, 5),
]

# 알트 기본값 (보수적)
ALT_BRACKETS: list[Bracket] = [
    Bracket(5_000,        0.010, 75),
    Bracket(25_000,       0.025, 50),
    Bracket(100_000,      0.050, 20),
    Bracket(250_000,      0.100, 10),
    Bracket(1_000_000,    0.125, 5),
    Bracket(float("inf"), 0.250, 2),
]


class MarginTierTable:
    def __init__(
        self,
        symbol_brackets: dict[str, list[Bracket]] | None = None,
        default_brackets: list[Bracket] | None = None,
    ) -> None:
        self.symbol_brackets = symbol_brackets or {
            "BTCUSDT": BTC_BRACKETS,
            "ETHUSDT": ETH_BRACKETS,
        }
        self.default_brackets = default_brackets or ALT_BRACKETS

    def _brackets_for(self, symbol: str) -> list[Bracket]:
        return self.symbol_brackets.get(symbol, self.default_brackets)

    def mm_rate(self, symbol: str, notional: float) -> float:
        for br in self._brackets_for(symbol):
            if notional <= br.max_notional:
                return br.mm_rate
        return self._brackets_for(symbol)[-1].mm_rate

    def max_leverage(self, symbol: str, notional: float) -> int:
        for br in self._brackets_for(symbol):
            if notional <= br.max_notional:
                return br.max_leverage
        return self._brackets_for(symbol)[-1].max_leverage
