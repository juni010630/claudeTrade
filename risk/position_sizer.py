"""포지션 크기 계산기."""
from __future__ import annotations

from signals.models import LeverageTier

# params.yaml leverage_tiers 섹션이 없을 때 쓰는 기본값
_DEFAULT_TIER_CONFIG: dict[str, dict] = {
    "SS": {"leverage": 15, "size_fraction": 0.35},
    "S":  {"leverage": 10, "size_fraction": 0.25},
    "A":  {"leverage": 5,  "size_fraction": 0.15},
    "B":  {"leverage": 3,  "size_fraction": 0.08},
    "C":  {"leverage": 2,  "size_fraction": 0.04},
}


class PositionSizer:
    def __init__(
        self,
        risk_per_trade: float = 0.01,
        tier_config: dict[str, dict] | None = None,
    ) -> None:
        """
        risk_per_trade: 자산 대비 위험 비율 (기본 1%)
        tier_config: params.yaml leverage_tiers 섹션
            {"S": {"leverage": 5, "size_fraction": 0.15}, ...}
        """
        self.risk_per_trade = risk_per_trade
        self._cfg = tier_config or _DEFAULT_TIER_CONFIG

    def _tier_params(self, tier: LeverageTier) -> tuple[int, float]:
        """(leverage, size_fraction) 반환."""
        cfg = self._cfg.get(tier.value, {})
        leverage = int(cfg.get("leverage", 1))
        size_fraction = float(cfg.get("size_fraction", 0.0))
        return leverage, size_fraction

    def calculate(
        self,
        tier: LeverageTier,
        equity: float,
        entry_price: float,
        sl_price: float,
    ) -> tuple[float, int]:
        """
        Returns:
            (size_usd, leverage) — 명목 포지션 크기(USD)와 레버리지
        """
        if tier == LeverageTier.NO_TRADE:
            return 0.0, 0

        leverage, size_fraction = self._tier_params(tier)

        # 명목 크기: 자산 × size_fraction × leverage
        notional = equity * size_fraction * leverage

        # 추가 안전장치: 실제 손실이 risk_per_trade를 넘지 않도록 캡
        if entry_price > 0 and sl_price > 0 and entry_price != sl_price:
            risk_pct = abs(entry_price - sl_price) / entry_price
            max_notional_by_risk = (equity * self.risk_per_trade) / risk_pct * leverage
            notional = min(notional, max_notional_by_risk)

        return round(notional, 2), leverage
