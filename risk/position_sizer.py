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
        max_notional_equity_mult: float = 3.0,
        max_notional_usd: float | None = None,  # 심볼별 절대 명목가 상한 ($). 호가창 유동성 현실 반영.
        initial_capital: float | None = None,
    ) -> None:
        """
        risk_per_trade: 자산 대비 위험 비율 (기본 1%)
        tier_config: params.yaml leverage_tiers 섹션
        max_notional_equity_mult: notional이 equity의 N배를 초과하지 못하도록 하드캡
            (기본 3.0 → 최대 equity × 3). 레버리지 컴파운딩 폭발 방지.
        initial_capital: 현재 미사용 (하위호환 보존용). equity 기반 sizing 사용.
            equity 비례 sizing → 수익 시 포지션 비중 일정, Sharpe·MDD 지표 유의미.
        """
        self.risk_per_trade = risk_per_trade
        self._cfg = tier_config or _DEFAULT_TIER_CONFIG
        self.max_notional_equity_mult = max_notional_equity_mult
        self.max_notional_usd = max_notional_usd
        self.initial_capital = initial_capital  # 하위호환 보존 (사이징 미사용)

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

        # equity 기반 sizing: 자본 증가 시 포지션도 비례 확대 (진정한 compound)
        # → 드로다운 시 자동 축소, Sharpe·MDD·return% 지표 왜곡 없음
        ref_capital = equity

        # 명목 크기: 기준자본 × size_fraction × leverage
        notional = ref_capital * size_fraction * leverage

        # 안전장치 1: 실제 손실이 risk_per_trade를 넘지 않도록 캡
        # 손실(USD) = notional × risk_pct (notional은 이미 레버리지 포함)
        # → notional ≤ ref_capital × risk_per_trade / risk_pct
        if entry_price > 0 and sl_price > 0 and entry_price != sl_price:
            risk_pct = abs(entry_price - sl_price) / entry_price
            max_notional_by_risk = (ref_capital * self.risk_per_trade) / risk_pct
            notional = min(notional, max_notional_by_risk)

        # 안전장치 2: 기준자본 대비 notional 하드캡 (초과 레버리지 방지)
        notional = min(notional, ref_capital * self.max_notional_equity_mult)

        # 안전장치 3: 절대 명목가 상한 (호가창 유동성 한계 — 알트 특히)
        if self.max_notional_usd is not None:
            notional = min(notional, self.max_notional_usd)

        return round(notional, 2), leverage
