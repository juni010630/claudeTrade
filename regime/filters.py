"""전략별 국면 적합성 필터."""
from __future__ import annotations

from regime.models import MarketRegime

# 전략명 → 허용 국면 집합
_ALL_REGIMES = {MarketRegime.TRENDING, MarketRegime.RANGING, MarketRegime.PRE_BREAKOUT}

_ELIGIBLE_REGIMES: dict[str, set[MarketRegime]] = {
    "ema_cross":          {MarketRegime.TRENDING, MarketRegime.PRE_BREAKOUT},
    "ema_cross_slow":     {MarketRegime.TRENDING, MarketRegime.PRE_BREAKOUT},
    # 횡보 판정은 전략 내부 게이트(use_regime_gate / max_adx)가 담당 — 1h ADX 기반
    # 엔진 RANGING은 거의 발동하지 않아 (2023조차 0건) 라우팅은 전체 허용으로 변경
    "mean_reversion":     _ALL_REGIMES,
    # 1d 슬로우 크로스 — 국면 무관 검증 (NEWEDGE_GREEDY_RESULTS.md), 게이트 없음
    "macross_d":          _ALL_REGIMES,
    "multi_tf_breakout":  {MarketRegime.TRENDING, MarketRegime.PRE_BREAKOUT},
    "funding_reversion":  _ALL_REGIMES,
    "volume_imbalance":   _ALL_REGIMES,
    # 15m 모멘텀 브레이크아웃 (Scalp 검증 포팅) — 국면 무게이트 (Scalp 벡터화와 동일)
    "momentum_breakout":  _ALL_REGIMES,
}


def is_strategy_eligible(regime: MarketRegime, strategy_name: str) -> bool:
    allowed = _ELIGIBLE_REGIMES.get(strategy_name, set())
    return regime in allowed
