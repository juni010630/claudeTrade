"""전략별 국면 적합성 필터."""
from __future__ import annotations

from regime.models import MarketRegime

# 전략명 → 허용 국면 집합
_ALL_REGIMES = {MarketRegime.TRENDING, MarketRegime.RANGING, MarketRegime.PRE_BREAKOUT}

_ELIGIBLE_REGIMES: dict[str, set[MarketRegime]] = {
    "ema_cross":          {MarketRegime.TRENDING, MarketRegime.PRE_BREAKOUT},
    "mean_reversion":     {MarketRegime.RANGING},
    "multi_tf_breakout":  {MarketRegime.TRENDING, MarketRegime.PRE_BREAKOUT},
    "funding_reversion":  _ALL_REGIMES,
    "volume_imbalance":   _ALL_REGIMES,
}


def is_strategy_eligible(regime: MarketRegime, strategy_name: str) -> bool:
    allowed = _ELIGIBLE_REGIMES.get(strategy_name, set())
    return regime in allowed
