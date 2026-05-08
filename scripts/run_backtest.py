"""백테스트 실행 → MetricsReport 출력.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --params config/params.yaml
    python scripts/run_backtest.py --start 2022-01-01 --end 2023-12-31
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from data.loader import DataLoader
from engine.backtest import BacktestEngine
from execution.broker import BacktestBroker
from execution.commission import CommissionModel
from execution.funding import FundingRateSimulator
from execution.slippage import SlippageModel
from regime.detector import RegimeDetector
from risk.circuit_breaker import CircuitBreaker
from risk.correlation import CorrelationFilter
from risk.guards import RiskGuards
from risk.position_sizer import PositionSizer
from signals.scorer import ConfluenceScorer
from strategies.ema_cross import EMACrossStrategy
from strategies.multi_tf_breakout import MultiTFBreakoutStrategy

import pandas as pd


def build_engine(p: dict, initial_capital: float, abort_mdd: float | None = None,
                  isolated_margin: bool = False, **engine_kwargs) -> BacktestEngine:
    """params.yaml 딕셔너리로 BacktestEngine을 생성합니다."""
    symbols = p["symbols"]
    r  = p.get("risk", {})
    e  = p.get("execution", {})
    rg = p.get("regime", {})
    sc = p.get("scorer", {})

    # ------------------------------------------------------------------
    # 활성화된 전략만 인스턴스화
    # ------------------------------------------------------------------
    strategy_map = {
        "ema_cross":          EMACrossStrategy,
        "multi_tf_breakout":  MultiTFBreakoutStrategy,
    }
    strategies = []
    for key, cls in strategy_map.items():
        cfg = p.get("strategies", {}).get(key)
        if cfg is None:
            continue  # config에 미정의 전략은 건너뜀
        if not cfg.get("enabled", True):
            continue
        cfg["symbols"] = symbols
        strategies.append(cls(cfg))

    if not strategies:
        raise ValueError(
            "활성화된 전략이 없습니다. params.yaml에서 enabled: true 전략을 하나 이상 설정하세요."
        )

    # ------------------------------------------------------------------
    # ML 소프트 스코링 (선택적)
    # ------------------------------------------------------------------
    ml_filter = None
    ml_cfg = sc.get("ml_soft_scoring", {})
    if ml_cfg.get("enabled", False):
        try:
            from strategies.ml_filter import MLModels, MLSignalFilter
            model_path = ml_cfg.get("model_path", "models/ml_filter.pkl")
            models = MLModels.load(model_path)
            ml_filter = MLSignalFilter(
                models=models,
                clf_threshold=0.0,  # soft scoring이므로 차단 없음
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("ML 모델 로드 실패: %s — ML 보너스 비활성", e)

    # ------------------------------------------------------------------
    # 체결 비용 모델
    # ------------------------------------------------------------------
    commission_model = CommissionModel(
        maker_rate=e.get("commission_maker", 0.0002),
        taker_rate=e.get("commission_taker", 0.0005),
    )
    slippage_model = SlippageModel(
        default_bps=e.get("default_slippage_bps", 5.0),
    )
    broker = BacktestBroker(
        commission_model=commission_model,
        slippage_model=slippage_model,
    )

    return BacktestEngine(
        initial_capital=initial_capital,
        strategies=strategies,
        regime_detector=RegimeDetector(
            primary_symbol=symbols[0],
            adx_period=rg.get("adx_period", 14),
            adx_trending_threshold=rg.get("adx_trending_threshold", 25.0),
            adx_ranging_threshold=rg.get("adx_ranging_threshold", 20.0),
            bb_period=rg.get("bb_period", 20),
            bb_std=rg.get("bb_std", 2.0),
            bb_width_lookback=rg.get("bb_width_lookback", 50),
            bb_width_squeeze_pct=rg.get("bb_width_squeeze_pct", 0.2),
            primary_tf=rg.get("primary_tf", "1h"),
        ),
        confluence_scorer=ConfluenceScorer(
            volume_ratio_threshold=sc.get("volume_ratio_threshold", 1.5),
            rsi_long_max=sc.get("rsi_long_max", 65.0),
            rsi_short_min=sc.get("rsi_short_min", 35.0),
            funding_long_max=sc.get("funding_long_max", 0.0003),
            funding_short_min=sc.get("funding_short_min", -0.0003),
            daily_ema_period=sc.get("daily_ema_period", 200),
            tier_ss_min_score=sc.get("tier_ss_min_score", 7),
            tier_s_min_score=sc.get("tier_s_min_score", 5),
            tier_a_min_score=sc.get("tier_a_min_score", 3),
            tier_b_min_score=sc.get("tier_b_min_score", 2),
            tier_c_min_score=sc.get("tier_c_min_score", 1),
            ml_filter=ml_filter,
            ml_bonus_threshold_1=ml_cfg.get("bonus_threshold_1", 0.6),
            ml_bonus_threshold_2=ml_cfg.get("bonus_threshold_2", 0.75),
            rsi_neutral_penalty=tuple(sc["rsi_neutral_penalty"]) if sc.get("rsi_neutral_penalty") else None,
        ),
        risk_guards=RiskGuards(
            max_positions=r.get("max_positions", 4),
            max_same_direction=r.get("max_same_direction", 3),
            daily_pause_threshold=r.get("daily_drawdown_pause", -0.05),
            daily_stop_threshold=r.get("daily_drawdown_stop", -0.08),
        ),
        circuit_breaker=CircuitBreaker(
            strategy_pause_losses=r.get("circuit_breaker_pause_losses", 5),
            global_stop_losses=r.get("circuit_breaker_stop_losses", 10),
            pause_duration_hours=r.get("circuit_breaker_pause_hours", 48),
        ),
        correlation_filter=CorrelationFilter(
            block_threshold=r.get("correlation_block_threshold", 0.9),
            lookback=r.get("correlation_lookback", 100),
        ),
        position_sizer=PositionSizer(
            risk_per_trade=r.get("risk_per_trade", 0.01),
            tier_config=p.get("leverage_tiers"),
            max_notional_usd=r.get("max_notional_usd"),
            max_notional_equity_mult=r.get("max_notional_equity_mult", 3.0),
        ),
        broker=broker,
        funding_simulator=FundingRateSimulator(
            interval_hours=e.get("funding_interval_hours", 8),
        ),
        max_hold_hours=p.get("engine", {}).get("max_hold_hours"),
        breakeven_trigger_r=p.get("engine", {}).get("breakeven_trigger_r"),
        trailing_r_mult=p.get("engine", {}).get("trailing_r_mult"),
        strategy_min_score=p.get("strategy_min_score"),
        strategy_block_hours=p.get("strategy_block_hours"),
        strategy_block_symbols=p.get("strategy_block_symbols"),
        abort_mdd_threshold=abort_mdd,
        isolated_margin=isolated_margin,
        **engine_kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="config/final_v13_eth.yaml", help="파라미터 파일 경로")
    parser.add_argument("--start", default=None, help="백테스트 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="백테스트 종료일 (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=None, help="초기 자본")
    parser.add_argument("--abort-mdd", type=float, default=None,
                        help="running peak 대비 이 비율(음수) 초과 DD 시 백테 조기 중단 (예: -0.35)")
    parser.add_argument("--isolated", action="store_true",
                        help="isolated margin 모드 (포지션별 독립 청산)")
    args = parser.parse_args()

    with open(args.params) as f:
        p = yaml.safe_load(f)

    bt = p.get("backtest", {})
    initial_capital = args.capital or bt.get("initial_capital", 100_000)
    since_str = args.start or bt.get("start", "2024-01-01")
    until_str = args.end or bt.get("end")

    since = pd.Timestamp(since_str, tz="UTC")
    until = pd.Timestamp(until_str, tz="UTC") if until_str else None

    data_cfg = p.get("data", {})
    loader = DataLoader(
        symbols=p["symbols"],
        timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=data_cfg.get("cache_dir", "data/cache"),
        lookback=data_cfg.get("lookback_bars", 300),
    )

    engine = build_engine(p, initial_capital, abort_mdd=args.abort_mdd,
                          isolated_margin=getattr(args, 'isolated', False))

    active = [s.name for s in engine.strategies]
    print(f"백테스트 시작: {since_str} ~ {until_str or '최신'}")
    print(f"초기 자본: ${initial_capital:,.0f}")
    print(f"활성 전략: {', '.join(active)}")
    if args.abort_mdd is not None:
        print(f"abort_mdd: {args.abort_mdd*100:.1f}%")
    print()

    snapshots = loader.iterate(since=since, until=until)
    report = engine.run(snapshots)
    report.print_summary()
    if report.aborted:
        print(f"\n[EARLY-STOP] aborted=True (running peak 대비 DD가 {args.abort_mdd*100:.1f}% 초과)")


if __name__ == "__main__":
    main()
