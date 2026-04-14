"""다기간 백테스트 검증 — 오버피팅 탐지.

Usage:
    python scripts/validate_periods.py --trial 16    # ext 결과 파일의 특정 트라이얼
    python scripts/validate_periods.py --params config/params_v2.yaml  # 파일 직접
    python scripts/validate_periods.py --trial 16 --trial 28 --trial 41v2

검증 기간:
  ① 2022 Bear  2022-01-01 ~ 2022-12-31
  ② 2023 Recov 2023-01-01 ~ 2023-12-31
  ③ 2024 Bull  2024-01-01 ~ 2025-03-01  (원래 최적화 구간)
  ④ Full       2022-01-01 ~ 2025-03-01
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

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
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum_breakout import MomentumBreakoutStrategy
from strategies.multi_tf_breakout import MultiTFBreakoutStrategy

PERIODS = [
    ("2022 Bear",  "2022-01-01", "2022-12-31"),
    ("2023 Recov", "2023-01-01", "2023-12-31"),
    ("2024 Bull",  "2024-01-01", "2025-03-01"),
    ("Full 3yr",   "2022-01-01", "2025-03-01"),
]

EXT_RESULTS = Path("optimize_v2_ext_results.json")
V2_RESULTS  = Path("optimize_v2_results.json")


def _set_nested(d: dict, dotkey: str, value: Any) -> None:
    keys = dotkey.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def apply_diff(base: dict, diff: dict) -> dict:
    p = copy.deepcopy(base)
    for key, (_, new_val) in diff.items():
        if key == "symbols":
            p["symbols"] = new_val
        else:
            _set_nested(p, key, new_val)
    return p


def build_engine(p: dict, initial_capital: float) -> BacktestEngine:
    symbols = p["symbols"]
    r  = p.get("risk", {})
    e  = p.get("execution", {})
    rg = p.get("regime", {})
    sc = p.get("scorer", {})

    strategy_map = {
        "momentum_breakout": MomentumBreakoutStrategy,
        "multi_tf_breakout": MultiTFBreakoutStrategy,
        "mean_reversion":    MeanReversionStrategy,
    }
    strategies = []
    for key, cls in strategy_map.items():
        cfg = p.get("strategies", {}).get(key, {})
        if cfg.get("enabled", True):
            cfg["symbols"] = symbols
            strategies.append(cls(cfg))

    commission_model = CommissionModel(
        maker_rate=e.get("commission_maker", 0.0002),
        taker_rate=e.get("commission_taker", 0.0005),
    )
    slippage_model = SlippageModel(default_bps=e.get("default_slippage_bps", 5.0))
    broker = BacktestBroker(commission_model=commission_model, slippage_model=slippage_model)

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
            tier_s_min_score=sc.get("tier_s_min_score", 6),
            tier_a_min_score=sc.get("tier_a_min_score", 4),
            tier_b_min_score=sc.get("tier_b_min_score", 2),
        ),
        risk_guards=RiskGuards(
            max_positions=r.get("max_positions", 5),
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
        ),
        broker=broker,
        funding_simulator=FundingRateSimulator(interval_hours=e.get("funding_interval_hours", 8)),
    )


def run_period(p: dict, start: str, end: str, cache_dir: str, tfs: list, primary_tf: str) -> dict:
    symbols = p["symbols"]
    loader = DataLoader(
        symbols=symbols,
        timeframes=tfs,
        primary_tf=primary_tf,
        cache_dir=cache_dir,
        lookback=300,
    )
    since = pd.Timestamp(start, tz="UTC")
    until = pd.Timestamp(end, tz="UTC")
    engine = build_engine(p, 100_000)
    snaps  = loader.iterate(since=since, until=until)
    report = engine.run(snaps)
    return {
        "return":  report.total_return_pct,
        "sharpe":  report.sharpe,
        "mdd":     report.max_drawdown,
        "trades":  report.total_trades,
        "wr":      report.win_rate,
        "calmar":  report.calmar,
    }


def validate(label: str, p: dict, cache_dir: str, tfs: list, primary_tf: str) -> None:
    syms = "+".join(s.replace("USDT","") for s in p["symbols"])
    print(f"\n{'='*65}")
    print(f"  {label}  [{syms}]")
    print(f"{'='*65}")
    print(f"  {'기간':<14}  {'수익률':>8}  {'Sharpe':>7}  {'MDD':>7}  {'거래':>5}  {'WR':>6}")
    print(f"  {'-'*60}")

    results = {}
    for name, start, end in PERIODS:
        try:
            r = run_period(p, start, end, cache_dir, tfs, primary_tf)
            results[name] = r
            print(f"  {name:<14}  {r['return']:>+7.1f}%  {r['sharpe']:>7.3f}  {r['mdd']:>6.1f}%  {r['trades']:>5}  {r['wr']:>5.1f}%")
        except Exception as ex:
            print(f"  {name:<14}  ERROR: {ex}")

    # 일관성 점수: 수익 구간 수 / 전체
    profitable = sum(1 for r in results.values() if r.get("return", -999) > 0)
    positive_sharpe = sum(1 for r in results.values() if r.get("sharpe", -999) > 0.5)
    print(f"\n  수익 구간: {profitable}/{len(PERIODS)}  Sharpe>0.5 구간: {positive_sharpe}/{len(PERIODS)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial",  type=int, action="append", dest="trials", default=[])
    parser.add_argument("--params", type=str, default=None, help="yaml 파일 직접 지정")
    args = parser.parse_args()

    # 기준 params 로드
    with open("config/params_v2.yaml") as f:
        base_params = yaml.safe_load(f)

    data_cfg   = base_params.get("data", {})
    cache_dir  = data_cfg.get("cache_dir", "data/cache")
    tfs        = base_params.get("timeframes", ["1h", "4h", "1d"])
    primary_tf = base_params.get("primary_timeframe", "1h")

    subjects = []

    # --params: yaml 파일 직접
    if args.params:
        with open(args.params) as f:
            p = yaml.safe_load(f)
        subjects.append((f"파일: {args.params}", p))

    # --trial N: ext 결과에서 찾기
    if args.trials:
        ext_data = json.loads(EXT_RESULTS.read_text()) if EXT_RESULTS.exists() else []
        v2_data  = json.loads(V2_RESULTS.read_text())  if V2_RESULTS.exists()  else []
        all_data = ext_data + v2_data

        for trial_num in args.trials:
            match = next((x for x in all_data if x["trial"] == trial_num), None)
            if match is None:
                print(f"Trial #{trial_num} 없음")
                continue
            p = apply_diff(base_params, match["params_diff"])
            s = match["summary"]
            label = (f"Trial #{trial_num}  "
                     f"sharpe={s['sharpe']:.3f}  ret={s['total_return']:+.1f}%  MDD={s['max_drawdown']:.1f}%")
            subjects.append((label, p))

    # 기본: 현재 params_v2.yaml
    if not subjects:
        subjects.append(("현재 params_v2.yaml (Trial #41)", base_params))

    for label, p in subjects:
        validate(label, p, cache_dir, tfs, primary_tf)

    print()


if __name__ == "__main__":
    main()
