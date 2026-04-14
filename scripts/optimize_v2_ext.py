"""V2 확장 최적화 — 종목 조합 + 복합 스코어(Sharpe×수익률 균형).

Usage:
    python scripts/optimize_v2_ext.py --trials 50 --metric combo
    python scripts/optimize_v2_ext.py --resume --trials 30
    python scripts/optimize_v2_ext.py --apply-best

복합 스코어: sharpe + 0.005 * return_pct
  → Sharpe를 우선하되, 수익률도 함께 극대화
"""
from __future__ import annotations

import argparse
import copy
import json
import random
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

# ─────────────────────────────────────────────
# 탐색 가능한 종목 조합
# ─────────────────────────────────────────────
SYMBOL_POOLS = [
    # 기존 조합
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"],
    # BTC/ETH 코어 + 알트
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"],
    ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"],
    ["BTCUSDT", "ETHUSDT", "BNBUSDT", "AVAXUSDT"],
    ["BTCUSDT", "ETHUSDT", "AVAXUSDT", "DOGEUSDT"],
    # 5종목 풀
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "AVAXUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "DOGEUSDT"],
    # 6종목
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "AVAXUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT", "DOGEUSDT"],
]

# ─────────────────────────────────────────────
# 탐색 공간 (종목 제외)
# ─────────────────────────────────────────────
SEARCH_SPACE: dict[str, Any] = {
    # 리스크
    "risk.risk_per_trade":          [0.005, 0.008, 0.01, 0.012, 0.015, 0.02],
    "risk.max_positions":           [4, 5, 6, 8],
    "risk.max_same_direction":      [2, 3, 4],
    "risk.daily_drawdown_pause":    [-0.04, -0.05, -0.06, -0.07],
    "risk.daily_drawdown_stop":     [-0.07, -0.08, -0.10, -0.12],

    # 레버리지 티어
    "leverage_tiers.S.leverage":      [8, 10, 12, 15, 20],
    "leverage_tiers.S.size_fraction": [0.20, 0.25, 0.30, 0.35, 0.40],
    "leverage_tiers.A.leverage":      [4, 5, 7, 10],
    "leverage_tiers.A.size_fraction": [0.10, 0.15, 0.20, 0.25],
    "leverage_tiers.B.leverage":      [2, 3, 4, 5],
    "leverage_tiers.B.size_fraction": [0.06, 0.08, 0.10, 0.12],

    # 국면 탐지
    "regime.adx_trending_threshold": [22.0, 25.0, 28.0, 30.0, 33.0],
    "regime.adx_ranging_threshold":  [15.0, 18.0, 20.0, 22.0],
    "regime.bb_width_squeeze_pct":   [0.15, 0.20, 0.25, 0.30, 0.35],

    # 스코어러
    "scorer.volume_ratio_threshold": [1.0, 1.2, 1.5, 1.8, 2.0],
    "scorer.rsi_long_max":           [55.0, 60.0, 65.0, 70.0, 75.0],
    "scorer.rsi_short_min":          [25.0, 30.0, 35.0, 40.0, 45.0],
    "scorer.tier_s_min_score":       [4, 5, 6, 7],
    "scorer.tier_a_min_score":       [3, 4, 5],
    "scorer.tier_b_min_score":       [1, 2, 3],

    # momentum_breakout
    "strategies.momentum_breakout.bb_std":            [1.5, 1.8, 2.0, 2.2, 2.5],
    "strategies.momentum_breakout.atr_tp_mult":       [2.0, 2.5, 3.0, 3.5, 4.0],
    "strategies.momentum_breakout.atr_sl_mult":       [0.6, 0.8, 1.0, 1.2],
    "strategies.momentum_breakout.volume_multiplier": [1.2, 1.5, 1.8, 2.0, 2.5],
    "strategies.momentum_breakout.volume_lookback":   [15, 20, 25],

    # multi_tf_breakout
    "strategies.multi_tf_breakout.bb_std_4h":         [1.8, 2.0, 2.2, 2.5],
    "strategies.multi_tf_breakout.bb_std_1h":         [1.5, 1.8, 2.0, 2.2],
    "strategies.multi_tf_breakout.rsi_long_min":      [45.0, 50.0, 55.0],
    "strategies.multi_tf_breakout.rsi_short_max":     [45.0, 50.0, 55.0],
    "strategies.multi_tf_breakout.volume_multiplier": [1.2, 1.5, 1.8, 2.0, 2.5],
    "strategies.multi_tf_breakout.atr_tp_mult":       [2.0, 2.5, 3.0, 3.5],
    "strategies.multi_tf_breakout.atr_sl_mult":       [0.7, 1.0, 1.2, 1.5],

    # mean_reversion
    "strategies.mean_reversion.bb_std":         [1.5, 1.8, 2.0, 2.2],
    "strategies.mean_reversion.rsi_oversold":   [25.0, 30.0, 35.0],
    "strategies.mean_reversion.rsi_overbought": [65.0, 70.0, 75.0, 80.0],
    "strategies.mean_reversion.atr_tp_mult":    [1.5, 2.0, 2.5, 3.0],
    "strategies.mean_reversion.atr_sl_mult":    [0.5, 0.7, 0.9, 1.0],
}

RESULTS_FILE = Path("optimize_v2_ext_results.json")
BASE_PARAMS   = "config/params_v2.yaml"

# ─────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────
def _set_nested(d: dict, dotkey: str, value: Any) -> None:
    keys = dotkey.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _get_nested(d: dict, keys: list[str]) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def sample_params(base: dict) -> dict:
    p = copy.deepcopy(base)
    # 종목 조합 랜덤 선택
    p["symbols"] = random.choice(SYMBOL_POOLS)
    # 파라미터 랜덤 선택
    for key, choices in SEARCH_SPACE.items():
        _set_nested(p, key, random.choice(choices))
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


def score_report(report, metric: str) -> float:
    """
    combo: Sharpe + 0.005 × return%
      → Sharpe 2.0 + 수익률 100% = 2.5점
      → Sharpe 2.1 + 수익률 40%  = 2.3점
    sharpe: 순수 Sharpe
    """
    mdd    = report.max_drawdown
    trades = report.total_trades

    if trades < 15:
        return -999.0
    if mdd < -35.0:
        return -999.0

    sharpe = report.sharpe if report.sharpe is not None else -999.0
    ret    = report.total_return_pct if report.total_return_pct is not None else -999.0

    if metric == "combo":
        if sharpe < 0:
            return sharpe
        return sharpe + 0.005 * ret

    elif metric == "sharpe":
        return sharpe

    elif metric == "total_return":
        if mdd < -40.0:
            return -999.0
        return ret

    return -999.0


def load_results() -> list[dict]:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return []


def save_results(results: list[dict]) -> None:
    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))


def _extract_diff(base: dict, new: dict) -> dict[str, tuple]:
    diff = {}
    # 종목 변경 체크
    if base.get("symbols") != new.get("symbols"):
        diff["symbols"] = (base.get("symbols"), new.get("symbols"))
    for key in SEARCH_SPACE:
        parts = key.split(".")
        b_val = _get_nested(base, parts)
        n_val = _get_nested(new, parts)
        if b_val != n_val:
            diff[key] = (b_val, n_val)
    return diff


def _apply_best_params(path: str, base: dict, diff: dict[str, tuple]) -> None:
    p = copy.deepcopy(base)
    for key, (_, new_val) in diff.items():
        if key == "symbols":
            p["symbols"] = new_val
        else:
            _set_nested(p, key, new_val)
    with open(path, "w") as f:
        yaml.dump(p, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",     type=int, default=30)
    parser.add_argument("--metric",     default="combo", choices=["combo", "sharpe", "total_return"])
    parser.add_argument("--params",     default=BASE_PARAMS)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--seed",       type=int, default=None)
    parser.add_argument("--apply-best", action="store_true")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    with open(args.params) as f:
        base_params = yaml.safe_load(f)

    bt = base_params.get("backtest", {})
    initial_capital = bt.get("initial_capital", 100_000)
    since = pd.Timestamp(bt.get("start", "2024-01-01"), tz="UTC")
    until_str = bt.get("end")
    until = pd.Timestamp(until_str, tz="UTC") if until_str else None

    data_cfg = base_params.get("data", {})

    results = load_results() if args.resume else []
    trial_offset = len(results)

    print(f"V2 확장 최적화 시작 — 목표: {args.metric.upper()}, 시도: {args.trials}회")
    print(f"종목 조합 후보: {len(SYMBOL_POOLS)}개")
    if results:
        best_so_far = max(results, key=lambda r: r["score"])
        bs = best_so_far["summary"]
        print(f"이전 최고: score={best_so_far['score']:.4f}  "
              f"ret={bs['total_return']:+.1f}%  sharpe={bs['sharpe']:.3f}  "
              f"MDD={bs['max_drawdown']:.1f}%  symbols={best_so_far.get('symbols', '?')}")
    print()

    for i in range(args.trials):
        trial_num = trial_offset + i + 1
        p = sample_params(base_params)
        symbols = p["symbols"]

        # 해당 종목 데이터 로더
        loader = DataLoader(
            symbols=symbols,
            timeframes=base_params["timeframes"],
            primary_tf=base_params.get("primary_timeframe", "1h"),
            cache_dir=data_cfg.get("cache_dir", "data/cache"),
            lookback=data_cfg.get("lookback_bars", 300),
        )

        try:
            engine  = build_engine(p, initial_capital)
            snaps   = loader.iterate(since=since, until=until)
            report  = engine.run(snaps)
            sc_val  = score_report(report, args.metric)

            result = {
                "trial":   trial_num,
                "score":   sc_val,
                "metric":  args.metric,
                "symbols": symbols,
                "summary": {
                    "total_return":  report.total_return_pct,
                    "sharpe":        report.sharpe,
                    "calmar":        report.calmar,
                    "max_drawdown":  report.max_drawdown,
                    "trade_count":   report.total_trades,
                    "win_rate":      report.win_rate,
                    "profit_factor": report.profit_factor,
                },
                "params_diff": _extract_diff(base_params, p),
            }
            results.append(result)
            save_results(results)

            syms_short = "+".join(s.replace("USDT","") for s in symbols)
            print(
                f"[{trial_num:>3}] score={sc_val:+.4f}  "
                f"ret={report.total_return_pct:+.1f}%  "
                f"sharpe={report.sharpe:.3f}  "
                f"MDD={report.max_drawdown:.1f}%  "
                f"trades={report.total_trades}  "
                f"[{syms_short}]"
            )

        except Exception as ex:
            print(f"[{trial_num:>3}] ERROR ({'+'.join(symbols)}): {ex}")
            continue

    if not results:
        print("결과 없음")
        return

    best = max(results, key=lambda r: r["score"])
    top5 = sorted(results, key=lambda r: r["score"], reverse=True)[:5]

    print("\n" + "=" * 70)
    print(f"  V2 확장 최적 결과 (Trial #{best['trial']})")
    print("=" * 70)
    bs = best["summary"]
    print(f"  종목:          {best.get('symbols', '?')}")
    print(f"  총 수익률:     {bs['total_return']:+.2f}%")
    print(f"  Sharpe:        {bs['sharpe']:.3f}")
    print(f"  Calmar:        {bs['calmar']:.3f}")
    print(f"  Max DD:        {bs['max_drawdown']:.1f}%")
    print(f"  거래 수:       {bs['trade_count']}")
    print(f"  승률:          {bs['win_rate']:.1f}%")
    print(f"  Profit Factor: {bs['profit_factor']:.3f}")
    print("\n  변경된 파라미터:")
    for k, (orig, new) in best["params_diff"].items():
        print(f"    {k}: {orig} → {new}")
    print()
    print("  Top-5:")
    for r in top5:
        rs = r["summary"]
        syms = "+".join(s.replace("USDT","") for s in r.get("symbols", []))
        print(f"    #{r['trial']:>3}  score={r['score']:+.4f}  "
              f"ret={rs['total_return']:+.1f}%  "
              f"sharpe={rs['sharpe']:.3f}  MDD={rs['max_drawdown']:.1f}%  [{syms}]")

    if args.apply_best:
        _apply_best_params(args.params, base_params, best["params_diff"])
        print(f"\n최적 파라미터가 {args.params}에 저장되었습니다.")


if __name__ == "__main__":
    main()
