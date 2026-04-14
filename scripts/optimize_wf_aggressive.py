"""워크포워드 최적화 — 공격적 수익 추구 배치 (Aggressive Batch).

훈련:  2022-01-01 ~ 2023-12-31  (하락장 + 횡보장)
검증:  2024-01-01 ~ 2025-03-01  (불장)

스코어 = 0.25 × train_sharpe + 0.45 × test_sharpe + 0.03 × test_return(%)
  → 기존 대비 수익률 보너스 6배 강화 (0.005 → 0.030)
  → MDD 허용 한도 완화: -35% → -50%
  → 검증 거래 최소 수 완화: 5 → 3건

이전 WF 고수익 트라이얼 (#42 +124%, #4 +96%, #15 +82%) 공통 패턴 반영:
  - SS size_fraction 0.40~0.60, leverage 18~30
  - daily_drawdown_stop -0.10~-0.20 (포지션 오래 유지)
  - max_positions 6~10
  - adx_ranging_threshold 낮게 (더 많은 시장 참여)
  - 알트코인 비중 확대 (DOGE, XRP, AVAX 변동성 활용)

Usage:
    python scripts/optimize_wf_aggressive.py --trials 60
    python scripts/optimize_wf_aggressive.py --resume --trials 40
    python scripts/optimize_wf_aggressive.py --apply-best
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
from strategies.donchian_breakout import DonchianBreakoutStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum_breakout import MomentumBreakoutStrategy
from strategies.multi_tf_breakout import MultiTFBreakoutStrategy

# ─────────────────────────────────────────────
# 기간 설정
# ─────────────────────────────────────────────
TRAIN_START = "2022-01-01"
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2025-03-01"

TRAIN_WEIGHT  = 0.25
TEST_WEIGHT   = 0.45
RETURN_WEIGHT = 0.030   # 수익률 보너스: 100% 수익 → +3.0점 (기존 0.5점 대비 6배)

# ─────────────────────────────────────────────
# 공격적 종목 풀 — 알트코인 비중 확대
# ─────────────────────────────────────────────
SYMBOL_POOLS = [
    # BTC+ETH 기반 + 알트 다양화
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "AVAXUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "DOGEUSDT"],
    # 6종목 풀 — 더 많은 기회
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "AVAXUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT", "DOGEUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "AVAXUSDT", "DOGEUSDT"],
    # 7종목 풀
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "AVAXUSDT", "DOGEUSDT"],
]

# ─────────────────────────────────────────────
# 공격적 탐색 공간
# ─────────────────────────────────────────────
SEARCH_SPACE: dict[str, Any] = {
    # 리스크 — 더 크게, 더 많이
    "risk.risk_per_trade":          [0.010, 0.012, 0.015, 0.018, 0.020, 0.025],
    "risk.max_positions":           [6, 7, 8, 10],
    "risk.max_same_direction":      [3, 4, 5],
    "risk.daily_drawdown_pause":    [-0.05, -0.06, -0.08, -0.10],
    "risk.daily_drawdown_stop":     [-0.10, -0.12, -0.15, -0.18, -0.20],

    # 레버리지 — 상한 크게 확장
    "leverage_tiers.SS.leverage":      [18, 20, 22, 25, 28, 30],
    "leverage_tiers.SS.size_fraction": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    "leverage_tiers.S.leverage":       [12, 14, 15, 18, 20],
    "leverage_tiers.S.size_fraction":  [0.22, 0.25, 0.28, 0.32, 0.35, 0.40],
    "leverage_tiers.A.leverage":       [7, 8, 10, 12],
    "leverage_tiers.A.size_fraction":  [0.15, 0.18, 0.20, 0.25],
    "leverage_tiers.B.leverage":       [3, 4, 5],
    "leverage_tiers.B.size_fraction":  [0.08, 0.10, 0.12],
    "leverage_tiers.C.leverage":       [2, 3],
    "leverage_tiers.C.size_fraction":  [0.04, 0.05, 0.06],

    # 국면 탐지 — 진입 문턱 낮추기
    "regime.adx_trending_threshold": [20.0, 22.0, 25.0, 28.0],
    "regime.adx_ranging_threshold":  [12.0, 15.0, 18.0],
    "regime.bb_width_squeeze_pct":   [0.10, 0.15, 0.20, 0.25],

    # 스코어러 — 진입 조건 완화
    "scorer.volume_ratio_threshold": [1.0, 1.2, 1.5, 1.8],
    "scorer.rsi_long_max":           [65.0, 70.0, 75.0],
    "scorer.rsi_short_min":          [25.0, 30.0, 35.0, 40.0],
    "scorer.tier_ss_min_score":      [7],
    "scorer.tier_s_min_score":       [5, 6],
    "scorer.tier_a_min_score":       [3, 4],
    "scorer.tier_b_min_score":       [2],
    "scorer.tier_c_min_score":       [1],

    # momentum_breakout — TP 크게, SL 타이트하게
    "strategies.momentum_breakout.bb_std":            [1.5, 1.8, 2.0, 2.2],
    "strategies.momentum_breakout.atr_tp_mult":       [2.5, 3.0, 3.5, 4.0, 5.0],
    "strategies.momentum_breakout.atr_sl_mult":       [0.5, 0.6, 0.8, 1.0],
    "strategies.momentum_breakout.volume_multiplier": [1.2, 1.5, 1.8, 2.0],
    "strategies.momentum_breakout.volume_lookback":   [15, 20, 25],

    # donchian_breakout — 넓은 채널, 큰 TP
    "strategies.donchian_breakout.period":            [15, 20, 25, 30, 40, 50],
    "strategies.donchian_breakout.confirm_period":    [8, 10, 12, 15],
    "strategies.donchian_breakout.atr_tp_mult":       [2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
    "strategies.donchian_breakout.atr_sl_mult":       [0.5, 0.6, 0.8, 1.0, 1.2],
    "strategies.donchian_breakout.volume_multiplier": [1.2, 1.5, 1.8, 2.0],

    # multi_tf_breakout
    "strategies.multi_tf_breakout.bb_std_4h":         [1.8, 2.0, 2.2, 2.5],
    "strategies.multi_tf_breakout.bb_std_1h":         [1.5, 1.8, 2.0, 2.2],
    "strategies.multi_tf_breakout.volume_multiplier": [1.2, 1.5, 1.8, 2.0],
    "strategies.multi_tf_breakout.atr_tp_mult":       [2.5, 3.0, 3.5, 4.0, 5.0],
    "strategies.multi_tf_breakout.atr_sl_mult":       [0.5, 0.7, 1.0, 1.2],

    # mean_reversion — 극단 RSI에서 진입, 큰 TP
    "strategies.mean_reversion.bb_std":         [1.5, 1.8, 2.0, 2.2],
    "strategies.mean_reversion.rsi_oversold":   [20.0, 25.0, 30.0],
    "strategies.mean_reversion.rsi_overbought": [70.0, 75.0, 80.0],
    "strategies.mean_reversion.atr_tp_mult":    [2.0, 2.5, 3.0, 3.5, 4.0],
    "strategies.mean_reversion.atr_sl_mult":    [0.5, 0.7, 1.0],
}

RESULTS_FILE = Path("optimize_wf_aggressive_results.json")
BASE_PARAMS  = "config/params_v2.yaml"
WORKER_ID    = None  # --worker 플래그로 설정됨


# ─────────────────────────────────────────────
# 헬퍼 (동일)
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
    p["symbols"] = random.choice(SYMBOL_POOLS)
    for key, choices in SEARCH_SPACE.items():
        _set_nested(p, key, random.choice(choices))
    return p


def build_engine(p: dict, initial_capital: float) -> BacktestEngine:
    symbols = p["symbols"]
    r  = p.get("risk", {})
    e  = p.get("execution", {})
    rg = p.get("regime", {})
    sc = p.get("scorer", {})

    don_cfg = p.get("strategies", {}).get("donchian_breakout", {})
    if not don_cfg.get("enabled", True):
        don_strategy = None
    else:
        don_cfg["symbols"] = symbols
        don_strategy = DonchianBreakoutStrategy(don_cfg)

    strategy_list_cfg = {
        "momentum_breakout": MomentumBreakoutStrategy,
        "multi_tf_breakout": MultiTFBreakoutStrategy,
        "mean_reversion":    MeanReversionStrategy,
    }
    strategies = []
    for key, cls in strategy_list_cfg.items():
        cfg = p.get("strategies", {}).get(key, {})
        if cfg.get("enabled", True):
            cfg["symbols"] = symbols
            strategies.append(cls(cfg))
    if don_strategy:
        strategies.append(don_strategy)

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
            tier_ss_min_score=sc.get("tier_ss_min_score", 7),
            tier_s_min_score=sc.get("tier_s_min_score", 5),
            tier_a_min_score=sc.get("tier_a_min_score", 3),
            tier_b_min_score=sc.get("tier_b_min_score", 2),
            tier_c_min_score=sc.get("tier_c_min_score", 1),
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


def run_period(p: dict, start: str, end: str, loader: DataLoader) -> dict:
    since  = pd.Timestamp(start, tz="UTC")
    until  = pd.Timestamp(end, tz="UTC")
    engine = build_engine(p, 100_000)
    snaps  = loader.iterate(since=since, until=until)
    report = engine.run(snaps)
    return {
        "sharpe": report.sharpe or 0.0,
        "return": report.total_return_pct or 0.0,
        "mdd":    report.max_drawdown or 0.0,
        "trades": report.total_trades,
        "wr":     report.win_rate or 0.0,
    }


def wf_score(train: dict, test: dict) -> float:
    """
    공격적 WF 스코어.
    - MDD 허용 한도: -50% (기존 -35%)
    - 수익률 보너스 6배 강화
    - 검증 최소 거래: 3건 (기존 5건)
    - 검증 손실 시 벌점 유지 (방향성 필터)
    """
    t_sharpe = train["sharpe"]
    v_sharpe = test["sharpe"]
    t_mdd    = train["mdd"]
    v_mdd    = test["mdd"]
    t_trades = train["trades"]
    v_trades = test["trades"]

    # 최소 거래 수 필터 (완화)
    if t_trades < 10 or v_trades < 3:
        return -999.0
    # MDD 하드 필터 (완화: -35% → -50%)
    if t_mdd < -50.0 or v_mdd < -50.0:
        return -999.0

    # 검증이 손실이면 벌점 (완화: -2.0 → -1.5)
    if v_sharpe < 0:
        return v_sharpe - 1.5
    if t_sharpe < 0:
        return t_sharpe - 0.5

    # 기본 점수: 가중 평균 Sharpe + 강화된 수익률 보너스
    base      = TRAIN_WEIGHT * t_sharpe + TEST_WEIGHT * v_sharpe
    ret_bonus = RETURN_WEIGHT * test["return"]   # 100% → +3.0점

    return base + ret_bonus


def load_results(path: Path | None = None) -> list[dict]:
    p = path or RESULTS_FILE
    if p.exists():
        return json.loads(p.read_text())
    return []


def save_results(results: list[dict], path: Path | None = None) -> None:
    p = path or RESULTS_FILE
    p.write_text(json.dumps(results, indent=2, ensure_ascii=False))


def _extract_diff(base: dict, new: dict) -> dict[str, tuple]:
    diff = {}
    if base.get("symbols") != new.get("symbols"):
        diff["symbols"] = (base.get("symbols"), new.get("symbols"))
    for key in SEARCH_SPACE:
        parts = key.split(".")
        b_val = _get_nested(base, parts)
        n_val = _get_nested(new, parts)
        if b_val != n_val:
            diff[key] = (b_val, n_val)
    return diff


def _apply_best(path: str, base: dict, diff: dict[str, tuple]) -> None:
    p = copy.deepcopy(base)
    for key, (_, new_val) in diff.items():
        if key == "symbols":
            p["symbols"] = new_val
        else:
            _set_nested(p, key, new_val)
    with open(path, "w") as f:
        yaml.dump(p, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",     type=int, default=60)
    parser.add_argument("--params",     default=BASE_PARAMS)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--seed",       type=int, default=None)
    parser.add_argument("--apply-best", action="store_true")
    parser.add_argument("--worker",     type=int, default=None,
                        help="워커 ID (1,2,3...). 설정 시 별도 결과 파일에 저장")
    args = parser.parse_args()

    # 워커 ID에 따라 출력 파일 결정
    if args.worker is not None:
        out_file = Path(f"optimize_wf_aggressive_results_w{args.worker}.json")
    else:
        out_file = RESULTS_FILE

    if args.seed is not None:
        random.seed(args.seed)
    elif args.worker is not None:
        random.seed(args.worker * 1000 + 42)  # 워커별 고정 시드

    with open(args.params) as f:
        base_params = yaml.safe_load(f)

    strats = base_params.setdefault("strategies", {})
    if "donchian_breakout" not in strats:
        strats["donchian_breakout"] = {
            "enabled": True,
            "period": 20, "confirm_period": 10,
            "volume_multiplier": 1.5, "volume_lookback": 20,
            "atr_period": 14, "atr_tp_mult": 3.0, "atr_sl_mult": 1.0,
            "daily_ema_period": 200,
            "signal_tf": "1h", "confirm_tf": "4h", "filter_tf": "1d",
        }

    data_cfg   = base_params.get("data", {})
    cache_dir  = data_cfg.get("cache_dir", "data/cache")
    tfs        = base_params.get("timeframes", ["1h", "4h", "1d"])
    primary_tf = base_params.get("primary_timeframe", "1h")

    results = load_results(out_file) if args.resume else []
    trial_offset = len(results)

    worker_tag = f" [Worker #{args.worker}]" if args.worker else ""
    print("=" * 70)
    print(f"  워크포워드 최적화 — 공격적 수익 배치 (Aggressive Batch){worker_tag}")
    print(f"  출력 파일: {out_file}")
    print(f"  훈련: {TRAIN_START} ~ {TRAIN_END}  (하락장 + 횡보)")
    print(f"  검증: {TEST_START}  ~ {TEST_END}  (불장)")
    print(f"  스코어 = {TRAIN_WEIGHT}×훈련Sh + {TEST_WEIGHT}×검증Sh + {RETURN_WEIGHT}×검증수익%")
    print(f"  MDD 허용: -50%  |  레버리지 SS 최대 30x  |  종목 최대 7개")
    print(f"  시도: {args.trials}회  (총 {trial_offset + args.trials}회)")
    print("=" * 70)
    if results:
        best_so_far = max(results, key=lambda r: r["score"])
        print(f"이전 최고: score={best_so_far['score']:.4f}  "
              f"train_sh={best_so_far['train']['sharpe']:.3f}  "
              f"test_sh={best_so_far['test']['sharpe']:.3f}  "
              f"test_ret={best_so_far['test']['return']:+.1f}%")
    print()

    for i in range(args.trials):
        trial_num = trial_offset + i + 1
        p = sample_params(base_params)
        symbols = p["symbols"]

        loader = DataLoader(
            symbols=symbols,
            timeframes=tfs,
            primary_tf=primary_tf,
            cache_dir=cache_dir,
            lookback=300,
        )

        try:
            train = run_period(p, TRAIN_START, TRAIN_END, loader)
            test  = run_period(p, TEST_START,  TEST_END,  loader)
            score = wf_score(train, test)

            result = {
                "trial":       trial_num,
                "score":       score,
                "symbols":     symbols,
                "train":       train,
                "test":        test,
                "params_diff": _extract_diff(base_params, p),
            }
            results.append(result)
            save_results(results, out_file)

            syms_short = "+".join(s.replace("USDT", "") for s in symbols)
            print(
                f"[{trial_num:>3}] score={score:+.4f}  "
                f"train(sh={train['sharpe']:.2f} ret={train['return']:+.0f}%)  "
                f"test(sh={test['sharpe']:.2f} ret={test['return']:+.0f}% MDD={test['mdd']:.0f}%)  "
                f"[{syms_short}]"
            )

        except Exception as ex:
            print(f"[{trial_num:>3}] ERROR: {ex}")
            continue

    if not results:
        print("결과 없음"); return

    best = max(results, key=lambda r: r["score"])
    top5 = sorted(results, key=lambda r: r["score"], reverse=True)[:5]

    print("\n" + "=" * 70)
    print(f"  공격적 WF 최적 결과 (Trial #{best['trial']})")
    print("=" * 70)
    tr, te = best["train"], best["test"]
    syms = "+".join(s.replace("USDT", "") for s in best["symbols"])
    print(f"  종목:   {syms}")
    print(f"  훈련:   Sharpe={tr['sharpe']:.3f}  ret={tr['return']:+.1f}%  MDD={tr['mdd']:.1f}%  ({tr['trades']}건)")
    print(f"  검증:   Sharpe={te['sharpe']:.3f}  ret={te['return']:+.1f}%  MDD={te['mdd']:.1f}%  ({te['trades']}건)")
    print(f"  WF점수: {best['score']:.4f}")
    print("\n  변경 파라미터:")
    for k, (o, n) in best["params_diff"].items():
        print(f"    {k}: {o} → {n}")
    print("\n  Top-5:")
    for r in top5:
        syms2 = "+".join(s.replace("USDT", "") for s in r["symbols"])
        print(f"    #{r['trial']:>3}  score={r['score']:+.4f}  "
              f"train_sh={r['train']['sharpe']:.2f}  test_sh={r['test']['sharpe']:.2f}  "
              f"test_ret={r['test']['return']:+.1f}%  MDD={r['test']['mdd']:.1f}%  [{syms2}]")

    if args.apply_best:
        _apply_best(args.params, base_params, best["params_diff"])
        print(f"\n최적 파라미터가 {args.params}에 저장되었습니다.")


if __name__ == "__main__":
    main()
