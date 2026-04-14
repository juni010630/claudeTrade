"""워크포워드 최적화 — 오버피팅 방지 핵심 도구.

훈련:  2022-01-01 ~ 2023-12-31  (하락장 + 횡보장)
검증:  2024-01-01 ~ 2025-03-01  (불장)

스코어 = 0.4 × train_sharpe + 0.6 × test_sharpe
  → 검증 성과를 더 중시. 훈련 과적합 시 검증이 낮아져 벌점.
  → 둘 다 양수여야 높은 점수.

전략: momentum_breakout + multi_tf_breakout + donchian_breakout + mean_reversion

Usage:
    python scripts/optimize_walkforward.py --trials 50
    python scripts/optimize_walkforward.py --resume --trials 30
    python scripts/optimize_walkforward.py --apply-best
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

TRAIN_WEIGHT = 0.4
TEST_WEIGHT  = 0.6

# ─────────────────────────────────────────────
# 종목 풀
# ─────────────────────────────────────────────
SYMBOL_POOLS = [
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"],
    ["BTCUSDT", "ETHUSDT", "BNBUSDT", "AVAXUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "AVAXUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "DOGEUSDT"],
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "AVAXUSDT"],
]

# ─────────────────────────────────────────────
# 탐색 공간
# ─────────────────────────────────────────────
SEARCH_SPACE: dict[str, Any] = {
    # 리스크
    "risk.risk_per_trade":          [0.006, 0.008, 0.01, 0.012, 0.015],
    "risk.max_positions":           [4, 5, 6],
    "risk.max_same_direction":      [2, 3, 4],
    "risk.daily_drawdown_pause":    [-0.04, -0.05, -0.06],
    "risk.daily_drawdown_stop":     [-0.08, -0.10, -0.12],

    # 레버리지 티어 — 5단계 세분화 (워크포워드: 과도한 레버리지 억제)
    "leverage_tiers.SS.leverage":      [12, 15, 18, 20],
    "leverage_tiers.SS.size_fraction": [0.25, 0.30, 0.35, 0.40],
    "leverage_tiers.S.leverage":       [8, 10, 12],
    "leverage_tiers.S.size_fraction":  [0.18, 0.22, 0.28],
    "leverage_tiers.A.leverage":       [4, 5, 7],
    "leverage_tiers.A.size_fraction":  [0.10, 0.15, 0.20],
    "leverage_tiers.B.leverage":       [2, 3, 4],
    "leverage_tiers.B.size_fraction":  [0.06, 0.08, 0.10],
    "leverage_tiers.C.leverage":       [1, 2],
    "leverage_tiers.C.size_fraction":  [0.03, 0.05],

    # 국면 탐지
    "regime.adx_trending_threshold": [22.0, 25.0, 28.0, 30.0],
    "regime.adx_ranging_threshold":  [15.0, 18.0, 20.0],
    "regime.bb_width_squeeze_pct":   [0.15, 0.20, 0.25, 0.30],

    # 스코어러
    "scorer.volume_ratio_threshold": [1.0, 1.2, 1.5, 1.8],
    "scorer.rsi_long_max":           [60.0, 65.0, 70.0],
    "scorer.rsi_short_min":          [30.0, 35.0, 40.0],
    "scorer.tier_ss_min_score":      [7],           # 완벽 신호만 (고정)
    "scorer.tier_s_min_score":       [5, 6],
    "scorer.tier_a_min_score":       [3, 4],
    "scorer.tier_b_min_score":       [2],
    "scorer.tier_c_min_score":       [1],

    # momentum_breakout
    "strategies.momentum_breakout.bb_std":            [1.5, 1.8, 2.0, 2.2],
    "strategies.momentum_breakout.atr_tp_mult":       [2.0, 2.5, 3.0, 3.5],
    "strategies.momentum_breakout.atr_sl_mult":       [0.6, 0.8, 1.0, 1.2],
    "strategies.momentum_breakout.volume_multiplier": [1.2, 1.5, 1.8, 2.0],
    "strategies.momentum_breakout.volume_lookback":   [15, 20, 25],

    # donchian_breakout
    "strategies.donchian_breakout.period":            [15, 20, 25, 30, 40],
    "strategies.donchian_breakout.confirm_period":    [8, 10, 12, 15],
    "strategies.donchian_breakout.atr_tp_mult":       [2.0, 2.5, 3.0, 3.5, 4.0],
    "strategies.donchian_breakout.atr_sl_mult":       [0.6, 0.8, 1.0, 1.2],
    "strategies.donchian_breakout.volume_multiplier": [1.2, 1.5, 1.8, 2.0],

    # multi_tf_breakout
    "strategies.multi_tf_breakout.bb_std_4h":         [1.8, 2.0, 2.2, 2.5],
    "strategies.multi_tf_breakout.bb_std_1h":         [1.5, 1.8, 2.0, 2.2],
    "strategies.multi_tf_breakout.volume_multiplier": [1.2, 1.5, 1.8, 2.0],
    "strategies.multi_tf_breakout.atr_tp_mult":       [2.0, 2.5, 3.0, 3.5],
    "strategies.multi_tf_breakout.atr_sl_mult":       [0.7, 1.0, 1.2],

    # mean_reversion
    "strategies.mean_reversion.bb_std":         [1.5, 1.8, 2.0, 2.2],
    "strategies.mean_reversion.rsi_oversold":   [25.0, 30.0, 35.0],
    "strategies.mean_reversion.rsi_overbought": [65.0, 70.0, 75.0],
    "strategies.mean_reversion.atr_tp_mult":    [1.5, 2.0, 2.5, 3.0],
    "strategies.mean_reversion.atr_sl_mult":    [0.5, 0.7, 1.0],
}

RESULTS_FILE = Path("optimize_wf_results.json")
BASE_PARAMS  = "config/params_v2.yaml"


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

    # donchian config (기본값 포함)
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
    워크포워드 복합 스코어.
    - 훈련/검증 모두 Sharpe > 0 이어야 양수 점수
    - 어느 한쪽이라도 손실(-) 이면 강하게 벌점
    - MDD -35% 초과 시 거부
    """
    t_sharpe = train["sharpe"]
    v_sharpe = test["sharpe"]
    t_mdd    = train["mdd"]
    v_mdd    = test["mdd"]
    t_trades = train["trades"]
    v_trades = test["trades"]

    # 최소 거래 수 필터
    if t_trades < 10 or v_trades < 5:
        return -999.0
    # MDD 하드 필터
    if t_mdd < -35.0 or v_mdd < -35.0:
        return -999.0

    # 검증이 손실이면 강력 벌점
    if v_sharpe < 0:
        return v_sharpe - 2.0
    if t_sharpe < 0:
        return t_sharpe - 1.0

    # 기본 점수: 가중 평균 Sharpe
    base = TRAIN_WEIGHT * t_sharpe + TEST_WEIGHT * v_sharpe

    # 수익률 보너스 (검증 기준): 100% 수익 → +0.5점
    ret_bonus = 0.005 * test["return"]

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
    parser.add_argument("--trials",     type=int, default=30)
    parser.add_argument("--params",     default=BASE_PARAMS)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--seed",       type=int, default=None)
    parser.add_argument("--apply-best", action="store_true")
    parser.add_argument("--worker",     type=int, default=None,
                        help="워커 ID. 설정 시 별도 결과 파일에 저장")
    args = parser.parse_args()

    if args.worker is not None:
        out_file = Path(f"optimize_wf_results_w{args.worker}.json")
    else:
        out_file = RESULTS_FILE

    if args.seed is not None:
        random.seed(args.seed)
    elif args.worker is not None:
        random.seed(args.worker * 1000 + 7)

    with open(args.params) as f:
        base_params = yaml.safe_load(f)

    # donchian_breakout 기본 설정이 없으면 추가
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

    print("=" * 65)
    print("  워크포워드 최적화")
    print(f"  훈련: {TRAIN_START} ~ {TRAIN_END}  (하락장 + 횡보)")
    print(f"  검증: {TEST_START}  ~ {TEST_END}  (불장)")
    print(f"  스코어 = {TRAIN_WEIGHT}×훈련Sharpe + {TEST_WEIGHT}×검증Sharpe + 수익률보너스")
    print(f"  전략: momentum + donchian + multi_tf + mean_reversion")
    print(f"  시도: {args.trials}회  (총 {trial_offset + args.trials}회)")
    print("=" * 65)
    if results:
        best_so_far = max(results, key=lambda r: r["score"])
        print(f"이전 최고: score={best_so_far['score']:.4f}  "
              f"train_sharpe={best_so_far['train']['sharpe']:.3f}  "
              f"test_sharpe={best_so_far['test']['sharpe']:.3f}  "
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

            syms_short = "+".join(s.replace("USDT","") for s in symbols)
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

    print("\n" + "=" * 65)
    print(f"  WF 최적 결과 (Trial #{best['trial']})")
    print("=" * 65)
    tr, te = best["train"], best["test"]
    syms = "+".join(s.replace("USDT","") for s in best["symbols"])
    print(f"  종목:   {syms}")
    print(f"  훈련:   Sharpe={tr['sharpe']:.3f}  ret={tr['return']:+.1f}%  MDD={tr['mdd']:.1f}%  ({tr['trades']}건)")
    print(f"  검증:   Sharpe={te['sharpe']:.3f}  ret={te['return']:+.1f}%  MDD={te['mdd']:.1f}%  ({te['trades']}건)")
    print(f"  WF점수: {best['score']:.4f}")
    print("\n  변경 파라미터:")
    for k, (o, n) in best["params_diff"].items():
        print(f"    {k}: {o} → {n}")
    print("\n  Top-5:")
    for r in top5:
        syms2 = "+".join(s.replace("USDT","") for s in r["symbols"])
        print(f"    #{r['trial']:>3}  score={r['score']:+.4f}  "
              f"train_sh={r['train']['sharpe']:.2f}  test_sh={r['test']['sharpe']:.2f}  "
              f"test_ret={r['test']['return']:+.1f}%  [{syms2}]")

    if args.apply_best:
        _apply_best(args.params, base_params, best["params_diff"])
        print(f"\n최적 파라미터가 {args.params}에 저장되었습니다.")


if __name__ == "__main__":
    main()
