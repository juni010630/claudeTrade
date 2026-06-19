"""[연구·screw_sweep] v20 기준 단일 variant 풀 백테 (프로덕션 build_engine 패리티).

v20 config에 override(JSON)를 deep-merge 후 그대로 build_engine→run.
override는 yaml을 직접 고치는 것과 동치 → 백테=라이브 패리티/look-ahead 불변 보장.

사용: python3 research/screw_sweep/run_variant.py <out_name> '<override_json>'
  예) capital_fraction 스윕:
      run_variant.py alloc_60_20_20 '{"strategy_capital_fraction":{"ema_cross":0.6,"multi_tf_breakout":0.6,"mean_reversion":0.2,"macross_d":0.2}}'
  예) DVOL 끄기:
      run_variant.py dvol_off '{"dvol_perbook":{"enabled":false}}'
  예) per-book 타깃:
      run_variant.py dvol_t35_t50 '{"dvol_perbook":{"targets":{"ema_cross":35,"multi_tf_breakout":35,"mean_reversion":50,"macross_d":50}}}'
  예) rpt:
      run_variant.py rpt_009 '{"risk":{"risk_per_trade":0.09}}'
출력: data/results/sw_<out_name>_daily.parquet  (+ stdout 풀 요약)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import yaml

from data.loader import DataLoader
from scripts.run_backtest import build_engine

BASE_CFG = "config/final_v20_dvol_perbook.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    """override를 base에 재귀 병합 (dict는 깊게, 그 외는 교체)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def main() -> None:
    out_name = sys.argv[1]
    override = json.loads(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else {}

    p = yaml.safe_load(open(BASE_CFG))
    deep_merge(p, override)

    bt = p["backtest"]
    loader = DataLoader(
        symbols=p["symbols"], timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=p["data"].get("cache_dir", "data/cache"),
        lookback=p["data"].get("lookback_bars", 300),
    )
    engine = build_engine(p, bt["initial_capital"], abort_mdd=None)
    print(f"[variant {out_name}] override={json.dumps(override)}", flush=True)
    snaps = loader.iterate(since=pd.Timestamp(bt["start"], tz="UTC"),
                           until=pd.Timestamp(bt["end"], tz="UTC"))
    report = engine.run(snaps)
    report.print_summary()
    eq = engine.equity_curve.to_series().resample("1D").last()
    out = f"data/results/sw_{out_name}_daily.parquet"
    eq.to_frame("eq").to_parquet(out)
    print(f"saved {out} rows={len(eq)}", flush=True)


if __name__ == "__main__":
    main()
