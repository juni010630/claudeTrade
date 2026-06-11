"""v17 컴포넌트 분리 엔진 백테 — 추세-only / 슬리브-only 일별 equity 추출.

config 파일 무수정: final_v17.yaml을 메모리에서 변형 (capital_fraction 1.0, 딥플로어 off).
사용: python3 scripts/newedge_component_run.py {trend|sleeve}
출력: data/results/comp_{trend|sleeve}_daily.parquet (일별 equity)
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from data.loader import DataLoader
from scripts.run_backtest import build_engine

MODE = sys.argv[1]

with open("config/final_v17.yaml") as f:
    p = yaml.safe_load(f)
p = copy.deepcopy(p)
p["risk"]["deep_floor_dd"] = None  # 컴포넌트 standalone은 풀사이즈 — 조기중단 방지

if MODE == "trend":
    p["strategies"]["mean_reversion"]["enabled"] = False
    p["strategy_capital_fraction"] = {"ema_cross": 1.0, "multi_tf_breakout": 1.0}
elif MODE == "sleeve":
    p["strategies"]["ema_cross"]["enabled"] = False
    p["strategies"]["multi_tf_breakout"]["enabled"] = False
    p["strategy_capital_fraction"] = {"mean_reversion": 1.0}
else:
    raise SystemExit("mode: trend|sleeve")

bt = p["backtest"]
loader = DataLoader(
    symbols=p["symbols"],
    timeframes=p["timeframes"],
    primary_tf=p.get("primary_timeframe", "1h"),
    cache_dir=p["data"].get("cache_dir", "data/cache"),
    lookback=p["data"].get("lookback_bars", 300),
)
engine = build_engine(p, bt["initial_capital"], abort_mdd=None)
print(f"[{MODE}] strategies: {[s.name for s in engine.strategies]}")
snapshots = loader.iterate(since=pd.Timestamp(bt["start"], tz="UTC"),
                           until=pd.Timestamp(bt["end"], tz="UTC"))
report = engine.run(snapshots)
report.print_summary()

eq = engine.equity_curve.to_series().resample("1D").last()
eq.to_frame("eq").to_parquet(f"data/results/comp_{MODE}_daily.parquet")
print(f"saved data/results/comp_{MODE}_daily.parquet rows={len(eq)}")
