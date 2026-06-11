"""백테 실행 + 풀 요약 출력 + 일별 equity parquet 저장.

사용: python3 scripts/run_bt_save.py <config.yaml> <out_name>
출력: data/results/<out_name>_daily.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from data.loader import DataLoader
from scripts.run_backtest import build_engine

cfg_path, out_name = sys.argv[1], sys.argv[2]
with open(cfg_path) as f:
    p = yaml.safe_load(f)
bt = p["backtest"]
loader = DataLoader(
    symbols=p["symbols"], timeframes=p["timeframes"],
    primary_tf=p.get("primary_timeframe", "1h"),
    cache_dir=p["data"].get("cache_dir", "data/cache"),
    lookback=p["data"].get("lookback_bars", 300),
)
engine = build_engine(p, bt["initial_capital"], abort_mdd=None)
snaps = loader.iterate(since=pd.Timestamp(bt["start"], tz="UTC"),
                       until=pd.Timestamp(bt["end"], tz="UTC"))
report = engine.run(snaps)
report.print_summary()
eq = engine.equity_curve.to_series().resample("1D").last()
eq.to_frame("eq").to_parquet(f"data/results/{out_name}_daily.parquet")
print(f"saved data/results/{out_name}_daily.parquet rows={len(eq)}")
