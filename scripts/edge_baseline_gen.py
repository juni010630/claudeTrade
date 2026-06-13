"""엣지부패 모니터 베이스라인 생성 (로컬 1회 실행).

v17 전구간 백테 equity curve(1h)에서 롤링 30d/90d 수익률 분포를 추출해
percentile 그리드(p0~p100)를 config/edge_baseline_v17.json에 저장.
edge_monitor.py가 라이브 롤링 수익률의 백분위를 이 그리드로 보간 산출.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

from data.loader import DataLoader
import scripts.run_backtest as rb

CONFIG = sys.argv[1] if len(sys.argv) > 1 else "config/final_v17.yaml"
OUT = sys.argv[2] if len(sys.argv) > 2 else "config/edge_baseline_v17.json"
WINDOWS = {"30d": 720, "90d": 2160}  # 1h 봉 수

p = yaml.safe_load(open(CONFIG))
bt = p.get("backtest", {})
loader = DataLoader(
    symbols=p["symbols"], timeframes=p["timeframes"],
    primary_tf=p.get("primary_timeframe", "1h"),
    cache_dir=p.get("data", {}).get("cache_dir", "data/cache"),
    lookback=p.get("data", {}).get("lookback_bars", 300),
)
engine = rb.build_engine(p, 100.0)
since = pd.Timestamp(bt.get("start", "2022-01-01"), tz="UTC")
until = pd.Timestamp(bt["end"], tz="UTC") if bt.get("end") else None
print(f"전구간 백테 실행 ({since.date()} ~ {until.date() if until is not None else '최신'})…")
report = engine.run(loader.iterate(since=since, until=until))
eq = engine.equity_curve.to_series()
print(f"equity curve {len(eq)}봉, 최종 ${eq.iloc[-1]:,.0f}")

grids = {}
pctl = list(range(101))
for name, w in WINDOWS.items():
    rets = (eq / eq.shift(w) - 1).dropna()
    grids[f"ret_{name}"] = {
        "percentiles": pctl,
        "values": [float(v) for v in np.percentile(rets.values, pctl)],
        "n_windows": int(len(rets)),
    }
    print(f"ret_{name}: n={len(rets)}, p5={grids[f'ret_{name}']['values'][5]:+.3f}, "
          f"p50={grids[f'ret_{name}']['values'][50]:+.3f}")

out = {
    "config": CONFIG,
    "period": [str(since.date()), str(until.date()) if until is not None else None],
    "final_equity": float(eq.iloc[-1]),
    "grids": grids,
}
Path(OUT).write_text(json.dumps(out, indent=2))
print(f"저장: {OUT}")
