"""최대 연속 손절 구간 찾기."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml
from data.loader import DataLoader
from scripts.run_backtest import build_engine


with open("config/final_v13_eth.yaml") as f:
    p = yaml.safe_load(f)

bt = p.get("backtest", {})
loader = DataLoader(
    symbols=p["symbols"], timeframes=p["timeframes"],
    primary_tf=p.get("primary_timeframe", "1h"),
    cache_dir=p.get("data", {}).get("cache_dir", "data/cache"),
    lookback=p.get("data", {}).get("lookback_bars", 300),
)
engine = build_engine(p, bt.get("initial_capital", 100))
engine.run(loader.iterate(
    since=pd.Timestamp(bt.get("start", "2022-01-01"), tz="UTC"),
    until=pd.Timestamp(bt.get("end"), tz="UTC") if bt.get("end") else None,
))

df = engine.ledger.to_dataframe()
df = df.sort_values("exit_time").reset_index(drop=True)
df["win"] = df["pnl"] > 0

# 연속 손절 구간 탐색
streaks = []
cur_len, cur_start = 0, None
for i, row in df.iterrows():
    if not row["win"]:
        if cur_len == 0:
            cur_start = i
        cur_len += 1
    else:
        if cur_len > 0:
            streaks.append((cur_len, cur_start, i - 1))
        cur_len = 0

if cur_len > 0:
    streaks.append((cur_len, cur_start, len(df) - 1))

streaks.sort(reverse=True)

print(f"전체 거래: {len(df)}건  손절: {(~df['win']).sum()}건  승률: {df['win'].mean()*100:.1f}%\n")
print("── 연속 손절 Top 5 ──────────────────────────────────────────────────")
for rank, (length, s, e) in enumerate(streaks[:5], 1):
    seg = df.iloc[s:e+1]
    total_loss = seg["pnl"].sum()
    eq_before = df.iloc[:s]["pnl"].sum() + bt.get("initial_capital", 100)
    dd_pct = total_loss / eq_before * 100
    print(f"\n  #{rank}  연속 {length}회 손절  ({seg.iloc[0]['exit_time'].date()} ~ {seg.iloc[-1]['exit_time'].date()})")
    print(f"       총손실: ${total_loss:,.2f}  진입당시 equity 대비 {dd_pct:.1f}%")
    print(f"       {'번호':>3} {'exit_time':^22} {'symbol':^10} {'dir':^6} {'strategy':^20} {'pnl':>10}")
    for _, r in seg.iterrows():
        print(f"       {r.name:>3} {str(r['exit_time'])[:19]:^22} {r['symbol']:^10} {r['direction']:^6} {r['strategy']:^20} ${r['pnl']:>9,.2f}")
