"""연속 손절 이후 equity 회복 분석."""
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

df = engine.ledger.to_dataframe().sort_values("exit_time").reset_index(drop=True)
initial = bt.get("initial_capital", 100)

# 거래별 누적 equity
df["cum_pnl"] = df["pnl"].cumsum()
df["equity"] = initial + df["cum_pnl"]

# 연속 손절 구간 찾기 (min 4연속)
streaks = []
cur_len, cur_start = 0, None
for i, row in df.iterrows():
    if row["pnl"] <= 0:
        if cur_len == 0:
            cur_start = i
        cur_len += 1
    else:
        if cur_len >= 4:
            streaks.append((cur_len, cur_start, i - 1))
        cur_len = 0
if cur_len >= 4:
    streaks.append((cur_len, cur_start, len(df) - 1))

streaks.sort(reverse=True)

print("── 연속 손절 구간별 회복 분석 ───────────────────────────────────────────\n")

for rank, (length, s, e) in enumerate(streaks, 1):
    seg = df.iloc[s:e+1]
    streak_start_eq = df.iloc[s]["equity"] - df.iloc[s]["pnl"]  # 첫 손절 직전 equity (= peak 근사)
    streak_end_eq   = df.iloc[e]["equity"]                       # 6연속 손절 끝 equity
    total_loss = streak_end_eq - streak_start_eq
    dd_pct = total_loss / streak_start_eq * 100

    # 회복: streak 이후 처음으로 streak_start_eq 초과한 거래
    post = df.iloc[e+1:] if e+1 < len(df) else pd.DataFrame()
    recovered = post[post["equity"] >= streak_start_eq]

    streak_end_date = seg.iloc[-1]["exit_time"]
    streak_start_date = seg.iloc[0]["exit_time"]

    if not recovered.empty:
        rec_row = recovered.iloc[0]
        rec_date = rec_row["exit_time"]
        days = (rec_date - streak_end_date).days
        trades_to_recover = recovered.index[0] - e
        recovered_flag = f"✅ 회복 완료"
        rec_info = f"{rec_date.date()} (손절 종료 후 {days}일, {trades_to_recover}거래)"
    else:
        recovered_flag = "⏳ 미회복 (백테스트 종료)"
        rec_info = "—"
        days = None

    print(f"  #{rank}  {length}연속 손절  {streak_start_date.date()} ~ {streak_end_date.date()}  {recovered_flag}")
    print(f"       손절 전 equity:  ${streak_start_eq:>12,.2f}")
    print(f"       손절 후 equity:  ${streak_end_eq:>12,.2f}  ({dd_pct:+.1f}%)")
    print(f"       총 손실:         ${total_loss:>12,.2f}")
    print(f"       회복 시점:       {rec_info}")

    # 회복까지의 거래 상세 (최대 10건)
    if not post.empty:
        show = post.iloc[:min(trades_to_recover if not recovered.empty else 10, 15)]
        print(f"       ── 회복 경로 ({len(show)}건 표시) ──")
        for _, r in show.iterrows():
            marker = "★" if r["equity"] >= streak_start_eq else " "
            print(f"       {marker} {str(r['exit_time'])[:10]}  {r['symbol']:>8}  {r['direction']:>5}  ${r['pnl']:>10,.2f}  → equity ${r['equity']:>12,.2f}")
    print()
