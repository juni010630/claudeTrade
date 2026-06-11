"""슬리브(mean_reversion) 심볼 확장 — IS 솔로 스캔.

후보 = 캐시에 2022-01부터 1d+펀딩 풀데이터 있는 미사용 심볼(BTCDOM 제외) + 현역 5개(기준점).
각 심볼을 'v17 설정 그대로, mean_reversion 단독, capital_fraction 1.0'으로 IS(2022-2024) 백테.

사전 등록 선별 규칙 (다음 단계에서 사용, 실행 전 고정):
  거래 ≥ 30 AND IS Sharpe ≥ 현역 5개 중앙값 → 상위 5개만 확장 후보
  → 확장 슬리브 v17 vs 현 v17을 IS·OOS 비교, OOS에서 Sharpe·수익 ≥ AND MDD +5pp 이내일 때만 채택.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from data.loader import DataLoader
import scripts.run_backtest as rb

CONFIG = "config/final_v17.yaml"
IS = ("2022-01-01", "2024-12-31")
INCUMBENTS = ["LTCUSDT", "UNIUSDT", "STORJUSDT", "ARPAUSDT", "BANDUSDT"]


def candidates() -> list[str]:
    p = yaml.safe_load(open(CONFIG))
    used = set(p["symbols"]) | set(p["strategies"]["mean_reversion"]["symbols"])
    out = []
    for f in sorted(Path("data/cache").glob("ohlcv_*_1d.parquet")):
        sym = f.stem.replace("ohlcv_", "").replace("_1d", "")
        if sym in used or sym == "BTCDOMUSDT":
            continue
        df = pd.read_parquet(f)
        if isinstance(df.index, pd.DatetimeIndex):
            start, end = df.index.min(), df.index.max()
        elif "timestamp" in df.columns:
            s = pd.to_datetime(df["timestamp"], utc=True)
            start, end = s.min(), s.max()
        else:
            continue
        if start > pd.Timestamp("2022-01-05", tz="UTC") or end < pd.Timestamp("2026-04-01", tz="UTC"):
            continue
        if not Path(f"data/cache/funding_{sym}_8h.parquet").exists():
            continue
        out.append(sym)
    return out


def run_solo(sym: str) -> dict:
    t0 = time.time()
    p = copy.deepcopy(yaml.safe_load(open(CONFIG)))
    p["symbols"] = ["ETHUSDT", sym] if sym != "ETHUSDT" else ["ETHUSDT"]
    p["strategies"]["ema_cross"]["enabled"] = False
    p["strategies"]["multi_tf_breakout"]["enabled"] = False
    p["strategies"]["mean_reversion"]["symbols"] = [sym]
    p["strategy_capital_fraction"]["mean_reversion"] = 1.0
    loader = DataLoader(
        symbols=p["symbols"], timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=p.get("data", {}).get("cache_dir", "data/cache"),
        lookback=p.get("data", {}).get("lookback_bars", 300),
    )
    engine = rb.build_engine(p, 100.0)
    report = engine.run(loader.iterate(
        since=pd.Timestamp(IS[0], tz="UTC"), until=pd.Timestamp(IS[1], tz="UTC")))
    return {
        "symbol": sym,
        "incumbent": sym in INCUMBENTS,
        "final": round(report.final_equity, 1),
        "ret": round(report.total_return_pct, 1),
        "mdd": round(report.max_drawdown, 2),
        "sharpe": round(report.sharpe, 3),
        "pf": round(report.profit_factor, 3),
        "wr": round(report.win_rate, 1),
        "trades": report.total_trades,
        "secs": round(time.time() - t0, 1),
    }


def main() -> None:
    cands = candidates()
    syms = INCUMBENTS + cands
    print(f"IS 솔로 스캔: 현역 {len(INCUMBENTS)} + 후보 {len(cands)} = {len(syms)}심볼")
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(run_solo, s): s for s in syms}
        done = 0
        for fut in as_completed(futs):
            done += 1
            s = futs[fut]
            try:
                r = fut.result()
                res.append(r)
                tag = "현역" if r["incumbent"] else "    "
                print(f"  [{done}/{len(syms)}] {tag} {r['symbol']:14} Sh{r['sharpe']:>6.2f}  "
                      f"PF{r['pf']:>5.2f}  수익{r['ret']:>+8.1f}%  MDD{r['mdd']:>7.1f}  "
                      f"거래{r['trades']:>3}  ({r['secs']}s)", flush=True)
            except Exception as e:
                print(f"  [{done}/{len(syms)}] {s} 실패: {e}", flush=True)

    print(f"\n총 소요: {time.time()-t0:.0f}s")
    Path("SLEEVE_SCAN_RESULTS.json").write_text(
        json.dumps({"IS": IS, "results": res}, ensure_ascii=False, indent=2))

    import statistics
    inc = [r for r in res if r["incumbent"]]
    med = statistics.median(r["sharpe"] for r in inc)
    print(f"\n현역 5개 IS Sharpe 중앙값 = {med:.3f}")
    elig = sorted([r for r in res if not r["incumbent"] and r["trades"] >= 30 and r["sharpe"] >= med],
                  key=lambda r: -r["sharpe"])
    print(f"선별 통과(거래≥30 & Sh≥중앙값): {len(elig)}개 → 상위 5 = "
          f"{[r['symbol'] for r in elig[:5]]}")
    print("저장: SLEEVE_SCAN_RESULTS.json")


if __name__ == "__main__":
    main()
