"""maker 진입의 경로 영향 엔진 시뮬 — 놓친 매매가 복리에 끼치는 비용 (프로덕션 무수정).

추세 후보(ema/multi) 진입을 후보 생성 단계에서 5m 체결 판정으로 필터:
  체결 = 체결시각(now+1h)부터 5분 윈도의 5m 봉이 limit(시그널 close)를 터치/관통.
  MAKER_ONLY  : 미체결 → 후보 드롭 (놓친 매매의 경로 비용 측정)
  FALLBACK    : 미체결 → 윈도 끝 5m close로 시장가 폴백 (entry/TP/SL 동반 시프트 = 권고안)
슬리브(mean_reversion)는 불변. 5m 데이터 없는 시점 후보는 판정 불가 → 통과(보수 아님, 카운트만).

주의: 5m 윈도 판정은 집행 시뮬레이션(TP/SL이 봉 고저로 체결 판정하는 것과 동급) — 신호 look-ahead 아님.
비용 모델은 BASE와 동일(전 진입 taker+슬리피지 유지) → FALLBACK이 BASE급이면 실제 maker 절감
(체결분 수수료 3bps+슬리피지 회피)이 얹혀 순우위라는 보수적 논증 구조.

런(사전선언 5개): BASE / NOOP(클래스 재할당+무필터, BASE와 일치 필수) /
MAKER_ONLY strict5 / MAKER_ONLY touch5 / FALLBACK strict5

사용: python scripts/maker_path_sim.py
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

CFG = "config/final_v17.yaml"
END = "2026-04-14"
TREND = {"ema_cross", "multi_tf_breakout"}
SYMS_5M = ["ETHUSDT", "BTCUSDT", "DOGEUSDT", "ARBUSDT", "FILUSDT", "ADAUSDT"]
WIN_BARS = 1  # 5분 윈도 = 5m 봉 1개


def _load_5m():
    out = {}
    for s in SYMS_5M:
        d = pd.read_parquet(f"data/cache/ohlcv_{s}_5m.parquet")
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        if d["timestamp"].dt.tz is not None:
            d["timestamp"] = d["timestamp"].dt.tz_localize(None)
        d = d.sort_values("timestamp")
        out[s] = (d["timestamp"].to_numpy(), d["low"].to_numpy(),
                  d["high"].to_numpy(), d["close"].to_numpy())
    return out


def run_one(args):
    tag, mode, strict = args  # mode: None=BASE, "noop", "only", "fallback"
    from data.loader import DataLoader
    from engine.backtest import BacktestEngine
    from scripts.run_backtest import build_engine

    p = yaml.safe_load(open(CFG))
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)

    if mode is not None:
        class _MakerSim(BacktestEngine):
            def _generate_all_candidates(self, snapshot, regime, prices):
                cands = super()._generate_all_candidates(snapshot, regime, prices)
                if self._mk_mode == "noop":
                    return cands
                out = []
                for c in cands:
                    if c["strategy"] not in TREND or c["symbol"] not in self._mk_5m:
                        out.append(c)
                        continue
                    ts, lo, hi, cl = self._mk_5m[c["symbol"]]
                    # snapshot.timestamp = 완성봉 close 시각(effective_time) = 체결 시각 그 자체
                    t0 = np.datetime64(snapshot.timestamp.tz_localize(None))
                    i0 = np.searchsorted(ts, t0)
                    i1 = i0 + WIN_BARS
                    if i0 >= len(ts) or ts[i0] != t0:
                        self._mk_stats["nodata"] += 1
                        out.append(c)  # 판정 불가 → 통과
                        continue
                    self._mk_stats["checked"] += 1
                    limit = c["entry_price"]
                    w_lo, w_hi = lo[i0:i1], hi[i0:i1]
                    if c["direction"] == "long":
                        filled = (w_lo <= limit).any() if not self._mk_strict else (w_lo < limit).any()
                    else:
                        filled = (w_hi >= limit).any() if not self._mk_strict else (w_hi > limit).any()
                    if filled:
                        out.append(c)
                        continue
                    if self._mk_mode == "only":
                        self._mk_stats["dropped"] += 1
                        continue
                    # fallback: 윈도 끝 시장가 — entry/TP/SL 동반 시프트 (거리 보존)
                    px = float(cl[i1 - 1])
                    shift = px - c["entry_price"]
                    c = dict(c)
                    c["entry_price"] = px
                    c["tp_price"] += shift
                    c["sl_price"] += shift
                    self._mk_stats["fallback"] += 1
                    out.append(c)
                return out

        eng.__class__ = _MakerSim
        eng._mk_mode = mode
        eng._mk_strict = strict
        eng._mk_5m = _load_5m()
        eng._mk_stats = {"checked": 0, "dropped": 0, "fallback": 0, "nodata": 0}

    report = eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                                    until=pd.Timestamp(END, tz="UTC")))
    eq = eng.equity_curve.to_series()
    daily = eq.resample("1D").last().pct_change().fillna(0)
    yearly = daily.groupby(daily.index.year).apply(lambda r: ((1 + r).prod() - 1) * 100)
    df = eng.ledger.to_dataframe()
    n_trend = int(df["strategy"].isin(TREND).sum()) if len(df) else 0
    stats = getattr(eng, "_mk_stats", {})
    return tag, round(report.final_equity, 2), round(report.sharpe, 3), \
        round(report.max_drawdown, 1), len(df), n_trend, dict(yearly), stats


def main():
    jobs = [("BASE", None, False),
            ("NOOP(무결성)", "noop", False),
            ("MAKER_ONLY·strict", "only", True),
            ("MAKER_ONLY·touch", "only", False),
            ("FALLBACK·strict", "fallback", True)]
    with ProcessPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(run_one, jobs))

    print("=== maker 경로 시뮬 (v17 전구간, 5분 윈도, 비용 모델 BASE 동일) ===")
    base = None
    for tag, f, sh, mdd, n, n_tr, yearly, st in results:
        ys = " ".join(f"{int(k)}:{round(v):+}" for k, v in yearly.items())
        extra = (f" | 판정{st.get('checked',0)} 드롭{st.get('dropped',0)}"
                 f" 폴백{st.get('fallback',0)} 미판정{st.get('nodata',0)}") if st else ""
        print(f"  {tag:18} ${f:>10,.2f} Sh{sh:6.3f} MDD{mdd:6.1f}% {n}건(추세{n_tr}){extra}  {ys}")
        if tag == "BASE":
            base = (f, n)
    noop = next(r for r in results if r[0].startswith("NOOP"))
    print(f"\n[무결성] NOOP ${noop[1]:,.2f}/{noop[4]}건 vs BASE ${base[0]:,.2f}/{base[1]}건"
          f" → {'일치' if (abs(noop[1]-base[0]) < 0.01 and noop[4] == base[1]) else '불일치(하네스 버그)'}")


if __name__ == "__main__":
    main()
