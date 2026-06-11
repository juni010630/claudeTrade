"""TRI 틸트 배분 백테 — 횡보/중립 = 50:50(현행), 추세장 = 70:30 (프로덕션 무수정).

재탕 대조 (기각 격자와의 차이):
  ① 슬리브 오버웨이트 레그 없음 — 일별·주간·월간 스위치를 죽인 쪽은 전부 "횡보→슬리브 과중".
    어제 G0에서 추세판정 주는 스프레드 +2.87%/주(4/5년) = 추세측 틸트만은 양의 사전증거.
  ② 신호 = TRI (검증 완료: OOS AUC 0.746, 플립 7.7/년 — TREND_INDEX_RESULTS.md), 히스테리시스 라벨.
  ③ 월간 스킴B(80:20/50:50, basket ER, 기각)와 신호·강도·주기 상이.

인과: TRI 행(1d 개봉스탬프 T)은 T+1d 00:00에 확정 → 가용시각 인덱스로 asof 조회.
변형(사전선언 — 스윕 금지): TILT73(추세 0.7/0.3), TILT64(0.6/0.4, 단조성 확인용), BASE, NOOP(무결성).

게이트 G-TILT: 전구간 Sharpe >= 1.937 AND MDD봉 >= -45% AND 최종수익 > $8,991
  AND IS·OOS 모두 개선 부호 AND TILT64가 BASE~TILT73 사이(단조성). 기각 시 기록만.

사용: python scripts/tri_tilt_bt.py
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

CFG = "config/final_v17.yaml"
END = "2026-04-14"
TREND_STRATS = ("ema_cross", "multi_tf_breakout")


def build_labels() -> pd.Series:
    """시장 TRI 히스테리시스 라벨, 가용시각(=1d 봉 마감시각) 인덱스 (tz-aware UTC)."""
    from data.cache import ParquetCache
    from regime.trend_index import trend_index, label_series

    tris = []
    for s in ("BTCUSDT", "ETHUSDT"):
        cache = ParquetCache("data/cache")
        d1 = cache.load(s, "1d").copy()
        d4 = cache.load(s, "4h").copy()
        for d in (d1, d4):
            d["timestamp"] = pd.to_datetime(d["timestamp"])
            if d["timestamp"].dt.tz is not None:
                d["timestamp"] = d["timestamp"].dt.tz_localize(None)
        tris.append(trend_index(d1.sort_values("timestamp").set_index("timestamp"),
                                d4.sort_values("timestamp").set_index("timestamp"))["tri"])
    mkt = (tris[0] + tris[1]) / 2
    lab = label_series(mkt)
    lab.index = (lab.index + pd.Timedelta(days=1)).tz_localize("UTC")  # 가용시각 = 봉 마감
    return lab


def run_one(args):
    tag, w_trend_in_trend, labels, start, until = args  # w None=BASE, w<0=NOOP(조회만)
    from data.loader import DataLoader
    from engine.backtest import BacktestEngine
    from scripts.run_backtest import build_engine

    p = yaml.safe_load(open(CFG))
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)

    if w_trend_in_trend is not None:
        class _TiltSim(BacktestEngine):
            def _generate_all_candidates(self, snapshot, regime, prices):
                lab = self._tilt_labels.asof(snapshot.timestamp)
                if self._tilt_w > 0 and isinstance(lab, str):
                    w = self._tilt_w if lab == "추세장" else 0.5
                    if w != self._tilt_cur:
                        self._tilt_cur = w
                        self._tilt_n += 1
                        self._strategy_capital_fraction = {
                            "ema_cross": w, "multi_tf_breakout": w,
                            "mean_reversion": round(1.0 - w, 2),
                        }
                return super()._generate_all_candidates(snapshot, regime, prices)

        eng.__class__ = _TiltSim
        eng._tilt_labels = labels
        eng._tilt_w = w_trend_in_trend
        eng._tilt_cur = 0.5
        eng._tilt_n = 0

    report = eng.run(loader.iterate(since=pd.Timestamp(start, tz="UTC"),
                                    until=pd.Timestamp(until, tz="UTC")))
    eq = eng.equity_curve.to_series()
    daily = eq.resample("1D").last().pct_change().fillna(0)
    yearly = daily.groupby(daily.index.year).apply(lambda r: ((1 + r).prod() - 1) * 100)
    return tag, round(report.final_equity, 2), round(report.sharpe, 3), \
        round(report.max_drawdown, 1), len(eng.ledger.to_dataframe()), \
        getattr(eng, "_tilt_n", 0), dict(yearly)


def main():
    labels = build_labels()
    dist = labels.value_counts()
    print(f"[TRI 라벨] {dist.to_dict()} | 기간 {labels.index[0]:%Y-%m-%d}~{labels.index[-1]:%Y-%m-%d}")

    F, IS_, OOS = ("2022-01-01", END), ("2022-01-01", "2024-12-31"), ("2025-01-01", END)
    jobs = [("BASE·full", None, labels, *F),
            ("NOOP·full", -1.0, labels, *F),
            ("T73·full", 0.7, labels, *F),
            ("T64·full", 0.6, labels, *F),
            ("BASE·IS", None, labels, *IS_), ("T73·IS", 0.7, labels, *IS_),
            ("BASE·OOS", None, labels, *OOS), ("T73·OOS", 0.7, labels, *OOS)]
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = {r[0]: r[1:] for r in ex.map(run_one, jobs)}

    print("\n=== TRI 틸트 (횡보/중립 50:50 · 추세 w:1-w) ===")
    for tag, (f, sh, mdd, n, nsw, yearly) in res.items():
        ys = " ".join(f"{int(k)}:{round(v):+}" for k, v in yearly.items())
        sw = f" 전환{nsw}" if nsw else ""
        print(f"  {tag:10} ${f:>10,.2f} Sh{sh:6.3f} MDD{mdd:6.1f}% {n}건{sw}  {ys}")

    b, n_, t73, t64 = res["BASE·full"], res["NOOP·full"], res["T73·full"], res["T64·full"]
    print(f"\n[무결성] NOOP ${n_[0]:,.2f}/{n_[3]}건 vs BASE ${b[0]:,.2f}/{b[3]}건 → "
          f"{'일치' if abs(n_[0]-b[0]) < 0.01 and n_[3] == b[3] else '불일치(버그)'}")

    g_sh = t73[1] >= b[1]
    g_mdd = t73[2] >= -45.0
    g_ret = t73[0] > b[0]
    is_up = res["T73·IS"][0] > res["BASE·IS"][0]
    oos_up = res["T73·OOS"][0] > res["BASE·OOS"][0]
    mono = min(b[0], t73[0]) <= t64[0] <= max(b[0], t73[0])
    print(f"[G-TILT] Sharpe {b[1]}→{t73[1]} ({'OK' if g_sh else 'FAIL'}) | "
          f"MDD {t73[2]}% ({'OK' if g_mdd else 'FAIL'}) | 수익 ${b[0]:,.0f}→${t73[0]:,.0f} ({'OK' if g_ret else 'FAIL'})")
    print(f"         IS {'↑' if is_up else '↓'} / OOS {'↑' if oos_up else '↓'} "
          f"({'OK' if is_up and oos_up else 'FAIL'}) | 단조성 T64=${t64[0]:,.0f} ({'OK' if mono else 'FAIL'})")
    ok = g_sh and g_mdd and g_ret and is_up and oos_up and mono
    print(f"\n판정: {'통과 — 채택 검토' if ok else '기각 — 50:50 유지'}")


if __name__ == "__main__":
    main()
