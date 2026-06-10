"""슬리브 '소소한 이득 여러번' 변형 스윕 — 대칭 TP/SL × 신호빈도(1d/12h/8h).

사용자 가설: TP/SL 1:1 + 진입을 12h/8h 단위로 → 잦은 작은 익절.
사전 기록: 4h 빈도는 건당 엣지 붕괴로 기각(4.23%→1.21%), TP 단독 스윕도 단조 악화.
12h/8h는 미검증 보간점 — 본 스윕으로 결판.

격리: 프로덕션 무수정. 12h/8h 봉은 1h 캐시 리샘플(origin=epoch → 00시 경계 정렬,
00:00 진입 셀은 현행 1d와 동일 정보집합). 중복진입 방지는 서브클래스 경계 게이트.
판정(사전선언): Sharpe ≥ 1.94(현행) AND MDD ≥ -42.4% 비악화 → 통과 시에만 2단계.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from data.loader import DataLoader
from strategies.mean_reversion import MeanReversionStrategy
import scripts.run_backtest as rb

CONFIG = "config/final_v17.yaml"
SRC = Path("data/cache")
DST = Path("data/freq_cache")

# (이름, signal_tf, 게이트시간, tp_mult, sl_mult)
VARIANTS = [
    ("1:1 tp1.0/sl1.0", "1d", None, 1.0, 1.0),
    ("1:1 tp1.5/sl1.5", "1d", None, 1.5, 1.5),
    ("1:1 tp2.0/sl2.0", "1d", None, 2.0, 2.0),
    ("12h 현행비율",     "12h", 12, 3.0, 2.0),
    ("8h 현행비율",      "8h",  8, 3.0, 2.0),
    ("12h + 1:1 1.5",   "12h", 12, 1.5, 1.5),
    ("8h + 1:1 1.5",    "8h",  8, 1.5, 1.5),
]


class GatedMR(MeanReversionStrategy):
    """signal_tf 12h/8h용 — 봉 경계 시각에만 신호 (1d의 hour==0 게이트와 등가)."""

    def __init__(self, cfg: dict, gate_hours: int) -> None:
        super().__init__(cfg)
        self._gate_h = gate_hours

    def generate_signals(self, snapshot, regime):
        if snapshot.timestamp.hour % self._gate_h != 0:
            return []
        return super().generate_signals(snapshot, regime)


def build_freq_cache(symbols: list[str]) -> None:
    DST.mkdir(parents=True, exist_ok=True)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for sym in symbols:
        # 1h/4h/1d + funding은 원본 심볼릭 링크
        for name in [f"ohlcv_{sym}_1h.parquet", f"ohlcv_{sym}_4h.parquet",
                     f"ohlcv_{sym}_1d.parquet", f"funding_{sym}_8h.parquet"]:
            src, dst = SRC / name, DST / name
            if src.exists() and not dst.exists():
                os.symlink(src.resolve(), dst)
        # 12h/8h 리샘플 (epoch 앵커 → 00시 경계 정렬)
        h1 = pd.read_parquet(SRC / f"ohlcv_{sym}_1h.parquet").set_index("timestamp")
        for tf in ("12h", "8h"):
            out = DST / f"ohlcv_{sym}_{tf}.parquet"
            if out.exists():
                continue
            r = (h1.resample(tf, origin="epoch", label="left", closed="left")
                 .agg(agg).dropna().reset_index())
            r.to_parquet(out)


def run(v):
    name, stf, gate_h, tp, sl = v
    p = yaml.safe_load(open(CONFIG))
    mr = p["strategies"]["mean_reversion"]
    mr["atr_tp_mult"], mr["atr_sl_mult"], mr["signal_tf"] = tp, sl, stf
    if stf not in p["timeframes"]:
        p["timeframes"] = p["timeframes"] + [stf]
    bt = p.get("backtest", {})
    loader = DataLoader(
        symbols=p["symbols"], timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=str(DST),
        lookback=p.get("data", {}).get("lookback_bars", 300),
    )
    engine = rb.build_engine(p, 100.0)
    if gate_h is not None:
        engine.strategies = [
            GatedMR(p["strategies"]["mean_reversion"], gate_h)
            if s.name == "mean_reversion" else s
            for s in engine.strategies
        ]
    since = pd.Timestamp(bt.get("start", "2022-01-01"), tz="UTC")
    until = pd.Timestamp(bt["end"], tz="UTC") if bt.get("end") else None
    report = engine.run(loader.iterate(since=since, until=until))
    mrt = [r for r in engine.ledger.records if r.strategy == "mean_reversion"]
    wins = sum(1 for r in mrt if r.pnl > 0)
    return {
        "name": name, "final": report.final_equity, "mdd": report.max_drawdown,
        "sharpe": report.sharpe, "trades": report.total_trades,
        "mr_n": len(mrt), "mr_wr": 100 * wins / len(mrt) if mrt else 0,
        "mr_pnl": sum(r.pnl for r in mrt),
    }


if __name__ == "__main__":
    p0 = yaml.safe_load(open(CONFIG))
    build_freq_cache(p0["symbols"])
    print(f"freq_cache 준비 완료. {len(VARIANTS)}변형 병렬 실행…")
    with ProcessPoolExecutor(max_workers=len(VARIANTS)) as ex:
        results = list(ex.map(run, VARIANTS))
    print(f"\n{'변형':18} {'최종$':>9} {'MDD%':>7} {'Sharpe':>7} {'거래':>5} "
          f"{'슬리브n':>7} {'WR%':>6} {'슬리브PnL':>10}")
    print("-" * 78)
    print(f"{'현행 (기록치)':18} {8991:>9,} {-42.4:>7.1f} {1.94:>7.2f} {805:>5} "
          f"{353:>7} {51.8:>6.1f} {733.1:>+10.1f}")
    for r in results:
        print(f"{r['name']:18} {r['final']:>9,.0f} {r['mdd']:>7.1f} {r['sharpe']:>7.2f} "
              f"{r['trades']:>5} {r['mr_n']:>7} {r['mr_wr']:>6.1f} {r['mr_pnl']:>+10.1f}")
