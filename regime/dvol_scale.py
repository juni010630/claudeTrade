"""DVOL(Deribit 내재변동성지수) 인버스 변동성타게팅 — 글로벌 사이즈 배수 스케줄 빌더.

거래일 D 배수 = clip(target / DVOL[D-lag_days], lo, hi). 1일 래그 = look-ahead 차단.
백테(run_backtest)·라이브(live_trade) 양쪽 동일 함수·동일 데이터 호출 → 패리티 보장.
(2026-06-17 채택: 세션 유일 엔진검증 통과 알파 — project_dvol_voltarget. IV는 선행지표라
 고변동(엣지약화) 구간 진입 전 노출축소. 실현변동성(RV) 타게팅과 달리 회복 안 놓침.)
"""
from __future__ import annotations

import pandas as pd


def build_dvol_schedule(dvol_path: str, target: float, clip_lo: float, clip_hi: float,
                        lag_days: int = 1) -> pd.Series:
    """일별 글로벌 사이즈 배수 (index=UTC 일자). 데이터 파일 없으면 fetch(배포 견고성)."""
    from pathlib import Path
    if not Path(dvol_path).exists():
        refresh_dvol_parquet(dvol_path)   # 서버 첫 배포 시 data/ gitignore로 파일 부재 → 자동 수집
    dvol = pd.read_parquet(dvol_path)["dvol_btc"]
    dvol.index = pd.to_datetime(dvol.index, utc=True)
    return (target / dvol.shift(lag_days)).clip(clip_lo, clip_hi).dropna()


def refresh_dvol_parquet(path: str = "data/regime/dvol_btc_full.parquet") -> bool:
    """Deribit에서 최신 BTC DVOL을 증분 병합 (라이브 일일 갱신용).
    실패 시 기존 유지·False (네트워크 일시장애에 봇 중단 금지)."""
    import json
    import urllib.request
    from pathlib import Path

    p = Path(path)
    old = pd.read_parquet(p)["dvol_btc"] if p.exists() else pd.Series(dtype=float)
    try:
        import time
        end = int(time.time() * 1000)
        start = end - 60 * 86400000  # 최근 60일
        url = ("https://www.deribit.com/api/v2/public/get_volatility_index_data"
               f"?currency=BTC&start_timestamp={start}&end_timestamp={end}&resolution=1D")
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.load(r)["result"]["data"]
    except Exception:
        return False
    new = pd.Series({pd.Timestamp(int(row[0]), unit="ms", tz="UTC").normalize(): float(row[4])
                     for row in data}, name="dvol_btc")
    merged = pd.concat([old, new])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    p.parent.mkdir(parents=True, exist_ok=True)
    merged.to_frame("dvol_btc").to_parquet(p)
    return True
