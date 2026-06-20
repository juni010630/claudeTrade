"""멀티 심볼/타임프레임 DataLoader — MarketSnapshot 이터레이터."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from data.cache import ParquetCache
from data.schemas import MarketSnapshot

logger = logging.getLogger(__name__)


class DataLoader:
    """
    캐시에서 데이터를 읽어 타임스탬프별로 MarketSnapshot을 생성한다.
    기준 타임프레임(primary_tf)의 봉 마감 시각을 이터레이션 기준으로 삼는다.
    """

    def __init__(
        self,
        symbols: list[str],
        timeframes: list[str],
        primary_tf: str = "1h",
        cache_dir: str | Path = "data/cache",
        lookback: int = 300,  # 각 타임스탬프에서 제공할 과거 봉 수
    ) -> None:
        self.symbols = symbols
        self.timeframes = timeframes
        self.primary_tf = primary_tf
        self.cache = ParquetCache(cache_dir)
        self.lookback = lookback

        # 전체 데이터 로드
        self._ohlcv: dict[str, dict[str, pd.DataFrame]] = {}
        self._funding: dict[str, pd.DataFrame] = {}

        for sym in symbols:
            self._ohlcv[sym] = {}
            for tf in timeframes:
                df = self.cache.load(sym, tf)
                if df is None:
                    raise FileNotFoundError(
                        f"캐시 없음: {sym} {tf}. scripts/fetch_data.py 먼저 실행하세요."
                    )
                self._ohlcv[sym][tf] = df.set_index("timestamp")

            funding_df = self.cache.load(sym, "8h", data_type="funding")
            self._funding[sym] = (
                funding_df.set_index("timestamp") if funding_df is not None else pd.DataFrame()
            )

        # 성능: (sym, tf)별 봉 close-time int64 배열 사전계산 → iterate에서 searchsorted O(log n).
        # (기존: 봉마다 전체 인덱스 + timedelta 덧셈 + 불리언 take O(n) — 다심볼에서 지배 비용)
        # ⚠️ asi8은 인덱스 단위(ns/us/ms)를 그대로 따르는데 비교 상대 Timestamp.value는 항상 ns —
        # pandas 3.x가 parquet을 ms로 읽으면 단위 불일치로 전 봉 통과(look-ahead) → as_unit("ns") 필수.
        self._close_i8: dict[tuple[str, str], np.ndarray] = {}
        # (sym, tf)별 "직전 갭 위치" 누적배열 — iterate에서 lookback 윈도가 캐시 내부 공백(비인접
        # 봉)을 가로질러 BB/RSI/ATR을 오염시키지 않도록 윈도 시작을 갭 이후로 클램프하는 데 사용.
        # 갭 없으면 전부 -1 → 클램프 미발동 → 무갭 경로 결과 비트동일.
        self._last_gap: dict[tuple[str, str], np.ndarray] = {}
        _gap_thr = {tf: (pd.Timedelta(tf) * 3.0).to_timedelta64() for tf in timeframes}
        for sym in symbols:
            for tf in timeframes:
                idx = self._ohlcv[sym][tf].index
                self._close_i8[(sym, tf)] = (idx + pd.Timedelta(tf)).as_unit("ns").asi8
                n = len(idx)
                gmark = np.full(n, -1, dtype=np.int64)
                if n >= 2:
                    diffs = idx.to_series().diff().to_numpy()
                    big = np.where(diffs > _gap_thr[tf])[0]
                    gmark[big] = big
                self._last_gap[(sym, tf)] = np.maximum.accumulate(gmark)
        self._fund_i8: dict[str, np.ndarray] = {
            sym: fd.index.as_unit("ns").asi8 for sym, fd in self._funding.items()
        }
        # (sym, tf)별 직전 윈도 캐시 — 같은 (start, end) 구간이면 프레임 재사용 (4h/1d는 봉 간 대부분 동일).
        # 스냅샷 프레임은 전 소비자 read-only (변형 없음 확인) → 내용 비트동일.
        self._win_cache: dict[tuple[str, str], tuple[int, int, pd.DataFrame]] = {}

        self._warn_data_gaps()

    def _warn_data_gaps(self, max_factor: float = 3.0) -> None:
        """캐시 내부의 큰 시간 공백 1회 경고. 위치기반 슬라이싱은 갭을 사이에 둔
        비인접 봉을 연속으로 제시(지표 오염) + 갭 구간엔 stale 봉이 현재가로 쓰임.
        엔진 _get_bars는 stale 봉을 제외하지만, 갭 자체는 데이터 백필로 해소해야 함."""
        gapped = []
        for sym in self.symbols:
            tf = self.primary_tf
            idx = self._ohlcv[sym][tf].index
            if len(idx) < 2:
                continue
            diffs = idx.to_series().diff()
            big = diffs[diffs > pd.Timedelta(tf) * max_factor]
            if len(big):
                gapped.append((sym, big.max(), big.idxmax()))
        if gapped:
            worst = max(gapped, key=lambda x: x[1])
            logger.warning(
                "데이터 갭 감지: %d/%d 심볼 %s 캐시에 내부 공백 — 최악 %s (%s 직전, ~%s). "
                "갭 구간을 지나는 백테는 stale 가격/지표 오염 위험 → fetch_data.py 백필 권장.",
                len(gapped), len(self.symbols), self.primary_tf,
                worst[1], worst[0], worst[2],
            )

    def iterate(
        self,
        since: pd.Timestamp | None = None,
        until: pd.Timestamp | None = None,
    ) -> Iterator[MarketSnapshot]:
        """기준 타임프레임의 각 봉에 대해 MarketSnapshot을 yield."""
        primary_sym = self.symbols[0]
        primary_df = self._ohlcv[primary_sym][self.primary_tf]
        all_timestamps = primary_df.index

        # lookback 기준은 전체 데이터에서의 위치 (since/until 필터 전)
        min_idx = self.lookback
        if since is not None:
            # since 이전에 lookback만큼 데이터가 있는지 확인
            first_valid = all_timestamps.searchsorted(since)
            min_idx = max(min_idx, first_valid)

        timestamps = all_timestamps[min_idx:]
        if since is not None:
            timestamps = timestamps[timestamps >= since]
        if until is not None:
            timestamps = timestamps[timestamps < until]

        primary_delta = pd.Timedelta(self.primary_tf)
        for ts in timestamps:

            bars: dict[str, dict[str, pd.DataFrame]] = {}
            # ts = primary_tf 봉의 open time. effective time = ts + primary_delta (봉 close 시점).
            # 각 TF의 봉은 close_time(= open + tf_delta) ≤ effective_time 인 것만 포함.
            # 이래야 미마감 4h/1d 봉의 미래 데이터를 사용하지 않음.
            effective_time = ts + primary_delta
            eff_i8 = effective_time.value
            for sym in self.symbols:
                bars[sym] = {}
                for tf in self.timeframes:
                    # close_time ≤ effective_time 인 행 수 = searchsorted (기존 마스크와 동일 결과)
                    end = int(np.searchsorted(self._close_i8[(sym, tf)], eff_i8, side="right"))
                    start = max(0, end - self.lookback)
                    # 갭 클램프: 윈도 [start,end) 안에 캐시 공백이 있으면 시작을 가장 최근 갭 이후로
                    # 올려 비인접 봉 splice 차단(지표 오염 방지). 무갭이면 g=-1 → 미발동(결과 불변).
                    if end > 0:
                        g = int(self._last_gap[(sym, tf)][end - 1])
                        if g > start:
                            start = g
                    cached = self._win_cache.get((sym, tf))
                    if cached is not None and cached[0] == start and cached[1] == end:
                        bars[sym][tf] = cached[2]
                    else:
                        sub = self._ohlcv[sym][tf].iloc[start:end].reset_index()
                        self._win_cache[(sym, tf)] = (start, end, sub)
                        bars[sym][tf] = sub

            funding_rates: dict[str, float] = {}
            for sym in self.symbols:
                fd = self._funding[sym]
                if not fd.empty:
                    pos = int(np.searchsorted(self._fund_i8[sym], eff_i8, side="right"))
                    funding_rates[sym] = float(fd["rate"].iloc[pos - 1]) if pos > 0 else 0.0
                else:
                    funding_rates[sym] = 0.0

            yield MarketSnapshot(
                timestamp=effective_time,  # close time 기준 (LiveFeed와 일치)
                bars=bars,
                funding_rates=funding_rates,
                open_interest={},   # 필요 시 별도 캐시에서 로드
                btc_dominance=0.0,
            )
