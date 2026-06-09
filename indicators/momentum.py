"""모멘텀 지표 — 순수 함수.

⚠️ RSI는 표준 Wilder(alpha=1/period) 평활이 아니라 ewm(span=period)를 쓴다. 외부
   도구(TA-Lib/TradingView)와 값이 체계적으로 다르나, 백테=라이브가 동일 함수를 공유하고
   모든 임계값이 이 정의로 캘리브레이션됨. Wilder로 바꾸면 전 백테가 변하니 변경 금지(변경 시
   연도별 전 파라미터 재검증 필수). 외부 비교용 컨벤션 차이일 뿐 결함 아님.
"""
from __future__ import annotations

import pandas as pd


def rsi(df: pd.DataFrame, period: int = 14, col: str = "close") -> pd.Series:
    delta = df[col].diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """현재 거래량 / 최근 N봉 평균 거래량."""
    avg = df["volume"].rolling(period).mean()
    return df["volume"] / avg.replace(0, float("nan"))
