"""[연구·screw_sweep] variant 일별 equity parquet → 전구간+연도별 지표 비교표.

연도별 = 전구간 복리 equity의 캘린더연 슬라이스(연속, 독립-$100 재시작 아님).
  → 스크리닝용 강건성 렌즈. 최종 후보는 독립 연도런으로 확정 권장.
Sharpe = mean(일수익)/std(일수익)*sqrt(365). MDD = min(eq/cummax - 1).

사용: python3 research/screw_sweep/analyze.py <name1> <name2> ...
      (name = data/results/sw_<name>_daily.parquet 의 <name>; 'baseline'은 v20_confirm)
인자 없으면 data/results/sw_*_daily.parquet 전부.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("data/results")
YEARS = {"2022": ("2022-01-01", "2022-12-31"),
         "2023": ("2023-01-01", "2023-12-31"),
         "2024": ("2024-01-01", "2024-12-31"),
         "25-26": ("2025-01-01", "2026-12-31")}


def metrics(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 3:
        return {"ret%": np.nan, "Sh": np.nan, "MDD%": np.nan}
    ret = eq.pct_change().dropna()
    sh = (ret.mean() / ret.std() * np.sqrt(365)) if ret.std() > 0 else 0.0
    mdd = (eq / eq.cummax() - 1).min()
    return {"ret%": (eq.iloc[-1] / eq.iloc[0] - 1) * 100, "Sh": sh, "MDD%": mdd * 100}


def resolve(name: str) -> Path:
    if name == "baseline":
        return RES / "v20_confirm_daily.parquet"
    p = RES / f"sw_{name}_daily.parquet"
    return p if p.exists() else RES / f"{name}_daily.parquet"


def main() -> None:
    names = sys.argv[1:]
    if not names:
        names = ["baseline"] + sorted(p.stem[3:-6] for p in RES.glob("sw_*_daily.parquet"))
    rows = []
    for nm in names:
        fp = resolve(nm)
        if not fp.exists():
            print(f"  (missing: {nm} -> {fp})")
            continue
        eq = pd.read_parquet(fp)["eq"]
        eq.index = pd.to_datetime(eq.index, utc=True)
        full = metrics(eq)
        row = {"variant": nm, "final$": eq.dropna().iloc[-1],
               "Sh": round(full["Sh"], 3), "MDD%": round(full["MDD%"], 1)}
        for yk, (s, e) in YEARS.items():
            m = metrics(eq.loc[s:e])
            row[f"{yk}_r%"] = round(m["ret%"], 0)
            row[f"{yk}_MDD"] = round(m["MDD%"], 0)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("variant")
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(df.to_string())


if __name__ == "__main__":
    main()
