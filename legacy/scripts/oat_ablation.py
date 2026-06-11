"""OAT(One-At-a-Time) 어블레이션 — 현 라이브(merged_noblock_sleeve)를 기준 기계로 놓고
'부품(조건)'을 하나씩만 변형(제거/느슨/강하게)해 그 한 개의 전체 성과 기여를 격리 측정.

원칙:
 - 변인은 매 변형당 단 하나(나머지 전부 고정).
 - 진입을 바꾸는 나사는 전부 풀 엔진(run). TP/SL 배수도 슬롯/쿨다운 연쇄 때문에 풀 엔진.
   (fast replay = 진입 고정이라 진입 안 바꾸는 나사에만 유효 → 별도 fast_sweep.py로 TP/SL 그리드)
 - 결과는 전구간 + 연도별 둘 다 저장(미세조정=경로 카오스 → 연도별 일관성으로 판정).

결과 저장: OAT_ABLATION_RESULTS.json (raw) + OAT_ABLATION.md (사람용 표).
프로덕션 코드 무수정 (run_backtest.build_engine 재사용).
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
DEL = "__DEL__"  # 키 삭제 센티넬


# ── config 변형 적용 ────────────────────────────────────────────────────────
def apply_ops(p: dict, ops: list) -> dict:
    for path, val in ops:
        node = p
        for k in path[:-1]:
            node = node.setdefault(k, {})
        if val == DEL:
            node.pop(path[-1], None)
        else:
            node[path[-1]] = val
    return p


def _yearly_ret(eq: pd.Series) -> dict:
    out = {}
    for y, seg in eq.groupby(eq.index.year):
        if len(seg) >= 2:
            out[str(int(y))] = round((seg.iloc[-1] / seg.iloc[0] - 1) * 100, 0)
    return out


# ── 워커 (ProcessPoolExecutor용 — 모듈 최상위 필수) ──────────────────────────
def run_one(spec: dict) -> dict:
    t0 = time.time()
    with open(CONFIG) as f:
        p = yaml.safe_load(f)
    apply_ops(p, spec["ops"])

    bt = p.get("backtest", {})
    since = pd.Timestamp(bt.get("start", "2022-01-01"), tz="UTC")
    until = pd.Timestamp(bt.get("end"), tz="UTC") if bt.get("end") else None
    loader = DataLoader(
        symbols=p["symbols"], timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=p.get("data", {}).get("cache_dir", "data/cache"),
        lookback=p.get("data", {}).get("lookback_bars", 300),
    )
    engine = rb.build_engine(p, 100.0)
    report = engine.run(loader.iterate(since=since, until=until))
    eq = engine.equity_curve.to_series()
    return {
        "name": spec["name"], "cat": spec["cat"], "dir": spec["dir"],
        "final": round(report.final_equity, 0),
        "ret": round(report.total_return_pct, 0),
        "mdd": round(report.max_drawdown, 1),        # 이미 % (봉기준)
        "sharpe": round(report.sharpe, 3),
        "wr": round(report.win_rate, 1),
        "pf": round(report.profit_factor, 2),
        "trades": report.total_trades,
        "yearly": _yearly_ret(eq),
        "secs": round(time.time() - t0, 1),
    }


# ── 어블레이션 매트릭스 (각 변형 = 나사 하나) ────────────────────────────────
# 베이스라인 값(merged_noblock_sleeve):
#  max_pos10 same_dir8 corr0.8 cooldown6 rpt0.07 CB(5/10) dailyDD off
#  BTC숏차단 위성티어게이트(A/S→ADA,ARB,FIL) max_hold336
#  scorer: ss5 s4 sss6 vol1.8 rsi(70/40)
#  ema: fast8 slow21 macd12/21/7 dema150 tp3.5 sl1.8
#  multi: bb_std1h2.2 4h2.0 rsi_p10 rsi(55/45) volmult1.8 tp4.0 sl2.1
#  size_bonus on(x1.25) capital 50:50
def build_specs() -> list:
    S = []
    def add(name, cat, d, ops): S.append({"name": name, "cat": cat, "dir": d, "ops": ops})

    add("BASELINE", "base", "—", [])

    # ── C. 리스크 게이트 ──
    add("BTC숏 허용",            "gate", "remove",  [[["symbol_block_directions","BTCUSDT"], DEL]])
    add("max_pos 10→5",         "gate", "tighten", [[["risk","max_positions"], 5]])
    add("max_pos 10→15",        "gate", "loosen",  [[["risk","max_positions"], 15]])
    add("same_dir 8→4",         "gate", "tighten", [[["risk","max_same_direction"], 4]])
    add("cooldown 6→0 (제거)",  "gate", "remove",  [[["risk","tp_cooldown_hours"], 0.0]])
    add("cooldown 6→12",        "gate", "tighten", [[["risk","tp_cooldown_hours"], 12.0]])
    add("corr 0.8→1.0 (off)",   "gate", "remove",  [[["risk","correlation_block_threshold"], 1.01]])
    add("corr 0.8→0.6",         "gate", "tighten", [[["risk","correlation_block_threshold"], 0.6]])
    add("위성티어게이트 제거",   "gate", "remove",  [[["tier_block_symbols"], DEL]])
    add("daily DD ON(-4/-10)",  "gate", "on",      [[["risk","daily_drawdown_pause"], -0.04],
                                                    [["risk","daily_drawdown_stop"], -0.10]])
    add("CB 강화(pause 5→3)",   "gate", "tighten", [[["risk","circuit_breaker_pause_losses"], 3]])
    add("CB 제거",              "gate", "off",     [[["risk","circuit_breaker_pause_losses"], 999],
                                                    [["risk","circuit_breaker_stop_losses"], 999]])
    add("max_hold 336→168",     "gate", "tighten", [[["engine","max_hold_hours"], 168]])
    add("max_hold 336→720",     "gate", "loosen",  [[["engine","max_hold_hours"], 720]])

    # ── B. 스코어러 티어/필터 ──
    add("SS게이트 5→4",         "score", "loosen",  [[["scorer","tier_ss_min_score"], 4]])
    add("SS게이트 5→6",         "score", "tighten", [[["scorer","tier_ss_min_score"], 6]])
    add("S게이트 4→3",          "score", "loosen",  [[["scorer","tier_s_min_score"], 3]])
    add("S게이트 4→5",          "score", "tighten", [[["scorer","tier_s_min_score"], 5]])
    add("vol_thr 1.8→1.5",      "score", "loosen",  [[["scorer","volume_ratio_threshold"], 1.5]])
    add("vol_thr 1.8→2.2",      "score", "tighten", [[["scorer","volume_ratio_threshold"], 2.2]])
    add("RSI게이트 느슨(75/35)", "score", "loosen", [[["scorer","rsi_long_max"], 75.0],
                                                     [["scorer","rsi_short_min"], 35.0]])
    add("RSI게이트 강화(65/45)", "score", "tighten",[[["scorer","rsi_long_max"], 65.0],
                                                     [["scorer","rsi_short_min"], 45.0]])

    # ── A. 전략 신호 파라미터 ──
    add("ema MACD 12→14",       "ema", "loosen",   [[["strategies","ema_cross","macd_fast"], 14]])
    add("ema fast 8→9",         "ema", "tighten",  [[["strategies","ema_cross","fast_period"], 9]])
    add("ema slow 21→20",       "ema", "tighten",  [[["strategies","ema_cross","slow_period"], 20]])
    add("ema 4h-EMA 150→200",   "ema", "tighten",  [[["strategies","ema_cross","daily_ema_period"], 200]])
    add("ema 4h-EMA 150→100",   "ema", "loosen",   [[["strategies","ema_cross","daily_ema_period"], 100]])
    add("ema TP 3.5→3.0",       "ema", "tighten",  [[["strategies","ema_cross","atr_tp_mult"], 3.0]])
    add("ema TP 3.5→4.0",       "ema", "loosen",   [[["strategies","ema_cross","atr_tp_mult"], 4.0]])
    add("ema SL 1.8→1.5",       "ema", "tighten",  [[["strategies","ema_cross","atr_sl_mult"], 1.5]])
    add("ema SL 1.8→2.1",       "ema", "loosen",   [[["strategies","ema_cross","atr_sl_mult"], 2.1]])

    add("multi BB1h 2.2→2.0",   "multi", "loosen", [[["strategies","multi_tf_breakout","bb_std_1h"], 2.0]])
    add("multi BB1h 2.2→2.5",   "multi", "tighten",[[["strategies","multi_tf_breakout","bb_std_1h"], 2.5]])
    add("multi RSI_p 10→14",    "multi", "tighten",[[["strategies","multi_tf_breakout","rsi_period"], 14]])
    add("multi RSI 느슨(50/50)","multi", "loosen", [[["strategies","multi_tf_breakout","rsi_long_min"], 50.0],
                                                    [["strategies","multi_tf_breakout","rsi_short_max"], 50.0]])
    add("multi vol 1.8→1.5",    "multi", "loosen", [[["strategies","multi_tf_breakout","volume_multiplier"], 1.5]])
    add("multi TP 4.0→3.5",     "multi", "tighten",[[["strategies","multi_tf_breakout","atr_tp_mult"], 3.5]])
    add("multi SL 2.1→1.8",     "multi", "tighten",[[["strategies","multi_tf_breakout","atr_sl_mult"], 1.8]])

    # ── D. 사이징 ──
    add("rpt 0.07→0.05",        "size", "tighten", [[["risk","risk_per_trade"], 0.05]])
    add("rpt 0.07→0.09",        "size", "loosen",  [[["risk","risk_per_trade"], 0.09]])
    add("size_bonus 제거",      "size", "remove",  [[["strategy_size_bonus"], DEL]])
    add("size_bonus_mult→1.0",  "size", "off",     [[["strategy_size_bonus_mult"], 1.0]])
    add("배분 40:60(추세↓)",    "size", "shift",   [[["strategy_capital_fraction","ema_cross"], 0.4],
                                                    [["strategy_capital_fraction","multi_tf_breakout"], 0.4],
                                                    [["strategy_capital_fraction","mean_reversion"], 0.6]])
    add("배분 60:40(추세↑)",    "size", "shift",   [[["strategy_capital_fraction","ema_cross"], 0.6],
                                                    [["strategy_capital_fraction","multi_tf_breakout"], 0.6],
                                                    [["strategy_capital_fraction","mean_reversion"], 0.4]])

    # ── 컴포넌트 제거 (전략 on/off) ──
    add("슬리브 OFF",           "comp", "off",     [[["strategies","mean_reversion","enabled"], False]])
    add("multi_tf OFF",         "comp", "off",     [[["strategies","multi_tf_breakout","enabled"], False]])
    add("ema_cross OFF",        "comp", "off",     [[["strategies","ema_cross","enabled"], False]])

    return S


def main() -> None:
    specs = build_specs()
    print(f"OAT 어블레이션 시작 — {len(specs)}개 변형 (베이스라인 포함), config={CONFIG}")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(run_one, s): s for s in specs}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                r = fut.result()
                results.append(r)
                print(f"  [{done}/{len(specs)}] {r['name']:22} "
                      f"${r['final']:>9,.0f}  MDD{r['mdd']:>6.1f}  Sh{r['sharpe']:>5.2f}  "
                      f"거래{r['trades']:>4}  ({r['secs']}s)", flush=True)
            except Exception as e:
                s = futs[fut]
                print(f"  [{done}/{len(specs)}] {s['name']:22} 실패: {e}", flush=True)
    print(f"\n총 소요: {time.time()-t0:.0f}s")

    base = next(r for r in results if r["name"] == "BASELINE")
    years = sorted({y for r in results for y in r["yearly"]})

    # cat 순서 보존
    order = {"base":0, "gate":1, "score":2, "ema":3, "multi":4, "size":5, "comp":6}
    spec_order = {s["name"]: i for i, s in enumerate(specs)}
    results.sort(key=lambda r: (order.get(r["cat"], 9), spec_order.get(r["name"], 99)))

    # ── JSON 저장 (raw 전부) ──
    out_json = {"config": CONFIG, "baseline": base, "years": years, "variants": results}
    Path("OAT_ABLATION_RESULTS.json").write_text(json.dumps(out_json, ensure_ascii=False, indent=2))

    # ── 마크다운 표 ──
    L = []
    L.append(f"# OAT 어블레이션 결과 — 현 라이브({CONFIG}) 기준 기계, 부품 1개씩 변형")
    L.append("")
    L.append(f"전구간 2022-01-01~{yaml.safe_load(open(CONFIG)).get('backtest',{}).get('end','?')}, "
             f"$100 시작, 봉기준 MDD. 변인 = 매 행당 나사 1개(나머지 고정).")
    L.append("")
    L.append(f"**BASELINE: ${base['final']:,.0f} / Sharpe {base['sharpe']:.2f} / "
             f"MDD {base['mdd']:.1f}% / 거래 {base['trades']} / WR {base['wr']}% / PF {base['pf']}**")
    L.append("")
    L.append("Δ = 베이스라인 대비. ⚠️=거래수 변화 작은데(±15) 성과 출렁 → 경로노이즈 의심.")
    L.append("")
    hdr = "| 변형 | 방향 | 최종$ | Δ$% | MDD% | ΔMDD | Sharpe | ΔSh | 거래 | " + " | ".join(years) + " |"
    L.append(hdr)
    L.append("|" + "---|" * (8 + len(years)))
    cur_cat = None
    catname = {"gate":"리스크 게이트","score":"스코어러","ema":"ema_cross","multi":"multi_tf",
               "size":"사이징","comp":"컴포넌트 on/off","base":"베이스라인"}
    for r in results:
        if r["cat"] != cur_cat:
            cur_cat = r["cat"]
            L.append(f"| **— {catname.get(cur_cat,cur_cat)} —** | | | | | | | | | " + " | "*(len(years)-1) + " |")
        d_final = (r["final"]/base["final"]-1)*100 if base["final"] else 0
        d_mdd = r["mdd"] - base["mdd"]
        d_sh = r["sharpe"] - base["sharpe"]
        noise = "⚠️" if abs(r["trades"]-base["trades"]) <= 15 and abs(d_final) > 25 and r["name"] != "BASELINE" else ""
        ycols = " | ".join(f"{r['yearly'].get(y,'—')}" for y in years)
        L.append(f"| {r['name']}{noise} | {r['dir']} | {r['final']:,.0f} | {d_final:+.0f}% | "
                 f"{r['mdd']:.1f} | {d_mdd:+.1f} | {r['sharpe']:.2f} | {d_sh:+.2f} | {r['trades']} | {ycols} |")
    L.append("")
    L.append("_연도 값 = 해당 연도 경로슬라이스 수익%. 단독 백테 아님(복리 경로 연속)._")
    Path("OAT_ABLATION.md").write_text("\n".join(L))
    print(f"\n저장 완료: OAT_ABLATION_RESULTS.json, OAT_ABLATION.md")


if __name__ == "__main__":
    main()
