"""라이브 엣지부패 모니터 — 매일 1회 (systemd timer), 프로덕션 무수정.

① 신호 패리티: ANCHOR부터 현재까지 v21d 리플레이 vs 라이브 실체결(trades.csv +
   state.json 보유 포지션) 진입 이벤트 대조. 불일치 1건 = WARN.
   (경로의존 때문에 윈도우 분할 금지 — 항상 ANCHOR부터 전체 리플레이)
② 체결 품질: 매칭된 진입의 라이브 vs 리플레이 가격 괴리(bps, +=불리).
   중앙값 > 15bps 또는 단건 > 50bps = WARN. (백테 가정 5bps)
③ 엣지부패: 일별 equity 로그의 30d/90d 수익률을 백테 분포
   (config/edge_baseline_v21d.json) 백분위로 환산. p5 미만 = ALERT.
   로그 31일 누적 전엔 침묵. 수익률만 비교(일별 표본 vs 1h 분포의 Sharpe 왜곡 방지).

자동 행동 없음 — 경보만. 성과 기반 자동 감속은 tail-cut/스트릭 기각 교훈상 금지,
파국 경로는 딥플로어(live_trade)가 담당.

알림: WARN/ALERT 시 즉시, 월요일엔 무소식이어도 주간 요약. 실행 실패도 텔레그램.
재배포(config 변경) 시: ANCHOR/CONFIG/BASELINE 갱신 + 베이스라인 재생성 필수.
(ANCHOR_CAPITAL은 라이브 잔고 자동조회로 대체 — 신호/슬리피지 패리티엔 무관해 정밀값 불요.)

사용:
  python scripts/edge_monitor.py                # 정기 실행 (timer)
  python scripts/edge_monitor.py --force-notify # 요약 강제 발송 (설치 검증용)
  python scripts/edge_monitor.py --no-balance   # 잔고 조회 생략 (로컬 테스트)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import ccxt
import numpy as np
import pandas as pd
import yaml

from data.loader import DataLoader
import scripts.run_backtest as rb

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("edge_monitor")

# ── 기준점 (재배포 시 갱신) ───────────────────────────────────────
ANCHOR = pd.Timestamp("2026-06-19 23:00", tz="UTC")  # ⚠️ v21d 실제 재시작 = 06-19 22:48 UTC → 첫 1h봉 23:00.
#   날짜만(00:00)으로 두면 재시작 전 ~23h를 리플레이가 v21d로 덮어, state.json 이월 포지션(앵커 이전
#   진입분)을 flat-start 리플레이가 앵커에서 재진입 → '리플레이에만 존재' 오경보(2026-06-20 UNI/ARPA MR 사례).
#   v21d = v21c + ema·multi early_exit_on_opp(반대 모멘텀 시 청산/리버스). 사이징·배분·진입 불변,
#   추세책 일부 포지션이 반대신호 시 조기청산 → v21c 보유분과 전환윈도 '리플레이/라이브전용' 오경보 가능(무시).
#   재배포 시 ANCHOR = 실제 systemctl 재시작 시각의 다음 1h봉 경계로 갱신할 것(systemctl show -p ExecMainStartTimestamp).
CONFIG = "config/final_v21d_eexit.yaml"   # 2026-06-19 v21d 배포 (v21c + 반대신호 조기청산)
BASELINE = "config/edge_baseline_v21d.json"  # 생성 전엔 ③ 자동 스킵 (31일 누적 후 발동)
# replay 시작 자본: 신호/슬리피지 패리티엔 무관(포지션 사이즈만 스케일하고, ③ 롤링은
# 실잔고 로그를 씀) → 정밀값 불필요. main에서 라이브 잔고를 API로 긁어 쓰고,
# 못 긁으면(--no-balance/조회 실패) 이 폴백 사용.
ANCHOR_CAPITAL_FALLBACK = 10000.0

# ── 파일 경로 ─────────────────────────────────────────────────────
CACHE_DIR = Path("data/edge_cache")
EQUITY_LOG = Path("data/edge_equity_log.csv")
TRADES_CSV = Path("trades.csv")
STATE_JSON = Path("data/state.json")

# ── 사전선언 임계 (사후 완화 금지) ────────────────────────────────
SLIP_MED_WARN_BPS = 15.0   # 매칭 진입 슬리피지 중앙값
SLIP_MAX_WARN_BPS = 50.0   # 단건 최대
PCTL_ALERT = 5.0           # 롤링 수익률 백테 백분위
MATCH_TOL = pd.Timedelta(hours=1)

_TF_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
_PREFETCH_BARS = 320  # 라이브 lookback 300 + 여유


def _load_env() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


# ── ① 데이터 갱신 (증분) ──────────────────────────────────────────
def refresh_cache(symbols: list[str], timeframes: list[str]) -> None:
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    now_ms = ex.milliseconds()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for sym in symbols:
        for tf in timeframes:
            path = CACHE_DIR / f"ohlcv_{sym}_{tf}.parquet"
            tf_ms = _TF_MS[tf]
            old = pd.read_parquet(path) if path.exists() else None
            since_ms = (int(old["timestamp"].max().timestamp() * 1000) + tf_ms
                        if old is not None and len(old)
                        else int(ANCHOR.timestamp() * 1000) - _PREFETCH_BARS * tf_ms)
            rows = []
            while since_ms < now_ms:
                batch = ex.fetch_ohlcv(sym, tf, since=since_ms, limit=1000)
                if not batch:
                    break
                rows.extend(batch)
                since_ms = batch[-1][0] + tf_ms
                if len(batch) < 1000:
                    break
            if rows:
                new = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
                new["timestamp"] = pd.to_datetime(new["timestamp"], unit="ms", utc=True)
                # 미완성 봉 제거 (봉시작 + tf가 현재 이후 = 아직 진행 중)
                new = new[new["timestamp"] + pd.Timedelta(milliseconds=tf_ms)
                          <= pd.Timestamp.now(tz="UTC")]
                df = (pd.concat([old, new]) if old is not None else new)
                df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
                df.to_parquet(path)
        # 펀딩 (scorer confluence 점수 + 펀딩 시뮬에 필요)
        fpath = CACHE_DIR / f"funding_{sym}_8h.parquet"
        fold = pd.read_parquet(fpath) if fpath.exists() else None
        fsince = (int(fold["timestamp"].max().timestamp() * 1000) + 1
                  if fold is not None and len(fold)
                  else int(ANCHOR.timestamp() * 1000) - 3 * 86_400_000)
        frows = []
        while True:
            batch = ex.fetch_funding_rate_history(sym, since=fsince, limit=1000)
            if not batch:
                break
            frows.extend(batch)
            fsince = batch[-1]["timestamp"] + 1
            if len(batch) < 1000:
                break
        if frows:
            fnew = pd.DataFrame([{
                "timestamp": pd.Timestamp(r["timestamp"], unit="ms", tz="UTC"),
                "symbol": r.get("symbol", sym),
                "rate": float(r.get("fundingRate") or 0.0),
            } for r in frows])
            fdf = (pd.concat([fold, fnew]) if fold is not None else fnew)
            fdf = fdf.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
            fdf.to_parquet(fpath)
    logger.info("캐시 갱신 완료 (%d심볼 × %s)", len(symbols), timeframes)


# ── ② 리플레이 ────────────────────────────────────────────────────
def run_replay(p: dict, capital: float):
    loader = DataLoader(
        symbols=p["symbols"], timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=str(CACHE_DIR),
        lookback=p.get("data", {}).get("lookback_bars", 300),
    )
    engine = rb.build_engine(p, capital)
    # loader.iterate의 since는 봉 open 기준, 스냅샷 timestamp는 close 기준(open+1h).
    # 라이브가 ANCHOR 정각에 처리한 스냅샷(= open ANCHOR-1h 봉)을 포함하려면 1h 앞당김.
    primary_delta = pd.Timedelta(p.get("primary_timeframe", "1h"))
    engine.run(loader.iterate(since=ANCHOR - primary_delta))
    return engine


# ── ③ 진입 이벤트 수집 ────────────────────────────────────────────
_EVENT_COLS = ["symbol", "strategy", "direction", "entry_time", "entry_price",
               "status", "exit_time", "exit_price", "exit_reason"]


def live_entries() -> pd.DataFrame:
    rows = []
    if TRADES_CSV.exists():
        df = pd.read_csv(TRADES_CSV)
        for _, r in df.iterrows():
            et = pd.Timestamp(r["entry_time"])
            et = et.tz_localize("UTC") if et.tzinfo is None else et.tz_convert("UTC")
            if et < ANCHOR:
                continue
            xt = pd.Timestamp(r["exit_time"])
            xt = xt.tz_localize("UTC") if xt.tzinfo is None else xt.tz_convert("UTC")
            rows.append(dict(symbol=r["symbol"], strategy=r["strategy"],
                             direction=r["direction"], entry_time=et,
                             entry_price=float(r["entry_price"]), status="closed",
                             exit_time=xt, exit_price=float(r["exit_price"]),
                             exit_reason=r.get("exit_reason", "")))
    if STATE_JSON.exists():
        data = json.loads(STATE_JSON.read_text())
        for sym, pos in data.get("positions", {}).items():
            ot = pd.Timestamp(pos["opened_at"])
            ot = ot.tz_localize("UTC") if ot.tzinfo is None else ot.tz_convert("UTC")
            if ot < ANCHOR:
                continue
            rows.append(dict(symbol=sym, strategy=pos["strategy"],
                             direction=pos["direction"], entry_time=ot,
                             entry_price=float(pos["entry_price"]), status="open",
                             exit_time=None, exit_price=None, exit_reason=None))
    return pd.DataFrame(rows, columns=_EVENT_COLS)


def replay_entries(engine) -> pd.DataFrame:
    rows = []
    for r in engine.ledger.records:
        rows.append(dict(symbol=r.symbol, strategy=r.strategy, direction=r.direction,
                         entry_time=r.entry_time, entry_price=r.entry_price,
                         status="closed", exit_time=r.exit_time,
                         exit_price=r.exit_price, exit_reason=r.exit_reason))
    for sym, pos in engine.tracker.snapshot().positions.items():
        rows.append(dict(symbol=sym, strategy=pos.strategy, direction=pos.direction,
                         entry_time=pos.opened_at, entry_price=pos.entry_price,
                         status="open", exit_time=None, exit_price=None, exit_reason=None))
    return pd.DataFrame(rows, columns=_EVENT_COLS)


# ── ④ 대조 ────────────────────────────────────────────────────────
def match_events(live: pd.DataFrame, rep: pd.DataFrame):
    matched, live_used = [], set()
    rep_only = []
    for _, rr in rep.iterrows():
        cand = None
        for li, lr in live.iterrows():
            if li in live_used:
                continue
            if (lr["symbol"] == rr["symbol"] and lr["strategy"] == rr["strategy"]
                    and lr["direction"] == rr["direction"]
                    and abs(lr["entry_time"] - rr["entry_time"]) <= MATCH_TOL):
                cand = li
                break
        if cand is None:
            rep_only.append(rr)
        else:
            live_used.add(cand)
            matched.append((live.loc[cand], rr))
    live_only = [lr for li, lr in live.iterrows() if li not in live_used]
    return matched, live_only, rep_only


def slippage_bps(live_row, rep_row) -> float:
    """진입가 괴리 bps. + = 라이브가 불리한 방향."""
    raw = (live_row["entry_price"] - rep_row["entry_price"]) / rep_row["entry_price"] * 1e4
    return raw if live_row["direction"] == "long" else -raw


# ── ⑤ 일별 equity 로그 + 롤링 백분위 ──────────────────────────────
def fetch_usdt_balance() -> float | None:
    """라이브 USDT 총잔고. 조회 실패 시 None."""
    try:
        ex = ccxt.binanceusdm({
            "apiKey": os.environ.get("BINANCE_API_KEY", ""),
            "secret": os.environ.get("BINANCE_SECRET", ""),
            "enableRateLimit": True,
        })
        return float(ex.fetch_balance().get("USDT", {}).get("total", 0) or 0) or None
    except Exception:
        logger.warning("잔고 조회 실패 — replay 폴백 자본 사용")
        return None


def append_equity_log(usdt: float | None) -> None:
    if usdt is None:
        return  # --no-balance or 조회 실패
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    if EQUITY_LOG.exists():
        with open(EQUITY_LOG) as f:
            if any(row.startswith(today) for row in f):
                return  # 오늘 이미 기록
    EQUITY_LOG.parent.mkdir(parents=True, exist_ok=True)
    new_file = not EQUITY_LOG.exists()
    with open(EQUITY_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["date", "equity"])
        w.writerow([today, f"{usdt:.2f}"])
    logger.info("equity 로그: %s $%.2f", today, usdt)


def rolling_percentiles() -> list[str]:
    """30d/90d 라이브 수익률 → 백테 분포 백분위. 표본 부족 시 빈 리스트."""
    if not EQUITY_LOG.exists() or not Path(BASELINE).exists():
        return []
    eq = pd.read_csv(EQUITY_LOG, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if len(eq) < 31:
        return [f"③ 롤링: 표본 {len(eq)}/31일 — 누적 중"]
    grids = json.loads(Path(BASELINE).read_text())["grids"]
    last_date = eq["date"].iloc[-1]
    last_eq = eq["equity"].iloc[-1]
    out = []
    for name, days in (("30d", 30), ("90d", 90)):
        # 행 위치가 아닌 '날짜 기준' lookback — 로그에 결손일(잔고조회 실패/서버 다운/
        # --no-balance)이 섞이면 positional iloc은 30/90일보다 더 과거를 참조해
        # 수익률·백분위가 왜곡된다(baseline은 연속 1h봉 기준이라 호라이즌 불일치).
        target = last_date - pd.Timedelta(days=days)
        prior = eq[eq["date"] <= target]
        if prior.empty:
            continue  # 아직 days일 치 히스토리 없음
        base_eq = prior["equity"].iloc[-1]
        ret = last_eq / base_eq - 1
        g = grids[f"ret_{name}"]
        pctl = float(np.interp(ret, g["values"], g["percentiles"]))
        flag = " 🚨ALERT" if pctl < PCTL_ALERT else ""
        out.append(f"③ {name} 수익률 {ret:+.1%} = 백테 p{pctl:.0f}{flag}")
    return out


# ── 메인 ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-notify", action="store_true")
    parser.add_argument("--no-balance", action="store_true")
    args = parser.parse_args()
    _load_env()

    from execution.notifier import TelegramNotifier
    notifier = TelegramNotifier.from_env()

    try:
        with open(CONFIG) as f:
            p = yaml.safe_load(f)
        usdt = None if args.no_balance else fetch_usdt_balance()
        capital = usdt if usdt else ANCHOR_CAPITAL_FALLBACK
        refresh_cache(p["symbols"], p["timeframes"])
        engine = run_replay(p, capital)
        live = live_entries()
        rep = replay_entries(engine)
        matched, live_only, rep_only = match_events(live, rep)
        append_equity_log(usdt)

        lines, severity = [], "OK"
        days = max(0, (pd.Timestamp.now(tz="UTC") - ANCHOR).days)
        lines.append(f"기간 {ANCHOR.date()}~ ({days}일) | 라이브 {len(live)}건 / 리플레이 {len(rep)}건")

        # ① 신호 패리티
        if live_only or rep_only:
            severity = "WARN"
            for lr in live_only:
                lines.append(f"① 라이브에만 존재: {lr['symbol']} {lr['direction']} "
                             f"{lr['strategy']} @{lr['entry_time']:%m-%d %H:%M}")
            for rr in rep_only:
                lines.append(f"① 리플레이에만 존재(라이브 미체결): {rr['symbol']} "
                             f"{rr['direction']} {rr['strategy']} @{rr['entry_time']:%m-%d %H:%M}")
        else:
            lines.append(f"① 신호 패리티: {len(matched)}/{len(matched)} 일치 ✓")

        # ② 체결 품질
        if matched:
            slips = [slippage_bps(l, r) for l, r in matched]
            med, mx = float(np.median(slips)), float(max(slips, key=abs))
            note = ""
            # 중앙값 기준은 표본 5건부터 (n=2로 매일 같은 WARN 반복 = 경보 피로 방지).
            # 단건 50bps 초과는 즉시 발동.
            if ((len(slips) >= 5 and abs(med) > SLIP_MED_WARN_BPS)
                    or abs(mx) > SLIP_MAX_WARN_BPS):
                severity = "WARN" if severity == "OK" else severity
                note = " ⚠️"
            lines.append(f"② 진입 슬리피지(n={len(slips)}): 중앙값 {med:+.1f}bps / "
                         f"최대 {mx:+.1f}bps (가정 5bps){note}")

        # ③ 엣지부패
        roll = rolling_percentiles()
        if any("ALERT" in s for s in roll):
            severity = "ALERT"
        lines.extend(roll)

        head = {"OK": "🩺", "WARN": "⚠️", "ALERT": "🚨"}[severity]
        body = f"{head} <b>엣지 모니터 [{severity}]</b>\n" + "\n".join(lines)
        print(body)
        is_monday = pd.Timestamp.now(tz="UTC").weekday() == 0
        if notifier.enabled and (severity != "OK" or is_monday or args.force_notify):
            notifier.notify_info(body)
            time.sleep(1)
    except Exception as e:
        logger.exception("edge_monitor 실패")
        if notifier.enabled:
            notifier.notify_info(f"🔧 edge_monitor 실행 실패: {type(e).__name__}: {str(e)[:200]}")
            time.sleep(1)
        sys.exit(1)


if __name__ == "__main__":
    main()
