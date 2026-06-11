"""새 매매 방식 greedy search — 레벨 0 스크리닝 (IS 2022-01-01 ~ 2024-12-31 전용).

기존 엣지(1h/4h 컨플루언스 추세, 1d RSI극단 역추세)와 메커니즘이 다른 7개 패밀리를
광역 유니버스(1d 데이터 2022-01 이전 시작 심볼)에서 standalone 평가.
OOS(2025~)는 봉인 — 데이터 자체를 IS 종료일로 절단해 로드.

신호 = t-1 완성봉, 진입 = t 시가. 양쪽터치 시 SL 우선(비관적). 비용 왕복 10bp.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

COST = 0.0010  # 왕복 10bp
IS_END = "2024-12-31"
CACHE = Path(__file__).parent.parent / "data" / "cache"


def load_universe(tf="1d", start_cut="2022-01-01"):
    out = {}
    for f in sorted(os.listdir(CACHE)):
        if not (f.startswith("ohlcv_") and f.endswith(f"_{tf}.parquet")):
            continue
        sym = f[6:-(len(tf) + 9)]
        df = pd.read_parquet(CACHE / f)
        df = df[df.timestamp <= IS_END]
        if len(df) < 200 or str(df.timestamp.iloc[0])[:10] > start_cut:
            continue
        out[sym] = df.reset_index(drop=True)
    return out


def ema(c, n):
    return c.ewm(span=n, adjust=False).mean()


def rsi(c, n):
    dd = c.diff()
    up = dd.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-dd.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def sim_trades(df, sig, tp_mult, sl_mult, max_hold):
    """sig[t] = t-1봉 기준 신호(+1/-1/0), t 시가 진입. 고정 ATR TP/SL + 보유캡.
    반환: (entry_ts, r) 리스트. SL 우선 체결(비관). 보유캡 도달 시 해당봉 종가 청산."""
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    ts = df["timestamp"].values
    a = df["__atr"].values
    n = len(df)
    trades = []
    i = 1
    while i < n:
        s = sig[i]
        if s == 0 or not np.isfinite(a[i - 1]) or a[i - 1] <= 0:
            i += 1
            continue
        e = o[i]
        if s > 0:
            tp, sl = e + tp_mult * a[i - 1], e - sl_mult * a[i - 1]
        else:
            tp, sl = e - tp_mult * a[i - 1], e + sl_mult * a[i - 1]
        r = None
        j = i
        while j < n:
            if s > 0:
                if l[j] <= sl:
                    r = (sl - e) / e
                    break
                if h[j] >= tp:
                    r = (tp - e) / e
                    break
            else:
                if h[j] >= sl:
                    r = (e - sl) / e
                    break
                if l[j] <= tp:
                    r = (e - tp) / e
                    break
            if j - i + 1 >= max_hold:
                r = (c[j] - e) / e * (1 if s > 0 else -1)
                break
            j += 1
        if r is None:  # 데이터 끝 — 마지막 종가 청산
            r = (c[-1] - e) / e * (1 if s > 0 else -1)
            j = n - 1
        trades.append((ts[i], r - COST))
        i = j + 1  # 청산봉 다음부터 (심볼당 1포지션)
    return trades


def agg(trades):
    if not trades:
        return None
    tdf = pd.DataFrame(trades, columns=["t", "r"])
    tdf["year"] = pd.to_datetime(tdf.t).dt.year
    gw = tdf.r[tdf.r > 0].sum()
    gl = -tdf.r[tdf.r <= 0].sum()
    pf = gw / gl if gl > 0 else 99.0
    yr = tdf.groupby("year").r.sum()
    yrs = {y: round(yr.get(y, 0) * 100) for y in [2022, 2023, 2024]}
    return {
        "n": len(tdf),
        "wr": round((tdf.r > 0).mean() * 100),
        "pf": round(pf, 2),
        "totR%": round(tdf.r.sum() * 100),
        "avgR_bp": round(tdf.r.mean() * 10000),
        "yr": yrs,
        "pos_years": sum(1 for v in yrs.values() if v > 0),
    }


# ── 패밀리 정의 (1d) ──────────────────────────────────────────────


def f1_donchian(df, N, exit_mode, tp=4.0, sl=2.0, hold=40):
    c = df["close"]
    hh = c.rolling(N).max().shift(1)  # t-1까지의 직전 N일 최고종가 (t-1 제외 위해 shift 추가)
    ll = c.rolling(N).min().shift(1)
    prev_c = c.shift(1)
    long_sig = (prev_c > hh.shift(1)).fillna(False)
    short_sig = (prev_c < ll.shift(1)).fillna(False)
    sig = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    if exit_mode == "tpsl":
        return sim_trades(df, sig, tp, sl, hold)
    # channel exit: 반대편 N//2 채널 이탈 종가 확인 후 다음 시가 청산 + SL 2ATR 보호
    o, h, l, cv = df["open"].values, df["high"].values, df["low"].values, c.values
    a = df["__atr"].values
    ts = df["timestamp"].values
    exit_hi = c.rolling(N // 2).max().shift(1).values
    exit_lo = c.rolling(N // 2).min().shift(1).values
    n = len(df)
    trades = []
    i = 1
    while i < n:
        s = sig[i]
        if s == 0 or not np.isfinite(a[i - 1]) or a[i - 1] <= 0:
            i += 1
            continue
        e = o[i]
        slp = e - s * sl * a[i - 1]
        r = None
        j = i
        while j < n - 1:
            if s > 0 and l[j] <= slp:
                r = (slp - e) / e
                break
            if s < 0 and h[j] >= slp:
                r = (e - slp) / e
                break
            if j > i and ((s > 0 and cv[j] < exit_lo[j]) or (s < 0 and cv[j] > exit_hi[j])):
                r = (o[j + 1] - e) / e * s
                break
            j += 1
        if r is None:
            r = (cv[-1] - e) / e * s
            j = n - 1
        trades.append((ts[i], r - COST))
        i = j + 1
    return trades


def f2_pullback(df, rsi_th, tp=3.0, sl=2.0, hold=15):
    c = df["close"]
    e50 = ema(c, 50)
    r3 = rsi(c, 3)
    up = (c.shift(1) > e50.shift(1)) & (e50.shift(1) > e50.shift(2))
    dn = (c.shift(1) < e50.shift(1)) & (e50.shift(1) < e50.shift(2))
    long_sig = up & (r3.shift(1) < rsi_th)
    short_sig = dn & (r3.shift(1) > 100 - rsi_th)
    sig = np.where(long_sig.fillna(False), 1, np.where(short_sig.fillna(False), -1, 0))
    return sim_trades(df, sig, tp, sl, hold)


def f3_squeeze(df, pct_th=0.25, N=20, tp=4.0, sl=2.0, hold=30):
    c = df["close"]
    natr = df["__atr"] / c
    rank = natr.rolling(100).rank(pct=True)
    squeezed = rank.shift(2) < pct_th  # 돌파 전봉 기준 압축 상태
    hh = c.rolling(N).max().shift(2)
    ll = c.rolling(N).min().shift(2)
    long_sig = squeezed & (c.shift(1) > hh)
    short_sig = squeezed & (c.shift(1) < ll)
    sig = np.where(long_sig.fillna(False), 1, np.where(short_sig.fillna(False), -1, 0))
    return sim_trades(df, sig, tp, sl, hold)


def f5_leadlag(alt_dfs, btc_df, th=0.02, hold=2):
    """BTC 전일 수익 |ret|>th → 다음날 알트 동방향 시가 진입, hold일 뒤 시가 청산."""
    btc = btc_df.set_index("timestamp")["close"]
    btc_ret = btc.pct_change()
    trades = []
    for sym, df in alt_dfs.items():
        if sym in ("BTCUSDT", "BTCDOMUSDT"):
            continue
        ts = df["timestamp"].values
        o = df["open"].values
        idx = pd.DatetimeIndex(df["timestamp"])
        prev_ret = btc_ret.reindex(idx).shift(1).values  # t봉에 BTC t-1봉 ret 정렬
        n = len(df)
        i = 1
        while i < n - hold:
            pr = prev_ret[i]
            if np.isfinite(pr) and abs(pr) > th:
                s = 1 if pr > 0 else -1
                r = (o[i + hold] - o[i]) / o[i] * s - COST
                trades.append((ts[i], r))
                i += hold
            else:
                i += 1
    return trades


def f7_tsmom(dfs, lookback=30, rebal=5):
    """상시 포지션 TSMOM: sign(ret_lookback), rebal일마다 갱신. 일별수익 합산."""
    rets = []
    for sym, df in dfs.items():
        c = df.set_index("timestamp")["close"]
        r = c.pct_change()
        mom = np.sign(c.pct_change(lookback))
        pos = mom.shift(1).copy()
        pos.iloc[: lookback + 1] = 0
        # rebal일마다만 포지션 변경
        keep = np.arange(len(pos)) % rebal != 0
        pos[keep] = np.nan
        pos = pos.ffill().fillna(0)
        turn = pos.diff().abs().fillna(0)
        rets.append(pos * r - turn * COST / 2)
    port = pd.concat(rets, axis=1).mean(axis=1).dropna()
    yr = port.groupby(port.index.year).sum()
    sharpe = port.mean() / port.std() * np.sqrt(365) if port.std() > 0 else 0
    eq = (1 + port).cumprod()
    mdd = ((eq - eq.cummax()) / eq.cummax()).min()
    return {
        "n": "daily",
        "wr": "-",
        "pf": round(sharpe, 2),
        "totR%": round(port.sum() * 100),
        "avgR_bp": round(port.mean() * 10000, 1),
        "yr": {y: round(yr.get(y, 0) * 100) for y in [2022, 2023, 2024]},
        "pos_years": sum(1 for y in [2022, 2023, 2024] if yr.get(y, 0) > 0),
        "mdd%": round(mdd * 100),
    }


# ── 1h 패밀리 ─────────────────────────────────────────────────────


def f4_shock(df, k=3.0, vmult=3.0, mode="follow", tp=3.0, sl=1.5, hold=24):
    c = df["close"]
    r1 = c.pct_change()
    sigma = r1.rolling(168).std()
    vma = df["volume"].rolling(168).mean()
    shock = (r1.abs() > k * sigma) & (df["volume"] > vmult * vma)
    direction = np.sign(r1) * (1 if mode == "follow" else -1)
    sig_ser = (shock.shift(1) * direction.shift(1)).fillna(0)
    sig = sig_ser.values.astype(int)
    return sim_trades(df, sig, tp, sl, hold)


def f6_volbreak(df_1h, k=0.5):
    """일중 변동성 돌파: 당일시가 ± k×전일레인지 첫 터치 진입, 당일 마지막 1h 종가 청산."""
    d = df_1h.copy()
    d["date"] = d.timestamp.dt.floor("D")
    trades = []
    days = list(d.groupby("date", sort=True))
    prev_range = None
    for di in range(1, len(days)):
        pday = days[di - 1][1]
        prev_range = pday.high.max() - pday.low.min()
        day = days[di][1]
        if prev_range <= 0 or len(day) < 12:
            continue
        dopen = day.open.iloc[0]
        up_lv = dopen + k * prev_range
        dn_lv = dopen - k * prev_range
        hi, lo, op = day.high.values, day.low.values, day.open.values
        up_hit = np.argmax(hi >= up_lv) if (hi >= up_lv).any() else 10**9
        dn_hit = np.argmax(lo <= dn_lv) if (lo <= dn_lv).any() else 10**9
        if up_hit == 10**9 and dn_hit == 10**9:
            continue
        if up_hit <= dn_hit:
            s, e = 1, max(op[up_hit], up_lv)
        else:
            s, e = -1, min(op[dn_hit], dn_lv)
        x = day.close.iloc[-1]
        trades.append((day.timestamp.iloc[0], (x - e) / e * s - COST))
    return trades


def run_1d_families():
    dfs = load_universe("1d")
    print(f"universe 1d: {len(dfs)} symbols")
    for df in dfs.values():
        df["__atr"] = atr(df, 14)
    results = {}

    for N in [30, 55, 100]:
        for em in ["tpsl", "channel"]:
            tr = []
            for df in dfs.values():
                tr += f1_donchian(df, N, em)
            results[f"F1_donchian_N{N}_{em}"] = agg(tr)

    for th in [10, 15, 25]:
        tr = []
        for df in dfs.values():
            tr += f2_pullback(df, th)
        results[f"F2_pullback_rsi{th}"] = agg(tr)

    for pct in [0.2, 0.3]:
        tr = []
        for df in dfs.values():
            tr += f3_squeeze(df, pct)
        results[f"F3_squeeze_p{int(pct*100)}"] = agg(tr)

    btc = dfs.get("BTCUSDT")
    for th, hold in [(0.02, 1), (0.02, 3), (0.04, 3)]:
        results[f"F5_leadlag_th{int(th*100)}_h{hold}"] = agg(f5_leadlag(dfs, btc, th, hold))

    for lb in [20, 30, 60]:
        results[f"F7_tsmom_lb{lb}"] = f7_tsmom(dfs, lb)

    return results


def run_1h_families():
    dfs = load_universe("1h")
    print(f"universe 1h: {len(dfs)} symbols")
    results = {}
    for df in dfs.values():
        df["__atr"] = atr(df, 14)
    for k, mode in [(3.0, "follow"), (4.0, "follow"), (3.0, "fade")]:
        tr = []
        for df in dfs.values():
            tr += f4_shock(df, k=k, mode=mode)
        results[f"F4_shock_k{k}_{mode}"] = agg(tr)
    sub = {s: d for s, d in list(dfs.items())}
    for k in [0.5, 0.7]:
        tr = []
        for df in sub.values():
            tr += f6_volbreak(df, k)
        results[f"F6_volbreak_k{k}"] = agg(tr)
    return results


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = {}
    if which in ("all", "1d"):
        results.update(run_1d_families())
    if which in ("all", "1h"):
        results.update(run_1h_families())
    print(f"\n{'variant':<28}{'n':>7}{'wr':>5}{'pf':>7}{'totR%':>8}{'avg_bp':>8}{'posY':>6}  yearly")
    for k, v in results.items():
        if v is None:
            print(f"{k:<28}  (no trades)")
            continue
        print(
            f"{k:<28}{v['n']:>7}{v['wr']:>5}{v['pf']:>7}{v['totR%']:>8}{v['avgR_bp']:>8}"
            f"{v['pos_years']:>6}  {v['yr']}"
        )


if __name__ == "__main__":
    main()
