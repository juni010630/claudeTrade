"""ADX 스코어링 스윕 — 4개 백테스트 병렬 실행."""
import subprocess, sys, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CONFIGS = [
    ("baseline",        "config/final_v13_eth.yaml"),
    ("ADX>40→+2",       "config/adx_strong_bonus.yaml"),
    ("ADX>50→차단",     "config/adx_high_cutoff.yaml"),
    ("ADX>40+2/55차단", "config/adx_combined.yaml"),
]

def run_bt(label, cfg):
    r = subprocess.run(
        [sys.executable, "scripts/run_backtest.py", "--params", cfg],
        capture_output=True, text=True
    )
    return label, r.stdout, r.stderr

def parse(out):
    def get(pat):
        m = re.search(pat, out)
        return m.group(1) if m else "—"
    return {
        "equity":  get(r"최종 자산[^$]*\$([\d,]+)"),
        "sharpe":  get(r"Sharpe[:\s]+([\d.]+)"),
        "mdd":     get(r"Max Drawdown[:\s]+([-\d.]+%)"),
        "wr":      get(r"승률[:\s]+([\d.]+%)"),
        "pf":      get(r"Profit Factor[:\s]+([\d.]+)"),
        "trades":  get(r"총 거래 수[:\s]+(\d+)"),
    }

print(f"{'백테스트':22s} {'거래수':>6} {'승률':>7} {'PF':>6} {'Sharpe':>7} {'MDD':>8} {'최종자산':>14}")
print("─" * 80)

results = {}
with ThreadPoolExecutor(max_workers=4) as ex:
    futs = {ex.submit(run_bt, lbl, cfg): lbl for lbl, cfg in CONFIGS}
    for fut in as_completed(futs):
        lbl, out, err = fut.result()
        results[lbl] = parse(out)
        print(f"  완료: {lbl}", flush=True)

print()
for lbl, _ in CONFIGS:
    m = results.get(lbl, {})
    print(f"{lbl:22s} {m.get('trades','—'):>6} {m.get('wr','—'):>7} "
          f"{m.get('pf','—'):>6} {m.get('sharpe','—'):>7} "
          f"{m.get('mdd','—'):>8} {m.get('equity','—'):>14}")

if __name__ == "__main__":
    pass
