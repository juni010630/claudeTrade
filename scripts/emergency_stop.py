"""긴급 전량청산 — 모든 열린 주문 취소 + 전 포지션 시장가 청산.

봇과 독립 실행 (라이브 루프 무수정). 서버에서:
    sudo systemctl stop trade-bot          # 1) 봇 먼저 정지 (재진입 방지)
    python scripts/emergency_stop.py       # 2) dry-run: 청산 대상 확인만
    python scripts/emergency_stop.py --yes # 3) 실제 청산

수기 청산 대신 이 절차 사용 — 주문취소→reduceOnly 시장가 순서 보장, 텔레그램 기록 남김.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# .env 로드 (live_trade와 동일 — python-dotenv 없이 직접 파싱)
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from execution.notifier import TelegramNotifier  # noqa: E402
from scripts.live_trade import build_exchange  # noqa: E402


def bot_running() -> bool:
    r = subprocess.run(["pgrep", "-f", "live_trade.py"], capture_output=True)
    return r.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="실제 청산 실행 (없으면 dry-run)")
    parser.add_argument("--demo", action="store_true", help="테스트넷")
    args = parser.parse_args()

    if bot_running():
        print("⛔ live_trade 프로세스가 아직 돌고 있다 — 먼저 정지할 것:")
        print("   sudo systemctl stop trade-bot")
        sys.exit(1)

    ex = build_exchange(demo=args.demo)
    positions = [p for p in ex.fetch_positions() if float(p.get("contracts") or 0) != 0]

    if not positions:
        print("열린 포지션 없음.")
        return

    print(f"열린 포지션 {len(positions)}개:")
    for p in positions:
        side = p.get("side", "?")
        print(f"  {p['symbol']:20} {side:5} qty={p['contracts']}  uPnL={p.get('unrealizedPnl')}")

    if not args.yes:
        print("\n[DRY-RUN] 실제 청산하려면 --yes 를 붙일 것.")
        return

    closed, failed = [], []
    for p in positions:
        sym = p["symbol"]
        side = p.get("side", "")
        close_side = "sell" if side == "long" else "buy"
        try:
            ex.cancel_all_orders(sym)
        except Exception as e:
            print(f"  주문취소 실패 {sym}: {e}")
        try:
            qty = ex.amount_to_precision(sym, abs(float(p["contracts"])))
            ex.create_order(sym, "market", close_side, qty, None, {"reduceOnly": True})
            closed.append(sym)
            print(f"  ✅ 청산: {sym} {close_side} {qty}")
        except Exception as e:
            failed.append(sym)
            print(f"  ❌ 청산 실패 {sym}: {e}")

    msg = f"🚨 긴급 전량청산 실행\n청산 {len(closed)}개: {', '.join(closed) or '-'}"
    if failed:
        msg += f"\n실패 {len(failed)}개: {', '.join(failed)} — 수동 확인 필요!"
    notifier = TelegramNotifier.from_env()
    notifier.send(msg)
    print(f"\n완료: 청산 {len(closed)} / 실패 {len(failed)} (텔레그램 전송됨)")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
