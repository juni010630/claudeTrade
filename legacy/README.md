# legacy/ — 일회성 실험 코드 아카이브 (2026-06-11 정리)

v18(`config/final_v18_triple.yaml`) 라이브 기준으로 운영·백테 경로에서 참조되지 않는
일회성 실험 코드를 모아둔 곳. **삭제가 아니라 보존** — 실험 기록(repo 루트 *.md)의
재현 코드가 여기 있다.

## 구성
- `scripts/` — 실험 스크립트 136개 (스윕/어블레이션/진단/검증 세션 산출물)
- `config/` — 실험 config 57개 (p2~p7, test_*, uni*, adx_*, bucket_* 등 + final_v18_triple_pools=기각된 v18b, 2026-06-12 추가)
- `tests/` — 죽은 테스트 (`test_full_backtest.py` — 삭제된 strategies.momentum_breakout import)

## 이동 기준 (2026-06-11 검증)
- 운영 진입점(live_trade, run_backtest, fetch_data, edge_monitor, edge_baseline_gen,
  oi_collector, emergency_stop, regime_now, trend_index_now) + tests/ 에서 import/경로/субprocess
  참조 전무 — 10개 에이전트 적대적 참조 스캔으로 확인
- 코어 패키지(engine, data, execution, …) 55개 모듈은 전부 도달 가능 → 이동 대상 없음

## ⚠️ 제자리 실행 불가
모든 스크립트가 `sys.path.insert(0, Path(__file__).parent.parent)` 패턴이라
여기서 실행하면 패키지 루트가 `legacy/`로 잡혀 깨진다. 또한 `config/X.yaml` 상대경로도
`legacy/config/`로 이동된 상태. **재실행하려면 해당 파일을 `scripts/`·`config/` 원위치로
복원할 것.** 옛 문서(*.md)에 적힌 `scripts/X.py` 경로는 `legacy/scripts/X.py`로 읽으면 된다.
