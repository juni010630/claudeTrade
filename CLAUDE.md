# claudeTrade — 자동매매 봇

## 프로젝트 개요
바이낸스 USDM 선물 자동매매 봇. 백테스트 엔진 + 라이브 트레이딩 인프라.

## 작업 원칙

신중함 > 속도. 사소한 작업은 판단껏 생략 가능.

### 1. 코딩 전에 생각하기
- 가정을 명시적으로 밝힐 것. 불확실하면 물어볼 것.
- 해석이 여러 가지면 제시할 것 — 조용히 하나를 고르지 말 것.
- 더 단순한 방법이 있으면 말할 것. 필요하면 반박할 것.
- 불명확하면 멈추고, 뭐가 헷갈리는지 명명하고, 질문할 것.

### 2. 단순함 우선
- 요청받은 것 이상의 기능 금지. 추측성 코드 금지.
- 한 번 쓰는 코드에 추상화 금지. 요청 없는 "유연성/설정 가능성" 금지.
- 불가능한 시나리오에 대한 에러 처리 금지.
- 200줄이 50줄로 가능하면 다시 쓸 것.
- 자문: "시니어 엔지니어가 과하다고 할까?" → 그렇다면 단순화.

### 3. 외과적 수정
- 꼭 필요한 부분만 건드릴 것. 인접 코드/주석/포맷 "개선" 금지.
- 안 깨진 것 리팩토링 금지. 기존 스타일을 따를 것.
- 무관한 dead code 발견 시 언급만 하고 삭제하지 말 것.
- 내 수정으로 orphan이 된 import/변수/함수는 제거할 것.
- 테스트: 변경된 모든 줄이 사용자 요청으로 직접 추적 가능해야 함.

### 4. 목표 기반 실행
- 작업을 검증 가능한 목표로 변환:
  - "버그 수정" → 재현 케이스 만들고 통과시키기
  - "전략/엔진 수정" → 수정 전후 백테스트 결과 비교 (의도한 차이만 발생 확인)
  - "리팩토링" → 전후 백테스트 결과 완전 동일 확인
- 다단계 작업은 `단계 → 검증 방법` 형태로 간단히 계획 제시.

### 5. 프로젝트 절대 규칙
- **look-ahead bias 절대 불허** — 전략/엔진 수정 시 미래 데이터 참조 여부 항상 확인
- **백테스트 = 라이브 일치** — config 새 키 추가 시 `run_backtest.py`와 `live_trade.py`의 build_engine 양쪽 모두 반영
- trailing/breakeven류 동적 SL 도입 시 5m 교차검증 필수 (1h 단독 결과 신뢰 금지)

## 백테스트 실행 규칙

- 여러 백테스트는 항상 병렬로 실행할 것 (subprocess, asyncio, concurrent.futures 등 상황에 맞게)
- 폴링 루프 금지 — 완료 신호(콜백, Future, Event, returncode 등)를 기다릴 것
- 결과는 완료 이벤트 수신 후 일괄 확인
- 최적화 결과는 연도별(2022~2025) 분리 검증 — 특정 연도 과최적화 기각

## 최종 전략 (final_v15_gate)

### ⚠️ 데이터 백필 사건 (2026-06-07)
기존 캐시의 ADA/FIL은 2024-02, ARB는 2024-02부터만 존재했다. 백필(ADA/FIL 2022-01~, ARB 2023-03~) 후
v13의 진짜 성과 = **2023년 -78%, full MDD -88.6%** (기존 문서 수치는 데이터 공백 착시).
→ 위성심볼 티어 게이트(v15)로 해결. **백테스트 수치 인용 전 캐시 데이터 범위 확인 필수.**

### 설정 파일
- `config/final_v16_slwide.yaml` — **라이브 운영본 (챔피언)** = v15 + multi_tf SL 1.8→2.1 ATR
- `config/final_v15_gate.yaml` — v13 + ADA/FIL/ARB SS티어(score 5+) 게이트
- `config/final_v13_eth.yaml` — 구버전 — 완전 데이터 기준 2023 -78% 취약
- `config/final_v14_pyramid.yaml` — v13 + ema_cross 한정 피라미딩. 완전 데이터에선 MDD 악화로 기본 비활성. 라이브 STOP_MARKET 증액 구현 완료 (testnet 미검증)

### 완전 데이터 재검증 (2026-06-08 전수 완료)
모든 상속 파라미터를 백필 데이터로 재검증함. 유지 확정: 시간대 차단 22개(진짜 OOS 통과 — 제거 시 -99%),
TP 3.5/4.0, 티어 임계(SSS6/SS5/S4), 쿨다운 6h, 보유 336h, SSS rpt 0.20, corr 0.8, regime 임계,
ema 8/21, multi BB 25/2.2σ/vol 1.8x. 유일한 변경 = multi SL 2.1 (v16).
교훈: 거래 ~10건짜리 미세조정은 경로 카오스(슬롯/쿨다운 연쇄)로 결과가 ±수배 출렁임 — 연도별 검증 필수.

### 전략 구성
- **ema_cross**: 1h EMA(8/21) 크로스 + MACD(12,21,7) 히스토그램 전환 + 4h EMA(150) 방향 필터 (ATR 14)
- **multi_tf_breakout**: 4h BB(25,2.0) 돌파 확인 + 1h BB(25,2.2) 브레이크아웃 + RSI(10) 모멘텀 + 거래량 1.8x 확인
- SS/S/A 티어 사용 (SS=5+, S=4, A=3 — confluence score 3점 이상)
- **ETH regime**: ETHUSDT ADX로 시장 상태 분류 (BTC 대비 3x 수익 개선)

### 이전 버전
- `config/final_fixed_v2.yaml` — BTC regime, RPT 0.07, max_pos 5 (원본)
- `config/final_aggressive_v1.json` — trailing SL 포함 (1h 성과 뻥튀기 확인 → 제거)

### 심볼 (ETH 첫 번째 = regime 기준)
`ETHUSDT`, `BTCUSDT`, `DOGEUSDT`, `ARBUSDT`, `FILUSDT`, `ADAUSDT`

### 리스크 (final_v13_eth)
- risk_per_trade: 0.099
- SS: 40x leverage / 30% size_fraction (score 5+)
- S: 25x leverage / 22% size_fraction (score 4)
- A: 10x leverage / 10% size_fraction (score 3)
- max_positions: 7, max_same_direction: 5
- correlation_block_threshold: 0.8
- daily_drawdown_pause: -4%, daily_drawdown_stop: -10%

### 엔진
- 순수 고정 TP/SL (breakeven/trailing 모두 없음)
- SL: 1.8x ATR, TP: 3.5x(ema, ATR14) / 4.0x(multi_tf) ATR
- max_hold_hours: 336 (14일)
- 시간대 차단 (전략별, 진입봉 UTC 기준):
  - ema_cross: UTC 6~11, 13~14, 16, 18~19, 21, 23 (13개 시간)
  - multi_tf_breakout: UTC 3~5, 7~10, 13, 16 (9개 시간)

### 5m 교차검증 (look-ahead bias 확인)
- 순수 고정 TP/SL: **1h vs 5m 괴리 0%** — 완벽 일치, look-ahead 없음
- 피라미딩(정적 트리거가 + 비관적 체결 순서): **1h vs 5m 완전 일치** (6심볼 5m 데이터로 검증)
- breakeven 1.0R: 1h vs 5m 괴리 94% — **1h에서 뻥튀기 확인** → 제거됨
- trailing도 동일 문제 → 제거됨

### 백테스트 성과 (완전 데이터, 2022-01-01 ~ 2026-04-23, $100 시작)
| | v13 (완전 데이터 재평가) | v15 | **v16 (라이브)** |
|---|---|---|---|
| 최종 자산 | $5.28M | $10.6M | **$19.3M** |
| Sharpe / Calmar | 2.723 / 13.1 | 2.921 / 21.1 | **3.049 / 26.7** |
| MDD | **-88.6%** | -65.8% | **-60.5%** |
| WR / PF / 거래 | 49.6% / 3.92 / 369 | 52.8% / 5.92 / 316 | 54.6% / 7.19 / 315 |

### 연도별 안정성 (독립 구간, 완전 데이터)
| 연도 | v13 $100→ | v15 $100→ | **v16 $100→** | v16 Sharpe |
|------|-----------|-----------|---------------|------------|
| 2022 (베어) | $6,584 | $5,738 | **$8,052** | 4.27 |
| 2023 (횡보) | **$22 (-78%)** | $87 | **$105 (+5%)** | 0.50 |
| 2024 (불) | $2,236 | $1,534 | $1,444 | 2.82 |
| 2025-26 | $17,112 | $16,450 | **$18,712** | 3.93 |

### 최소 시드
- 이론적 최소: $18 (BTC 최소 notional $100 충족)
- 추천: $50~100

## 검증 완료 항목

### 로직 무결성
- 30건 랜덤 매매 로그 수동 검증: entry=1h close 일치, SL/TP bar high/low 내 체결 확인
- multi_tf_breakout curr_upper vs prev_upper: 120건 전수 비교 → 차이 0건
- look-ahead bias: 없음 (DataLoader effective_time 마스크 정상)

### 엔진 결정론성
- dry-run vs 백테스트 42건 비교: **PnL 차이 0.000000** (완전 일치)

### 실거래 검증
- testnet vs dry-run: 시그널 타이밍/방향 동일, equity 차이 0.047%
- testnet 체결가 = 5m bar close 정확 일치

### stale pricing fix
- `engine/backtest.py`: signal_tf ≠ primary_tf일 때 entry_price를 primary_tf close로 보정
- TP/SL도 price_shift만큼 동일하게 이동 (ATR 거리 유지)

### 4h offset 강건성
- 0h~3h offset 테스트: 모든 offset에서 Sharpe > 1.0
- 0h(바이낸스 기본)이 최적 (Sharpe 2.16), 2h가 최약 (1.23)
- 바이낸스 전용이면 문제 없음

### 시간대 필터
- UTC [6,7,8,9,16] 차단 → Sharpe 2.15→2.61 개선
- UTC 6-9 = 아시아 오후 비활성, UTC 16 = US 개장 직후 노이즈

## 코드 구조

### 핵심 모듈
- `engine/backtest.py` — 백테스트 + 라이브 공용 엔진 (BacktestEngine)
- `data/loader.py` — 캐시 parquet → MarketSnapshot 이터레이터
- `data/live_feed.py` — ccxt 실시간 피드 (5m/1h/4h/1d, freshness 검증)
- `strategies/` — 전략 클래스 (ema_cross, multi_tf_breakout 등 13개)
- `execution/live_broker.py` — 바이낸스 주문 전송
- `execution/sl_poller.py` — 테스트넷 SL 폴링 (STOP_MARKET 미지원 대체)
- `scripts/live_trade.py` — 1h 라이브 트레이딩 러너
- `scripts/live_validation.py` — 5m 검증용 러너

### 설정 파일
- `config/final_v13_eth.yaml` — **최종 전략 설정** (현재 라이브 운영)
- `config/final_fixed_v2.yaml` — 이전 버전 (BTC regime)
- `config/final_aggressive_v1.json` — 이전 전략 (trailing 포함, 1h 성과 뻥튀기 확인됨)
- `config/params.yaml` — 기본 설정
- `config/params_v2.yaml` — WF 최적화 기반 설정

### 환경변수 (.env)
```
BINANCE_API_KEY=...
BINANCE_SECRET=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

## 주의사항
- 모든 시간은 UTC 기준 (바이낸스 서버 기준)
- 테스트넷과 메인넷 가격이 다름 (0.1~0.5%) → 테스트넷에서 백테스트 대조 불가
- `max_notional_usd` 미설정 — 대규모 시드에서는 유동성 한계 주의
- equity 기반 복리 사이징 → 시드 크기와 관계없이 %수익률 동일
- 원격 서버 배포는 git이 아닌 scp 사용 (158.180.90.152)
