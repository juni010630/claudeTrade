# claudeTrade — 자동매매 봇

## 프로젝트 개요
바이낸스 USDM 선물 자동매매 봇. 백테스트 엔진 + 라이브 트레이딩 인프라.

## 최종 전략 (final_v13_eth)

### 설정 파일
`config/final_v13_eth.yaml` (ETH regime, 고정 TP/SL, 136+ 백테스트 최적화)

### 전략 구성
- **ema_cross**: 1h EMA(8/21) 크로스 + MACD(12,21,7) 히스토그램 전환 + 4h EMA(150) 방향 필터 (ATR 14)
- **multi_tf_breakout**: 4h BB(25,2.0) 돌파 확인 + 1h BB(25,2.2) 브레이크아웃 + RSI(10) 모멘텀 + 거래량 1.8x 확인
- SS/S/A 티어 사용 (SS=5+, S=4, A=3 — confluence score 3점 이상)
- **ETH regime**: ETHUSDT ADX로 시장 상태 분류 (BTC 대비 3x 수익 개선)

### 이전 버전
- `config/final_fixed_v2.yaml` — BTC regime, RPT 0.07, max_pos 5 (원본)
- `config/final_aggressive_v1.json` — trailing SL 포함 (1h 성과 뻥튀기 확인 → 제거)

## 백테스트 실행 규칙

- 여러 백테스트는 항상 병렬로 실행할 것 (subprocess, asyncio, concurrent.futures 등 상황에 맞게)
- 폴링 루프 금지 — 완료 신호(콜백, Future, Event, returncode 등)를 기다릴 것
- 결과는 완료 이벤트 수신 후 일괄 확인

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
- 시간대 차단: UTC [6,7,8,9,16] (한국 [15,16,17,18,01]시)

### 5m 교차검증 (look-ahead bias 확인)
- 순수 고정 TP/SL: **1h vs 5m 괴리 0%** — 완벽 일치, look-ahead 없음
- breakeven 1.0R: 1h vs 5m 괴리 94% — **1h에서 뻥튀기 확인** → 제거됨
- trailing도 동일 문제 → 제거됨

### 백테스트 성과 (final_v13_eth, ETH regime)
- Full (2022-2026): Sharpe 2.25, MDD -50%, $100→$200,122
- 2025-2026 (16개월): Sharpe 3.12, MDD -42%, $100→$4,963
- WR ~46.7%, PF ~1.95, 거래 480건, 최대 연속 손절 6회
- **1h = 5m 동일 결과** (subbar 교차검증 통과)

### 연도별 안정성
| 연도 | $100→ | Sharpe | MDD |
|------|-------|--------|-----|
| 2022 (베어) | $590 | 1.53 | -47% |
| 2023 (횡보) | ~$146 | 0.90 | -50% |
| 2024 (불 시작) | ~$336 | 1.62 | -50% |
| 2025 (4개월) | $3,251 | 2.87 | -39% |

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
- `config/final_fixed_v2.yaml` — **최종 전략 설정** (trailing 제거, 고정 TP/SL + BE 1.0R)
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

## dead code 정리 (완료)
- `engine/backtest.py`: `_check_tp_sl`, `_update_dynamic_sl` 제거
- 프로젝트 루트 JSON 176개, log 37개 삭제
- 디버그 스크립트 (`_audit_backtest.py` 등) 삭제

## 주의사항
- 모든 시간은 UTC 기준 (바이낸스 서버 기준)
- 테스트넷과 메인넷 가격이 다름 (0.1~0.5%) → 테스트넷에서 백테스트 대조 불가
- `max_notional_usd` 미설정 — 대규모 시드에서는 유동성 한계 주의
- equity 기반 복리 사이징 → 시드 크기와 관계없이 %수익률 동일
