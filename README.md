# claudeTrade — Binance USDM Futures 자동매매 봇

EMA Cross + Multi-TF Breakout 전략 기반 자동매매.
ETH ADX regime 필터, 고정 TP/SL, 1h 시그널.

## 빠른 시작

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. .env 설정 (.env.example 복사 후 API 키 입력)
cp .env.example .env
# BINANCE_API_KEY, BINANCE_SECRET 입력

# 3. 데이터 다운로드 (최초 1회)
python scripts/fetch_data.py --params config/final_v13_eth.yaml

# 4. 실행
python scripts/live_trade.py --params config/final_v13_eth.yaml           # 테스트넷
python scripts/live_trade.py --params config/final_v13_eth.yaml --no-demo # 실계정
```

## 바이낸스 API 키 발급

### 1. API 키 생성

1. [바이낸스](https://www.binance.com) 로그인
2. 우측 상단 프로필 아이콘 → **API 관리** (또는 https://www.binance.com/my/settings/api-management)
3. **API 키 만들기** 클릭 → 라벨 입력 (예: `trade-bot`) → **다음**
4. 이메일 인증 + 2FA 인증 완료
5. **API Key**와 **Secret Key**가 표시됨 → **Secret Key는 이때만 보여주니까 반드시 복사해둘 것**

### 2. API 키 권한 설정

생성된 API 키 옆 **수정** 클릭:

- **읽기 전용** (Enable Reading): 체크
- **선물 주문** (Enable Futures): 체크
- **현물 주문** (Enable Spot & Margin Trading): 체크 해제 (불필요)
- **출금** (Enable Withdrawals): **반드시 체크 해제** (보안)

### 3. IP 허용 목록 (필수 권장)

바이낸스는 IP 제한 없는 API 키를 90일 후 자동 삭제함. **반드시 IP를 등록해야 함.**

1. API 수정 화면에서 **IP 접근 제한** → **허용된 IP 주소만 접근** 선택
2. 내 공인 IP 확인:
   ```bash
   curl -s ifconfig.me
   ```
   또는 브라우저에서 https://ifconfig.me 접속
3. 확인된 IP 주소를 **허용 IP** 칸에 입력 → **확인**

> **주의**: 가정용 인터넷은 IP가 바뀔 수 있음. IP 바뀌면 봇이 거래 못 함 → 바이낸스에서 IP 업데이트 필요.
> 고정 IP가 없으면 클라우드 서버(AWS, Vultr 등)에서 돌리는 게 안전함.

### 4. .env 파일에 입력

```bash
cp .env.example .env
```

`.env` 파일을 열어서 복사해둔 키 입력:

```
BINANCE_API_KEY=AbCdEf1234567890...
BINANCE_SECRET=xYz9876543210...
```

### 5. 테스트넷 키 (선택 — 먼저 연습하고 싶으면)

실계정 말고 테스트넷에서 먼저 돌려볼 수 있음:

1. https://testnet.binancefuture.com 접속 → GitHub로 로그인
2. 상단 **API Key** 메뉴에서 키 생성 (IP 제한 없음)
3. `.env`에 테스트넷 키 입력 → `python scripts/live_trade.py` (기본이 테스트넷)

> 테스트넷은 가상 자금이라 돈 안 나감. 실계정 전환은 `--no-demo` 플래그 추가.

---

## 바이낸스 선물 설정

실행 전 바이낸스에서 아래 설정 필요:

- **선물 거래 활성화** (바이낸스 앱 → 선물 → 퀴즈 통과)
- **마진 모드**: Cross
- **레버리지** (심볼별 수동 설정):

| 심볼 | SS (40x) | S (25x) | A (10x) |
|------|----------|---------|---------|
| ETHUSDT | O | O | O |
| BTCUSDT | O | O | O |
| DOGEUSDT | O | O | O |
| ARBUSDT | O | O | O |
| FILUSDT | O | O | O |
| ADAUSDT | O | O | O |

> 봇이 자동으로 주문 시 레버리지를 설정하지만, 바이낸스 최대 레버리지 한도가 심볼마다 다르므로 미리 확인 권장.

## 주요 명령

```bash
# 백테스트 (검증용)
python scripts/run_backtest.py --params config/final_v13_eth.yaml

# 테스트넷 (기본, --no-demo 없으면 자동 테스트넷)
python scripts/live_trade.py --params config/final_v13_eth.yaml

# 실계정 (주의!)
python scripts/live_trade.py --params config/final_v13_eth.yaml --no-demo

# 즉시 1회 실행 (연결 테스트)
python scripts/live_trade.py --params config/final_v13_eth.yaml --snap-now

# 주문 없이 로그만
python scripts/live_trade.py --params config/final_v13_eth.yaml --dry-run
```

## 텔레그램 알림 (선택)

1. @BotFather에서 봇 생성 → 토큰 획득
2. 봇에 메시지 전송 후 chat_id 확인
3. `.env`에 `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` 입력

## 리스크 설정 (final_v13_eth)

- risk_per_trade: 9.9%
- SS (score 5+): 40x / 30% size
- S (score 4): 25x / 22% size (**ADAUSDT 차단** — WR 11% 독성 제거)
- A (score 3): 10x / 10% size
- max_positions: 7
- daily_drawdown_pause: -4%, stop: -10%
- 최소 시드: $50 추천

## 백테스트 성과 (2022-04 ~ 2026-04)

| 지표 | 값 |
|------|-----|
| 초기 자본 | $100 |
| 최종 자산 | $464,077 |
| Sharpe | 2.503 |
| MDD | -48.7% |
| 승률 | 48.5% |
| Profit Factor | 2.18 |
| 총 거래 | 433건 |

### 티어별 성과

| 티어 | 거래 | 승률 | PF | PnL |
|------|------|------|-----|-----|
| SS (5+점, 40x) | 254 | 48.0% | 3.43 | +$440,753 |
| S (4점, 25x) | 114 | 53.5% | 0.95 | -$9,976 |
| A (3점, 10x) | 65 | 41.5% | 3.09 | +$33,201 |

### 연도별 안정성

| 연도 | $100→ | Sharpe | MDD |
|------|-------|--------|-----|
| 2022 (베어) | $590 | 1.53 | -47% |
| 2023 (횡보) | ~$146 | 0.90 | -50% |
| 2024 (불 시작) | ~$336 | 1.62 | -50% |
| 2025 (4개월) | $3,251 | 2.87 | -39% |

## 시그널 덤프/리플레이

사이징·리스크 파라미터 튜닝 시 시그널 생성을 한 번만 수행하고 반복 재계산:

```bash
# 1. 시그널 덤프 (풀 백테스트 + 시그널 수집, ~7분)
python scripts/verify_replay.py  # dump → replay 결과 일치 검증

# 2. 리플레이 튜닝 실험 (병렬 실행)
python scripts/tune_s_tier.py    # S 티어 사이징/차단 실험 4가지 병렬
```

- `run_dump()` → `(MetricsReport, signals_df)`: 후보 시그널 parquet 저장
- `run_replay(signals_df, snapshots)` → `MetricsReport`: 전략 계산 스킵, 사이징만 재계산
- 검증 결과: PnL 차이 $0.000000 (완전 일치)
