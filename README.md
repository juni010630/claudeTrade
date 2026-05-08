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

## 바이낸스 설정

실행 전 바이낸스에서 아래 설정 필요:

- **선물 거래 활성화**
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
- S (score 4): 25x / 22% size
- A (score 3): 10x / 10% size
- max_positions: 7
- daily_drawdown_pause: -4%, stop: -10%
- 최소 시드: $50 추천

## 백테스트 성과 (2022-2026)

- Sharpe: 2.25, MDD: -50%
- WR: ~47%, PF: ~1.95
- $100 → $200,122 (full period)
