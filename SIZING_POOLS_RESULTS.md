# C축 — 가상 서브계좌(사이징 풀) + 월간 리밸: 구현 완료·활성화 기각 (2026-06-10)

배경: 현행 `strategy_capital_fraction`은 총 equity 비례 = **연속 리밸**(추세 복리가 매 진입마다
슬리브 기준자본으로 누수, 그 역도 동일). "월간리밸 필수" 가설(블렌드 사이징 실체) 검증 위해
풀별 가상 equity 사이징 + 월 경계 가상 재분배를 엔진에 구현.

## 구현 (config `sizing_pools`, 기본 off)

- `risk/models.py` PortfolioState.pool_cash / `portfolio/tracker.py` 풀 cash 미러링(_pool_add:
  진입비용·청산 PnL·증액비용·펀딩) + pool_equity/init_pools/rebalance_pools
- `engine/backtest.py` 월 경계 첫 봉 가상 재분배(강제청산 없음) + 사이징 cap_base=풀 equity
- `portfolio/state_store.py` pool_cash·last_rebal_month 영속 / `live_trade.py` 복원 배선
- **백테=라이브 양쪽 배선 완료. 패리티: off 시 v17 비트동일 $8,991.21/805건 확인.**

## 결과 (`scripts/sizing_pools_check.py`, trend/sleeve 50:50 월간)

| | base(연속 리밸) | pools ON(월간) |
|---|---|---|
| 전구간 | $8,991 / Sh 1.937 / MDD -42.4% | $10,970(+22%) / Sh 1.926 / **MDD -45.1%** |
| IS(22~24) | $699 / 1.321 | $790(+13%) / **1.343(↑)** |
| OOS(25~) | $1,376 / 3.437 / -35.5% | $1,490(+8%) / 3.313 / -37.5% |

거래수 805 동일(사이징만 변경, 시그널 불변). 가설 방향은 확인됨(전 셀 수익↑, IS Sharpe↑).

## 판정: **활성화 기각 (G-C: Sharpe 1.937→1.926 비열위 위반 + MDD -45.1% 한도 0.1pp 초과)**

- 사전등록 게이트 원칙상 경계 케이스도 기각. 리밸 주기 스윕으로 통과점 찾기 = 다중검정 오염이라 금지
- **동일 프론티어의 더 단순한 대안 존재**: 정적 55:45가 $11,589/-44.8%/Sh1.912로
  수익 더 높고 MDD 더 얕음(코드 변경 없는 config 한 줄). 풀 구조가 주는 추가 가치 없음
- 코드는 inert 보존(off, 패리티 검증됨) — 추후 비중 변경과 결합 실험 시 재사용 가능

교훈: "월간리밸 필수"의 실체 = 수익↑/MDD↑ 리스크 다이얼이지 공짜 점심 아님. 연속 리밸의
상호 완충(한쪽 손실이 양쪽 사이징을 즉시 줄임)이 바로 MDD -42.4의 출처였음.
