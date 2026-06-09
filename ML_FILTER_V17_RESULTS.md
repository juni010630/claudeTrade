# ML 메타라벨 필터 — v17 재시도 결과 (2026-06-10, 기각)

1차 시도(2026-06-07, v13 293건, OOS AUC 0.487 기각) 후 v17 기준 재시도. **사전 고정 게이트(STEP 4) 미달로 기각.**

## 설정
- 데이터셋: v17 전구간 백테 ledger, **추세전략만**(ema 237 + multi 215 = 452건, 슬리브 제외), 승률 46.2%, NaN 0
- 학습: `train_ml_filter.py` — Purged K-Fold(5, hold/embargo 14d) + walk-forward(train ~2024 / test 2025~)
- 적용 배선: scorer ML 훅에 추세전략 가드 추가(`signals/scorer.py:173` — ema/multi만 적용, 슬리브 비오염)

## 학습 결과 (STEP 3 게이트 통과, 단 경고 2)
- 음성테스트: 셔플 AUC 0.490 ✓ (누수 없음)
- Purged CV 평균: LR 0.531 / LGBM 0.508
- Walk-forward OOS: LR 0.544 / **LGBM 0.615** → 규칙대로 LGBM 채택
- ⚠️ LGBM은 CV(0.508)와 WF(0.615) 불일치 — 수용기준 3 위반 의심(test 113건 노이즈)
- ⚠️ permutation 1위 = hour_cos(+0.095), 2위 vol_ratio(+0.084) — hour-of-day는 과적합 전과 축

## 백테 비교 (STEP 4, eval_ml_filter.py 15개 병렬)

OOS 2025-01-01~2026-04-14, $100 시작:

| 모드 | 거래 | WR | PF | Sharpe | MDD | 최종$ |
|---|---|---|---|---|---|---|
| baseline | 205 | 54.6% | 2.23 | 3.44 | -35.5% | 1,376 |
| bonus | 205 | = | = | = | = | = (모델 prob 최대 ~0.57 < 임계 0.6 → 발동 0) |
| hardcut(0.45) | 139 | 56.1% | 2.08 | 3.12 | -19.6% | 484 |

연도별 PF (baseline→hardcut): 2022 1.43→2.33, 2023 1.03→1.61, 2024 1.21→2.62 — **전부 학습구간 인샘플 착시**. OOS인 2025만 유효.

## 판정
- 게이트 1 (OOS PF·Sharpe ≥ baseline): **미달** — hardcut PF 2.23→2.08, Sharpe 3.44→3.12, 수익 $1,376→$484
- 게이트 3 (CV=WF 방향 일치): LGBM 불일치
- MDD 반토막(-35.5→-19.6%)은 실재하나 수익 2/3 포기 대가 — 리스크조정으로도 열위
- **결론: 약한 신호(AUC ~0.55)는 존재하나 거래 컷의 복리 손실을 못 이김. 기각.**

## 상태
- `config/final_v17.yaml` `ml_soft_scoring.enabled: false` 유지 (라이브 무영향)
- `models/ml_filter.pkl` = 이번 LGBM (gitignore, 미사용)
- scorer 추세전략 가드 / builder mean_reversion 제외 / eval 파싱 수정은 향후 재시도용으로 유지
- 재시도 조건: 신호레벨 데이터셋(미체결 후보 솔로 TP/SL 라벨링, `_generate_all_candidates` 활용)으로 n 수천 확보 — 단 hour 피처 제외 또는 별도 검증 권장
