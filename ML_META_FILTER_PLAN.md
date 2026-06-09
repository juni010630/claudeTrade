# ML 메타라벨 필터 — 실행 설계서

> 상태: **설계 확정, 구현 대기**. 다음 세션(모델 교체 후)이 이 문서만으로 바로 착수할 수 있도록 코드 앵커·시그니처·수용기준까지 명시함.
> 작성 맥락: `final_v13_eth` 전략에 ML을 **2차 필터(meta-labeling)** 로 부착. 규칙 기반 진입은 그대로 두고, 진입 후보의 P(win)을 예측해 컷/보너스로 활용.

---

## 0. 확정된 결정 (재논의 불필요)

| 항목 | 결정 |
|------|------|
| ML 역할 | **메타라벨 필터** (규칙 진입 후보 → P(win) 예측) |
| 라벨 `y` | **이진 win = `pnl > 0`** (timeout도 pnl 부호로 처리) |
| 적용 모드 | **하드컷 + 보너스 둘 다 구현 후 백테스트 비교** |
| 모델 | **LogisticRegression(L2) 베이스라인 + 얕은 LightGBM** 비교, 단순한 쪽 우선 |
| 풀링 | **단일 풀링 모델 + `strategy`/`direction`을 피처로** (전략별 분리는 데이터 부족, 후순위) |
| 검증 | **철저하게**: walk-forward + Purged K-Fold(Embargo) + 연도별 + calibration + permutation importance |
| 추가 검증 | **5m 교차검증** (필터 적용 후에도 1h=5m 일치 유지 확인) |

### 절대 원칙 (위반 시 전체 폐기)
1. **무누수(no look-ahead)**: 모든 피처는 진입봉 `idx`까지의 데이터 `bars[:idx+1]`만 사용. ([[feedback_no_lookahead]])
2. **OOS·연도별 우선**: in-sample 개선은 무시. OOS(2025~26)와 연도별(2022/23/24)에서 살아남지 못하면 bucket-block처럼 깔끔히 기각. ([[project_bucket_block_result]])
3. **임계값·하이퍼파라미터는 train에서만 결정**. test를 보고 튜닝 금지.
4. **데이터 작음(449~627건) 경고**: 복잡 모델 즉시 암기. 피처 ≤12개, 모델 강한 규제.

---

## 1. 데이터 파이프라인

### 1.1 라벨 소스 (이미 존재)
- `engine/backtest.py` `BacktestEngine.run()` → `engine.ledger.to_dataframe()`
- `portfolio/ledger.py:17` `TradeRecord` 컬럼: `trade_id, symbol, strategy, direction, entry_price, exit_price, size_usd, leverage, pnl, commission, slippage_cost, funding_paid, entry_time, exit_time, exit_reason, regime_at_entry, confluence_score`
- **라벨**: `y = (pnl > 0).astype(int)`

### 1.2 피처 소스 (일부 존재 + 신규 추출 필요)
- 기존: `run_fill_dump()` (`engine/backtest.py:331`) → 진입 전수 + `regime, adx, bb_width_pct, sl_dist, score, tier`
- 신규: 아래 §2 피처를 진입봉 컨텍스트에서 추출해야 함 → **데이터셋 빌더가 직접 `compute_features()`를 호출**해 채움

### 1.3 조인 키
- `(entry_time, symbol, strategy, direction)` 4-튜플로 ledger ↔ 피처 조인.
- ⚠️ 동일 봉·동일 심볼에 ema/multi 동시 진입 가능 → `strategy`까지 키에 포함 필수.
- 조인 후 행 수 == ledger 거래 수 검증 (누락/중복 0 확인).

### 1.4 산출물
- `data/ml_dataset.parquet` — 컬럼: `[entry_time, symbol, strategy, direction, <features...>, pnl, exit_reason, y]`
- 빌더: `scripts/build_ml_dataset.py` (신규)

---

## 2. 피처 명세 (무누수, 진입봉 `[:idx+1]`만)

설계 핵심: **규칙(scorer 7항목)이 이미 이진 임계로 쓰는 값을 "연속값"으로 다시 제공** → ML이 임계 근처의 미묘함을 학습. 이것이 메타라벨이 가치를 더하는 메커니즘.

| # | 피처 | 정의 | 비고 |
|---|------|------|------|
| 1 | `adx` | 진입봉 ADX(14) | 규칙은 25 임계만 봄 → 연속값 |
| 2 | `bb_width_pct` | BB 밴드폭 % | 스퀴즈 정도 |
| 3 | `rsi_1h` | 1h RSI(진입봉) | 규칙은 70/40 임계만 |
| 4 | `vol_ratio` | 현재거래량 / 20봉평균 | 규칙은 1.8 임계만 |
| 5 | `atr_pct` | `sl_dist / entry_price` | 변동성 정규화 |
| 6 | `dist_ema200_1d` | `(close - ema200_1d) / close` | 일봉 추세 강도 (방향 부호 포함) |
| 7 | `funding` | 진입 시 펀딩비 | 연속값 |
| 8 | `hour_utc` | `entry_time.hour` | 시간대 차단 잔여 신호 (순환 인코딩 sin/cos 고려) |
| 9 | `direction_long` | long=1, short=0 | |
| 10 | `strategy_ema` | ema_cross=1, multi=0 | 풀링 모델 식별자 |

확장 후보(1차 통과 후): `regime_onehot`, `btc_corr`, `recent_win_streak`, 4h 모멘텀.

### 2.1 무누수 보증
- `compute_features(bars_1h, bars_4h, bars_1d, idx)` — `idx`는 진입봉 인덱스. 내부에서 **절대 `idx` 이후 행 참조 금지**.
- 빌더는 백테스트가 실제로 진입을 결정한 시점의 스냅샷과 동일한 `idx`를 써야 함 → 가능하면 `run_fill_dump` 경로에서 피처를 함께 적재(별도 재계산보다 안전). 별도 재계산 시 entry_price/sl_dist 역산으로 동일성 검증.

---

## 3. 모델 명세

### 3.1 전처리
- 수치 피처 표준화(StandardScaler) — **train fold에서만 fit**, test는 transform.
- `hour_utc`는 sin/cos 2피처로 순환 인코딩 또는 그대로(트리는 그대로 OK).
- 스케일러/인코더는 모델과 함께 직렬화.

### 3.2 모델 후보
1. **LogisticRegression** — `penalty='l2', C` 그리드 `{0.01,0.1,1.0}`, `class_weight=None`(클래스 ~47:53 균형).
2. **LightGBM** — `max_depth∈{2,3}, n_estimators≤100, learning_rate≤0.05, min_child_samples 크게(≥30), subsample=0.8, colsample=0.8, reg_lambda↑`.

선택 규칙: **OOS에서 LGBM이 Logistic 대비 유의미하게 낫지 않으면 Logistic 채택** (단순성·과적합저항 우선).

### 3.3 직렬화 — `strategies/ml_filter.py` (신규)
기존 배선(`scripts/run_backtest.py:74-80`, `signals/scorer.py:151-170`)이 요구하는 인터페이스를 **정확히** 구현:

```python
class MLModels:
    """학습된 모델 + 전처리기 + 피처 순서 번들."""
    @classmethod
    def load(cls, path: str) -> "MLModels": ...
    def save(self, path: str) -> None: ...

class MLSignalFilter:
    def __init__(self, models: MLModels, clf_threshold: float = 0.0): ...
    def predict(self, feat: dict | np.ndarray) -> dict:
        # 반드시 {"clf_prob": float} 반환 (scorer.py:163이 이 키 사용)
        ...

def compute_features(bars_1h, bars_4h, bars_1d, idx: int) -> dict | None:
    # 무누수. 데이터 부족 시 None (scorer.py:161이 None 가드)
    ...
```
- 기본 경로: `models/ml_filter.pkl` (`run_backtest.py:75`).
- `compute_features`는 빌더와 라이브 scorer가 **같은 함수**를 호출 → 학습/추론 피처 일관성 보장.

---

## 4. 검증 설계 (본체)

### 4.1 누수 차단 CV
- **Purged K-Fold + Embargo**: 거래 보유기간이 겹침(`max_hold_hours: 336` = 14일) → 표준 K-fold 누수.
  - fold 경계에서 train 샘플 중 test 구간과 보유기간이 겹치는 것 제거(purge).
  - test fold 직후 14일 embargo (그 기간 train 샘플 제외).
  - 참고: López de Prado, *Advances in Financial ML* ch.7.
- **시간순 walk-forward**: train `2022~2024` / test `2025~2026` 고정 split (가장 정직, 당신 이력과 일관).

### 4.2 평가 지표
- 분류: AUC, **calibration 곡선**(prob 0.7 → 실제 ~70% win 여야 함), Brier score.
- **전략 단위(핵심)**: 필터 적용 백테스트의 OOS PF/Sharpe/MDD/WR/거래수 — baseline 대비.
  - 백테스트 비교는 `scripts/run_backtest.py`에 `scorer.ml_soft_scoring.enabled` on/off로 수행.

### 4.3 연도별 무붕괴 (필수)
- 2022/2023/2024/2025 각각 PF·Sharpe 출력 → 어느 해도 baseline 대비 크게 악화 없어야 함. ([[project_bucket_block_result]] 연도별 검증 원칙)

### 4.4 견고성 추가 테스트
- **permutation importance**: 피처 셔플 시 이득 사라지는지(우연/누수 감지).
- **feature shuffle 음성테스트**: 라벨 셔플 학습 → AUC≈0.5 확인(파이프라인 누수 없음 증명).
- **5m 교차검증**: 필터 적용 후 1h vs 5m subbar 결과 일치 유지 확인 (`engine/backtest.py` subbar 경로, `scripts/verify_replay.py`/`live_validation.py` 참조). ([[feedback_backtest_validation]])

---

## 5. 적용 모드 (둘 다 구현·비교)

기존 scorer 훅(`signals/scorer.py:151-170`)은 **보너스 모드**가 이미 구현됨. 하드컷은 추가 필요.

### 5.1 보너스 모드 (구현됨)
- `clf_prob ≥ bonus_threshold_1(0.6)` → +1점, `≥ bonus_threshold_2(0.75)` → +2점 → tier 승급 → 사이징↑.
- config: `scorer.ml_soft_scoring.{bonus_threshold_1, bonus_threshold_2}`.

### 5.2 하드컷 모드 (신규)
- `clf_prob < cut_threshold(예 0.45)` → **진입 스킵**.
- `MLSignalFilter.clf_threshold` 활용(현재 `0.0`으로 무차단). scorer 또는 candidate 필터 단계에서 `clf_prob < clf_threshold`면 NO_TRADE.
- config 신규 키: `scorer.ml_soft_scoring.cut_threshold`.

### 5.3 비교 판정
- baseline / 보너스 / 하드컷 3개를 OOS·연도별로 비교 → 승자 채택. 임계값은 **train에서만** 결정.

---

## 6. Config 스키마 (추가/확인)

`config/final_v13_eth.yaml`의 `scorer:` 아래 (현재 미존재 → 신규):
```yaml
scorer:
  ml_soft_scoring:
    enabled: false           # 기본 off (모델 검증 통과 후 on)
    model_path: models/ml_filter.pkl
    mode: bonus              # bonus | hardcut  (5절)
    bonus_threshold_1: 0.6
    bonus_threshold_2: 0.75
    cut_threshold: 0.45      # hardcut 모드 전용 (신규)
```
- ⚠️ **백테=실거래 일치**: `scripts/run_backtest.py build_engine`과 `scripts/live_trade.py`의 엔진 빌드가 이 키를 **양쪽 동일** 반영해야 함. ([[project_backtest_live_parity]], [[project_remote_deploy]])

---

## 7. 산출물 (파일별 데드라인 체크리스트)

| 파일 | 신규/수정 | 내용 |
|------|-----------|------|
| `scripts/build_ml_dataset.py` | 신규 | 백테스트 1회 → ledger+피처 조인 → `data/ml_dataset.parquet`. 조인 무결성 검증 포함 |
| `strategies/ml_filter.py` | 신규 | `MLModels`, `MLSignalFilter`, `compute_features` (§3.3 인터페이스 정확 준수) |
| `scripts/train_ml_filter.py` | 신규 | 데이터셋 로드 → Purged CV + walk-forward 학습 → Logistic vs LGBM → `models/ml_filter.pkl` 저장 + 리포트(AUC/calibration/permutation) |
| `scripts/eval_ml_filter.py` | 신규 | baseline/보너스/하드컷 백테스트 비교 + 연도별 + 5m 교차검증 리포트 |
| `signals/scorer.py` | 수정 | 하드컷 모드 분기 추가(현재 보너스만). `mode` 파라미터 |
| `scripts/run_backtest.py` | 수정 | `ml_soft_scoring.mode`, `cut_threshold` 배선 |
| `scripts/live_trade.py` | 수정 | 동일 ML 키 반영 (parity) |
| `config/final_v13_eth.yaml` | 수정 | §6 스키마 추가 (enabled:false) |
| `models/.gitignore` | 신규 | `.pkl` 추적 정책 결정 |

---

## 8. 실행 순서 (단계별 게이트)

```
STEP 1  build_ml_dataset.py 작성 → data/ml_dataset.parquet 생성
        ✓ 게이트: 행수==ledger 거래수, win 비율 ~47%, 피처 NaN 0
STEP 2  strategies/ml_filter.py compute_features 작성
        ✓ 게이트: 음성테스트(라벨셔플 AUC≈0.5), 무누수 단위테스트
STEP 3  train_ml_filter.py — Logistic + LGBM, Purged CV + walk-forward
        ✓ 게이트: OOS AUC>0.5 유의, calibration 양호, permutation 이득 존재
STEP 4  eval_ml_filter.py — baseline vs 보너스 vs 하드컷
        ✓ 게이트: OOS PF·Sharpe 개선 AND 연도별 무붕괴 AND walk-forward=purged 일관
STEP 5  5m 교차검증 — 필터 적용 후 1h=5m 일치
        ✓ 게이트: 괴리 ~0%
STEP 6  통과 시만 config enabled:true + live parity 반영 + 원격 배포
```

### 최종 수용 기준 (사전 고정, 사후 변경 금지)
1. OOS(2025~26) PF·Sharpe ≥ baseline.
2. 연도별 2022/23/24 어느 해도 크게 악화 없음.
3. walk-forward와 Purged-CV 결과가 같은 방향.
4. calibration 곡선 대각선 근처.
5. 5m 교차검증 괴리 ~0%.
6. permutation/라벨셔플 음성테스트 통과(누수 없음 증명).

**하나라도 미달 → 폐기.** (적응형 TP/SL·역추세 오버레이가 전부 그렇게 기각됨: [[project_adaptive_tpsl_result]], [[project_winrate_plan]])

---

## 9. 알려진 함정 (체크포인트)

- **보유기간 겹침 누수** → Purged+Embargo 없으면 백테스트 Sharpe 뻥튀기 (trailing 때와 동일 패턴).
- **데이터 작음(≤627건)** → 딥/깊은 트리 금지. 피처 ≤12, 강 규제.
- **조인 키 누락** → 동일봉 ema/multi 동시진입 시 strategy 미포함하면 라벨 오정렬.
- **train/test 누설** → 스케일러를 전체 데이터에 fit하면 누수. fold 내부 fit만.
- **임계값 사후 튜닝** → test 보고 threshold 고르면 OOS 의미 상실.
- **parity 깨짐** → live_trade에 ML 키 미반영 시 백테≠실거래.
