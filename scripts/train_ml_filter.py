"""ML 메타라벨 필터 학습.

Purged K-Fold + Embargo CV + walk-forward 검증.
LogisticRegression vs LightGBM 비교 후 OOS 성능 좋은 모델 저장.

Usage:
    python scripts/train_ml_filter.py
    python scripts/train_ml_filter.py --data data/ml_dataset.parquet --out models/ml_filter.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from strategies.ml_filter import MLModels, FEATURE_COLS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ml_dataset.parquet")
    parser.add_argument("--out", default="models/ml_filter.pkl")
    parser.add_argument("--wf-split", default="2025-01-01", help="walk-forward train/test 분할일")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"데이터: {len(df)}행, 승률: {df['y'].mean():.1%}")
    print(f"피처: {FEATURE_COLS}")

    X = df[FEATURE_COLS].values
    y = df["y"].values
    timestamps = df["timestamp"].values

    split_date = pd.Timestamp(args.wf_split, tz="UTC")
    train_mask = df["timestamp"] < split_date
    test_mask = ~train_mask

    print(f"\nwalk-forward: train {train_mask.sum()}건, test {test_mask.sum()}건")

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    ts_train = timestamps[train_mask]
    ts_test = timestamps[test_mask]

    # ── 음성 테스트: 라벨 셔플 → AUC ≈ 0.5 ──────────────────────
    print("\n[음성 테스트] 라벨 셔플 AUC...")
    y_shuffled = np.random.permutation(y_train)
    sc_tmp = StandardScaler().fit(X_train)
    X_tr_s = sc_tmp.transform(X_train)
    X_te_s = sc_tmp.transform(X_test)
    lr_null = LogisticRegression(C=0.1, max_iter=500).fit(X_tr_s, y_shuffled)
    null_auc = roc_auc_score(y_test, lr_null.predict_proba(X_te_s)[:, 1])
    print(f"  Null AUC (셔플): {null_auc:.4f}  (≈0.5 이어야 함)")
    if abs(null_auc - 0.5) > 0.05:
        print("  [WARN] 셔플 AUC가 0.5에서 너무 멀다 — 누수 가능성 재확인 필요")

    # ── Purged K-Fold CV ────────────────────────────────────────────
    hold_days = 14  # max_hold_hours=336 = 14일
    embargo_days = 14
    n_folds = 5
    print(f"\nPurged K-Fold (k={n_folds}, hold={hold_days}d, embargo={embargo_days}d)...")

    cv_aucs_lr, cv_aucs_lgbm = [], []
    folds = _purged_kfold(ts_train, n_folds, hold_days, embargo_days)

    for fold_i, (tr_idx, val_idx) in enumerate(folds):
        if len(tr_idx) < 20 or len(val_idx) < 5:
            continue
        Xtr, ytr = X_train[tr_idx], y_train[tr_idx]
        Xval, yval = X_train[val_idx], y_train[val_idx]

        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xval_s = scaler.transform(Xval)

        lr = LogisticRegression(C=0.1, max_iter=500).fit(Xtr_s, ytr)
        auc_lr = roc_auc_score(yval, lr.predict_proba(Xval_s)[:, 1]) if len(np.unique(yval)) > 1 else 0.5
        cv_aucs_lr.append(auc_lr)

        if HAS_LGBM:
            lgbm = _make_lgbm().fit(Xtr_s, ytr)
            auc_lgbm = roc_auc_score(yval, lgbm.predict_proba(Xval_s)[:, 1]) if len(np.unique(yval)) > 1 else 0.5
            cv_aucs_lgbm.append(auc_lgbm)

        print(f"  fold {fold_i+1}: train={len(tr_idx)}, val={len(val_idx)}"
              f"  LR={auc_lr:.4f}" + (f"  LGBM={auc_lgbm:.4f}" if HAS_LGBM and cv_aucs_lgbm else ""))

    print(f"\nCV 평균 AUC — LR: {np.mean(cv_aucs_lr):.4f}"
          + (f"  LGBM: {np.mean(cv_aucs_lgbm):.4f}" if cv_aucs_lgbm else ""))

    # ── Walk-forward (최종 OOS 평가) ────────────────────────────────
    print(f"\nWalk-forward OOS ({args.wf_split} 이후)...")
    scaler_wf = StandardScaler().fit(X_train)
    X_tr_wf = scaler_wf.transform(X_train)
    X_te_wf = scaler_wf.transform(X_test)

    lr_wf = LogisticRegression(C=0.1, max_iter=500).fit(X_tr_wf, y_train)
    lr_proba = lr_wf.predict_proba(X_te_wf)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_proba) if len(np.unique(y_test)) > 1 else 0.5
    lr_brier = brier_score_loss(y_test, lr_proba)
    print(f"  LogisticRegression — AUC: {lr_auc:.4f}, Brier: {lr_brier:.4f}")

    lgbm_auc, lgbm_brier = 0.0, 1.0
    lgbm_proba = None
    if HAS_LGBM:
        lgbm_wf = _make_lgbm().fit(X_tr_wf, y_train)
        lgbm_proba = lgbm_wf.predict_proba(X_te_wf)[:, 1]
        lgbm_auc = roc_auc_score(y_test, lgbm_proba) if len(np.unique(y_test)) > 1 else 0.5
        lgbm_brier = brier_score_loss(y_test, lgbm_proba)
        print(f"  LightGBM           — AUC: {lgbm_auc:.4f}, Brier: {lgbm_brier:.4f}")

    # ── 모델 선택 (단순한 쪽 우선) ──────────────────────────────────
    if HAS_LGBM and lgbm_auc > lr_auc + 0.01:
        print("\n→ LightGBM 채택 (OOS AUC가 LR보다 >1% 우수)")
        best_clf = lgbm_wf
        best_proba = lgbm_proba
        model_type = "lgbm"
    else:
        print("\n→ LogisticRegression 채택 (단순성 우선)")
        best_clf = lr_wf
        best_proba = lr_proba
        model_type = "logistic"

    # ── Calibration 곡선 ─────────────────────────────────────────
    if len(np.unique(y_test)) > 1 and best_proba is not None:
        frac_pos, mean_pred = calibration_curve(y_test, best_proba, n_bins=5, strategy="quantile")
        print("\nCalibration (예측 확률 vs 실제 승률):")
        for mp, fp in zip(mean_pred, frac_pos):
            bar = "█" * int(fp * 20)
            print(f"  pred={mp:.2f}  actual={fp:.2f}  {bar}")

    # ── Permutation importance ──────────────────────────────────────
    print("\nPermutation importance (AUC drop):")
    X_te_perm = X_te_wf.copy()
    base_auc = roc_auc_score(y_test, best_proba) if len(np.unique(y_test)) > 1 else 0.5
    importances = []
    for i, col in enumerate(FEATURE_COLS):
        X_perm = X_te_perm.copy()
        X_perm[:, i] = np.random.permutation(X_perm[:, i])
        p_prob = best_clf.predict_proba(X_perm)[:, 1]
        drop = base_auc - (roc_auc_score(y_test, p_prob) if len(np.unique(y_test)) > 1 else 0.5)
        importances.append((col, drop))
    importances.sort(key=lambda x: -x[1])
    for col, drop in importances:
        bar = "▓" * max(0, int(drop * 200))
        print(f"  {col:20s}  drop={drop:+.4f}  {bar}")

    # ── 저장 ────────────────────────────────────────────────────────
    models = MLModels(clf=best_clf, scaler=scaler_wf, feature_cols=FEATURE_COLS, model_type=model_type)
    models.save(args.out)
    print(f"\n[완료] 모델 저장: {args.out}")

    # ── 게이트 ──────────────────────────────────────────────────────
    oos_auc = lgbm_auc if model_type == "lgbm" else lr_auc
    if oos_auc > 0.5:
        print(f"[GATE] OOS AUC {oos_auc:.4f} > 0.5 ✓")
    else:
        print(f"[GATE FAIL] OOS AUC {oos_auc:.4f} ≤ 0.5 — 모델 예측력 없음, 사용 금지")


def _make_lgbm():
    """얕은 LightGBM (과적합 방지)."""
    return lgb.LGBMClassifier(
        max_depth=3,
        n_estimators=80,
        learning_rate=0.04,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        reg_alpha=1.0,
        random_state=42,
        verbose=-1,
    )


def _purged_kfold(
    timestamps: np.ndarray,
    n_folds: int,
    hold_days: int,
    embargo_days: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """보유기간 겹침 제거: test fold 경계에서 hold_days purge + embargo 적용."""
    ts = pd.to_datetime(timestamps, utc=True)
    n = len(ts)
    fold_size = n // n_folds
    folds = []

    for k in range(n_folds):
        val_start = k * fold_size
        val_end = (k + 1) * fold_size if k < n_folds - 1 else n
        val_idx = np.arange(val_start, val_end)

        val_ts_start = ts[val_start]
        val_ts_end = ts[val_end - 1]

        # 학습 인덱스: val 범위 밖에서 purge + embargo 적용
        train_idx = []
        for i in range(n):
            if val_start <= i < val_end:
                continue
            t = ts[i]
            # val 시작 직전 hold_days purge (학습샘플이 val 구간과 겹칠 수 있음)
            if val_ts_start - pd.Timedelta(days=hold_days) <= t < val_ts_start:
                continue
            # val 종료 직후 embargo
            if val_ts_end < t <= val_ts_end + pd.Timedelta(days=embargo_days):
                continue
            train_idx.append(i)

        folds.append((np.array(train_idx), val_idx))

    return folds


if __name__ == "__main__":
    main()
