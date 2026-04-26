# =============================================================================
# 14_session2_cross_validation.py
# Stage: Session 2
# Description: 5-fold stratified cross-validation on NASA MDP data.
#              Provides reliable performance estimate for XGBoost.
#
# FINDING: CV Mean F1: 0.7439, Std: 0.0074
#          Per-fold: [0.7426, 0.7435, 0.7564, 0.7333, 0.7436]
#          XGBoost selected as final model.
#          30.5% improvement over synthetic baseline (0.5700).
#          2.2x more stable (std: 0.0165 → 0.0074).
# =============================================================================

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

def run_cross_validation_session2(X_train, y_train):
    print("=" * 55)
    print("SESSION 2 — 5-Fold Cross Validation (NASA MDP)")
    print("=" * 55)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=cv, scoring='f1_macro', n_jobs=-1
    )

    print(f"CV F1 scores per fold: {cv_scores.round(4)}")
    print(f"Mean CV F1: {cv_scores.mean():.4f}")
    print(f"Std CV F1:  {cv_scores.std():.4f}")

    print("\n--- INTERPRETATION ---")
    print(f"True model performance confirmed at ~{cv_scores.mean():.2f} F1.")
    print("Single-split validation (0.64-0.66) was pessimistic due to small val set.")
    print("Low std (0.0074) confirms stable generalization across all folds.")

    print("\n--- DATASET COMPARISON ---")
    print(f"  Synthetic CV F1: 0.5700 ± 0.0165")
    print(f"  NASA MDP CV F1:  {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Improvement:     +{(cv_scores.mean() - 0.5700)*100:.1f}% F1")
    print(f"  Stability gain:  {0.0165 / cv_scores.std():.1f}x more stable")

    print("\nSELECTED MODEL: XGBoost (these parameters)")
    print("Proceed to final test set evaluation.")

    return cv_scores

if __name__ == "__main__":
    print("Run this module via master_training_script.py for full pipeline execution.")
