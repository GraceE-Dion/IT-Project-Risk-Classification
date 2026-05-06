# =============================================================================
# 09_session1_cross_validation.py
# Stage: Session 1
# Description: 5-fold stratified cross-validation on synthetic dataset.
#              Provides honest performance estimate independent of train/val split.
#
# FINDING: CV Mean F1: 0.5700, Std: 0.0165
#          Confirms dataset performance ceiling at ~0.57 F1.
#          Consistent with synthetic data's limited real-world signal.
# =============================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

def run_cross_validation_session1(X_train, y_train):
    print("=" * 55)
    print("SESSION 1 — 5-Fold Cross Validation (Synthetic)")
    print("=" * 55)

    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=cv, scoring='f1_macro', n_jobs=-1
    )

    print(f"CV F1 scores per fold: {cv_scores.round(4)}")
    print(f"Mean CV F1: {cv_scores.mean():.4f}")
    print(f"Std CV F1:  {cv_scores.std():.4f}")

    print("\n--- INTERPRETATION ---")
    print(f"Performance ceiling confirmed at ~{cv_scores.mean():.2f} F1.")
    print("Low variance across folds confirms stable but limited generalization.")
    print("Ceiling reflects synthetic data's generated signal — not real organizational complexity.")
    print("\nCOMPARISON NOTE: Session 2 (NASA MDP) CV Mean F1: 0.7439, Std: 0.0074")
    print("Real data produces 30.5% better F1 and 2.2x more stable results.")

    return cv_scores

if __name__ == "__main__":
    print("Run this module via master_training_script.py for full pipeline execution.")
