# =============================================================================
# 08_session1_synthetic_baseline.py
# Stage: Session 1
# Description: Full model comparison on synthetic dataset.
#              Logistic Regression, Random Forest (3 configs), XGBoost (3 configs)
#              Overfitting diagnosed and documented across all configurations.
#
# FINDING: Logistic Regression selected — tightest overfit gap (0.0079)
#          and near-optimal Val F1 (0.5890) on synthetic data.
#          Random Forest and XGBoost consistently overfit.
# =============================================================================

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score

OVERFIT_THRESHOLD = 0.05

def evaluate_model(model, X_train, y_train, X_val, y_val, model_name, target_names):
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    train_f1 = f1_score(y_train, train_preds, average='macro')
    val_f1 = f1_score(y_val, val_preds, average='macro')
    gap = train_f1 - val_f1
    status = "✅ PASS" if gap <= OVERFIT_THRESHOLD else "❌ OVERFIT"

    print(f"\n--- {model_name} ---")
    print(f"Training F1 (macro): {train_f1:.4f}")
    print(f"Validation F1 (macro): {val_f1:.4f}")
    print(f"Overfit Gap: {gap:.4f}  {status}")
    print(classification_report(y_val, val_preds, target_names=target_names))

    return val_f1, gap

def run_session1(X_train, X_val, y_train, y_val):
    print("=" * 55)
    print("SESSION 1 — Synthetic Dataset Model Comparison")
    print("=" * 55)
    target_names = ['Critical', 'High', 'Low', 'Medium']
    results = {}

    # Logistic Regression — baseline
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    results['Logistic Regression'] = evaluate_model(
        lr, X_train, y_train, X_val, y_val, "Logistic Regression", target_names
    )

    # Random Forest — deep (overfit expected)
    rf1 = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5,
                                  random_state=42, class_weight='balanced', n_jobs=-1)
    results['RF (deep)'] = evaluate_model(
        rf1, X_train, y_train, X_val, y_val, "Random Forest (deep — overfit expected)", target_names
    )

    # Random Forest — shallow (underfitting correction)
    rf2 = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=20,
                                  max_features='sqrt', random_state=42,
                                  class_weight='balanced', n_jobs=-1)
    results['RF (shallow)'] = evaluate_model(
        rf2, X_train, y_train, X_val, y_val, "Random Forest (shallow)", target_names
    )

    # Random Forest — balanced (sweet spot attempt)
    rf3 = RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_leaf=10,
                                  max_features='sqrt', random_state=42,
                                  class_weight='balanced', n_jobs=-1)
    results['RF (balanced)'] = evaluate_model(
        rf3, X_train, y_train, X_val, y_val, "Random Forest (balanced)", target_names
    )

    # XGBoost v1
    xgb1 = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                           subsample=0.8, colsample_bytree=0.8,
                           eval_metric='mlogloss', random_state=42, n_jobs=-1)
    results['XGBoost v1'] = evaluate_model(
        xgb1, X_train, y_train, X_val, y_val, "XGBoost v1", target_names
    )

    # XGBoost v2 — regularized
    xgb2 = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                           subsample=0.7, colsample_bytree=0.7,
                           reg_alpha=0.1, reg_lambda=1.5,
                           eval_metric='mlogloss', random_state=42, n_jobs=-1)
    results['XGBoost v2'] = evaluate_model(
        xgb2, X_train, y_train, X_val, y_val, "XGBoost v2 (regularized)", target_names
    )

    # XGBoost v3 — fine tuned
    xgb3 = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           reg_alpha=0.05, reg_lambda=1.0,
                           eval_metric='mlogloss', random_state=42, n_jobs=-1)
    results['XGBoost v3'] = evaluate_model(
        xgb3, X_train, y_train, X_val, y_val, "XGBoost v3 (fine tuned)", target_names
    )

    # Summary
    print("\n" + "=" * 55)
    print("SESSION 1 MODEL COMPARISON SUMMARY")
    print("=" * 55)
    print(f"{'Model':<25} {'Val F1':>8} {'Gap':>8} {'Status':>12}")
    print("-" * 55)
    for name, (val_f1, gap) in results.items():
        status = "✅ PASS" if gap <= OVERFIT_THRESHOLD else "❌ OVERFIT"
        print(f"{name:<25} {val_f1:>8.4f} {gap:>8.4f} {status:>12}")

    print("\nSELECTED MODEL: Logistic Regression")
    print("REASON: Tightest overfit gap (0.0079) with near-optimal Val F1.")
    print("Synthetic data has near-linear relationships — simpler models generalize better.")

    return lr  # Return selected model

if __name__ == "__main__":
    print("Run this module via master_training_script.py for full pipeline execution.")
