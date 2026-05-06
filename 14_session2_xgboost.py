# =============================================================================
# 13_session2_xgboost.py
# Stage: Session 2
# Description: XGBoost tuning across 3 configurations on NASA MDP data.
#              Progressive regularization applied across versions.
#
# RESULTS:
#   XGB v1 (depth=3):        Val F1: 0.6589 | Gap: 0.1499 ❌
#   XGB v2 (strong reg):     Val F1: 0.6468 | Gap: 0.0955 ⚠️
#   XGB v3 (medium reg):     Val F1: 0.6441 | Gap: 0.1339 ❌
#
# DECISION: Single-split validation scores are pessimistic (small val set).
#           Proceed to 5-fold cross-validation for honest performance estimate.
#           XGB v2 parameters selected as base for CV.
# =============================================================================

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score

OVERFIT_THRESHOLD = 0.05
TARGET_NAMES = ['Clean', 'Defective']

def evaluate_xgb(model, X_train, y_train, X_val, y_val, name):
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    train_f1 = f1_score(y_train, train_preds, average='macro')
    val_f1 = f1_score(y_val, val_preds, average='macro')
    gap = train_f1 - val_f1
    status = "✅ PASS" if gap <= OVERFIT_THRESHOLD else "❌ OVERFIT"

    print(f"\n--- {name} ---")
    print(f"Training F1:   {train_f1:.4f}")
    print(f"Validation F1: {val_f1:.4f}")
    print(f"Overfit Gap:   {gap:.4f}  {status}")
    print(classification_report(y_val, val_preds, target_names=TARGET_NAMES))

    return val_f1, gap

def run_xgboost_session2(X_train, y_train, X_val, y_val):
    print("=" * 55)
    print("SESSION 2 — XGBoost Tuning (NASA MDP)")
    print("=" * 55)
    results = {}

    # XGBoost v1 — initial
    xgb1 = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                           subsample=0.8, colsample_bytree=0.8,
                           reg_alpha=0.1, reg_lambda=1.0,
                           eval_metric='logloss', random_state=42, n_jobs=-1)
    results['XGB v1'] = evaluate_xgb(xgb1, X_train, y_train, X_val, y_val,
                                      "XGBoost v1 (depth=3, lr=0.1)")

    # XGBoost v2 — strong regularization
    xgb2 = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.05,
                           subsample=0.7, colsample_bytree=0.7,
                           reg_alpha=0.5, reg_lambda=2.0,
                           eval_metric='logloss', random_state=42, n_jobs=-1)
    results['XGB v2'] = evaluate_xgb(xgb2, X_train, y_train, X_val, y_val,
                                      "XGBoost v2 (depth=2, strong reg)")

    # XGBoost v3 — medium regularization
    xgb3 = XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.05,
                           subsample=0.75, colsample_bytree=0.75,
                           reg_alpha=0.3, reg_lambda=1.5,
                           eval_metric='logloss', random_state=42, n_jobs=-1)
    results['XGB v3'] = evaluate_xgb(xgb3, X_train, y_train, X_val, y_val,
                                      "XGBoost v3 (depth=2, medium reg, n=200)")

    # Summary
    print("\n" + "=" * 55)
    print("XGBOOST TUNING SUMMARY")
    print("=" * 55)
    print(f"{'Config':<12} {'Val F1':>8} {'Gap':>8} {'Status':>12}")
    print("-" * 45)
    for name, (val_f1, gap) in results.items():
        status = "✅ PASS" if gap <= OVERFIT_THRESHOLD else "❌ OVERFIT"
        print(f"{name:<12} {val_f1:>8.4f} {gap:>8.4f} {status:>12}")

    print("\nOBSERVATION: Single-split val scores (0.64-0.66) are pessimistic.")
    print("Small validation set (1,087 rows) causes variance in estimates.")
    print("NEXT STEP: 5-fold cross-validation for reliable performance estimate.")

if __name__ == "__main__":
    print("Run this module via master_training_script.py for full pipeline execution.")
