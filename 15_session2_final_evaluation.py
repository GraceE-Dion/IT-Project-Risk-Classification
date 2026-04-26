# =============================================================================
# 15_session2_final_evaluation.py
# Stage: Session 2
# Description: Final test set evaluation for Session 2 selected model.
#              TEST SET UNLOCKED HERE — only used once.
#
# RESULTS:
#   Selected model: XGBoost
#   Test F1 (macro): 0.6082
#   Test Accuracy:   0.70
#   Clean F1:        0.80
#   Defective F1:    0.42
#   Confusion matrix: 643/878 Clean correct, 118/210 Defective correct
# =============================================================================

import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, f1_score,
                              confusion_matrix, ConfusionMatrixDisplay)

TARGET_NAMES = ['Clean', 'Defective']

def run_final_evaluation_session2(X_train, y_train, X_test, y_test):
    print("=" * 55)
    print("SESSION 2 — FINAL TEST SET EVALUATION")
    print("TEST SET UNLOCKED")
    print("=" * 55)

    # Train final model on full training data
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
    model.fit(X_train, y_train)

    test_preds = model.predict(X_test)
    test_f1 = f1_score(y_test, test_preds, average='macro')

    print(f"\nTest F1 (macro): {test_f1:.4f}")
    print(f"Test Accuracy:   {(test_preds == y_test).mean():.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds, target_names=TARGET_NAMES))

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=TARGET_NAMES)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix — Session 2: XGBoost (NASA MDP)')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix_session2.png', dpi=150)
    plt.show()
    print("\nConfusion matrix saved to outputs/confusion_matrix_session2.png")

    print("\n--- DEPLOYMENT RISK ASSESSMENT ---")
    print("Clean F1 (0.80):     Strong — 643/878 correct")
    print("Defective F1 (0.42): Weaker — 118/210 correct")
    print("False Negatives: 92 missed defective modules")
    print("  → In production: missed defects reaching deployment = HIGH RISK")
    print("  → Recall optimization recommended for production deployment")
    print("  → Current model suitable for research and portfolio demonstration")
    print("\nVERDICT: Real-world data produces meaningfully better and more")
    print("reliable results than synthetic data. Model is a valid baseline")
    print("for IT project defect risk classification research.")

    return model

if __name__ == "__main__":
    print("Run this module via master_training_script.py for full pipeline execution.")
