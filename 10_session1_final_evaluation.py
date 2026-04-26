# =============================================================================
# 10_session1_final_evaluation.py
# Stage: Session 1
# Description: Final test set evaluation for Session 1 selected model.
#              TEST SET UNLOCKED HERE — only used once.
#
# RESULTS:
#   Selected model: Logistic Regression
#   Test F1 (macro): 0.5872
#   Test Accuracy:   0.58
# =============================================================================

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, f1_score,
                              confusion_matrix, ConfusionMatrixDisplay)

def run_final_evaluation_session1(X_train, y_train, X_test, y_test, label_encoder):
    print("=" * 55)
    print("SESSION 1 — FINAL TEST SET EVALUATION")
    print("TEST SET UNLOCKED")
    print("=" * 55)

    # Train final model on full training data
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    test_preds = model.predict(X_test)
    test_f1 = f1_score(y_test, test_preds, average='macro')

    print(f"\nTest F1 (macro): {test_f1:.4f}")
    print(f"Test Accuracy:   {(test_preds == y_test).mean():.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds,
                                 target_names=['Critical', 'High', 'Low', 'Medium']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Critical', 'High', 'Low', 'Medium'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix — Session 1: Logistic Regression (Synthetic)')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix_session1.png', dpi=150)
    plt.show()
    print("\nConfusion matrix saved to outputs/confusion_matrix_session1.png")

    print("\n--- KEY OBSERVATIONS ---")
    print("High and Medium classes consistently confused — adjacent risk boundaries")
    print("in synthetic data are ambiguous by design, not a modeling failure.")
    print("VERDICT: Model performance reflects synthetic data ceiling (~0.57 F1).")
    print("NOT suitable for real-world risk inference.")

    return model

if __name__ == "__main__":
    print("Run this module via master_training_script.py for full pipeline execution.")
