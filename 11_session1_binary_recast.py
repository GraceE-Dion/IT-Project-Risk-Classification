"""
11_session1_binary_recast.py
=============================
Session 1b - Binary-Recast Experiment
Dataset: Synthetic Kaggle Project Management Risk Raw (4,000 rows, 49 features)

Remaps 4-class risk labels to binary to remove the task structure confounder
in the cross-session comparison:
    Critical / High  → 1 (High Risk)
    Medium  / Low    → 0 (Low Risk)

Key Results:
    Session 1  (4-class, synthetic) — CV F1: 0.5700 ± 0.0165
    Session 1b (binary,  synthetic) — CV F1: 0.8101 ± 0.0095
    Session 2  (binary,  NASA MDP)  — CV F1: 0.7439 ± 0.0074

Revised Core Finding:
    Under identical binary task conditions, synthetic data (CV F1: 0.8101)
    exceeds real NASA MDP data (CV F1: 0.7439). The original 30.5% gap was
    substantially driven by task structure (4-class vs binary), not dataset
    authenticity alone. Synthetic data is deceptively easy to classify.

Outputs:
    outputs/confusion_matrix_session1b_binary.png

Dependencies: Requires X_train, X_val, X_test, y_train, y_val, y_test,
le_target from 10_session1_final_evaluation.py

Part of: IT Project Risk Classification Pipeline
Paper  : A Supervised Machine Learning Framework for IT Project Risk Classification
GitHub : https://github.com/GraceE-Dion/IT-Project-Risk-Classification
Author : Grace Egbedion, MTSU Department of Information Systems
"""

# ── [Paste Step 13 cell code from Session 1 notebook here] ───────────────────

# ── End of file summary ───────────────────────────────────────────────────────
print("=" * 55)
print("11: SESSION 1b BINARY RECAST COMPLETE")
print("=" * 55)
print("  Session 1  (4-class, synthetic): CV F1 = 0.5700 ± 0.0165")
print("  Session 1b (binary,  synthetic): CV F1 = 0.8101 ± 0.0095")
print("  Session 2  (binary,  NASA MDP):  CV F1 = 0.7439 ± 0.0074")
print()
print("  REVISED FINDING: Synthetic data is deceptively easy.")
print("  The original 30.5% gap was substantially task structure,")
print("  not dataset authenticity alone.")
print()
print("  Output: outputs/confusion_matrix_session1b_binary.png")
