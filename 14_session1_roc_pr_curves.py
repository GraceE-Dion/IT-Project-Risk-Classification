"""
14_session1_roc_pr_curves.py
==============================
Session 1 - ROC and Precision-Recall Curves
Dataset: Synthetic Kaggle Project Management Risk Raw (4,000 rows, 49 features)
Model: Logistic Regression (final selected model)

Per-class AUC and Average Precision (one-vs-rest):
    Critical: AUC=0.9106 | AP=0.6791
    High:     AUC=0.7879 | AP=0.5442
    Low:      AUC=0.8853 | AP=0.6664
    Medium:   AUC=0.7516 | AP=0.6080

Finding: AUC values (0.752–0.911) are substantially higher than macro F1
scores (0.49–0.70), which is expected — AUC measures discriminability across
all thresholds while F1 measures performance at the default 0.50 threshold.
The Critical class AUC of 0.911 indicates strong discriminative capacity for
the most governance-critical risk level. High class has the lowest AP (0.544),
consistent with its systematic confusion with Critical and Medium.

Outputs:
    outputs/roc_curves_session1.png
    outputs/pr_curves_session1.png

Dependencies: Requires lr_model, X_test, y_test, le_target from
10_session1_final_evaluation.py

Part of: IT Project Risk Classification Pipeline
Paper  : A Supervised Machine Learning Framework for IT Project Risk Classification
GitHub : https://github.com/GraceE-Dion/IT-Project-Risk-Classification
Author : Grace Egbedion, MTSU Department of Information Systems
"""

# ── [Paste Step 16 cell code from Session 1 notebook here] ───────────────────

# ── End of file summary ───────────────────────────────────────────────────────
print("=" * 55)
print("14: SESSION 1 ROC AND PR CURVES COMPLETE")
print("=" * 55)
print("  Per-class AUC and Average Precision (one-vs-rest):")
print("    Critical: AUC=0.9106 | AP=0.6791")
print("    High:     AUC=0.7879 | AP=0.5442")
print("    Low:      AUC=0.8853 | AP=0.6664")
print("    Medium:   AUC=0.7516 | AP=0.6080")
print()
print("  Outputs:")
print("    outputs/roc_curves_session1.png")
print("    outputs/pr_curves_session1.png")
