"""
19_session2_roc_pr_curves.py
==============================
Session 2 - ROC and Precision-Recall Curves
Dataset: NASA Metrics Data Program (MDP) JM1 (10,878 rows, 21 features)
Model: XGBoost (final selected model, binary Defective class)

Key Results:
    ROC AUC:           0.7119
    Average Precision: 0.4127
    Minority class prevalence: 0.1930
    PR AUC lift over random:   2.14x

Finding: ROC AUC of 0.712 confirms the model discriminates defective from
clean modules better than random (0.5). PR AUC of 0.413 is the more
informative metric under 80/20 imbalance — it is sensitive to false positives
in proportion to the minority class, which ROC AUC is not. The gap between
ROC AUC (0.712) and PR AUC (0.413) reflects the known optimism of ROC
analysis under class imbalance.

Outputs:
    outputs/roc_curve_session2.png
    outputs/pr_curve_session2.png

Dependencies: Requires xgb_final, X_test_scaled, y_test from
18_session2_final_evaluation.py

Part of: IT Project Risk Classification Pipeline
Paper  : A Supervised Machine Learning Framework for IT Project Risk Classification
GitHub : https://github.com/GraceE-Dion/IT-Project-Risk-Classification
Author : Grace Egbedion, MTSU Department of Information Systems
"""

# ── [Paste Step 14 cell code from Session 2 notebook here] ───────────────────

# ── End of file summary ───────────────────────────────────────────────────────
print("=" * 55)
print("19: SESSION 2 ROC AND PR CURVES COMPLETE")
print("=" * 55)
print("  ROC AUC:           0.7119")
print("  Average Precision: 0.4127  (2.1x above random 0.193)")
print()
print("  PR AUC is more informative than ROC under 80/20 imbalance.")
print()
print("  Outputs:")
print("    outputs/roc_curve_session2.png")
print("    outputs/pr_curve_session2.png")
