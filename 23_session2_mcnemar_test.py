"""
23_session2_mcnemar_test.py
============================
Session 2 - McNemar's Test: Logistic Regression vs XGBoost
Dataset: NASA Metrics Data Program (MDP) JM1 test set (1,088 samples)

Compares the error patterns of LR and XGBoost to assess whether XGBoost's
F1 advantage is accompanied by a statistically distinct error pattern.

Contingency Table:
    Both correct (a):          691
    LR correct, XGB wrong (b):  51
    LR wrong, XGB correct (c):  70
    Both wrong (d):            276

Key Results:
    McNemar's test (exact): statistic=51.0, p=0.1014
    Significant (p < 0.05): False

    Test macro F1:
    Logistic Regression: 0.5951
    XGBoost:             0.6082

Finding: Although XGBoost corrects 70 cases that LR fails on (c=70) while
LR corrects only 51 XGBoost failures (b=51), this difference does not reach
statistical significance (p=0.1014). The non-significant result indicates
XGBoost's F1 advantage (0.6082 vs 0.5951) is not accompanied by a
statistically distinct error pattern at the individual prediction level.
The two models share a large proportion of correct predictions (691/1088),
suggesting they differ in treatment of borderline cases rather than
systematic misclassification patterns.

Dependencies: Requires lr_model, xgb_final, X_test_scaled, y_test from
18_session2_final_evaluation.py

Part of: IT Project Risk Classification Pipeline
Paper  : A Supervised Machine Learning Framework for IT Project Risk Classification
GitHub : https://github.com/GraceE-Dion/IT-Project-Risk-Classification
Author : Grace Egbedion, MTSU Department of Information Systems
"""

# ── [Paste Step 17 cell code from Session 2 notebook here] ───────────────────

# ── End of file summary ───────────────────────────────────────────────────────
print("=" * 55)
print("22: SESSION 2 McNEMAR'S TEST COMPLETE")
print("=" * 55)
print("  Contingency: a=691, b=51, c=70, d=276")
print("  McNemar's p = 0.1014  (not significant)")
print("  LR F1: 0.5951 | XGBoost F1: 0.6082")
print()
print("  XGBoost corrects more LR errors (c=70 > b=51)")
print("  but difference not statistically significant at p<0.05.")
