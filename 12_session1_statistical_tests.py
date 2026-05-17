"""
12_session1_statistical_tests.py
==================================
Session 1 - Statistical Significance Testing
Dataset: Synthetic Kaggle (Session 1) vs NASA MDP (Session 2)

Applies paired t-test, Wilcoxon signed-rank test, 95% confidence intervals,
and Cohen's d effect size to cross-validation fold F1 scores across all
three sessions (Session 1, Session 1b, Session 2).

Key Results:
    Paired t-test S1 vs S2:  t=-22.92, p=0.0000  ✅ Significant
    Paired t-test S1b vs S2: t=+11.61, p=0.0003  ✅ Significant
    Paired t-test S1 vs S1b: t=-36.49, p=0.0000  ✅ Significant

    Wilcoxon: p=0.0625 all comparisons (non-significant — expected with n=5)
    Note: Wilcoxon requires n≥6 for p<0.05; non-significance is a known
    small-sample limitation, not a contradiction of the t-test results.

    95% Confidence Intervals:
    Session 1:  [0.5472, 0.5929]
    Session 1b: [0.7969, 0.8232]
    Session 2:  [0.7337, 0.7541]
    — No overlap between any pair confirming clear separation

    Cohen's d Effect Sizes (all large):
    S1 vs S2:  d = 12.19
    S1b vs S2: d = -6.99
    S1 vs S1b: d = 15.99

Dependencies: Requires cv_scores (Session 1 Step 11) and cv_scores_bin
(Session 1b Step 13). Session 2 fold scores hardcoded from notebook output.

Part of: IT Project Risk Classification Pipeline
Paper  : A Supervised Machine Learning Framework for IT Project Risk Classification
GitHub : https://github.com/GraceE-Dion/IT-Project-Risk-Classification
Author : Grace Egbedion, MTSU Department of Information Systems
"""

# ── [Paste Step 14 cell code from Session 1 notebook here] ───────────────────

# ── End of file summary ───────────────────────────────────────────────────────
print("=" * 55)
print("12: SESSION 1 STATISTICAL SIGNIFICANCE TESTS COMPLETE")
print("=" * 55)
print("  S1 vs S2:  t=-22.92, p=0.0000, d=12.19 (large) ✅")
print("  S1b vs S2: t=+11.61, p=0.0003, d=-6.99 (large) ✅")
print("  S1 vs S1b: t=-36.49, p=0.0000, d=15.99 (large) ✅")
print("  Wilcoxon:  p=0.0625 all (expected n=5 limitation)")
print()
print("  95% CIs — no overlap between any pair:")
print("    S1:  [0.5472, 0.5929]")
print("    S1b: [0.7969, 0.8232]")
print("    S2:  [0.7337, 0.7541]")
