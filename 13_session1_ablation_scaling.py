"""
13_session1_ablation_scaling.py
=================================
Session 1 - Ablation: Logistic Regression Without StandardScaler
Dataset: Synthetic Kaggle Project Management Risk Raw (4,000 rows, 49 features)

Isolates the contribution of StandardScaler to Session 1 model performance
by retraining Logistic Regression on unscaled raw features under identical
split and evaluation conditions.

Key Results:
    LR WITHOUT scaling: Val F1=0.5170 | Test F1=0.5115 | CV=0.4656 ± 0.0085
    LR WITH scaling:    Val F1=0.5890 | Test F1=0.5872 | CV=0.5700 ± 0.0165

    Scaling contribution:
    Val  F1 delta: +0.0720
    Test F1 delta: +0.0757
    CV   F1 delta: +0.1044

Finding: StandardScaler contributes a meaningful +0.10 CV F1 improvement.
LR without scaling failed to converge (ConvergenceWarning at max_iter=1000),
confirming that unscaled mixed-range features make gradient-based optimisation
substantially more difficult. Scaling is a methodological requirement, not
a convenience. However, the +0.10 delta cannot account for the full
cross-session gap (0.1739 CV F1 between S1 and S2), supporting the
interpretation that dataset provenance is the primary explanatory factor.

Dependencies: Requires df, y_encoded from preprocessing steps.

Part of: IT Project Risk Classification Pipeline
Paper  : A Supervised Machine Learning Framework for IT Project Risk Classification
GitHub : https://github.com/GraceE-Dion/IT-Project-Risk-Classification
Author : Grace Egbedion, MTSU Department of Information Systems
"""

# ── [Paste Step 15 cell code from Session 1 notebook here] ───────────────────

# ── End of file summary ───────────────────────────────────────────────────────
print("=" * 55)
print("13: SESSION 1 ABLATION — SCALING COMPLETE")
print("=" * 55)
print("  LR WITHOUT scaling: CV F1 = 0.4656 ± 0.0085")
print("  LR WITH    scaling: CV F1 = 0.5700 ± 0.0165")
print("  Scaling contribution: +0.1044 CV F1")
print()
print("  LR without scaling failed to converge (ConvergenceWarning)")
print("  Scaling is a methodological requirement, not optional.")
print("  But +0.10 delta cannot explain the full cross-session gap.")
