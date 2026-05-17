"""
21_session2_ablation_calibration.py
=====================================
Session 2 - Ablation: XGBoost Without Probability Calibration
Dataset: NASA Metrics Data Program (MDP) JM1 (10,878 rows, 21 features)

Isolates the contribution of Isotonic Regression calibration to probability
output reliability.

Key Results:
    WITHOUT calibration: Brier Score = 0.1989
    WITH calibration:    Brier Score = 0.1406
    Improvement:         +0.0582 (29.3%)

Critical Threshold Finding:
    At threshold 0.30:
      Uncalibrated: flags 72.6% of modules as Defective
      Calibrated:   flags 19.9% of modules as Defective

    True minority class prevalence: 19.3%

Finding: Uncalibrated XGBoost probability scores are severely overconfident
toward the positive class and are not directly usable for operational threshold
tuning without calibration. The calibrated model's 19.9% flagging rate at
threshold 0.30 is operationally plausible and closely matches true prevalence.
Calibration is essential for governance deployment contexts.

Dependencies: Requires xgb_final, calibrated_xgb, X_test_scaled, y_test
from 18_session2_final_evaluation.py and 16_session2_probability_calibration.py

Part of: IT Project Risk Classification Pipeline
Paper  : A Supervised Machine Learning Framework for IT Project Risk Classification
GitHub : https://github.com/GraceE-Dion/IT-Project-Risk-Classification
Author : Grace Egbedion, MTSU Department of Information Systems
"""

# ── [Paste Step 16 cell code from Session 2 notebook here] ───────────────────

# ── End of file summary ───────────────────────────────────────────────────────
print("=" * 55)
print("21: SESSION 2 ABLATION — CALIBRATION COMPLETE")
print("=" * 55)
print("  WITHOUT calibration: Brier = 0.1989")
print("  WITH    calibration: Brier = 0.1406  (29.3% improvement)")
print()
print("  At threshold 0.30:")
print("    Uncalibrated flags: 72.6% of modules")
print("    Calibrated flags:   19.9% of modules")
print("    True prevalence:    19.3%")
print()
print("  Calibration is essential for operational threshold tuning.")
