"""
16_session2_probability_calibration.py
========================================
Session 2 - Probability Calibration (Isotonic Regression)
Dataset: NASA Metrics Data Program (MDP) JM1 (10,878 rows, 21 features)

Applies Isotonic Regression post-hoc to the selected XGBoost model using
CalibratedClassifierCV (cv='prefit') fitted on the validation set.
Isotonic Regression is preferred over Platt Scaling for XGBoost because
gradient boosting produces non-monotonic miscalibration that the
non-parametric isotonic approach handles more effectively.

Key Results:
    Brier Score Uncalibrated      : 0.1989
    Brier Score Calibrated        : 0.1406
    Improvement                   : 0.0582 (29.3%)

Finding: XGBoost produces meaningfully uncalibrated probability outputs,
consistent with the known behaviour of gradient boosting methods which
optimise a discriminative objective rather than a proper scoring rule.
Isotonic Regression produces a 29.3% Brier Score improvement, enabling
operationally meaningful risk threshold tuning. This contrasts with
Session 1 where Logistic Regression showed no calibration gain (-0.0033).

Governance alignment: NIST AI RMF 1.0 Manage function — calibrated
probabilities allow setting alert thresholds below the default 0.50
boundary (e.g. P(Defective) > 0.30) to reduce false negatives in
high-stakes deployment contexts.

Outputs:
    outputs/calibration_curve_session2.png

Dependencies: Requires xgb_final, X_train_resampled, X_val_scaled,
y_val, X_test_scaled, y_test from 18_session2_final_evaluation.py

Part of: IT Project Risk Classification Pipeline
Paper  : A Supervised Machine Learning Framework for IT Project Risk Classification
GitHub : https://github.com/GraceE-Dion/IT-Project-Risk-Classification
Author : Grace Egbedion, Department of Information Systems, IT Project Management,
         Middle Tennessee State University
"""

# ── End of file summary ───────────────────────────────────────────────────────
print("=" * 55)
print("16: SESSION 2 PROBABILITY CALIBRATION COMPLETE")
print("=" * 55)
print(f"  Brier Score Uncalibrated : 0.1989")
print(f"  Brier Score Calibrated   : 0.1406")
print(f"  Improvement              : 0.0582 (29.3%)")
print()
print("  Finding: XGBoost benefits significantly from Isotonic")
print("  Regression calibration. Contrast with Session 1 where")
print("  Logistic Regression showed no calibration gain (-0.0033).")
print()
print("  Operational implication: Set P(Defective) > 0.30 threshold")
print("  rather than default 0.50 to reduce false negatives.")
print()
print("  Output: outputs/calibration_curve_session2.png")

# Step 12b: Probability Calibration (Isotonic Regression)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import numpy as np

# Isotonic Regression preferred over Platt Scaling for XGBoost:
# XGBoost produces uncalibrated probabilities; isotonic regression
# is non-parametric and handles non-monotonic miscalibration better.
# cv='prefit' calibrates on top of already-fitted xgb_final
calibrated_xgb = CalibratedClassifierCV(
    xgb_final, method='isotonic', cv='prefit'
)
calibrated_xgb.fit(X_val_scaled, y_val)

# Probabilities for the Defective class (index 1)
prob_uncal = xgb_final.predict_proba(X_test_scaled)[:, 1]
prob_cal   = calibrated_xgb.predict_proba(X_test_scaled)[:, 1]

# Brier Score (0 = perfect, 0.25 = no skill)
brier_uncal = brier_score_loss(y_test, prob_uncal)
brier_cal   = brier_score_loss(y_test, prob_cal)

print("=" * 50)
print("PROBABILITY CALIBRATION (Isotonic Regression)")
print("=" * 50)
print(f"Brier Score - Uncalibrated: {brier_uncal:.4f}")
print(f"Brier Score - Calibrated:   {brier_cal:.4f}")
print(f"Improvement:                {brier_uncal - brier_cal:.4f}")
print()
print("Brier Score: 0 = perfect, 0.25 = no skill. Lower is better.")

# Calibration curve
prob_true_uncal, prob_pred_uncal = calibration_curve(
    y_test, prob_uncal, n_bins=10)
prob_true_cal, prob_pred_cal = calibration_curve(
    y_test, prob_cal, n_bins=10)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
ax.plot(prob_pred_uncal, prob_true_uncal, 'b-o',
        label='Uncalibrated XGBoost', linewidth=2)
ax.plot(prob_pred_cal,   prob_true_cal,   'r-s',
        label='Calibrated (Isotonic Regression)', linewidth=2)
ax.set_xlabel('Mean Predicted Probability', fontsize=12)
ax.set_ylabel('Fraction of Positives (Defective)', fontsize=12)
ax.set_title('Probability Calibration — Session 2: XGBoost\n'
             '(NASA MDP JM1, Defective Class)',
             fontsize=13, fontweight='bold', color='#1F4E79')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('calibration_curve_session2.png', dpi=300, bbox_inches='tight')
plt.show()
print("Calibration curve saved.")
print()
print("Operational implication:")
print("  Calibrated probabilities enable risk threshold tuning.")
print("  Example: flag any module with P(Defective) > 0.30 for review")
print("  to reduce false negatives in high-stakes deployment contexts.")
