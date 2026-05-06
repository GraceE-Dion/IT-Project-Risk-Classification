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
