# Step 12b: Probability Calibration (Platt Scaling)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt
import numpy as np

# Apply Platt Scaling on validation set using the already-fitted lr_model
# cv='prefit' means we calibrate on top of the existing model without refitting
calibrated_model = CalibratedClassifierCV(lr_model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_val, y_val)

# Get calibrated probabilities on test set
test_proba_calibrated = calibrated_model.predict_proba(X_test)
test_proba_uncalibrated = lr_model.predict_proba(X_test)

# Brier Score (lower = better calibrated, 0 = perfect)
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import label_binarize

classes = list(range(len(le_target.classes_)))
y_test_bin = label_binarize(y_test, classes=classes)

brier_uncal = np.mean([
    brier_score_loss(y_test_bin[:, i], test_proba_uncalibrated[:, i])
    for i in range(len(classes))
])
brier_cal = np.mean([
    brier_score_loss(y_test_bin[:, i], test_proba_calibrated[:, i])
    for i in range(len(classes))
])

print("=" * 50)
print("PROBABILITY CALIBRATION RESULTS (Platt Scaling)")
print("=" * 50)
print(f"Brier Score - Uncalibrated: {brier_uncal:.4f}")
print(f"Brier Score - Calibrated:   {brier_cal:.4f}")
print(f"Improvement: {brier_uncal - brier_cal:.4f}")
print()
print("Brier Score interpretation: 0 = perfect, 0.25 = no skill")
print("Lower is better. Calibration corrects over/under-confidence")
print("in probability outputs, enabling risk threshold tuning.")

# Calibration curve — one per class
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, class_name in enumerate(le_target.classes_):
    prob_true_uncal, prob_pred_uncal = calibration_curve(
        y_test_bin[:, i], test_proba_uncalibrated[:, i], n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(
        y_test_bin[:, i], test_proba_calibrated[:, i], n_bins=10)

    axes[i].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    axes[i].plot(prob_pred_uncal, prob_true_uncal, 'b-o',
                 label='Uncalibrated', linewidth=2)
    axes[i].plot(prob_pred_cal,   prob_true_cal,   'r-s',
                 label='Calibrated (Platt)', linewidth=2)
    axes[i].set_xlabel('Mean Predicted Probability')
    axes[i].set_ylabel('Fraction of Positives')
    axes[i].set_title(f'Calibration Curve — Class: {class_name}')
    axes[i].legend(fontsize=9)
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Probability Calibration: Logistic Regression (Session 1)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('calibration_curves_session1.png', dpi=300, bbox_inches='tight')
plt.show()

print()
print("Calibration curves saved to calibration_curves_session1.png")
print()
print("Operational implication:")
print("  Calibrated probabilities allow risk threshold tuning.")
print("  Example: flag any project with P(Critical) > 0.35 for review,")
print("  rather than using the default 0.50 classification boundary.")
print("  This is directly relevant to NIST AI RMF 'Manage' function.")
