# =============================================================================
# 06_class_balancing.py
# Stage: Data
# Description: SMOTE applied to training set only to handle class imbalance.
#              Validation and test sets retain original distribution to ensure
#              evaluation reflects real-world conditions.
#
# Applied to: Session 2 (NASA MDP) — 80/20 clean/defective split
# Not required for: Session 1 (synthetic) — class_weight='balanced' used instead
# =============================================================================

import numpy as np
from imblearn.over_sampling import SMOTE

def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE to training set only.

    SMOTE (Synthetic Minority Oversampling Technique) generates synthetic
    samples for the minority class by interpolating between existing samples.

    IMPORTANT: SMOTE is applied to training data ONLY.
    Applying SMOTE to validation or test sets would create an optimistic
    bias in evaluation metrics that does not reflect real deployment conditions.

    Session 2 class distribution:
      Before SMOTE: 7,021 Clean (80.7%) vs 1,682 Defective (19.3%)
      After SMOTE:  7,021 Clean (50%) vs 7,021 Defective (50%)
    """
    print("=" * 55)
    print("CLASS BALANCING — SMOTE")
    print("=" * 55)

    print(f"Before SMOTE:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, cnt in zip(unique, counts):
        label = "Clean" if cls == 0 else "Defective"
        pct = cnt / len(y_train) * 100
        print(f"  Class {cls} ({label}): {cnt} ({pct:.1f}%)")

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"\nAfter SMOTE:")
    unique, counts = np.unique(y_resampled, return_counts=True)
    for cls, cnt in zip(unique, counts):
        label = "Clean" if cls == 0 else "Defective"
        pct = cnt / len(y_resampled) * 100
        print(f"  Class {cls} ({label}): {cnt} ({pct:.1f}%)")

    print(f"\nResampled train shape: {X_resampled.shape}")
    print("\nNOTE: SMOTE applied to training set only.")
    print("Validation and test sets retain original 80/20 distribution.")

    return X_resampled, y_resampled

if __name__ == "__main__":
    # Example usage with dummy imbalanced data
    X = np.random.randn(100, 5)
    y = np.array([0] * 80 + [1] * 20)
    apply_smote(X, y)
