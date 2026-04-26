# =============================================================================
# 07_train_val_test_split.py
# Stage: Data
# Description: Stratified 80/10/10 split.
#              Test set is LOCKED until final evaluation only.
#              Stratification preserves class proportions across all splits.
# =============================================================================

import numpy as np
from sklearn.model_selection import train_test_split

def split_data(X, y, random_state=42):
    """
    Stratified 80/10/10 train/validation/test split.

    Overfitting prevention:
    - Test set is locked and only used in the final evaluation step
    - All model selection and hyperparameter tuning uses validation set only
    - Stratify ensures class balance is preserved in each split

    Session 1 (synthetic):
      Train: 3,200 | Validation: 400 | Test: 400

    Session 2 (NASA MDP):
      Train: 8,703 | Validation: 1,087 | Test: 1,088
    """
    print("=" * 55)
    print("TRAIN / VALIDATION / TEST SPLIT (80/10/10)")
    print("=" * 55)

    # Step 1: Split off 10% test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=0.10,
        random_state=random_state,
        stratify=y
    )

    # Step 2: Split remaining 90% into 80% train / 10% validation
    # 0.111 * 90% ≈ 10% of total
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.111,
        random_state=random_state,
        stratify=y_train_val
    )

    print(f"Train size:      {X_train.shape}")
    print(f"Validation size: {X_val.shape}")
    print(f"Test size:       {X_test.shape}")

    print(f"\nTrain class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} ({cnt/len(y_train)*100:.1f}%)")

    print(f"\n{'='*55}")
    print("TEST SET IS LOCKED UNTIL FINAL EVALUATION")
    print(f"{'='*55}")

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    X = np.random.randn(1000, 10)
    y = np.array([0] * 800 + [1] * 200)
    split_data(X, y)
