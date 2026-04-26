# =============================================================================
# 05_feature_scaling.py
# Stage: Data
# Description: StandardScaler fit on training set only.
#              Applied separately to validation and test sets.
#              CRITICAL: Scaler is never fit on validation or test data
#              to prevent data leakage.
# =============================================================================

import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_features(X_train, X_val, X_test):
    """
    Fit StandardScaler on training set only.
    Transform validation and test sets using training statistics.

    This prevents data leakage — validation and test distributions
    must remain unknown to the scaler during fitting.
    """
    print("=" * 55)
    print("FEATURE SCALING")
    print("=" * 55)

    scaler = StandardScaler()

    # Fit ONLY on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform validation and test using training statistics
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set mean (first 3 features): {scaler.mean_[:3].round(4)}")
    print(f"Training set std  (first 3 features): {scaler.scale_[:3].round(4)}")
    print(f"\nScaled train shape: {X_train_scaled.shape}")
    print(f"Scaled val shape:   {X_val_scaled.shape}")
    print(f"Scaled test shape:  {X_test_scaled.shape}")
    print("\nNOTE: Scaler fit on training set only. Validation and test sets")
    print("transformed using training statistics to prevent data leakage.")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

if __name__ == "__main__":
    # Example usage with dummy data
    X_train = np.random.randn(100, 5)
    X_val = np.random.randn(20, 5)
    X_test = np.random.randn(20, 5)
    scale_features(X_train, X_val, X_test)
