# =============================================================================
# 04_preprocessing.py
# Stage: Data
# Description: Missing value imputation, label encoding, byte string decoding,
#              feature/target separation for both datasets
# =============================================================================

import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------------------
# Session 1 — Synthetic Dataset Preprocessing
# -----------------------------------------------------------------------------
def preprocess_synthetic(filepath):
    print("=" * 55)
    print("SESSION 1 — Synthetic Dataset Preprocessing")
    print("=" * 55)

    df = pd.read_csv(filepath)
    print(f"Initial shape: {df.shape}")
    print(f"Missing values before: {df.isnull().sum().sum()}")

    # Handle missing values
    # Missing columns: Tech_Environment_Stability (~2,500), Risk_Management_Maturity (~600),
    # Change_Control_Maturity (~550)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    print(f"Missing values after imputation: {df.isnull().sum().sum()}")

    # Separate target and drop ID
    X = df.drop(columns=['Risk_Level', 'Project_ID'])
    y = df['Risk_Level']

    # Encode categorical features
    le = LabelEncoder()
    for col in X.select_dtypes(include='object').columns:
        X[col] = le.fit_transform(X[col])

    # Encode target
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    print(f"\nTarget classes: {le_target.classes_}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y_encoded.shape}")

    return X.values, y_encoded, le_target

# -----------------------------------------------------------------------------
# Session 2 — NASA MDP Preprocessing
# -----------------------------------------------------------------------------
def preprocess_nasa_mdp(filepath):
    print("\n" + "=" * 55)
    print("SESSION 2 — NASA MDP Preprocessing")
    print("=" * 55)

    # Load ARFF
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    print(f"Initial shape: {df.shape}")

    # Decode byte string labels from ARFF format
    # Raw labels appear as b'N' and b'Y' — must decode to string before encoding
    df['label'] = df['label'].str.decode('utf-8')
    print(f"\nTarget distribution after decoding:")
    print(df['label'].value_counts())

    # Encode target: N → 0 (Clean), Y → 1 (Defective)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    print(f"\nEncoded classes: {le.classes_} → {le.transform(le.classes_)}")

    # Verify no missing or infinite values
    print(f"\nMissing values: {df.isnull().sum().sum()}")
    print(f"Infinite values: {np.isinf(df.select_dtypes(include='number')).sum().sum()}")

    # Separate features and target
    X = df.drop(columns=['label'])
    y = df['label']

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    return X.values, y.values, le

if __name__ == "__main__":
    preprocess_synthetic(
        "/kaggle/input/datasets/ka66ledata/project-management-risk-raw/project_risk_raw_dataset.csv"
    )
    preprocess_nasa_mdp("JM1.arff")
