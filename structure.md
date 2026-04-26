# 📂 Project File Structure

This repository is organized into a modular pipeline covering both development sessions — from synthetic baseline through to NASA Raw MDP real-world data modeling.

---

## The Complete Workflow

| File | Stage | Description |
|---|---|---|
| `01_setup_environment.py` | Setup | Installs all dependencies and verifies environment |
| `02_data_acquisition.py` | Data | Downloads synthetic dataset from Kaggle and NASA MDP JM1.arff from GitHub |
| `03_data_inspection.py` | Data | Shape, column names, target distribution, missing values, data authenticity check |
| `04_preprocessing.py` | Data | Missing value imputation, label encoding, byte string decoding, feature/target separation |
| `05_feature_scaling.py` | Data | StandardScaler fit on training set; applied to validation and test sets |
| `06_class_balancing.py` | Data | SMOTE applied to training set only; class distribution before and after |
| `07_train_val_test_split.py` | Data | Stratified 80/10/10 split — test set locked until final evaluation |
| `08_session1_synthetic_baseline.py` | Session 1 | Full pipeline on synthetic dataset — Logistic Regression, Random Forest, XGBoost |
| `09_session1_cross_validation.py` | Session 1 | 5-fold stratified CV on synthetic dataset — true performance ceiling: 0.57 F1 |
| `10_session1_final_evaluation.py` | Session 1 | Logistic Regression final test evaluation — Test F1: 0.5872 |
| `11_session2_nasa_mdp_baseline.py` | Session 2 | Logistic Regression baseline on NASA MDP data — Val F1: 0.6232, Gap: 0.0440 |
| `12_session2_random_forest.py` | Session 2 | Random Forest tuning — 3 configurations, all overfit (negative finding) |
| `13_session2_xgboost.py` | Session 2 | XGBoost tuning — 3 configurations with regularization progression |
| `14_session2_cross_validation.py` | Session 2 | 5-fold CV on NASA MDP — Mean F1: 0.7439, Std: 0.0074 (selected model) |
| `15_session2_final_evaluation.py` | Session 2 | XGBoost final test evaluation — Test F1: 0.6082, Accuracy: 0.70 |
| `16_dataset_comparison.py` | Analysis | Side-by-side comparison: synthetic vs NASA MDP performance metrics |
| `master_training_script.py` | Full Pipeline | End-to-end pipeline from data acquisition to final evaluation and comparison |
| `requirements.txt` | Setup | All dependencies for full reproduction |

---

## 🌳 Repository Tree

```
IT-Project-Risk-Classification/
│
├── notebooks/
│   ├── 01_synthetic_baseline.ipynb         # Session 1 — Full Kaggle notebook
│   └── 02_nasa_mdp_real_data.ipynb         # Session 2 — Full Kaggle notebook
│
├── data/
│   ├── raw/                                # Original datasets (do not modify)
│   │   ├── project_risk_raw_dataset.csv    # Synthetic dataset (Kaggle)
│   │   └── JM1.arff                        # NASA MDP raw data (GitHub)
│   └── processed/                          # Cleaned, encoded, scaled datasets
│
├── outputs/
│   ├── confusion_matrix_session1.png       # Session 1 confusion matrix
│   ├── confusion_matrix_session2.png       # Session 2 confusion matrix
│   ├── class_distribution_session1.png     # Session 1 class distribution
│   └── missing_values_session1.png         # Session 1 missing values chart
│
├── 01_setup_environment.py
├── 02_data_acquisition.py
├── 03_data_inspection.py
├── 04_preprocessing.py
├── 05_feature_scaling.py
├── 06_class_balancing.py
├── 07_train_val_test_split.py
├── 08_session1_synthetic_baseline.py
├── 09_session1_cross_validation.py
├── 10_session1_final_evaluation.py
├── 11_session2_nasa_mdp_baseline.py
├── 12_session2_random_forest.py
├── 13_session2_xgboost.py
├── 14_session2_cross_validation.py
├── 15_session2_final_evaluation.py
├── 16_dataset_comparison.py
├── master_training_script.py
├── requirements.txt
├── structure.md
└── README.md
```

---

## 🚀 Production and Documentation

**`master_training_script.py`**
The full end-to-end pipeline. Run this file to replicate the entire two-session development from data acquisition to final evaluation and dataset comparison in one execution.

**`requirements.txt`**
Install all dependencies with:
```bash
pip install -r requirements.txt
```

**`notebooks/`**
Both Kaggle notebooks are included for full reproducibility. Each notebook mirrors its corresponding `.py` files and can be run independently on Kaggle's free CPU environment. No GPU required.

**`README.md`**
The main project report covering both development sessions, overfitting prevention protocol, performance metrics, dataset comparison, and AI governance principles.
