# IT Project Risk Classification
### A Machine Learning Framework for Cybersecurity Governance and Human-Factor Risk Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()

---

## Overview

This project develops a supervised machine learning pipeline to classify IT project risk outcomes using governance, human-factor, and cybersecurity compliance features. The work sits at the intersection of **IT project management**, **cybersecurity governance**, and **data-driven risk analytics** — contributing to the broader research agenda of improving project delivery outcomes through predictive human-factor modeling.

This project is part of a research portfolio supporting work on the SECURE-EXEC™ framework for cybersecurity execution and governance.

---

## Research Motivation

IT project failures remain disproportionately tied to human-factor risks — governance gaps, compliance drift, team dynamics, and stakeholder misalignment — rather than purely technical failures. Existing risk models underweight these dimensions. This project builds a classification model that:

- Identifies early-stage risk signals in IT projects
- Surfaces human-factor and governance features as predictive variables
- Provides an interpretable, auditable framework aligned with NIST, CMMC, and ISO 27001 risk taxonomies

---

## Project Structure

```
it-project-risk-classification/
│
├── data/
│   ├── raw/                  # Original downloaded datasets (do not modify)
│   └── processed/            # Cleaned, feature-engineered datasets
│
├── notebooks/
│   ├── 01_synthetic_baseline.ipynb     # Session 1 — Synthetic dataset pipeline
│   └── 02_nasa_mdp_real_data.ipynb     # Session 2 — NASA Raw MDP pipeline
│
├── src/
│   ├── preprocess.py         # Data cleaning and feature engineering
│   ├── train.py              # Model training with cross-validation
│   ├── evaluate.py           # Metrics, confusion matrix, ROC-AUC
│   └── utils.py              # Helper functions
│
├── models/                   # Saved model artifacts (.pkl, .joblib)
├── outputs/                  # Plots, reports, final metrics
├── tests/                    # Unit tests for src modules
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Datasets

### Session 1 — Synthetic Baseline
**Source:** Kaggle — Project Management Risk Raw (ka66ledata)  
**Rows:** 4,000 | **Features:** 49 | **Target:** Risk Level (Critical / High / Medium / Low)  
**Note:** AI-generated dataset. Used for baseline methodology demonstration only. Not suitable for real-world inference.

### Session 2 — NASA Raw MDP Data
**Source:** NASA Metrics Data Program (MDP) via [NASADefectDataset](https://github.com/klainfo/NASADefectDataset/tree/master/OriginalData/MDP) — JM1.arff  
**Rows:** 10,878 | **Features:** 21 | **Target:** Defect Label (Clean / Defective)  
**License:** Public domain — peer-reviewed, widely cited in software engineering research  
**Key Features:** LOC, Cyclomatic Complexity, Halstead Metrics, Branch Count, Module Dependencies

---

## Methodology

### 1. Exploratory Data Analysis
- Class distribution analysis
- Correlation heatmap
- Missing value assessment
- Feature importance preview (mutual information)

### 2. Preprocessing
- Null imputation strategy (median for numeric, mode for categorical)
- Label encoding / one-hot encoding
- Feature scaling (StandardScaler for linear models)
- Class imbalance handling — **SMOTE** if needed

### 3. Models Evaluated
| Model | Rationale |
|---|---|
| Logistic Regression | Interpretable baseline |
| Random Forest | Handles mixed features, resistant to overfitting |
| Gradient Boosting (XGBoost) | Strong tabular performance |
| Support Vector Machine | Effective in high-dimensional space |
| MLP Neural Network | Captures non-linear risk interactions |

### 4. Overfitting Prevention Protocol

This project applies a strict **generalization discipline** from day one:

| Control | Implementation |
|---|---|
| Data split | 80% train / 10% validation / 10% test — test set locked until final evaluation |
| Cross-validation | 5-fold stratified CV on training set |
| Early stopping | Patience = 10 epochs on validation loss (for MLP) |
| Regularization | L2 / weight decay; Dropout (0.3–0.5) for neural network |
| Tree depth control | `max_depth`, `min_samples_leaf` for ensemble models |
| Learning curves | Plotted every run — training vs. validation accuracy/loss |
| Generalization threshold | Validation accuracy must be within 3–5% of training accuracy |

**Diagnosis flags monitored:**
- Training accuracy >> Validation accuracy → Overfitting → Increase regularization, reduce model complexity
- Both low → Underfitting → Add features, reduce regularization
- Validation loss rising while training loss falls → Early stopping triggered

### 5. Evaluation Metrics

Primary metrics (in order of priority for imbalanced risk data):
1. **F1 Score (macro)** — balances precision and recall across classes
2. **Recall** — critical for catching high-risk projects (false negatives are costly)
3. **ROC-AUC** — overall discriminative power
4. **Precision** — reduces false alarms
5. **Accuracy** — reported but not primary for imbalanced classes

---

## Results

### Dataset Comparison

| | Synthetic Dataset | NASA Raw MDP |
|---|---|---|
| CV F1 (macro) | 0.5700 | **0.7439** |
| CV Std | 0.0165 | **0.0074** |
| Rows | 4,000 | 10,878 |
| Data Quality | Simulated | Real NASA ✅ |
| Inference Ready | ❌ | ✅ |
| Improvement | baseline | **+30.5% F1, 2.2x more stable** |

### Session 1 — Synthetic Dataset

| Model | Val F1 (macro) | Overfit Gap | Status |
|---|---|---|---|
| Logistic Regression | 0.5890 | 0.0079 | ✅ Selected |
| Random Forest | 0.5590 | 0.3775 | ❌ Overfit |
| XGBoost | 0.5920 | 0.0805 | ❌ |
| XGBoost 5-Fold CV | 0.5700 ± 0.0165 | — | ✅ |

**Final Test F1 (Logistic Regression):** 0.5872 | **Accuracy:** 0.58

### Session 2 — NASA Raw MDP Data

| Model | Val F1 (macro) | Overfit Gap | Status |
|---|---|---|---|
| Logistic Regression | 0.6232 | 0.0440 | ✅ Clean |
| Random Forest | 0.6492 | 0.1402 | ❌ Overfit |
| XGBoost 5-Fold CV | 0.7439 ± 0.0074 | — | ✅ Selected |

**Final Test F1 (XGBoost):** 0.6082 | **Accuracy:** 0.70

---

## Key Findings

1. **Data quality critically impacts model performance** — Real NASA MDP data produced 30.5% better F1 and 2.2x more stable cross-validation results than the synthetic dataset.

2. **Synthetic data is not suitable for real-world inference** — Models trained on AI-generated project data fail to capture real organizational complexity, governance gaps, and human-factor interdependencies.

3. **Logistic Regression is a competitive baseline** — On the synthetic dataset, Logistic Regression (F1: 0.5890, gap: 0.0079) outperformed Random Forest and XGBoost in generalization, suggesting largely linear feature-target relationships in synthetic data.

4. **XGBoost generalizes best on real data** — On NASA MDP data, XGBoost achieved a 5-fold CV F1 of 0.7439 with low variance (std: 0.0074), demonstrating strong and stable generalization on authentic software project metrics.

5. **Class imbalance requires explicit handling** — The 80/20 clean/defective split in NASA data required SMOTE augmentation to prevent model bias toward the majority class.

6. **Future work** requires datasets with explicit governance, human-factor, and cybersecurity compliance features aligned with NIST SP 800-37 and CMMC frameworks for direct IT project risk classification.

---

## Governance & Compliance Alignment

This project's feature taxonomy is designed to map to established cybersecurity and project governance frameworks:

- **NIST SP 800-37** — Risk Management Framework
- **CMMC Level 2/3** — Human-factor access and control compliance indicators
- **ISO 27001** — Information security risk treatment features
- **PMI PMBOK** — Project performance domain variables

---

## Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
imbalanced-learn>=0.11
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
joblib>=1.3
```

Install with:
```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/GraceE-Dion/IT-Project-Risk-Classification.git
cd IT-Project-Risk-Classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebooks in order
jupyter notebook notebooks/01_synthetic_baseline.ipynb
jupyter notebook notebooks/02_nasa_mdp_real_data.ipynb
```

> Notebooks were developed and executed on Kaggle (free CPU). No GPU required.

---

## Author

**Grace Egbedion**  
Technical Program Manager | Cybersecurity Governance Specialist | PhD Candidate, Computational & Data Science  
[GitHub](https://github.com/GraceE-Dion) | [LinkedIn](#)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you reference this work:
```
Egbedion, G. (2025). IT Project Risk Classification: A Machine Learning Framework 
for Cybersecurity Governance and Human-Factor Risk Analytics. GitHub Repository.
https://github.com/GraceE-Dion/IT-Project-Risk-Classification
```
