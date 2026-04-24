# IT Project Risk Classification
### A Machine Learning Framework for Cybersecurity Governance and Human-Factor Risk Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange)]()

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
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_tuning_and_regularization.ipynb
│   └── 05_final_evaluation.ipynb
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

## Dataset

**Primary Source:** Kaggle / PROMISE Repository  
**Target Variable:** Project outcome — Risk Level (High / Medium / Low) or Success / Failure  
**Key Feature Categories:**

| Category | Example Features |
|---|---|
| Governance | Compliance score, audit frequency, policy adherence rate |
| Human Factors | Team size, turnover rate, stakeholder engagement score |
| Project Parameters | Duration, budget variance, scope change count |
| Cybersecurity | Security controls implemented, incident count, vulnerability exposure |
| Delivery | Sprint velocity, milestone slippage, defect rate |

> Dataset citation and license details added upon download.

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

> To be populated after training runs.

| Model | F1 (macro) | Recall | ROC-AUC | Overfit Gap |
|---|---|---|---|---|
| Logistic Regression | - | - | - | - |
| Random Forest | - | - | - | - |
| XGBoost | - | - | - | - |
| MLP | - | - | - | - |

---

## Key Findings

> To be populated after analysis.

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
git clone https://github.com/GraceE-Dion/it-project-risk-classification.git
cd it-project-risk-classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place raw dataset in data/raw/

# 4. Run notebooks in order
jupyter notebook notebooks/01_eda.ipynb
```

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
https://github.com/GraceE-Dion/it-project-risk-classification
```
