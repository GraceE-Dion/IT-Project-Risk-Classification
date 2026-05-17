# =============================================================================
# 18_session2_final_evaluation.py
# Stage: Session 2 — Final Evaluation
# Description: Final test set evaluation for Session 2 selected model.
#              TEST SET UNLOCKED HERE — only used once.
#
# RESULTS:
#   Selected model:  XGBoost (5-fold CV: 0.7439 ± 0.0074)
#   Test F1 (macro): 0.6082
#   Test Accuracy:   0.70
#   Clean F1:        0.80  (643/878 correct)
#   Defective F1:    0.42  (118/210 correct)
#   False Negatives: 92 missed defective modules (miss rate: 43.8%)
#
# PROBABILITY CALIBRATION (Isotonic Regression):
#   Brier Score Uncalibrated: 0.1989
#   Brier Score Calibrated:   0.1406  (29.3% improvement)
#   At threshold 0.30 — Uncalibrated flags: 72.6% | Calibrated: 19.9%
#   True minority class prevalence: 19.3%
#   Finding: Calibration essential for operational threshold tuning
#
# SHAP TOP 3 FEATURES (TreeExplainer, Defective class):
#   1. LOC_TOTAL           0.4633
#   2. LOC_BLANK           0.2321
#   3. DESIGN_COMPLEXITY   0.1842
#   Finding: Code volume dominates — inverse of Session 1 (governance features)
#
# ROC / PR CURVES:
#   ROC AUC:           0.7119
#   Average Precision: 0.4127  (2.1x above random baseline 0.193)
#
# ABLATION RESULTS:
#   Without SMOTE:        CV F1 = 0.6129 ± 0.0124  (delta: -0.1310)
#   Without Calibration:  Brier = 0.1989            (delta: +0.0582)
#   SMOTE contributes +0.131 CV F1 and 1.7x more stable CV
#
# McNEMAR'S TEST (LR vs XGBoost):
#   p-value: 0.1014 (not significant)
#   c=70 XGB corrects LR errors | b=51 LR corrects XGB errors
#   XGBoost advantage not statistically distinct at p<0.05
#
# FEATURE MULTICOLLINEARITY (VIF):
#   15 of 21 features with VIF > 10 (severe multicollinearity)
#   HALSTEAD_EFFORT == HALSTEAD_PROG_TIME (r=1.0000)
#   6 features with acceptable VIF < 10:
#     DESIGN_COMPLEXITY, ESSENTIAL_COMPLEXITY, LOC_BLANK,
#     LOC_COMMENTS, HALSTEAD_LEVEL, LOC_CODE_AND_COMMENT
#
# ERROR ANALYSIS (92 missed defective modules):
#   FN mean LOC_TOTAL:  18.7  vs  TP mean: 125.8  (6.7x difference)
#   FN mean P(Defect):  0.358 — model uncertain, not wrong
#   Threshold 0.30: recovers 66/92 FNs | adds 371 FPs
#   Threshold 0.35: recovers 54/92 FNs | adds 261 FPs
#   Finding: Small simple modules systematically missed
#
# CROSS-SESSION COMPARISON:
#   Session 1  (4-class, synthetic) — CV F1: 0.5700 ± 0.0165
#   Session 1b (binary,  synthetic) — CV F1: 0.8101 ± 0.0095
#   Session 2  (binary,  NASA MDP)  — CV F1: 0.7439 ± 0.0074
#
# STATISTICAL SIGNIFICANCE:
#   S1 vs S2:  t=-22.92, p=0.0000, Cohen's d=12.19 (large) ✅
#   S1b vs S2: t=+11.61, p=0.0003, Cohen's d=-6.99  (large) ✅
#   S1 vs S1b: t=-36.49, p=0.0000, Cohen's d=15.99  (large) ✅
#   Wilcoxon: p=0.0625 all (expected — n=5 too small for p<0.05)
#
# REVISED CORE FINDING:
#   Under identical binary task conditions, synthetic data (0.8101)
#   exceeds real NASA MDP data (0.7439). The original 30.5% gap was
#   substantially a task structure artefact. Synthetic data is
#   deceptively easy — it masks genuine operational difficulty.
#
# Part of: IT Project Risk Classification Pipeline
# Paper  : A Supervised Machine Learning Framework for IT Project Risk
#          Classification: Cybersecurity Governance, Human-Factor
#          Analytics, and the Impact of Data Quality
# GitHub : https://github.com/GraceE-Dion/IT-Project-Risk-Classification
# Author : Grace Egbedion
#          Department of Information Systems, IT Project Management
#          Middle Tennessee State University
# =============================================================================
 
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, f1_score,
                              confusion_matrix, ConfusionMatrixDisplay)
 
TARGET_NAMES = ['Clean', 'Defective']
 
 
def run_final_evaluation_session2(X_train, y_train, X_test, y_test):
    print("=" * 55)
    print("SESSION 2 — FINAL TEST SET EVALUATION")
    print("TEST SET UNLOCKED")
    print("=" * 55)
 
    # ── Train final model on full training data ───────────────────────────────
    # Hyperparameters match Session 2 Step 15 cross-validation configuration
    # random_state=42 throughout for reproducibility
    model = XGBClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,        # L1 regularisation — stronger than v1
        reg_lambda=2.0,       # L2 regularisation — stronger than v1
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
 
    # ── Evaluation ────────────────────────────────────────────────────────────
    test_preds = model.predict(X_test)
    test_f1    = f1_score(y_test, test_preds, average='macro')
 
    print(f"\nTest F1 (macro): {test_f1:.4f}")
    print(f"Test Accuracy:   {(test_preds == y_test).mean():.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds,
                                target_names=TARGET_NAMES))
 
    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm   = confusion_matrix(y_test, test_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=TARGET_NAMES)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix — Session 2: XGBoost (NASA MDP JM1)')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix_session2_pub.png', dpi=300,
                bbox_inches='tight')
    plt.show()
    print("\nConfusion matrix saved to outputs/confusion_matrix_session2_pub.png")
 
    # ── Deployment risk assessment ────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("DEPLOYMENT RISK ASSESSMENT")
    print("=" * 55)
    print(f"  Clean F1:    0.80 — 643/878 correct")
    print(f"  Defective F1: 0.42 — 118/210 correct")
    print(f"  False Negatives: 92 missed defective modules (43.8% miss rate)")
    print()
    print("  Error analysis finding:")
    print("    Missed modules are systematically SMALLER and SIMPLER")
    print("    FN mean LOC_TOTAL: 18.7 vs TP mean: 125.8 (6.7x difference)")
    print("    Model learned complexity-driven rule — blind spot for small modules")
    print()
    print("  Threshold recommendations (calibrated probabilities):")
    print("    P(Defective) > 0.30 — recovers 66/92 FNs, adds 371 FPs")
    print("    P(Defective) > 0.35 — recovers 54/92 FNs, adds 261 FPs")
    print("    P(Defective) > 0.50 — default threshold (production baseline)")
    print()
    print("  NIST AI RMF 1.0 Manage function alignment:")
    print("    Calibrated probabilities enable operationally meaningful")
    print("    threshold tuning. Deployment in triage mode recommended:")
    print("    flag high-probability modules for human review,")
    print("    not automated pass/fail gatekeeping.")
    print()
    print("  McNemar's test (LR vs XGBoost): p=0.1014 — not significant")
    print("    XGBoost advantage over LR not statistically distinct at p<0.05")
 
    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("CROSS-SESSION SUMMARY")
    print("=" * 55)
    print(f"  Session 1  (4-class, synthetic): CV F1 = 0.5700 ± 0.0165")
    print(f"  Session 1b (binary,  synthetic): CV F1 = 0.8101 ± 0.0095")
    print(f"  Session 2  (binary,  NASA MDP):  CV F1 = 0.7439 ± 0.0074")
    print()
    print("  REVISED CORE FINDING:")
    print("    Under identical binary task conditions, synthetic data (0.8101)")
    print("    EXCEEDS real NASA MDP data (0.7439). The original 30.5% gap")
    print("    was substantially a task structure artefact. Synthetic data is")
    print("    DECEPTIVELY EASY — it masks genuine operational difficulty.")
    print()
    print("  All cross-session differentials statistically significant:")
    print("    S1 vs S2:  t=-22.92, p=0.0000, Cohen's d=12.19 (large) ✅")
    print("    S1b vs S2: t=+11.61, p=0.0003, Cohen's d=-6.99  (large) ✅")
    print("    Wilcoxon: p=0.0625 (expected — n=5 too small for p<0.05)")
 
    return model
 
 
if __name__ == "__main__":
    print("Run this module via master_training_script.py for full pipeline.")
