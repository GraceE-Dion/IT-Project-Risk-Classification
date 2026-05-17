
Copy

# =============================================================================
# 27_dataset_comparison.py
# Stage: Analysis — Cross-Session Comparison
# Description: Side-by-side comparison of synthetic vs NASA MDP performance
#              across all three sessions including binary recast.
#
# CROSS-SESSION RESULTS:
#   Session 1  (4-class, synthetic) — CV F1: 0.5700 ± 0.0165
#   Session 1b (binary,  synthetic) — CV F1: 0.8101 ± 0.0095
#   Session 2  (binary,  NASA MDP)  — CV F1: 0.7439 ± 0.0074
#
# REVISED CORE FINDING:
#   Under identical binary task conditions, synthetic data (CV F1: 0.8101)
#   EXCEEDS real NASA MDP data (CV F1: 0.7439). The original 30.5% gap was
#   substantially a task structure artefact (4-class vs binary). Synthetic
#   data is DECEPTIVELY EASY — it produces inflated metrics that do not
#   reflect genuine operational difficulty.
#
# STATISTICAL SIGNIFICANCE:
#   S1 vs S2:  t=-22.92, p=0.0000, Cohen's d=12.19 (large) ✅
#   S1b vs S2: t=+11.61, p=0.0003, Cohen's d=-6.99  (large) ✅
#   S1 vs S1b: t=-36.49, p=0.0000, Cohen's d=15.99  (large) ✅
#   Wilcoxon:  p=0.0625 all (expected — n=5 too small for p<0.05)
#
# 95% CONFIDENCE INTERVALS:
#   Session 1:  [0.5472, 0.5929]
#   Session 1b: [0.7969, 0.8232]
#   Session 2:  [0.7337, 0.7541]
#   No overlap between any pair
#
# ADDITIONAL FINDINGS:
#   Dummy classifier floor: S1=0.1289 | S2=0.4465
#   ML lift over dummy:     S1=+0.4576 | S2=+0.1616
#   Brier Score:            S1=0.1390 (no cal gain) | S2=0.1989→0.1406 (29.3%)
#   SMOTE contribution:     +0.1310 CV F1, 1.7x more stable
#   Scaling contribution:   +0.1044 CV F1 (S1 ablation)
#   Top SHAP feature:       S1=Org_Process_Maturity (0.69) | S2=LOC_TOTAL (0.46)
#   ROC AUC:                S1=0.911 (Critical) | S2=0.7119
#   PR AUC:                 S1=0.679 (Critical) | S2=0.4127
#   VIF > 10:               S2 = 15 of 21 features
#   Error analysis:         92 FNs, FN LOC_TOTAL mean=18.7 vs TP=125.8 (6.7x)
#   McNemar's (LR vs XGB):  p=0.1014 (not significant)
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
import numpy as np
 
 
def run_dataset_comparison():
    print("=" * 65)
    print("DATASET COMPARISON — SYNTHETIC vs NASA RAW MDP (3 SESSIONS)")
    print("=" * 65)
 
    # ── Results from all three sessions ──────────────────────────────────────
    results = {
        "Session 1\n(4-class synthetic)": {
            "cv_f1": 0.5700, "cv_std": 0.0165,
            "test_f1": 0.5872, "test_accuracy": 0.58,
            "best_model": "Logistic Regression",
            "dummy_f1": 0.1289, "brier_uncal": 0.1390,
            "brier_cal": 0.1423, "top_shap": "Org_Process_Maturity (0.69)",
            "rows": 4000, "features": 49, "task": "4-class",
            "roc_auc": "0.911 (Critical)", "pr_auc": "0.679 (Critical)",
        },
        "Session 1b\n(binary synthetic)": {
            "cv_f1": 0.8101, "cv_std": 0.0095,
            "test_f1": 0.8220, "test_accuracy": 0.82,
            "best_model": "Logistic Regression",
            "dummy_f1": "N/A", "brier_uncal": "N/A",
            "brier_cal": "N/A", "top_shap": "—",
            "rows": 4000, "features": 49, "task": "binary (recast)",
            "roc_auc": "—", "pr_auc": "—",
        },
        "Session 2\n(binary NASA MDP)": {
            "cv_f1": 0.7439, "cv_std": 0.0074,
            "test_f1": 0.6082, "test_accuracy": 0.70,
            "best_model": "XGBoost",
            "dummy_f1": 0.4465, "brier_uncal": 0.1989,
            "brier_cal": 0.1406, "top_shap": "LOC_TOTAL (0.46)",
            "rows": 10878, "features": 21, "task": "binary",
            "roc_auc": "0.7119", "pr_auc": "0.4127",
        }
    }
 
    # ── Print comparison table ────────────────────────────────────────────────
    s1  = results["Session 1\n(4-class synthetic)"]
    s1b = results["Session 1b\n(binary synthetic)"]
    s2  = results["Session 2\n(binary NASA MDP)"]
 
    print(f"\n{'Metric':<28} {'Session 1':>16} {'Session 1b':>16} {'Session 2':>16}")
    print("-" * 80)
    print(f"{'Task':<28} {'4-class synth':>16} {'binary synth':>16} {'binary NASA':>16}")
    print(f"{'Rows':<28} {s1['rows']:>16,} {s1b['rows']:>16,} {s2['rows']:>16,}")
    print(f"{'Features':<28} {s1['features']:>16} {s1b['features']:>16} {s2['features']:>16}")
    print(f"{'CV F1 (macro)':<28} {s1['cv_f1']:>16.4f} {s1b['cv_f1']:>16.4f} {s2['cv_f1']:>16.4f}")
    print(f"{'CV Std':<28} {s1['cv_std']:>16.4f} {s1b['cv_std']:>16.4f} {s2['cv_std']:>16.4f}")
    print(f"{'Test F1 (macro)':<28} {s1['test_f1']:>16.4f} {s1b['test_f1']:>16.4f} {s2['test_f1']:>16.4f}")
    print(f"{'Test Accuracy':<28} {s1['test_accuracy']:>16.2f} {s1b['test_accuracy']:>16.2f} {s2['test_accuracy']:>16.2f}")
    print(f"{'Best Model':<28} {'LR':>16} {'LR':>16} {'XGBoost':>16}")
    print(f"{'Dummy Classifier F1':<28} {'0.1289':>16} {'N/A':>16} {'0.4465':>16}")
    print(f"{'Brier (uncal)':<28} {'0.1390':>16} {'N/A':>16} {'0.1989':>16}")
    print(f"{'Brier (cal)':<28} {'0.1423 (none)':>16} {'N/A':>16} {'0.1406 (+29.3%)':>16}")
    print(f"{'Top SHAP feature':<28} {'Org_Process_Mat':>16} {'—':>16} {'LOC_TOTAL':>16}")
    print(f"{'ROC AUC':<28} {'0.911 (Crit)':>16} {'—':>16} {'0.7119':>16}")
    print(f"{'PR AUC':<28} {'0.679 (Crit)':>16} {'—':>16} {'0.4127':>16}")
 
    # ── Plot 1: Three-session CV F1 comparison ────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
 
    session_labels = ['Session 1\n4-class\nSynthetic',
                      'Session 1b\nBinary\nSynthetic',
                      'Session 2\nBinary\nNASA MDP']
    cv_f1_vals = [0.5700, 0.8101, 0.7439]
    cv_std_vals = [0.0165, 0.0095, 0.0074]
    colors = ['#d9534f', '#f0ad4e', '#5cb85c']
 
    # CV F1 bar chart
    bars = axes[0].bar(session_labels, cv_f1_vals, yerr=cv_std_vals,
                       color=colors, capsize=8, width=0.5,
                       edgecolor='white', linewidth=0.8)
    axes[0].set_title('Cross-Validation F1 (macro)\nAll Three Sessions',
                      fontsize=11, fontweight='bold', color='#1F4E79')
    axes[0].set_ylabel('CV F1 Score', fontsize=10)
    axes[0].set_ylim(0, 0.95)
    for i, (v, s) in enumerate(zip(cv_f1_vals, cv_std_vals)):
        axes[0].text(i, v + s + 0.015, f'{v:.4f}',
                     ha='center', fontsize=9, fontweight='bold')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)
 
    # CV Std bar chart
    axes[1].bar(session_labels, cv_std_vals, color=colors, width=0.5,
                edgecolor='white', linewidth=0.8)
    axes[1].set_title('CV Standard Deviation\n(Lower = More Stable)',
                      fontsize=11, fontweight='bold', color='#1F4E79')
    axes[1].set_ylabel('CV Std', fontsize=10)
    axes[1].set_ylim(0, 0.022)
    for i, v in enumerate(cv_std_vals):
        axes[1].text(i, v + 0.0005, f'{v:.4f}',
                     ha='center', fontsize=9, fontweight='bold')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].grid(axis='y', linestyle='--', alpha=0.4)
 
    # Test F1 bar chart
    test_f1_vals = [0.5872, 0.8220, 0.6082]
    axes[2].bar(session_labels, test_f1_vals, color=colors, width=0.5,
                edgecolor='white', linewidth=0.8)
    axes[2].set_title('Final Test F1 (macro)\nHeld-out Test Set',
                      fontsize=11, fontweight='bold', color='#1F4E79')
    axes[2].set_ylabel('Test F1 Score', fontsize=10)
    axes[2].set_ylim(0, 0.95)
    for i, v in enumerate(test_f1_vals):
        axes[2].text(i, v + 0.015, f'{v:.4f}',
                     ha='center', fontsize=9, fontweight='bold')
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].grid(axis='y', linestyle='--', alpha=0.4)
 
    plt.suptitle(
        'Cross-Session Performance Comparison\n'
        'Session 1b reveals synthetic data is deceptively easy to classify',
        fontsize=13, fontweight='bold', color='#1F4E79'
    )
    plt.tight_layout()
    plt.savefig('outputs/dataset_comparison.png', dpi=300,
                bbox_inches='tight')
    plt.show()
    print("\nComparison chart saved to outputs/dataset_comparison.png")
 
    # ── Key research findings ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("KEY RESEARCH FINDINGS")
    print("=" * 65)
    print()
    print("1. REVISED CORE FINDING: Synthetic data is deceptively easy.")
    print("   Binary synthetic (0.8101) > binary NASA MDP (0.7439).")
    print("   Original 30.5% gap substantially a task structure artefact.")
    print()
    print("2. All cross-session differentials statistically significant:")
    print("   S1 vs S2:  t=-22.92, p=0.0000, Cohen's d=12.19 (large) ✅")
    print("   S1b vs S2: t=+11.61, p=0.0003, Cohen's d=-6.99  (large) ✅")
    print("   Wilcoxon:  p=0.0625 (expected — n=5 too small for p<0.05)")
    print()
    print("3. Optimal model shifts: LR (synthetic) → XGBoost (real data).")
    print("   SHAP explains the shift: governance features dominate in")
    print("   synthetic data; code volume features dominate in real data.")
    print()
    print("4. SMOTE contributes +0.131 CV F1 and 1.7x more stable CV.")
    print("   Calibration essential: uncalibrated XGBoost flags 72.6%")
    print("   at threshold 0.30 vs 19.9% calibrated (true rate: 19.3%).")
    print()
    print("5. Small-module blind spot: missed defects 6.7x smaller")
    print("   than detected (FN LOC_TOTAL: 18.7 vs TP: 125.8).")
    print()
    print("6. Severe multicollinearity: 15/21 features VIF > 10.")
    print("   HALSTEAD_EFFORT == HALSTEAD_PROG_TIME (r=1.0000).")
    print()
    print("7. Random Forest consistently overfits on SMOTE-augmented data:")
    print("   6 configurations, 2 sessions, all failed the 0.05 gate.")
    print()
    print("8. Dataset provenance verification is a required step:")
    print("   Kaggle-preprocessed NASA MDP normalised all features to [0,1]")
    print("   destroying authentic distributional variation.")
 
 
if __name__ == "__main__":
    run_dataset_comparison()
