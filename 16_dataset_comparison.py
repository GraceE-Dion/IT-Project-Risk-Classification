# =============================================================================
# 16_dataset_comparison.py
# Stage: Analysis
# Description: Side-by-side comparison of synthetic vs NASA MDP performance.
#              Core research finding: data quality drives model reliability.
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np

def run_dataset_comparison():
    print("=" * 60)
    print("DATASET COMPARISON — SYNTHETIC vs NASA RAW MDP")
    print("=" * 60)

    # Results from both sessions
    results = {
        "Synthetic Dataset": {
            "cv_f1": 0.5700,
            "cv_std": 0.0165,
            "test_f1": 0.5872,
            "test_accuracy": 0.58,
            "best_model": "Logistic Regression",
            "rows": 4000,
            "features": 49,
            "real_data": False
        },
        "NASA Raw MDP (JM1)": {
            "cv_f1": 0.7439,
            "cv_std": 0.0074,
            "test_f1": 0.6082,
            "test_accuracy": 0.70,
            "best_model": "XGBoost",
            "rows": 10878,
            "features": 21,
            "real_data": True
        }
    }

    # Print comparison table
    print(f"\n{'Metric':<30} {'Synthetic':>15} {'NASA MDP':>15} {'Change':>12}")
    print("-" * 75)
    print(f"{'Rows':<30} {results['Synthetic Dataset']['rows']:>15,} {results['NASA Raw MDP (JM1)']['rows']:>15,}")
    print(f"{'Features':<30} {results['Synthetic Dataset']['features']:>15} {results['NASA Raw MDP (JM1)']['features']:>15}")
    print(f"{'CV F1 (macro)':<30} {results['Synthetic Dataset']['cv_f1']:>15.4f} {results['NASA Raw MDP (JM1)']['cv_f1']:>15.4f} {'+30.5%':>12}")
    print(f"{'CV Std':<30} {results['Synthetic Dataset']['cv_std']:>15.4f} {results['NASA Raw MDP (JM1)']['cv_std']:>15.4f} {'2.2x stable':>12}")
    print(f"{'Test F1 (macro)':<30} {results['Synthetic Dataset']['test_f1']:>15.4f} {results['NASA Raw MDP (JM1)']['test_f1']:>15.4f}")
    print(f"{'Test Accuracy':<30} {results['Synthetic Dataset']['test_accuracy']:>15.2f} {results['NASA Raw MDP (JM1)']['test_accuracy']:>15.2f}")
    print(f"{'Best Model':<30} {'Logistic Regression':>15} {'XGBoost':>15}")
    print(f"{'Real Data':<30} {'No':>15} {'Yes':>15}")
    print(f"{'Inference Ready':<30} {'No':>15} {'Closer':>15}")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # CV F1 comparison
    datasets = ['Synthetic\nDataset', 'NASA Raw\nMDP (JM1)']
    cv_f1 = [0.5700, 0.7439]
    cv_std = [0.0165, 0.0074]
    colors = ['#d9534f', '#5cb85c']

    axes[0].bar(datasets, cv_f1, yerr=cv_std, color=colors, capsize=8, width=0.5)
    axes[0].set_title('Cross-Validation F1 (macro)\nSynthetic vs Real Data', fontsize=12)
    axes[0].set_ylabel('CV F1 Score')
    axes[0].set_ylim(0, 0.9)
    for i, (v, s) in enumerate(zip(cv_f1, cv_std)):
        axes[0].text(i, v + s + 0.01, f'{v:.4f}', ha='center', fontsize=10)

    # CV Std comparison (stability)
    axes[1].bar(datasets, cv_std, color=colors, width=0.5)
    axes[1].set_title('CV Standard Deviation\n(Lower = More Stable)', fontsize=12)
    axes[1].set_ylabel('CV Std')
    axes[1].set_ylim(0, 0.025)
    for i, v in enumerate(cv_std):
        axes[1].text(i, v + 0.0005, f'{v:.4f}', ha='center', fontsize=10)

    plt.suptitle('Data Quality Impact on Model Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/dataset_comparison.png', dpi=150)
    plt.show()
    print("\nComparison chart saved to outputs/dataset_comparison.png")

    print("\n--- KEY RESEARCH FINDINGS ---")
    improvement = (0.7439 - 0.5700) / 0.5700 * 100
    stability = 0.0165 / 0.0074
    print(f"1. Real data produced {improvement:.1f}% better CV F1 than synthetic data.")
    print(f"2. Real data produced {stability:.1f}x more stable results (lower CV std).")
    print("3. Best model type shifted: Logistic Regression (synthetic) → XGBoost (real).")
    print("4. Data quality, not model complexity, is the primary performance driver.")
    print("5. Synthetic datasets are not suitable for real-world risk inference.")

if __name__ == "__main__":
    run_dataset_comparison()
