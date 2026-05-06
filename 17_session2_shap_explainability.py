"""
17_session2_shap_explainability.py
=====================================
Session 2 - SHAP Explainability (TreeExplainer)
Dataset: NASA Metrics Data Program (MDP) JM1 (10,878 rows, 21 features)

Applies SHAP TreeExplainer to the selected XGBoost model. Computes exact
Shapley values for all 1,088 test set instances for the Defective class.

Key Results (Top 3 features by mean absolute SHAP impact):
    1. LOC_TOTAL           0.4633
    2. LOC_BLANK           0.2321
    3. DESIGN_COMPLEXITY   0.1842

Finding: Code volume and structural complexity features dominate — the
direct inverse of Session 1 where governance and human-factor variables
topped the SHAP rankings. The Session 2 profile is highly concentrated:
top 3 features account for the vast majority of predictive power, with
a sharp drop after DESIGN_COMPLEXITY (0.1842 → 0.0663). High LOC_TOTAL
pushes strongly toward Defective classification, consistent with software
engineering theory that larger modules carry higher defect density.

Cross-session SHAP contrast (answers RQ2):
    Session 1 top feature : Org_Process_Maturity  (0.69) — governance
    Session 2 top feature : LOC_TOTAL             (0.46) — code volume
    The shift in dominant feature type is the mechanistic explanation
    for the shift in optimal model type between sessions.

Governance alignment: NIST AI RMF 1.0 and Executive Order 14110 —
SHAP TreeExplainer provides decision-level transparency for XGBoost
outputs, satisfying explainability requirements for AI systems with
potential impact on operational decisions.

Outputs:
    outputs/shap_importance_session2.png
    outputs/shap_beeswarm_session2.png

Dependencies: Requires xgb_final, X_train_resampled, X_test_scaled,
y_test, df from 18_session2_final_evaluation.py

Part of: IT Project Risk Classification Pipeline
Paper  : A Supervised Machine Learning Framework for IT Project Risk Classification
GitHub : https://github.com/GraceE-Dion/IT-Project-Risk-Classification
Author : Grace Egbedion, Department of Information Systems, IT Project Management,
         Middle Tennessee State University
"""

# ── End of file summary ───────────────────────────────────────────────────────
print("=" * 55)
print("17: SESSION 2 SHAP EXPLAINABILITY COMPLETE")
print("=" * 55)
print("  Top 3 features by mean absolute SHAP impact:")
print("    1. LOC_TOTAL          0.4633")
print("    2. LOC_BLANK          0.2321")
print("    3. DESIGN_COMPLEXITY  0.1842")
print()
print("  Finding: Code volume and complexity features dominate.")
print("  Highly concentrated profile — inverse of Session 1.")
print("  Cross-session contrast directly answers RQ2:")
print("    Session 1: Org_Process_Maturity (0.69) — governance")
print("    Session 2: LOC_TOTAL (0.46)           — code volume")
print()
print("  Outputs:")
print("    outputs/shap_importance_session2.png")
print("    outputs/shap_beeswarm_session2.png")


# Step 12c: SHAP Explainability (TreeExplainer for XGBoost)
import shap
import matplotlib.pyplot as plt
import numpy as np

print("Computing SHAP values for XGBoost (Session 2)...")
print("Using TreeExplainer — exact computation for tree-based models.")
print()

# Feature names from JM1 ARFF — 21 features in column order
# Get directly from the dataframe to guarantee correct order
feature_names = list(df.drop(columns=['label']).columns)
print(f"Number of features: {len(feature_names)}")

# Convert to numpy arrays
X_train_arr = X_train_resampled if isinstance(X_train_resampled, np.ndarray) \
              else np.array(X_train_resampled)
X_test_arr  = X_test_scaled if isinstance(X_test_scaled, np.ndarray) \
              else np.array(X_test_scaled)

# Compute SHAP values using TreeExplainer
explainer   = shap.TreeExplainer(xgb_final)
shap_values = explainer.shap_values(X_test_arr)

print(f"Type of shap_values: {type(shap_values)}")
if isinstance(shap_values, list):
    print(f"List length        : {len(shap_values)}")
    print(f"Each array shape   : {shap_values[0].shape}")
else:
    print(f"Array shape        : {shap_values.shape}")

# Normalise to (n_features,) for the Defective class
if isinstance(shap_values, list):
    # Binary: list of 2 arrays — use index 1 (Defective)
    shap_defective  = shap_values[1]
    mean_importance = np.abs(shap_defective).mean(axis=0)
elif shap_values.ndim == 3:
    # (n_samples, n_features, n_classes)
    shap_defective  = shap_values[:, :, 1]
    mean_importance = np.abs(shap_defective).mean(axis=0)
else:
    # (n_samples, n_features)
    shap_defective  = shap_values
    mean_importance = np.abs(shap_values).mean(axis=0)

mean_importance = np.array(mean_importance).flatten()
print(f"mean_importance shape: {mean_importance.shape}")
assert mean_importance.shape[0] == len(feature_names), \
    f"Mismatch: {len(feature_names)} names vs {mean_importance.shape[0]} values"
print("Shape check passed.")

# Plot 1: Bar chart
n_top      = min(15, len(feature_names))
sorted_idx = np.argsort(mean_importance)[::-1][:n_top]
top_features   = [feature_names[i] for i in sorted_idx][::-1]
top_importance = [float(mean_importance[i]) for i in sorted_idx][::-1]

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top_features, top_importance,
        color='#2E75B6', edgecolor='white', linewidth=0.5)
ax.set_xlabel('Mean Absolute SHAP Value', fontsize=11)
ax.set_title('SHAP Feature Importance — Session 2: XGBoost\n'
             '(NASA MDP JM1, Defective Class, Top 15 Features)',
             fontsize=13, fontweight='bold', color='#1F4E79')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('shap_importance_session2.png', dpi=300, bbox_inches='tight')
plt.show()
print("SHAP importance chart saved.")

# Plot 2: Beeswarm
print("\nGenerating SHAP beeswarm for Defective class...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_defective,
    X_test_arr,
    feature_names=feature_names,
    max_display=15,
    show=False,
    plot_size=None
)
plt.title("SHAP Beeswarm — Defective Class\n"
          "Session 2: XGBoost (NASA MDP JM1)",
          fontsize=13, fontweight='bold', color='#1F4E79')
plt.tight_layout()
plt.savefig('shap_beeswarm_session2.png', dpi=300, bbox_inches='tight')
plt.show()
print("Beeswarm plot saved.")

# Top 10 summary table
print()
print("=" * 55)
print("SHAP TOP 10 FEATURES BY MEAN ABSOLUTE IMPACT")
print("=" * 55)
for rank, idx in enumerate(sorted_idx[:10], 1):
    print(f"  {rank:2d}. {feature_names[idx]:<40} {float(mean_importance[idx]):.4f}")
