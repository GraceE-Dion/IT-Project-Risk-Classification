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
