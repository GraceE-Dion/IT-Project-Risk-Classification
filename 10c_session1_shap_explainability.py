# Step 12c: SHAP Explainability (LinearExplainer for Logistic Regression)
import shap
import matplotlib.pyplot as plt
import numpy as np
 
print("Computing SHAP values for Logistic Regression (Session 1)...")
 
# ── Feature names ─────────────────────────────────────────────────────────────
try:
    feature_names = list(df.drop(columns=['Risk_Level', 'Project_ID']).columns)
except:
    feature_names = [f'Feature_{i}' for i in range(X_test.shape[1])]
 
n_features = len(feature_names)
print(f"Number of features: {n_features}")
 
# ── Convert to numpy if needed ────────────────────────────────────────────────
X_train_arr = X_train if isinstance(X_train, np.ndarray) else np.array(X_train)
X_test_arr  = X_test  if isinstance(X_test,  np.ndarray) else np.array(X_test)
 
# ── Compute SHAP values ───────────────────────────────────────────────────────
masker    = shap.maskers.Independent(X_train_arr)
explainer = shap.LinearExplainer(lr_model, masker=masker)
shap_raw  = explainer.shap_values(X_test_arr)
 
# Diagnose the raw output shape
print(f"Type of shap_raw : {type(shap_raw)}")
if isinstance(shap_raw, list):
    print(f"List length      : {len(shap_raw)}")
    print(f"Each array shape : {shap_raw[0].shape}")
else:
    print(f"Array shape      : {shap_raw.shape}")
 
# ── Normalise to per-feature mean absolute importance ────────────────────────
if isinstance(shap_raw, list):
    # List of (n_samples, n_features) — one per class
    # Average abs values over samples per class, then average over classes
    mean_importance = np.mean(
        [np.abs(sv).mean(axis=0) for sv in shap_raw], axis=0
    )
    shap_for_critical = shap_raw[0]   # class 0 = Critical
 
elif shap_raw.ndim == 3:
    # Shape (n_samples, n_features, n_classes)
    mean_importance   = np.abs(shap_raw).mean(axis=0).mean(axis=1)
    shap_for_critical = shap_raw[:, :, 0]
 
elif shap_raw.ndim == 2 and shap_raw.shape[1] == n_features:
    # Shape (n_samples, n_features) — binary or already reduced
    mean_importance   = np.abs(shap_raw).mean(axis=0)
    shap_for_critical = shap_raw
 
elif shap_raw.ndim == 2 and shap_raw.shape[1] != n_features:
    # Shape (n_samples, n_classes * n_features) — rare flattened multiclass
    n_classes = lr_model.classes_.shape[0]
    reshaped  = shap_raw.reshape(shap_raw.shape[0], n_classes, n_features)
    mean_importance   = np.abs(reshaped).mean(axis=0).mean(axis=0)
    shap_for_critical = reshaped[:, 0, :]
 
else:
    raise ValueError(f"Unexpected shap_raw shape: {shap_raw.shape}")
 
mean_importance = np.array(mean_importance).flatten()
print(f"mean_importance shape after normalisation: {mean_importance.shape}")
assert mean_importance.shape[0] == n_features, \
    f"Still mismatched: {mean_importance.shape[0]} vs {n_features}"
print("Shape check passed.")
 
# ── Plot 1: Bar chart of global feature importance ────────────────────────────
n_top      = min(15, n_features)
sorted_idx = np.argsort(mean_importance)[::-1][:n_top]
 
top_features   = [feature_names[i] for i in sorted_idx][::-1]
top_importance = [float(mean_importance[i]) for i in sorted_idx][::-1]
 
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top_features, top_importance,
        color='#2E75B6', edgecolor='white', linewidth=0.5)
ax.set_xlabel('Mean Absolute SHAP Value', fontsize=11)
ax.set_title('SHAP Feature Importance — Session 1: Logistic Regression\n'
             '(Synthetic Dataset, Top 15 Features)',
             fontsize=13, fontweight='bold', color='#1F4E79')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('shap_importance_session1.png', dpi=300, bbox_inches='tight')
plt.show()
print("SHAP importance chart saved.")
 
# ── Plot 2: Beeswarm for Critical class ───────────────────────────────────────
print("\nGenerating SHAP beeswarm for Critical risk class...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_for_critical,
    X_test_arr,
    feature_names=feature_names,
    max_display=15,
    show=False,
    plot_size=None
)
plt.title("SHAP Beeswarm — Critical Risk Class\n"
          "Session 1: Logistic Regression (Synthetic Dataset)",
          fontsize=13, fontweight='bold', color='#1F4E79')
plt.tight_layout()
plt.savefig('shap_beeswarm_session1.png', dpi=300, bbox_inches='tight')
plt.show()
print("Beeswarm plot saved.")
 
# ── Top 10 summary table ──────────────────────────────────────────────────────
print()
print("=" * 55)
print("SHAP TOP 10 FEATURES BY MEAN ABSOLUTE IMPACT")
print("=" * 55)
for rank, idx in enumerate(sorted_idx[:10], 1):
    print(f"  {rank:2d}. {feature_names[idx]:<40} {float(mean_importance[idx]):.4f}")
 
print()
print("Note: In Session 1 (synthetic data), SHAP values are expected")
print("to be relatively evenly distributed across features, reflecting")
print("the synthetic dataset's near-linear, low-signal feature structure.")
print("This even distribution is itself evidence of the synthetic data's")
print("limited discriminating power — no single feature dominates because")
print("the generative process produced artificially uniform contributions.")
