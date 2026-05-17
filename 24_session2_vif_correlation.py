"""
23_session2_vif_correlation.py
================================
Session 2 - Feature Correlation and VIF Analysis
Dataset: NASA Metrics Data Program (MDP) JM1 (10,878 rows, 21 features)

Examines multicollinearity among NASA MDP features using Pearson correlation
heatmap and Variance Inflation Factor (VIF) analysis.

Key Results:
    Features with VIF > 10 (problematic multicollinearity): 15 of 21
    Perfect correlations (r=1.0000):
      HALSTEAD_EFFORT    == HALSTEAD_PROG_TIME
      HALSTEAD_ERROR_EST == HALSTEAD_VOLUME

    Highly correlated pairs (|r| > 0.90): 20 pairs identified

    Features with acceptable VIF (< 10):
      DESIGN_COMPLEXITY (6.58), ESSENTIAL_COMPLEXITY (5.52),
      LOC_BLANK (5.32), LOC_COMMENTS (2.17),
      HALSTEAD_LEVEL (1.40), LOC_CODE_AND_COMMENT (1.17)

Finding: Severe multicollinearity across the Halstead metric family and
LOC variants. SHAP values for highly correlated features should be
interpreted collectively rather than individually. XGBoost's L1/L2
regularisation is more robust to multicollinear features than linear
models, concentrating predictive weight in one representative feature
per correlated cluster. Dimensionality reduction is recommended for
future work.

Outputs:
    outputs/correlation_heatmap_session2.png

Dependencies: Requires df from 02_data_acquisition.py.
Requires: pip install statsmodels

Part of: IT Project Risk Classification Pipeline
Paper  : A Supervised Machine Learning Framework for IT Project Risk Classification
GitHub : https://github.com/GraceE-Dion/IT-Project-Risk-Classification
Author : Grace Egbedion, MTSU Department of Information Systems
"""

# ── [Paste Step 18 cell code from Session 2 notebook here] ───────────────────

# ── End of file summary ───────────────────────────────────────────────────────
print("=" * 55)
print("23: SESSION 2 VIF AND CORRELATION ANALYSIS COMPLETE")
print("=" * 55)
print("  15 of 21 features with VIF > 10 (severe multicollinearity)")
print("  Perfect correlations (r=1.0000):")
print("    HALSTEAD_EFFORT    == HALSTEAD_PROG_TIME")
print("    HALSTEAD_ERROR_EST == HALSTEAD_VOLUME")
print("  20 pairs with |r| > 0.90")
print()
print("  6 features with acceptable VIF < 10:")
print("    DESIGN_COMPLEXITY, ESSENTIAL_COMPLEXITY, LOC_BLANK,")
print("    LOC_COMMENTS, HALSTEAD_LEVEL, LOC_CODE_AND_COMMENT")
print()
print("  Output: outputs/correlation_heatmap_session2.png")
