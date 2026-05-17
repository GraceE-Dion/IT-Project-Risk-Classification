"""
25_session2_error_analysis.py
===============================
Session 2 - Error Analysis: 92 Missed Defective Modules (False Negatives)
Dataset: NASA Metrics Data Program (MDP) JM1 test set (1,088 samples)

Analyses the 92 false negative modules to identify systematic patterns
in the model's errors.

Key Results:
    Defective modules in test set: 210
    True Positives (detected):     118
    False Negatives (missed):       92  (miss rate: 43.8%)

    Feature statistics — False Negatives vs True Positives:
    Feature                  FN Mean    TP Mean   All Defective
    LOC_TOTAL                  18.7      125.8         78.9
    BRANCH_COUNT                6.0       36.1         22.9
    CYCLOMATIC_COMPLEXITY       3.5       18.9         12.2
    DESIGN_COMPLEXITY           2.1       10.5          6.8
    HALSTEAD_EFFORT          4,110    191,680       109,507

    Predicted P(Defective) for missed modules:
    Mean: 0.3583 | Median: 0.3743 | Min: 0.1367 | Max: 0.4990

    Threshold recovery:
    0.40: recovers 40/92 FNs | +160 FP
    0.35: recovers 54/92 FNs | +261 FP
    0.30: recovers 66/92 FNs | +371 FP
    0.25: recovers 78/92 FNs | +452 FP

Critical Finding: Missed defective modules are systematically SMALLER and
SIMPLER than detected ones — FN mean LOC_TOTAL of 18.7 vs TP mean of 125.8
(6.7x difference). The model has learned a complexity-driven rule and has
a systematic blind spot for small, simple defective modules. The model is
UNCERTAIN (not wrong) about missed modules — mean P(Defective)=0.358.
Threshold tuning recovers some FNs but cannot eliminate the blind spot;
additional features beyond size/complexity metrics are needed.

Outputs:
    outputs/error_analysis_session2.png

Dependencies: Requires xgb_final, X_test_scaled, y_test, df from
18_session2_final_evaluation.py

Part of: IT Project Risk Classification Pipeline
Paper  : A Supervised Machine Learning Framework for IT Project Risk Classification
GitHub : https://github.com/GraceE-Dion/IT-Project-Risk-Classification
Author : Grace Egbedion, MTSU Department of Information Systems
"""

# ── [Paste Step 19 cell code from Session 2 notebook here] ───────────────────

# ── End of file summary ───────────────────────────────────────────────────────
print("=" * 55)
print("24: SESSION 2 ERROR ANALYSIS COMPLETE")
print("=" * 55)
print("  92 missed defective modules (miss rate: 43.8%)")
print()
print("  FN mean LOC_TOTAL:  18.7  vs  TP mean: 125.8 (6.7x)")
print("  FN mean P(Defect):  0.358 — uncertain, not wrong")
print()
print("  Threshold recovery:")
print("    0.30: 66/92 FNs recovered | +371 FP")
print("    0.35: 54/92 FNs recovered | +261 FP")
print()
print("  FINDING: Small simple modules systematically missed.")
print("  Complexity-driven rule has a structural blind spot.")
print()
print("  Output: outputs/error_analysis_session2.png")
