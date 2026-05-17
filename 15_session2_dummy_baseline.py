"""
11_session2_dummy_baseline.py
==============================
Session 2 - Dummy Classifier Baseline (Performance Floor)
Dataset: NASA Metrics Data Program (MDP) JM1 (10,878 rows, 21 features)

Establishes the majority-class prediction floor before any ML modelling.
The dummy classifier predicts Clean for every instance, achieving zero
recall on the Defective class. This provides the baseline against which
the ML model's lift is measured.

Key Results:
    Dummy Validation F1 (macro) : 0.4465
    Dummy Test F1       (macro) : 0.4466
    ML lift over dummy  (val)   : 0.1767
    ML lift over dummy  (test)  : 0.1616

Note: The higher dummy F1 compared to Session 1 (0.1289) reflects the
binary 80/20 class structure — predicting Clean correctly 80% of the time
inflates the majority class contribution to macro F1. The smaller ML lift
over dummy compared to Session 1 (0.46) reflects genuine minority class
detection difficulty under real-world imbalance, not a weaker model.
The XGBoost model detects 118 of 210 defective modules that the dummy
classifier misses entirely.

Dependencies: Requires X_train_scaled, y_train, X_val_scaled, y_val,
X_test_scaled, y_test from 07_train_val_test_split.py

Part of: IT Project Risk Classification Pipeline
Paper  : A Supervised Machine Learning Framework for IT Project Risk Classification
GitHub : https://github.com/GraceE-Dion/IT-Project-Risk-Classification
Author : Grace Egbedion, Department of Information Systems, IT Project Management,
         Middle Tennessee State University
"""

# ── End of file summary ───────────────────────────────────────────────────────
print("=" * 55)
print("11: SESSION 2 DUMMY CLASSIFIER BASELINE COMPLETE")
print("=" * 55)
print(f"  Dummy Validation F1 (macro) : 0.4465")
print(f"  Dummy Test F1       (macro) : 0.4466")
print()
print(f"  ML lift over dummy (val)    : 0.1767")
print(f"  ML lift over dummy (test)   : 0.1616")
print()
print("  This is the performance floor — the minimum achievable")
print("  without any learning. All ML models must exceed this.")



# Step 7b: Dummy Classifier Baseline (Performance Floor)
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)

dummy_val_preds = dummy.predict(X_val)
dummy_test_preds = dummy.predict(X_test)

dummy_val_f1  = f1_score(y_val,  dummy_val_preds,  average='macro')
dummy_test_f1 = f1_score(y_test, dummy_test_preds, average='macro')

print("=" * 50)
print("DUMMY CLASSIFIER BASELINE (Most Frequent Class)")
print("=" * 50)
print(f"Validation F1 (macro): {dummy_val_f1:.4f}")
print(f"Test F1       (macro): {dummy_test_f1:.4f}")
print()
print("Interpretation:")
print(f"  ML model lift over dummy (val):  {0.5890 - dummy_val_f1:.4f}")
print(f"  ML model lift over dummy (test): {0.5872 - dummy_test_f1:.4f}")
print()
print("Note: The dummy classifier predicts the majority class for every")
print("instance. This is the floor — the minimum performance achievable")
print("without any learning. The lift above this floor represents the")
print("actual value the ML model provides.")
