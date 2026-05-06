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
