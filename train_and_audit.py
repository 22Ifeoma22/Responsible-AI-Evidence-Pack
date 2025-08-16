import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import shap
import matplotlib.pyplot as plt
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate

# === 1. Load dataset (Adult dataset from OpenML) ===
print("🔍 Loading dataset...")
adult = fetch_openml(data_id=1590, as_frame=True)  # Adult Income dataset
X = adult.data
y = (adult.target == ">50K").astype(int)  # Binary: 1 if income > 50K

# Sensitive feature: sex
sensitive = X["sex"]

# === 2. Train-test split ===
print("📊 Splitting data...")
X_train, X_test, y_train, y_test, sex_train, sex_test = train_test_split(
    X, y, sensitive, test_size=0.3, random_state=42
)

# Drop sensitive attribute from training features
X_train = X_train.drop(columns=["sex"])
X_test = X_test.drop(columns=["sex"])

# === 3. Train model ===
print("🤖 Training model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"✅ Model accuracy: {acc:.3f}")

# === 4. Fairness Audit (Fairlearn) ===
print("📊 Running fairness audit...")
metrics = {
    "accuracy": accuracy_score,
    "selection_rate": selection_rate,
    "TPR": true_positive_rate,
    "FPR": false_positive_rate,
}

frame = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=sex_test)
audit_results = frame.by_group

# Save fairness audit results
audit_results.to_csv("artifacts/fairness_audit.csv")
print("📂 Saved fairness audit to artifacts/fairness_audit.csv")

# === 5. Explainability (SHAP) ===
print("🔍 Running SHAP explainability...")
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test[:100])  # subset for speed

plt.figure()
shap.summary_plot(shap_values[1], X_test[:100], show=False)
plt.savefig("artifacts/shap_summary.png")
print("📂 Saved SHAP summary to artifacts/shap_summary.png")

# === 6. Governance Audit Trail ===
with open("governance/audit_trail.md", "w") as f:
    f.write("# Governance Audit Trail\n")
    f.write(f"- Model accuracy: {acc:.3f}\n")
    f.write(f"- Fairness audit saved at artifacts/fairness_audit.csv\n")
    f.write(f"- SHAP summary saved at artifacts/shap_summary.png\n")

print("✅ Audit complete! Artifacts and governance trail updated.")
