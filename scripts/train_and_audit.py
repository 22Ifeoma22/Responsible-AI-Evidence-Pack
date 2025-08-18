import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate

warnings.filterwarnings("ignore", category=UserWarning)

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------
# 1) Load data
# --------------------------
def load_adult() -> pd.DataFrame:
    """
    Load the Adult dataset. If you already have it in data/adult.csv, load from there.
    Otherwise, we download from OpenML via pandas for a quick demo.
    """
    local_csv = Path("data") / "adult.csv"
    if local_csv.exists():
        df = pd.read_csv(local_csv)
    else:
        # Fallback: try OpenML via fetch from URL hosted mirrors (basic demo).
        url = "https://raw.githubusercontent.com/amueller/ml-workshop-1-of-4/master/datasets/adult.csv"
        df = pd.read_csv(url)
    # Normalize column names a bit (strip, lower)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def prepare_xy(df: pd.DataFrame, target_col: str = "income"):

    # Ensure target exists; common target name is 'income' in adult (<=50K / >50K)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data. Columns: {df.columns.tolist()}")

    y = df[target_col].astype(str)  # keep as string classification target
    X = df.drop(columns=[target_col])

    # Some adult CSVs call sensitive attribute 'sex'
    sensitive_col = "sex" if "sex" in X.columns else None
    sensitive = X[sensitive_col].copy() if sensitive_col else None

    return X, y, sensitive_col, sensitive

# --------------------------
# 2) Build pipeline with OHE
# --------------------------
def build_pipeline(X: pd.DataFrame):
    # Identify categorical vs numeric columns
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # OneHotEncoder param changed in sklearn ≥1.2: use sparse_output
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # for older sklearn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preproc = ColumnTransformer(
        transformers=[
            ("cat", ohe, cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("prep", preproc), ("clf", clf)])
    return pipe

# --------------------------
# 3) Fairness metrics helper
# --------------------------
def run_fairness(y_true, y_pred, sensitive: pd.Series, out_path: Path):
    metrics = {
        "selection_rate": selection_rate,
        "tpr": true_positive_rate,
        "fpr": false_positive_rate,
    }
    mf = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive)
    mf_df = mf.by_group
    mf_df.to_csv(out_path, index=True)

# --------------------------
# 4) Optional SHAP
# --------------------------
def try_shap_summary(pipe: Pipeline, X_train: pd.DataFrame, out_path: Path):
    # SHAP is optional; try it, but don’t fail the pipeline if it's missing
    try:
        import shap
        explainer = shap.TreeExplainer(pipe.named_steps["clf"])
        # Transform training data through preprocessor
        Xt = pipe.named_steps["prep"].transform(X_train)
        shap_values = explainer.shap_values(Xt)
        shap.summary_plot(shap_values, Xt, show=False)
        import matplotlib.pyplot as plt
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return True
    except Exception as e:
        print(f"[INFO] SHAP not available or failed: {e}")
        return False

# --------------------------
# 5) Main
# --------------------------
def main():
    print("📥 Loading dataset...")
    df = load_adult()
    X, y, sensitive_col, sensitive = prepare_xy(df, target_col="income")
    print(f"Data shape: {df.shape}. Target positive rate: {(y == '>50K').mean():.3f} (if 'income').")

    print("🔀 Splitting...")
    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.2, random_state=42, stratify=stratify
    )

    print("🏗️ Building pipeline...")
    pipe = build_pipeline(X_train)

    print("🏃 Training model...")
    pipe.fit(X_train, y_train)

    print("🔎 Predicting...")
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.3f}")

    # Save metrics JSON
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps({"accuracy": acc}, indent=2))

    # Save model (optional)
    joblib.dump(pipe, ARTIFACTS_DIR / "model.joblib")

    # Fairness audit (if we had a sensitive column)
    if sensitive_col is not None:
        print(f"⚖️ Running fairness audit by '{sensitive_col}'...")
        out_csv = ARTIFACTS_DIR / "fairness_audit.csv"
        run_fairness(y_test, y_pred, s_test, out_csv)
        print(f"📄 Saved fairness audit -> {out_csv}")
    else:
        print("⚠️ No sensitive column found; skipping fairness audit.")

    # Explainability attempt
    print("🧠 Trying SHAP summary...")
    shap_ok = try_shap_summary(pipe, X_train, ARTIFACTS_DIR / "shap_summary.png")
    if shap_ok:
        print("🖼️ Saved SHAP summary -> artifacts/shap_summary.png")
    else:
        # Fallback: simple permutation importance
        try:
            from sklearn.inspection import permutation_importance
            import matplotlib.pyplot as plt

            Xt = pipe.named_steps["prep"].transform(X_test)
            result = permutation_importance(pipe.named_steps["clf"], Xt, y_test, n_repeats=3, random_state=42, n_jobs=-1)
            imp = result.importances_mean
            plt.figure(figsize=(6,4))
            plt.bar(range(len(imp)), imp)
            plt.title("Permutation Importance (proxy)")
            plt.tight_layout()
            plt.savefig(ARTIFACTS_DIR / "feature_importance.png", dpi=150)
            plt.close()
            print("🖼️ Saved permutation importance -> artifacts/feature_importance.png")
        except Exception as e:
            print(f"[INFO] Permutation importance also failed: {e}")

    print("✅ Done.")

if __name__ == "__main__":
    main()
