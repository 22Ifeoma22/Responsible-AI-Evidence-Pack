# scripts/train_and_audit.py
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

  
skl_version = sklearn.__version__

# Fairness
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate

# Optional: SHAP (wrap in try/except so failures don't break the run)
warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------
# Utility paths
# -----------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
GOV_DIR = REPO_ROOT / "governance"

ARTIFACTS_DIR.mkdir(exist_ok=True)
GOV_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


# -----------------------
# Load data
# -----------------------
def load_adult() -> pd.DataFrame:
    """
    Loads Adult dataset.
    - If a local CSV exists in data/, it uses that.
    - Else, fetches from OpenML and saves a small copy locally.
    """
    local_csv = DATA_DIR / "adult.csv"
    if local_csv.exists():
        df = pd.read_csv(local_csv)
        return df

    # Fallback: fetch from OpenML
    from sklearn.datasets import fetch_openml
    adult = fetch_openml("adult", version=2, as_frame=True)
    df = adult.frame.copy()
    # Normalize column names for consistency
    df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]
    # Save a local copy for reproducibility
    df.to_csv(local_csv, index=False)
    return df


# -----------------------
# Main
# -----------------------
def main() -> None:
    print("Loading dataset...")
    df = load_adult()

    # Target and sensitive attributes (adjust if your local CSV uses other names)
    # OpenML Adult uses "class" as target, with values like <=50K, >50K
    target_col = "class" if "class" in df.columns else "income"
    sensitive_col = "sex" if "sex" in df.columns else "Sex"

    if target_col not in df.columns:
        raise ValueError(f"Could not find target column '{target_col}' in dataset columns: {df.columns.tolist()}")
    if sensitive_col not in df.columns:
        raise ValueError(f"Could not find sensitive column '{sensitive_col}' in dataset columns: {df.columns.tolist()}")

    # Drop rows with missing target
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    # Features/labels
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(str)  # make sure it's string for classification

    # Keep a copy of sensitive feature for fairness calc later
    sensitive_series = X[sensitive_col].astype(str)

    # Train/test split
    # You can stratify by the label for balanced classes
    print("Splitting data...")
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, y, sensitive_series, test_size=0.25, random_state=42, stratify=y
    )

    # -----------------------
    # Preprocessing + Model
    # -----------------------
    # Identify categorical vs numeric columns
    cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    # Because some sklearn versions changed the arg name, use sparse=False for widest compatibility
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),

from packaging import version

if version.parse(skl_version) >= version.parse("1.2"):
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
else:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", ohe, cat_cols),
    ],
    remainder="drop",
)
     
        ],
        remainder="drop",
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", clf),
    ])

    # Train
    print("Training model...")
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")

    # -----------------------
    # Fairness audit
    # -----------------------
    print("Running fairness audit...")
    metrics = {
        "selection_rate": selection_rate,
        "tpr": true_positive_rate,
        "fpr": false_positive_rate,
    }
    mf = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=A_test)
    fairness_df = mf.by_group
    print(fairness_df)

    # Save fairness audit CSV
    fairness_path = ARTIFACTS_DIR / "fairness_audit.csv"
    fairness_df.to_csv(fairness_path, index=True)
    print(f"Saved fairness audit -> {fairness_path}")

    # -----------------------
    # Optional: SHAP summary
    # -----------------------
    shap_path = ARTIFACTS_DIR / "shap_summary.png"
    try:
        import shap
        import matplotlib.pyplot as plt

        # Get a small, preprocessed sample to speed up SHAP
        X_small = X_test.sample(n=min(100, len(X_test)), random_state=42)
        X_small_proc = pipe.named_steps["prep"].transform(X_small)

        # For tree models, TreeExplainer is fast
        explainer = shap.TreeExplainer(pipe.named_steps["clf"])
        shap_values = explainer.shap_values(X_small_proc)

        plt.figure(figsize=(10, 6))
        # shap.summary_plot works with numpy arrays; feature names come from the ColumnTransformer
        # Build feature names for OHE output:
        ohe = pipe.named_steps["prep"].named_transformers_["cat"]
        ohe_features = []
        if hasattr(ohe, "get_feature_names_out"):
            ohe_features = ohe.get_feature_names_out(cat_cols).tolist()
        else:
            # fallback if using older sklearn
            ohe_features = [f"{c}_{i}" for c in cat_cols for i in range(1000)]  # not perfect, but avoids crash

        feature_names = num_cols + ohe_features
        shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values,
                          X_small_proc, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(shap_path, dpi=150)
        plt.close()
        print(f"Saved SHAP summary -> {shap_path}")
    except Exception as e:
        print(f"[WARN] Skipping SHAP plot: {e}")

    # -----------------------
    # Governance audit trail
    # -----------------------
    md = []
    md.append("## Audit Run\n")
    md.append(f"- Accuracy: **{acc:.3f}**\n")
    md.append(f"- Fairness audit: `{fairness_path.name}`\n")
    if shap_path.exists():
        md.append(f"- Importance plot: `{shap_path.name}`\n")
    else:
        md.append("- Importance plot: (not generated in this run)\n")

    audit_md_path = GOV_DIR / "audit_trail.md"
    with open(audit_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Saved governance audit -> {audit_md_path}")


if __name__ == "__main__":
    main()
