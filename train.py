from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "churn.csv"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)
    target = "Churn"
    if target not in df.columns:
        raise ValueError(f"Expected target column '{target}' in {DATA_PATH}")

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2, class_weight="balanced",
            random_state=42
        ),
    }

    results = {}
    best_name, best_pipeline, best_auc = None, None, -1
    for name, model in candidates.items():
        pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        prob = pipeline.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, prob))
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "roc_auc": auc,
            "classification_report": classification_report(
                y_test, pred, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        if auc > best_auc:
            best_name, best_pipeline, best_auc = name, pipeline, auc

    joblib.dump(best_pipeline, ARTIFACTS / "churn_pipeline.joblib")
    summary = {
        "selected_model": best_name,
        "selection_metric": "ROC-AUC on held-out test split",
        "dataset_rows": int(len(df)),
        "test_rows": int(len(X_test)),
        "target_positive_rate": float(y.mean()),
        "results": results,
        "note": "Synthetic demo data; metrics are illustrative, not real-world evidence."
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "selected_model": best_name,
        "results": {
            name: {"accuracy": round(val["accuracy"], 3), "roc_auc": round(val["roc_auc"], 3)}
            for name, val in results.items()
        },
        "saved_to": str(ARTIFACTS / "churn_pipeline.joblib"),
    }, indent=2))

if __name__ == "__main__":
    main()
