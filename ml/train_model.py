"""Train the term-deposit model and save it with joblib.

Run from the project root:
    python ml/train_model.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make "from ml import config" work when running "python ml/train_model.py"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml import config as cfg


def load_and_clean():
    """Load the CSV and return a cleaned DataFrame."""
    if not cfg.DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {cfg.DATA_PATH}. "
            "Put bank-additional-full.csv inside the data/ folder."
        )

    df = pd.read_csv(cfg.DATA_PATH, sep=";")
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns")

    # Remove rows that are identical in ALL original columns, so the same
    # record cannot end up in both the training set and the test set
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(df)} fully duplicated rows")

    # Keep only the columns we need (this also drops "duration", which
    # is only known after the call, and would leak the answer)
    df = df[cfg.FEATURES + [cfg.TARGET]].copy()

    # Tidy text columns
    for col in cfg.CATEGORICAL_FEATURES + [cfg.TARGET]:
        df[col] = df[col].astype(str).str.strip()

    # Target: yes -> 1, no -> 0
    df[cfg.TARGET] = df[cfg.TARGET].map({"yes": 1, "no": 0})
    if df[cfg.TARGET].isna().any():
        raise ValueError("Target column contains values other than yes/no")

    # Missing values: this dataset stores "unknown" as text, which we keep as
    # its own category. Any real blanks are filled inside the pipeline.
    missing = df.isna().sum()
    missing = missing[missing > 0]
    print("Missing values (real blanks):", "none" if missing.empty else "")
    if not missing.empty:
        print(missing.to_string())

    # Warn if the data has category values the web form does not offer
    for col, allowed in cfg.CHOICES.items():
        extra = set(df[col].unique()) - set(allowed)
        if extra:
            print(f"WARNING: {col} has values not in config.CHOICES: {sorted(extra)}")

    return df


def build_pipeline():
    """Preprocessing + Random Forest in one object."""
    numeric_steps = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_steps, cfg.NUMERIC_FEATURES),
        ("cat", categorical_steps, cfg.CATEGORICAL_FEATURES),
    ])

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",   # only ~11% of clients say yes
        random_state=cfg.RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([("preprocess", preprocessor), ("model", model)])


def main():
    df = load_and_clean()

    X = df[cfg.FEATURES]
    y = df[cfg.TARGET]

    # Stratified split keeps the same yes/no ratio in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.TEST_SIZE,
        stratify=y,
        random_state=cfg.RANDOM_STATE,
    )
    print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    print(f"Share of 'yes' in train: {y_train.mean():.1%} | in test: {y_test.mean():.1%}")

    pipeline = build_pipeline()
    print("Training the model...")
    pipeline.fit(X_train, y_train)

    # Evaluate on data the model has never seen
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }

    print("\n=== Test-set results ===")
    for name, value in metrics.items():
        print(f"{name:>10}: {value:.3f}")
    print("\nConfusion matrix (rows = actual no/yes, columns = predicted no/yes):")
    print(confusion_matrix(y_test, pred))
    print("\n" + classification_report(y_test, pred, target_names=["no", "yes"]))

    # Save the whole pipeline (preprocessing + model) in one file
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, cfg.MODEL_PATH, compress=3)

    info = {
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": cfg.FEATURES,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    cfg.METRICS_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"Model saved to   {cfg.MODEL_PATH}")
    print(f"Metrics saved to {cfg.METRICS_PATH}")


if __name__ == "__main__":
    main()
    