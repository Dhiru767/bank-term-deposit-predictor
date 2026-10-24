import json
from pathlib import Path
C = []
def _cell(kind, text):
    lines = text.strip().split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c
def md(s): C.append(_cell("markdown", s))
def code(s): C.append(_cell("code", s))

md("""
# Term Deposit Subscription: Model Training

This notebook walks through the training of the model used by the web app. It reuses the same cleaning and pipeline code as `ml/train_model.py` (imported, not copied), so the results match the script exactly.

**Steps:** 1) load and clean data, 2) explore, 3) split, 4) baseline, 5) train, 6) evaluate, 7) feature importance, 8) load the saved model the way the app does.
""")

code("""
import sys
from pathlib import Path

# Find the project root whether the notebook runs from notebooks/ or the root
ROOT = Path.cwd()
if not (ROOT / "ml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve, classification_report,
)
from sklearn.model_selection import train_test_split

from ml import config as cfg
from ml.train_model import load_and_clean, build_pipeline

pd.set_option("display.max_columns", 30)
print("Project root:", ROOT)
""")

md("## 1. Load and clean the data\nDuplicates are removed, `duration` is dropped (it is only known after the call, so it would leak the answer), and the target is mapped yes -> 1, no -> 0.")
code("""
df = load_and_clean()
df.head()
""")
code("""
print(df.shape)
df.dtypes
""")

md("## 2. Exploratory analysis")
code("""
counts = df[cfg.TARGET].value_counts().sort_index()
share = df[cfg.TARGET].mean()
print(f"Share of clients who subscribed: {share:.1%}")

fig, ax = plt.subplots(figsize=(4, 3.5))
ax.bar(["no", "yes"], counts.values, color=["#9aa5b1", "#2b7a78"])
ax.set_title("Class balance")
ax.set_ylabel("Clients")
for i, v in enumerate(counts.values):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom")
plt.show()
""")
code("""
def rate_plot(ax, series, title, rotate=0):
    rate = df.groupby(series)[cfg.TARGET].mean().sort_values(ascending=False) * 100
    ax.bar(rate.index.astype(str), rate.values, color="#2b7a78")
    ax.axhline(df[cfg.TARGET].mean() * 100, color="#c0392b", ls="--", lw=1, label="overall")
    ax.set_title(title)
    ax.set_ylabel("Subscription rate (%)")
    ax.tick_params(axis="x", rotation=rotate)

age_group = pd.cut(df["age"], [0, 25, 35, 45, 55, 65, 120],
                   labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"])

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
rate_plot(axes[0, 0], df["month"], "By month of contact")
rate_plot(axes[0, 1], df["job"], "By job", rotate=45)
rate_plot(axes[1, 0], df["poutcome"], "By outcome of previous campaign")
rate_plot(axes[1, 1], age_group, "By age group")
axes[0, 0].legend()
plt.tight_layout()
plt.show()
""")
md("**Reading the plots:** only about 11% of clients subscribe, so the classes are imbalanced. A previous successful campaign is by far the strongest signal, and the rate also varies a lot by month and age group.")

md("## 3. Train/test split\nStratified, so the yes/no ratio is the same in both sets. Same size and random seed as the script.")
code("""
X = df[cfg.FEATURES]
y = df[cfg.TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=cfg.TEST_SIZE, stratify=y, random_state=cfg.RANDOM_STATE
)
print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
print(f"'yes' share  train: {y_train.mean():.1%} | test: {y_test.mean():.1%}")
""")

md("## 4. Baseline: always predict \"no\"\nThis shows why accuracy alone is misleading on imbalanced data.")
code("""
base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
bp = base.predict(X_test)
print(f"Baseline accuracy: {accuracy_score(y_test, bp):.3f}")
print(f"Baseline recall:   {recall_score(y_test, bp):.3f}  (finds no subscribers at all)")
""")

md("## 5. Train the model\nA scikit-learn `Pipeline`: imputation + one-hot encoding + Random Forest (300 trees, `min_samples_leaf=5`, `class_weight=\"balanced\"`).")
code("""
pipeline = build_pipeline()
pipeline.fit(X_train, y_train)
pipeline
""")

md("## 6. Evaluate on the test set")
code("""
pred = pipeline.predict(X_test)
proba = pipeline.predict_proba(X_test)[:, 1]

results = pd.DataFrame({
    "Random Forest": {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    },
    "Always 'no'": {
        "accuracy": accuracy_score(y_test, bp),
        "precision": precision_score(y_test, bp, zero_division=0),
        "recall": recall_score(y_test, bp),
        "f1": f1_score(y_test, bp),
        "roc_auc": 0.5,
    },
}).round(3)
results
""")
code("""
print(classification_report(y_test, pred, target_names=["no", "yes"]))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["no", "yes"]).plot(
    ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion matrix")

fpr, tpr, _ = roc_curve(y_test, proba)
axes[1].plot(fpr, tpr, color="#2b7a78", label=f"Random Forest (AUC = {roc_auc_score(y_test, proba):.3f})")
axes[1].plot([0, 1], [0, 1], "--", color="grey", label="Chance")
axes[1].set_xlabel("False positive rate")
axes[1].set_ylabel("True positive rate")
axes[1].set_title("ROC curve")
axes[1].legend(loc="lower right")
plt.tight_layout()
plt.show()
""")
md("**Interpretation:** the baseline reaches high accuracy but finds zero subscribers. The forest gives up some accuracy to find more of them (higher recall), which is the useful trade-off for a marketing campaign. Precision is modest because subscribers are rare.")

md("## 7. Feature importance")
code("""
names = pipeline.named_steps["preprocess"].get_feature_names_out()
importances = pipeline.named_steps["model"].feature_importances_
top = pd.Series(importances, index=names).sort_values(ascending=False).head(15)
top.index = [n.split("__", 1)[-1] for n in top.index]

fig, ax = plt.subplots(figsize=(8, 5))
top[::-1].plot(kind="barh", ax=ax, color="#2b7a78")
ax.set_title("Top 15 features")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.show()
""")

md("## 8. Load the saved model, as the web app does\nThe app loads `model.joblib` created by `python ml/train_model.py`. This cell loads that file and predicts for one client from the test set.")
code("""
if not cfg.MODEL_PATH.exists():
    print("model.joblib not found. Run `python ml/train_model.py` from the project root first.")
else:
    saved = joblib.load(cfg.MODEL_PATH)
    client = X_test.iloc[[0]]
    print(client.T)
    p = saved.predict_proba(client)[0, 1]
    print(f"\\nPredicted: {'yes' if p >= 0.5 else 'no'} | probability of yes: {p:.3f}")
    print(f"Actual outcome: {'yes' if y_test.iloc[0] == 1 else 'no'}")
""")

nb = {
    "cells": C,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell{i:02d}"
out = Path("notebooks")
out.mkdir(exist_ok=True)
(out / "train_model.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Created notebooks/train_model.ipynb")
