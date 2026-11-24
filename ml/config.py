"""Single source of truth for the ML part of the project.

Both the training script and the Django app import from here, so the
features used for training can never drift away from the features
used by the web form.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Files
DATA_PATH = BASE_DIR / "data" / "bank-additional-full.csv"
MODEL_DIR = BASE_DIR / "ml" / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# Target: did the client subscribe to a term deposit? (yes / no)
TARGET = "y"

# Input features (what the web form will ask for)
NUMERIC_FEATURES = ["age", "campaign", "previous"]
CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Allowed values for each categorical feature (used by the form later)
CHOICES = {
    "job": [
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown",
    ],
    "marital": ["divorced", "married", "single", "unknown"],
    "education": [
        "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
        "professional.course", "university.degree", "unknown",
    ],
    "housing": ["no", "yes", "unknown"],
    "loan": ["no", "yes", "unknown"],
    "contact": ["cellular", "telephone"],
    "month": ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "poutcome": ["failure", "nonexistent", "success"],
}

# Training settings
TEST_SIZE = 0.2
RANDOM_STATE = 42