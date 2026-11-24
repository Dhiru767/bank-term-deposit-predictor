"""Loads the saved model once and makes predictions for the web app.

The model is NOT trained here. Train it first with:
    python ml/train_model.py
"""

from functools import lru_cache

import joblib
import pandas as pd

from ml import config as cfg

# Probability at or above this value is reported as "yes"
THRESHOLD = 0.5


class ModelNotFoundError(Exception):
    """The trained model file does not exist yet."""


class PredictionError(Exception):
    """The model could not make a prediction from the given input."""


@lru_cache(maxsize=1)
def load_model():
    """Load the model from disk the first time, then reuse it."""
    if not cfg.MODEL_PATH.exists():
        raise ModelNotFoundError(
            "The trained model file was not found. "
            "Run: python ml/train_model.py"
        )
    return joblib.load(cfg.MODEL_PATH)


def predict(inputs):
    """Take a dict of the 11 features and return the model's answer.

    Returns: {"prediction": "yes" or "no", "probability": float between 0 and 1}
    """
    # 1. Check that nothing is missing
    missing = [name for name in cfg.FEATURES if inputs.get(name) in (None, "")]
    if missing:
        raise PredictionError("Missing values for: " + ", ".join(missing))

    # 2. Convert numbers and check categories
    values = {}
    for name in cfg.NUMERIC_FEATURES:
        try:
            values[name] = float(inputs[name])
        except (TypeError, ValueError):
            raise PredictionError(f"{name} must be a number.")
    for name in cfg.CATEGORICAL_FEATURES:
        if inputs[name] not in cfg.CHOICES[name]:
            raise PredictionError(f"'{inputs[name]}' is not a valid value for {name}.")
        values[name] = inputs[name]

    # 3. Ask the model (columns in the same order as training)
    model = load_model()
    row = pd.DataFrame([values], columns=cfg.FEATURES)
    try:
        class_index = list(model.classes_).index(1)
        probability = float(model.predict_proba(row)[0][class_index])
    except Exception as exc:
        raise PredictionError(f"The model could not process this input: {exc}") from exc

    return {
        "prediction": "yes" if probability >= THRESHOLD else "no",
        "probability": probability,
    }
