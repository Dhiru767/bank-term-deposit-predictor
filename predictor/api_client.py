"""Calls the prediction API (the FastAPI service) from Django.

If the API cannot be reached, the app falls back to the local model so it keeps
working. Start the API from the project root with:
    uvicorn api_service.app:app --port 8001
"""

import json
import logging
import os
import sys

import requests

from . import ml_service

logger = logging.getLogger(__name__)

API_URL = os.getenv("PREDICTION_API_URL", "http://127.0.0.1:8001")
TIMEOUT_SECONDS = float(os.getenv("PREDICTION_API_TIMEOUT", "5"))


def _running_tests():
    """True when started with 'python manage.py test' (tests use the local model)."""
    return len(sys.argv) > 1 and sys.argv[1] == "test"


def _error_text(response):
    """Turn an error response from the API into a readable message."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, list):
        detail = "; ".join(
            ".".join(str(part) for part in item.get("loc", [])[1:]) + ": " + str(item.get("msg"))
            for item in detail
        )
    return detail or f"The prediction service returned status {response.status_code}."


def predict(inputs, use_api=None):
    """Return {"prediction": "yes"/"no", "probability": float}.

    Calls the API first. If the API is down or answers in an unexpected way, the
    local model is used instead. Raises the same errors as ml_service.predict.
    """
    if use_api is None:
        use_api = not _running_tests()
    if not use_api:
        return ml_service.predict(inputs)

    try:
        response = requests.post(
            API_URL.rstrip("/") + "/predict",
            data=json.dumps(inputs, default=float),
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        logger.warning("Prediction API not reachable at %s; using the local model.", API_URL)
        return ml_service.predict(inputs)

    if response.status_code == 200:
        try:
            data = response.json()
            return {
                "prediction": data["prediction"],
                "probability": float(data["probability"]),
                "source": "api",
            }
        except (ValueError, KeyError, TypeError):
            logger.warning("Unexpected answer from the prediction API; using the local model.")
            return ml_service.predict(inputs)
    if response.status_code == 422:
        raise ml_service.PredictionError(_error_text(response))
    if response.status_code == 503:
        raise ml_service.ModelNotFoundError(_error_text(response))

    logger.warning("Prediction API returned status %s; using the local model.", response.status_code)
    return ml_service.predict(inputs)
