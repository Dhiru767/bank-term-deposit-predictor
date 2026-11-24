from unittest.mock import patch

from django.test import SimpleTestCase
from fastapi.testclient import TestClient

from predictor import ml_service

from .app import app

VALID_INPUT = {
    "age": 35,
    "job": "admin.",
    "marital": "single",
    "education": "high.school",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "campaign": 2,
    "previous": 0,
    "poutcome": "nonexistent",
}


class ApiServiceTests(SimpleTestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_valid_prediction(self):
        response = self.client.post("/predict", json=VALID_INPUT)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn(body["prediction"], ("yes", "no"))
        self.assertGreaterEqual(body["probability"], 0.0)
        self.assertLessEqual(body["probability"], 1.0)

    def test_unknown_category_returns_422(self):
        response = self.client.post("/predict", json=dict(VALID_INPUT, job="wizard"))
        self.assertEqual(response.status_code, 422)

    def test_age_out_of_range_returns_422(self):
        response = self.client.post("/predict", json=dict(VALID_INPUT, age=500))
        self.assertEqual(response.status_code, 422)

    def test_missing_model_returns_503(self):
        error = ml_service.ModelNotFoundError("The trained model file was not found.")
        with patch.object(ml_service, "predict", side_effect=error):
            response = self.client.post("/predict", json=VALID_INPUT)
        self.assertEqual(response.status_code, 503)
