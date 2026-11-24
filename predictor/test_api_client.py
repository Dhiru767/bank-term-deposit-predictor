from unittest.mock import Mock, patch

import requests
from django.test import SimpleTestCase

from . import api_client, ml_service

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


def fake_response(status, body):
    response = Mock()
    response.status_code = status
    response.json.return_value = body
    return response


class ApiClientTests(SimpleTestCase):
    def test_successful_api_answer_is_returned(self):
        answer = fake_response(200, {"prediction": "yes", "probability": 0.7})
        with patch.object(api_client.requests, "post", return_value=answer) as post:
            result = api_client.predict(VALID_INPUT, use_api=True)
        self.assertEqual(result["prediction"], "yes")
        self.assertEqual(result["probability"], 0.7)
        self.assertEqual(result["source"], "api")
        self.assertTrue(post.call_args[0][0].endswith("/predict"))

    def test_api_422_becomes_prediction_error(self):
        answer = fake_response(422, {"detail": "'wizard' is not a valid value for job."})
        with patch.object(api_client.requests, "post", return_value=answer):
            with self.assertRaises(ml_service.PredictionError):
                api_client.predict(VALID_INPUT, use_api=True)

    def test_api_503_becomes_model_not_found_error(self):
        answer = fake_response(503, {"detail": "The trained model file was not found."})
        with patch.object(api_client.requests, "post", return_value=answer):
            with self.assertRaises(ml_service.ModelNotFoundError):
                api_client.predict(VALID_INPUT, use_api=True)

    def test_unreachable_api_falls_back_to_local_model(self):
        with patch.object(api_client.requests, "post", side_effect=requests.ConnectionError):
            result = api_client.predict(VALID_INPUT, use_api=True)
        self.assertIn(result["prediction"], ("yes", "no"))

    def test_use_api_false_never_calls_the_api(self):
        with patch.object(api_client.requests, "post") as post:
            result = api_client.predict(VALID_INPUT, use_api=False)
        post.assert_not_called()
        self.assertIn(result["prediction"], ("yes", "no"))

    def test_tests_use_the_local_model_by_default(self):
        with patch.object(api_client.requests, "post") as post:
            api_client.predict(VALID_INPUT)
        post.assert_not_called()
