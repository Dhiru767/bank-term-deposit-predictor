from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from . import ml_service
from .models import PredictionHistory
from .test_ml import VALID_INPUT, make_record

User = get_user_model()


class PredictionViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("dhiru", "d@example.com", "Str0ng-Pass-993")
        self.other = User.objects.create_user("sam", "s@example.com", "Str0ng-Pass-993")
        self.client.login(username="dhiru", password="Str0ng-Pass-993")

    def test_pages_require_login(self):
        self.client.logout()
        for name in ("predictor:predict", "predictor:history", "predictor:dashboard"):
            url = reverse(name)
            response = self.client.get(url)
            self.assertRedirects(response, f"{reverse('accounts:login')}?next={url}")

    def test_form_page_loads(self):
        response = self.client.get(reverse("predictor:predict"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "March to December")

    def test_valid_submission_saves_prediction_and_shows_result(self):
        response = self.client.post(reverse("predictor:predict"), VALID_INPUT)
        record = PredictionHistory.objects.get()
        self.assertRedirects(response, reverse("predictor:result", args=[record.pk]))
        self.assertEqual(record.user, self.user)
        self.assertEqual(record.job, "admin.")
        self.assertEqual(record.age, 35)
        self.assertIn(record.prediction, ("yes", "no"))
        self.assertTrue(0.0 <= record.probability <= 1.0)

    def test_invalid_submission_saves_nothing(self):
        response = self.client.post(reverse("predictor:predict"), dict(VALID_INPUT, age=5))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(PredictionHistory.objects.count(), 0)

    def test_missing_model_shows_message_and_saves_nothing(self):
        error = ml_service.ModelNotFoundError("model missing for test")
        with patch("predictor.ml_service.predict", side_effect=error):
            response = self.client.post(reverse("predictor:predict"), VALID_INPUT)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "model missing for test")
        self.assertEqual(PredictionHistory.objects.count(), 0)

    def test_owner_can_open_result(self):
        record = make_record(self.user)
        response = self.client.get(reverse("predictor:result", args=[record.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Estimated probability")

    def test_other_users_result_is_not_found(self):
        record = make_record(self.other)
        response = self.client.get(reverse("predictor:result", args=[record.pk]))
        self.assertEqual(response.status_code, 404)

    def test_history_shows_only_own_predictions(self):
        make_record(self.user, job="technician")
        make_record(self.other, job="housemaid")
        response = self.client.get(reverse("predictor:history"))
        self.assertContains(response, "technician")
        self.assertNotContains(response, "housemaid")

    def test_history_is_paginated(self):
        for _ in range(12):
            make_record(self.user)
        page1 = self.client.get(reverse("predictor:history"))
        page2 = self.client.get(reverse("predictor:history"), {"page": 2})
        self.assertEqual(len(page1.context["page"].object_list), 10)
        self.assertEqual(len(page2.context["page"].object_list), 2)

    def test_dashboard_counts(self):
        make_record(self.user, prediction="yes", probability=0.8)
        make_record(self.user, prediction="no", probability=0.1)
        make_record(self.user, prediction="no", probability=0.2)
        response = self.client.get(reverse("predictor:dashboard"))
        self.assertEqual(response.context["total"], 3)
        self.assertEqual(response.context["yes_count"], 1)
        self.assertEqual(response.context["no_count"], 2)


class PublicPageTests(TestCase):
    def test_home_and_about_are_public(self):
        for name in ("predictor:home", "predictor:about"):
            self.assertEqual(self.client.get(reverse(name)).status_code, 200)
