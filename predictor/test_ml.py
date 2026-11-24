from pathlib import Path
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.test import TestCase

from ml import config as ml_config

from . import ml_service
from .forms import PredictionForm
from .models import PredictionHistory

User = get_user_model()

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


def make_record(user, **overrides):
    """Save one PredictionHistory row without running the model."""
    data = dict(VALID_INPUT, prediction="no", probability=0.2)
    data.update(overrides)
    return PredictionHistory.objects.create(user=user, **data)


class ModelLoadTests(TestCase):
    """The trained model file must exist and load."""

    def test_model_file_exists(self):
        self.assertTrue(
            ml_config.MODEL_PATH.exists(),
            "model.joblib is missing. Run: python ml/train_model.py",
        )

    def test_model_loads_and_can_predict_probabilities(self):
        model = ml_service.load_model()
        self.assertTrue(hasattr(model, "predict_proba"))

    def test_model_has_both_classes(self):
        model = ml_service.load_model()
        self.assertEqual(list(model.classes_), [0, 1])

    def test_missing_model_file_raises_clear_error(self):
        self.addCleanup(ml_service.load_model.cache_clear)
        ml_service.load_model.cache_clear()
        with patch.object(ml_config, "MODEL_PATH", Path("does-not-exist.joblib")):
            with self.assertRaises(ml_service.ModelNotFoundError):
                ml_service.load_model()


class MlServiceTests(TestCase):
    def test_valid_input_returns_prediction_and_probability(self):
        result = ml_service.predict(VALID_INPUT)
        self.assertIn(result["prediction"], ("yes", "no"))
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)

    def test_label_follows_the_threshold(self):
        result = ml_service.predict(VALID_INPUT)
        expected = "yes" if result["probability"] >= ml_service.THRESHOLD else "no"
        self.assertEqual(result["prediction"], expected)

    def test_numbers_given_as_text_are_accepted(self):
        result = ml_service.predict(dict(VALID_INPUT, age="35", campaign="2"))
        self.assertIn(result["prediction"], ("yes", "no"))

    def test_unknown_category_is_rejected(self):
        with self.assertRaises(ml_service.PredictionError):
            ml_service.predict(dict(VALID_INPUT, job="wizard"))

    def test_missing_value_is_rejected(self):
        data = dict(VALID_INPUT)
        del data["age"]
        with self.assertRaises(ml_service.PredictionError):
            ml_service.predict(data)

    def test_non_numeric_age_is_rejected(self):
        with self.assertRaises(ml_service.PredictionError):
            ml_service.predict(dict(VALID_INPUT, age="abc"))


class PredictionFormTests(TestCase):
    def test_valid_data_is_accepted(self):
        self.assertTrue(PredictionForm(data=VALID_INPUT).is_valid())

    def test_age_below_minimum_is_rejected(self):
        form = PredictionForm(data=dict(VALID_INPUT, age=5))
        self.assertFalse(form.is_valid())
        self.assertIn("age", form.errors)

    def test_age_above_maximum_is_rejected(self):
        self.assertFalse(PredictionForm(data=dict(VALID_INPUT, age=150)).is_valid())

    def test_missing_field_is_rejected(self):
        data = dict(VALID_INPUT)
        del data["job"]
        form = PredictionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn("job", form.errors)

    def test_invalid_choice_is_rejected(self):
        self.assertFalse(PredictionForm(data=dict(VALID_INPUT, month="jan")).is_valid())

    def test_negative_previous_contacts_rejected(self):
        self.assertFalse(PredictionForm(data=dict(VALID_INPUT, previous=-1)).is_valid())

    def test_month_choices_match_config(self):
        months = [value for value, _ in PredictionForm().fields["month"].choices if value]
        self.assertEqual(months, ml_config.CHOICES["month"])


class PredictionHistoryModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("dhiru", "d@example.com", "Str0ng-Pass-993")

    def test_string_and_percent(self):
        record = make_record(self.user, prediction="yes", probability=0.171)
        self.assertEqual(record.probability_percent, 17.1)
        self.assertIn("Likely to subscribe", str(record))

    def test_newest_record_comes_first(self):
        first = make_record(self.user)
        second = make_record(self.user)
        self.assertEqual(list(PredictionHistory.objects.all()), [second, first])

    def test_deleting_user_deletes_their_history(self):
        make_record(self.user)
        self.user.delete()
        self.assertEqual(PredictionHistory.objects.count(), 0)

    def test_probability_must_be_between_0_and_1(self):
        record = make_record(self.user, probability=1.5)
        with self.assertRaises(ValidationError):
            record.full_clean()
