from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class PredictionHistory(models.Model):
    """One saved prediction: who asked, what they entered, what the model said."""

    RESULT_CHOICES = [
        ("yes", "Likely to subscribe"),
        ("no", "Unlikely to subscribe"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="predictions",
    )

    # Input values (the same 11 features the model was trained on)
    age = models.PositiveSmallIntegerField()
    campaign = models.PositiveSmallIntegerField()
    previous = models.PositiveSmallIntegerField()
    job = models.CharField(max_length=30)
    marital = models.CharField(max_length=30)
    education = models.CharField(max_length=30)
    housing = models.CharField(max_length=30)
    loan = models.CharField(max_length=30)
    contact = models.CharField(max_length=30)
    month = models.CharField(max_length=30)
    poutcome = models.CharField(max_length=30)

    # Model output
    prediction = models.CharField(max_length=3, choices=RESULT_CHOICES)
    probability = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Model's probability (0 to 1) that the client subscribes.",
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "prediction"
        verbose_name_plural = "prediction history"

    def __str__(self):
        return f"{self.user} - {self.get_prediction_display()} ({self.probability_percent}%)"

    @property
    def probability_percent(self):
        """Probability as a percentage, e.g. 17.1"""
        return round(self.probability * 100, 1)
