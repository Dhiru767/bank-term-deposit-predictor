from django.contrib import admin

from .models import PredictionHistory


@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ("user", "prediction", "probability_percent", "job", "age", "created_at")
    list_filter = ("prediction", "month", "job")
    search_fields = ("user__username",)
    readonly_fields = ("created_at",)
