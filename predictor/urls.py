from django.urls import path

from . import views

app_name = "predictor"

urlpatterns = [
    path("", views.home, name="home"),
    path("about/", views.about, name="about"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("predict/", views.predict, name="predict"),
    path("predict/result/<int:pk>/", views.result, name="result"),
    path("history/", views.history, name="history"),
]
