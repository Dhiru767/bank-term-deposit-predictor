import json
import logging

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db import DatabaseError
from django.shortcuts import get_object_or_404, redirect, render

from ml import config as ml_config

from . import api_client, ml_service
from .forms import PredictionForm
from .models import PredictionHistory

logger = logging.getLogger(__name__)


def home(request):
    return render(request, "home.html")


def about(request):
    """Project information. Shows the real test results saved by train_model.py."""
    info = None
    try:
        info = json.loads(ml_config.METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("metrics.json not found or unreadable")
    return render(request, "about.html", {"info": info})


@login_required
def dashboard(request):
    history = request.user.predictions.all()
    context = {
        "recent": history[:5],
        "total": history.count(),
        "yes_count": history.filter(prediction="yes").count(),
        "no_count": history.filter(prediction="no").count(),
    }
    return render(request, "dashboard.html", context)


@login_required
def predict(request):
    if request.method == "POST":
        form = PredictionForm(request.POST)
        if form.is_valid():
            try:
                result = api_client.predict(form.cleaned_data)
            except ml_service.ModelNotFoundError as exc:
                logger.error("Model file missing: %s", exc)
                messages.error(request, str(exc))
            except ml_service.PredictionError as exc:
                messages.error(request, str(exc))
            else:
                try:
                    record = PredictionHistory.objects.create(
                        user=request.user,
                        prediction=result["prediction"],
                        probability=result["probability"],
                        **form.cleaned_data,
                    )
                except DatabaseError:
                    logger.exception("Could not save the prediction")
                    messages.error(
                        request,
                        "The prediction was made but could not be saved. Please try again.",
                    )
                else:
                    return redirect("predictor:result", pk=record.pk)
    else:
        form = PredictionForm()

    return render(request, "predict.html", {"form": form})


@login_required
def result(request, pk):
    # Only the owner can open a result (other users get a 404)
    record = get_object_or_404(PredictionHistory, pk=pk, user=request.user)
    return render(request, "result.html", {"record": record})


@login_required
def history(request):
    paginator = Paginator(request.user.predictions.all(), 10)
    page = paginator.get_page(request.GET.get("page"))
    return render(request, "history.html", {"page": page})

