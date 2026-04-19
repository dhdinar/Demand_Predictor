"""URL routes for demand prediction API."""

from django.urls import path

from . import view


urlpatterns = [
    path("health/", view.health, name="api-health"),
    path("model-info/", view.model_info, name="api-model-info"),
    path("weekly-prediction/", view.weekly_prediction, name="api-weekly-prediction"),
]
