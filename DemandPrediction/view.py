"""HTTP views for demand prediction API endpoints."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .api_service import get_or_train_model, predict_weekly_rows


def _parse_json(request: HttpRequest) -> Dict[str, Any]:
    if not request.body:
        return {}
    try:
        return json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON body") from exc


@require_GET
def health(request: HttpRequest) -> JsonResponse:
    return JsonResponse(
        {
            "status": "ok",
            "service": "demand-prediction-api",
        }
    )


@csrf_exempt
@require_POST
def model_info(request: HttpRequest) -> JsonResponse:
    try:
        payload = _parse_json(request)
        cached = get_or_train_model(
            csv_path=payload.get("csv_path"),
            learning_rate=float(payload.get("learning_rate", 0.01)),
            epochs=int(payload.get("epochs", 1000)),
            use_rolling_feature=bool(payload.get("use_rolling_feature", False)),
            retrain=bool(payload.get("retrain", False)),
        )
    except ValueError as exc:
        return JsonResponse({"status": "error", "message": str(exc)}, status=400)

    return JsonResponse(
        {
            "status": "ok",
            "model": {
                "trained_at": cached.trained_at,
                "csv_path": cached.csv_path,
                "feature_names": cached.model_state.feature_names,
                "learning_rate": cached.learning_rate,
                "epochs": cached.epochs,
                "use_rolling_feature": cached.use_rolling_feature,
                "final_loss": round(cached.final_loss, 6),
            },
        }
    )


@csrf_exempt
@require_POST
def weekly_prediction(request: HttpRequest) -> JsonResponse:
    """Predict weekly units_sold for one or more products."""
    try:
        payload = _parse_json(request)
        cached = get_or_train_model(
            csv_path=payload.get("csv_path"),
            learning_rate=float(payload.get("learning_rate", 0.01)),
            epochs=int(payload.get("epochs", 1000)),
            use_rolling_feature=bool(payload.get("use_rolling_feature", False)),
            retrain=bool(payload.get("retrain", False)),
        )

        weekly_data = payload.get("weekly_data") or payload.get("data")
        if not isinstance(weekly_data, list):
            raise ValueError("weekly_data must be a JSON array")

        predictions: List[Dict[str, Any]] = predict_weekly_rows(cached, weekly_data)
    except ValueError as exc:
        return JsonResponse({"status": "error", "message": str(exc)}, status=400)
    except Exception:
        return JsonResponse(
            {"status": "error", "message": "Internal server error"},
            status=500,
        )

    return JsonResponse(
        {
            "status": "ok",
            "model": {
                "trained_at": cached.trained_at,
                "feature_names": cached.model_state.feature_names,
                "final_loss": round(cached.final_loss, 6),
            },
            "predictions": predictions,
        }
    )
