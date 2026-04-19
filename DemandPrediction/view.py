"""HTTP views for demand prediction API endpoints."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .api_service import get_or_train_model, predict_weekly_rows, export_mysql_to_csv

from django.conf import settings


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

        db_settings = settings.DATABASES["default"]
        host = db_settings["HOST"]
        port = int(db_settings["PORT"])
        dbname = db_settings["NAME"]
        user = db_settings["USER"]
        password = db_settings["PASSWORD"]

        product_ids = payload.get("product_ids")
        if not product_ids or not isinstance(product_ids, list):
            raise ValueError("product_ids must be provided as a list")

        # Read SQL query from file
        sql_path = payload.get("sql_path") or "sql/weekly_demand_features.sql"
        with open(sql_path, "r", encoding="utf-8") as f:
            sql_query = f.read()

        # Filter for product_ids and current week only
        from datetime import date
        current_week = date.today().strftime("%Y-%m-%d")
        sql_query += f"\nWHERE product_id IN ({','.join(str(int(pid)) for pid in product_ids)})"
        sql_query += f" AND week = '{current_week}'"

        csv_path = payload.get("csv_path") or "data.csv"
        export_mysql_to_csv(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            output_csv=csv_path,
            sql_query=sql_query,
        )

        cached = get_or_train_model(
            csv_path=csv_path,
            learning_rate=float(payload.get("learning_rate", 0.01)),
            epochs=int(payload.get("epochs", 1000)),
            use_rolling_feature=bool(payload.get("use_rolling_feature", False)),
            retrain=bool(payload.get("retrain", False)),
        )

        # Load the just-exported CSV and filter for current week and product_ids
        import csv as pycsv
        weekly_data = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = pycsv.DictReader(f)
            for row in reader:
                if int(row["product_id"]) in product_ids:
                    weekly_data.append(row)
        if not weekly_data:
            raise ValueError("No data found for the given product_ids and current week.")

        predictions: List[Dict[str, Any]] = predict_weekly_rows(cached, weekly_data)
    except ValueError as exc:
        return JsonResponse({"status": "error", "message": str(exc)}, status=400)
    except Exception as exc:
        return JsonResponse(
            {"status": "error", "message": f"Internal server error: {exc}"},
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
