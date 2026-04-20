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

        # Strip trailing semicolon and ORDER BY, inject WHERE, then re-add ORDER BY
        from datetime import date, timedelta
        today = date.today()
        current_week = (today - timedelta(days=today.weekday())).strftime("%Y-%m-%d")

        sql_query = sql_query.strip().rstrip(";")

        # Remove the final ORDER BY line so we can inject WHERE before it
        order_by_clause = "ORDER BY t.product_id, t.week"
        if order_by_clause in sql_query:
            sql_query = sql_query[:sql_query.rfind(order_by_clause)].rstrip()

        ids_str = ','.join(str(int(pid)) for pid in product_ids)
        sql_query += f"\nWHERE t.product_id IN ({ids_str})"
        sql_query += f"\nORDER BY t.product_id, t.week"

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

        # Clean empty strings to 0 in CSV before training
        import csv as pycsv
        cleaned_rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = pycsv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                cleaned_row = {k: (v if v != '' else '0') for k, v in row.items()}
                cleaned_rows.append(cleaned_row)

        if fieldnames is not None:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = pycsv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(cleaned_rows)
        else:
            # If no fieldnames, just clear the file
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                pass



        # Debug: Read the exported CSV and print some info (optional, can be removed later)
        import csv as pycsv
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = pycsv.DictReader(f)
            rows_debug = list(reader)
            print(f"[DEBUG] CSV rows count: {len(rows_debug)}")
            if rows_debug:
                print(f"[DEBUG] First row: {rows_debug[0]}")
                print(f"[DEBUG] Unique weeks: {set(r['week'] for r in rows_debug)}")
                print(f"[DEBUG] Unique product_ids: {set(r['product_id'] for r in rows_debug)}")



        print(f"[DEBUG] About to train model...")
        try:
            cached = get_or_train_model(
                csv_path=csv_path,
                learning_rate=float(payload.get("learning_rate", 0.01)),
                epochs=int(payload.get("epochs", 1000)),
                use_rolling_feature=bool(payload.get("use_rolling_feature", False)),
                retrain=bool(payload.get("retrain", False)),
            )
            print(f"[DEBUG] Model trained OK")
        except Exception as e:
            print(f"[DEBUG] get_or_train_model FAILED: {type(e).__name__}: {e}")
            raise

        cached = get_or_train_model(
            csv_path=csv_path,
            learning_rate=float(payload.get("learning_rate", 0.01)),
            epochs=int(payload.get("epochs", 1000)),
            use_rolling_feature=bool(payload.get("use_rolling_feature", False)),
            retrain=bool(payload.get("retrain", False)),
        )

        # Load the just-exported CSV and filter for current week and product_ids
        # weekly_data = []
        # with open(csv_path, "r", encoding="utf-8") as f:
        #     reader = pycsv.DictReader(f)
        #     for row in reader:
        #         if int(row["product_id"]) in product_ids:
        #             weekly_data.append(row)
        # if not weekly_data:
        #     raise ValueError("No data found for the given product_ids and current week.")
        
        latest_rows = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = pycsv.DictReader(f)
            for row in reader:
                pid = int(row["product_id"])
                if pid in product_ids:
                    existing = latest_rows.get(pid)
                    if existing is None or row["week"] > existing["week"]:
                        latest_rows[pid] = row

        
        weekly_data = list(latest_rows.values())
        if not weekly_data:
            raise ValueError("No data found for the given product_ids.")
        

        print(f"[DEBUG weekly_data] {weekly_data}")
        print(f"[DEBUG feature_names] {cached.model_state.feature_names}")

        predictions: List[Dict[str, Any]] = predict_weekly_rows(cached, weekly_data)

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
    except ValueError as exc:
        return JsonResponse({"status": "error", "message": str(exc)}, status=400)
    except Exception as exc:
        return JsonResponse(
            {"status": "error", "message": f"Internal server error: {exc}"},
            status=500,
        )
