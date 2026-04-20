
import pymysql
import csv
def export_mysql_to_csv(
    host: str,
    port: int,
    dbname: str,
    user: str,
    password: str,
    output_csv: str | Path,
    sql_query: str,
) -> None:
    """Export query output to CSV using MySQL."""
    conn = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=dbname,
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(sql_query)
            rows = cur.fetchall()
            if not rows:
                raise ValueError("No data returned from SQL query.")
            with open(output_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    finally:
        conn.close()

"""Service helpers for training and serving weekly demand predictions."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Sequence, Tuple
from zoneinfo import ZoneInfo

from demand_prediction_pipeline import (
    FEATURE_COLUMNS,
    ModelState,
    load_data,
    predict,
    preprocess,
    train,
)


@dataclass
class CachedModel:
    model_state: ModelState
    trained_at: str
    csv_path: str
    learning_rate: float
    epochs: int
    use_rolling_feature: bool
    final_loss: float


_CACHE_LOCK = Lock()
_MODEL_CACHE: Dict[Tuple[str, float, int, bool], CachedModel] = {}


def _dhaka_now_iso() -> str:
    return datetime.now(ZoneInfo("Asia/Dhaka")).isoformat()


def _resolve_csv_path(csv_path: str | None) -> str:
    candidate = Path(csv_path or "data.csv")
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if not candidate.exists():
        raise ValueError(f"CSV file not found: {candidate}")
    return str(candidate)


def _feature_names(rows: Sequence[Dict[str, Any]], use_rolling_feature: bool) -> List[str]:
    names = FEATURE_COLUMNS.copy()
    if use_rolling_feature:
        if not rows or "rolling_3wk_avg_units_sold" not in rows[0]:
            raise ValueError(
                "rolling_3wk_avg_units_sold column is required when use_rolling_feature=true"
            )
        names.append("rolling_3wk_avg_units_sold")
    return names


def get_or_train_model(
    csv_path: str | None,
    learning_rate: float,
    epochs: int,
    use_rolling_feature: bool,
    retrain: bool,
) -> CachedModel:
    resolved_csv_path = _resolve_csv_path(csv_path)
    cache_key = (resolved_csv_path, learning_rate, epochs, use_rolling_feature)

    with _CACHE_LOCK:
        if not retrain and cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

    rows = load_data(resolved_csv_path)
    feature_names = _feature_names(rows, use_rolling_feature)
    X, y, mean, std = preprocess(rows, feature_names)

    # Suppress per-epoch logs for API requests.
    weights, bias, loss_history = train(
        X,
        y,
        learning_rate=learning_rate,
        epochs=epochs,
        print_every=0,
    )

    model_state = ModelState(
        weights=weights,
        bias=bias,
        feature_mean=mean,
        feature_std=std,
        feature_names=feature_names,
    )
    cached = CachedModel(
        model_state=model_state,
        trained_at=_dhaka_now_iso(),
        csv_path=resolved_csv_path,
        learning_rate=learning_rate,
        epochs=epochs,
        use_rolling_feature=use_rolling_feature,
        final_loss=loss_history[-1] if loss_history else 0.0,
    )

    with _CACHE_LOCK:
        _MODEL_CACHE[cache_key] = cached

    return cached


def predict_weekly_rows(
    cached_model: CachedModel,
    weekly_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not weekly_rows:
        raise ValueError("weekly_data must contain at least one row")

    feature_names = cached_model.model_state.feature_names
    outputs: List[Dict[str, Any]] = []

    for item in weekly_rows:
        missing = [name for name in feature_names if name not in item]
        if missing:
            raise ValueError(f"Missing feature(s) in input row: {missing}")

        #vector = [float(item[name]) for name in feature_names]
        vector = [float(item[name]) if item[name] != '' else 0.0 for name in feature_names]
        y_hat = predict(vector, cached_model.model_state)[0]
        outputs.append(
            {
                "product_id": item.get("product_id"),
                "week": item.get("week"),
                "predicted_units_sold": round(float(y_hat), 4),
            }
        )

    return outputs
