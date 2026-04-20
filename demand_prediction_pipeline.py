"""Demand prediction pipeline using PostgreSQL features and manual gradient descent.

This script does the following:
1) Optionally exports weekly feature data from PostgreSQL to CSV.
2) Loads CSV data.
3) Trains a linear regression model with gradient descent (no sklearn training).
4) Exposes predict(new_data, model_state) for inference.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


FEATURE_COLUMNS = [
    "prev_units_sold",
    "wishlist_count",
    "cart_total",
    "unique_message_users",
]
TARGET_COLUMN = "units_sold"


@dataclass
class ModelState:
    weights: List[float]
    bias: float
    feature_mean: List[float]
    feature_std: List[float]
    feature_names: List[str]


def read_sql_query(sql_path: str | Path) -> str:
    """Read SQL text from file."""
    return Path(sql_path).read_text(encoding="utf-8")


def export_sql_to_csv(
    host: str,
    port: int,
    dbname: str,
    user: str,
    password: str,
    output_csv: str | Path,
    sql_path: str | Path,
) -> None:
    """Export query output to CSV using MySQL."""
    try:
        import pymysql
    except ImportError as exc:
        raise ImportError(
            "pymysql is required for export. Install with: pip install pymysql"
        ) from exc

    query = read_sql_query(sql_path).strip().rstrip(";")

    connection = pymysql.connect(
        host=host,
        port=int(port),
        user=user,
        password=password,
        database=dbname,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            if not rows:
                raise ValueError("No data returned from SQL query.")
            fieldnames = rows[0].keys()
            with open(output_csv, "w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    finally:
        connection.close()


def load_data(csv_path: str | Path) -> List[Dict[str, Any]]:
    """Load feature dataset from CSV and perform basic checks."""
    required_cols = set(FEATURE_COLUMNS + [TARGET_COLUMN])

    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")

        missing = required_cols - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

        for row in reader:
            rows.append(dict(row))

    return rows


def preprocess(
    rows: Sequence[Dict[str, Any]],
    feature_columns: Sequence[str],
) -> Tuple[List[List[float]], List[float], List[float], List[float]]:
    """Select features and target, then normalize features with mean/std.

    Returns
    -------
    X_norm : np.ndarray
        Normalized feature matrix.
    y : np.ndarray
        Target vector.
    mean : np.ndarray
        Feature-wise mean from training data.
    std_safe : np.ndarray
        Feature-wise std with zeros replaced by 1 to avoid division by zero.
    """
    if not rows:
        raise ValueError("Input dataset is empty")

    X_raw: List[List[float]] = []
    y: List[float] = []
    for row in rows:
        # Clean empty strings in feature columns
        cleaned_features = [float(row[col]) if row[col] != '' else 0.0 for col in feature_columns]
        X_raw.append(cleaned_features)
        y_val = float(row[TARGET_COLUMN]) if row[TARGET_COLUMN] != '' else 0.0
        y.append(y_val)

    n_samples = len(X_raw)
    n_features = len(feature_columns)

    mean = [0.0] * n_features
    for j in range(n_features):
        mean[j] = sum(X_raw[i][j] for i in range(n_samples)) / n_samples

    std = [0.0] * n_features
    for j in range(n_features):
        variance = sum((X_raw[i][j] - mean[j]) ** 2 for i in range(n_samples)) / n_samples
        std[j] = math.sqrt(variance)

    # Prevent division by zero for constant columns.
    std_safe = [value if value != 0 else 1.0 for value in std]

    X_norm: List[List[float]] = []
    for sample in X_raw:
        X_norm.append([(sample[j] - mean[j]) / std_safe[j] for j in range(n_features)])

    return X_norm, y, mean, std_safe


def compute_mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Compute mean squared error loss."""
    n = len(y_true)
    return sum((y_pred[i] - y_true[i]) ** 2 for i in range(n)) / n


def train(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    learning_rate: float = 0.01,
    epochs: int = 1000,
    print_every: int = 100,
) -> Tuple[List[float], float, List[float]]:
    """Train linear regression with manual gradient descent."""
    n_samples = len(X)
    if n_samples == 0:
        raise ValueError("Training data is empty")
    n_features = len(X[0])

    weights = [0.0] * n_features
    bias = 0.0
    loss_history: List[float] = []

    for epoch in range(1, epochs + 1):
        predictions = [sum(X[i][j] * weights[j] for j in range(n_features)) + bias for i in range(n_samples)]
        errors = [predictions[i] - y[i] for i in range(n_samples)]

        loss = compute_mse(y, predictions)
        loss_history.append(loss)

        grad_w = [0.0] * n_features
        for j in range(n_features):
            grad_w[j] = (2.0 / n_samples) * sum(X[i][j] * errors[i] for i in range(n_samples))
        grad_b = (2.0 / n_samples) * sum(errors)

        for j in range(n_features):
            weights[j] -= learning_rate * grad_w[j]
        bias -= learning_rate * grad_b

        if print_every > 0 and (epoch % print_every == 0 or epoch == 1):
            print(f"Epoch {epoch:4d}/{epochs} - MSE: {loss:.6f}")

    return weights, bias, loss_history


def predict(new_data: Sequence[Sequence[float]] | Sequence[float], model_state: ModelState) -> List[float]:
    """Predict units_sold for new samples.

    new_data can be a single sample with 4 values or multiple samples.
    Feature order must match model_state.feature_names.
    """
    if not new_data:
        raise ValueError("new_data cannot be empty")

    if isinstance(new_data[0], (int, float)):
        samples = [[float(v) for v in new_data]]  # type: ignore[arg-type]
    else:
        samples = [[float(v) for v in sample] for sample in new_data]  # type: ignore[list-item]

    if len(samples[0]) != len(model_state.feature_names):
        raise ValueError(
            f"Expected {len(model_state.feature_names)} features, got {len(samples[0])}"
        )

    predictions: List[float] = []
    for sample in samples:
        normalized = [
            (sample[j] - model_state.feature_mean[j]) / model_state.feature_std[j]
            for j in range(len(model_state.feature_names))
        ]
        y_hat = sum(normalized[j] * model_state.weights[j] for j in range(len(model_state.feature_names)))
        y_hat += model_state.bias
        predictions.append(y_hat)

    return predictions


def run_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    """Run export, load, preprocess, train, and a sample prediction."""
    if args.export_from_db:
        export_sql_to_csv(
            host=args.db_host,
            port=args.db_port,
            dbname=args.db_name,
            user=args.db_user,
            password=args.db_password,
            output_csv=args.csv_path,
            sql_path=args.sql_path,
        )
        print(f"Export completed: {args.csv_path}")

    feature_columns = FEATURE_COLUMNS.copy()
    rows = load_data(args.csv_path)

    if args.use_rolling_feature:
        # Only include the rolling feature if present in the dataset.
        if not rows or "rolling_3wk_avg_units_sold" not in rows[0]:
            raise ValueError(
                "--use-rolling-feature was passed but column rolling_3wk_avg_units_sold is missing"
            )
        feature_columns.append("rolling_3wk_avg_units_sold")

    X, y, mean, std_safe = preprocess(rows, feature_columns)

    weights, bias, loss_history = train(
        X,
        y,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        print_every=args.print_every,
    )

    model_state = ModelState(
        weights=weights,
        bias=bias,
        feature_mean=mean,
        feature_std=std_safe,
        feature_names=feature_columns,
    )

    print("\nFinal model parameters")
    print("Weights:", model_state.weights)
    print("Bias:", model_state.bias)

    # Example prediction using first row feature values from the dataset.
    sample_raw = [float(rows[0][col]) if rows[0][col] != '' else 0.0 for col in feature_columns]
    sample_pred = predict(sample_raw, model_state)[0]
    print("\nExample prediction")
    print("Input features:", dict(zip(feature_columns, sample_raw)))
    print("Predicted units_sold:", round(float(sample_pred), 4))

    return {
        "model_state": model_state,
        "loss_history": loss_history,
        "example_prediction": float(sample_pred),
    }


def parse_args() -> argparse.Namespace:
    """Define CLI arguments for flexible pipeline runs."""
    parser = argparse.ArgumentParser(description="Demand prediction pipeline")

    parser.add_argument("--csv-path", type=str, default="data.csv", help="Input/output CSV path")
    parser.add_argument(
        "--sql-path",
        type=str,
        default="sql/weekly_demand_features.sql",
        help="Path to SQL file for export",
    )

    parser.add_argument("--export-from-db", action="store_true", help="Export SQL result to CSV first")
    parser.add_argument("--db-host", type=str, default="localhost")
    parser.add_argument("--db-port", type=int, default=3306)
    parser.add_argument("--db-name", type=str, default="tumii")
    parser.add_argument("--db-user", type=str, default="root")
    parser.add_argument("--db-password", type=str, default="")

    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument(
        "--use-rolling-feature",
        action="store_true",
        help="Include rolling_3wk_avg_units_sold as an additional feature when available",
    )

    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()

    # --- MySQL connection test for debugging ---
    try:
        import pymysql
        print(f"[DEBUG] Testing MySQL connection to {cli_args.db_host}:{cli_args.db_port} as {cli_args.db_user}...")
        conn = pymysql.connect(
            host=cli_args.db_host,
            port=int(cli_args.db_port),
            user=cli_args.db_user,
            password=cli_args.db_password,
            database=cli_args.db_name,
            connect_timeout=5
        )
        conn.close()
        print("[DEBUG] MySQL connection successful!")
    except Exception as e:
        print(f"[ERROR] MySQL connection failed: {type(e).__name__}: {e}")
        import sys
        sys.exit(1)

    run_pipeline(cli_args)
