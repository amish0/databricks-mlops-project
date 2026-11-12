"""Evaluate a trained sales model and write DVC-compatible metrics."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def evaluate(data_path: Path, model_path: Path, params_path: Path, metrics_path: Path) -> dict:
    data = pd.read_csv(data_path)
    with params_path.open(encoding="utf-8") as file:
        params = yaml.safe_load(file)
    artifact = joblib.load(model_path)
    features = artifact["features"]
    target = artifact["target"]

    _, X_test, _, y_test = train_test_split(
        data[features],
        data[target],
        test_size=float(params["test_size"]),
        random_state=int(params["random_state"]),
    )
    predictions = artifact["model"].predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("Evaluation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.3f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed_sales.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/sales_model.joblib"))
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--metrics", type=Path, default=Path("reports/metrics.json"))
    args = parser.parse_args()
    evaluate(args.data, args.model, args.params, args.metrics)


if __name__ == "__main__":
    main()
