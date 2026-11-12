"""Train a sales revenue model using the configured training split."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURES = ["product", "region", "units_sold", "unit_price", "marketing_spend"]
TARGET = "revenue"


def load_params(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        params = yaml.safe_load(file)
    if not isinstance(params, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return params


def train(data_path: Path, model_path: Path, params_path: Path) -> None:
    data = pd.read_csv(data_path)
    missing = set(FEATURES + [TARGET]).difference(data.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {', '.join(sorted(missing))}")
    if len(data) < 3:
        raise ValueError("At least three records are required to train the model.")

    params = load_params(params_path)
    random_state = int(params["random_state"])
    test_size = float(params["test_size"])
    model_params = params["model"]

    X_train, _, y_train, _ = train_test_split(
        data[FEATURES],
        data[TARGET],
        test_size=test_size,
        random_state=random_state,
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categories", OneHotEncoder(handle_unknown="ignore"), ["product", "region"]),
        ],
        remainder="passthrough",
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=int(model_params["n_estimators"]),
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "target": TARGET}, model_path)
    print(f"Trained model on {len(X_train)} records -> {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed_sales.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/sales_model.joblib"))
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args()
    train(args.data, args.model, args.params)


if __name__ == "__main__":
    main()
