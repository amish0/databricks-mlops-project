"""Validate raw sales data and create a model-ready dataset."""

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "date",
    "product",
    "region",
    "units_sold",
    "unit_price",
    "marketing_spend",
}


def transform_sales(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Load, validate, and enrich raw sales data."""
    sales = pd.read_csv(input_path)
    missing_columns = REQUIRED_COLUMNS.difference(sales.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")
    if sales.empty:
        raise ValueError("The input sales file contains no records.")
    if sales[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError("Required sales fields cannot contain missing values.")

    sales["date"] = pd.to_datetime(sales["date"], errors="raise")
    for column in ("units_sold", "unit_price", "marketing_spend"):
        sales[column] = pd.to_numeric(sales[column], errors="raise")
    if (sales["units_sold"] < 0).any() or (sales["unit_price"] < 0).any():
        raise ValueError("Units sold and unit price must be non-negative.")
    sales["revenue"] = sales["units_sold"] * sales["unit_price"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sales.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    return sales


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/sales.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed_sales.csv"))
    args = parser.parse_args()

    sales = transform_sales(args.input, args.output)
    print(f"Processed {len(sales)} sales records -> {args.output}")


if __name__ == "__main__":
    main()
