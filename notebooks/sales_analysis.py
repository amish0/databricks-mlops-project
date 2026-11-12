# Databricks notebook source
# MAGIC %md
# MAGIC # Sales demo: explore the data and score the model
# MAGIC
# MAGIC Run the ETL and training stages first (`dvc repro`), or run this file after
# MAGIC generating `data/processed_sales.csv` and `models/sales_model.joblib`.

# COMMAND ----------

from pathlib import Path

import joblib
import pandas as pd


ROOT = Path.cwd()
raw_path = ROOT / "data" / "sales.csv"
processed_path = ROOT / "data" / "processed_sales.csv"
model_path = ROOT / "models" / "sales_model.joblib"

if not processed_path.exists():
    raise FileNotFoundError("Run `dvc repro` from the project root before this analysis.")

sales = pd.read_csv(processed_path, parse_dates=["date"])
print(f"Rows: {len(sales)}")
print(f"Total revenue: ${sales['revenue'].sum():,.2f}")
print("\nRevenue by product:")
print(sales.groupby("product")["revenue"].sum().sort_values(ascending=False))
print("\nRevenue by region:")
print(sales.groupby("region")["revenue"].sum().sort_values(ascending=False))

# COMMAND ----------

if model_path.exists():
    artifact = joblib.load(model_path)
    example = sales[artifact["features"]].iloc[[0]]
    prediction = artifact["model"].predict(example)[0]
    actual = sales[artifact["target"]].iloc[0]
    print(f"Example actual revenue:    ${actual:,.2f}")
    print(f"Example predicted revenue: ${prediction:,.2f}")
else:
    print("Model not found; run `dvc repro` to train it.")
