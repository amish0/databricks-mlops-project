# Databricks notebook source
# MAGIC %md
# MAGIC # Sales MLOps pipeline
# MAGIC
# MAGIC Runs the repository ETL, training, and evaluation stages in order.
# MAGIC Generated artifacts live on the job cluster's local disk for this demo run.

# COMMAND ----------

from pathlib import Path
import sys
import tempfile

import pandas as pd


dbutils.widgets.text("project_root", "")
project_root = Path(dbutils.widgets.get("project_root"))
if not project_root.is_dir():
    raise FileNotFoundError(f"Bundle project directory not found: {project_root}")

for required_file in ("data/sales.csv", "params.yaml", "etl/sales_etl.py", "ml/train.py", "ml/evaluate.py"):
    if not (project_root / required_file).is_file():
        raise FileNotFoundError(f"Required project file not found: {project_root / required_file}")

sys.path.insert(0, str(project_root))
from etl.sales_etl import transform_sales
from ml.evaluate import evaluate
from ml.train import train

output_root = Path(tempfile.mkdtemp(prefix="sales-mlops-demo-"))
processed_path = output_root / "processed_sales.csv"
model_path = output_root / "sales_model.joblib"
metrics_path = output_root / "metrics.json"
params_path = project_root / "params.yaml"

transform_sales(project_root / "data" / "sales.csv", processed_path)
train(processed_path, model_path, params_path)
metrics = evaluate(processed_path, model_path, params_path, metrics_path)

print(f"Pipeline artifacts for this run: {output_root}")
print(f"Evaluation metrics: {metrics}")
