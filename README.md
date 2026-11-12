# Databricks MLOps Sales Demo

A compact, reproducible sales-revenue ML pipeline. The demo uses synthetic
sample data and runs locally with Python and DVC; the analysis script is also
formatted as a Databricks notebook export.

## What it does

1. Validates raw sales records and calculates revenue (`units_sold * unit_price`).
2. Trains a Random Forest regression pipeline using product, region, units sold,
   unit price, and marketing spend.
3. Evaluates the model on a deterministic holdout split and saves MAE, RMSE, and
   R-squared metrics.
4. Tracks the stages, dependencies, parameters, model, and metrics with DVC.

## Quick start

Use Python 3.10 or later. From the project root:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
dvc repro
```

The pipeline creates:

- `data/processed_sales.csv` — validated sales records with calculated revenue
- `models/sales_model.joblib` — fitted preprocessing and regression pipeline
- `reports/metrics.json` — holdout evaluation metrics

To review recorded metrics:

```bash
dvc metrics show
```

To run a stage by itself after its prerequisites have been generated:

```bash
python etl/sales_etl.py
python ml/train.py
python ml/evaluate.py
```

To open the Databricks notebook, import `notebooks/sales_analysis.py`. Locally,
run it from the project root with Python after `dvc repro` to print summary
statistics and an example prediction.

## Deploy the pipeline to Databricks

The GitHub Actions workflow deploys a Databricks Asset Bundle when changes are
pushed to `main`, or when started manually from the Actions tab. It deploys a
job that runs the ETL, training, and evaluation stages in
`notebooks/databricks_sales_pipeline.py`.

Configure these repository settings in **Settings → Secrets and variables →
Actions** before running the workflow:

| Name | Type | Value |
| --- | --- | --- |
| `DATABRICKS_HOST` | Variable | Workspace URL, such as `https://<workspace-host>` |
| `DATABRICKS_NODE_TYPE_ID` | Variable | A node type available in this workspace's cloud and region |
| `DATABRICKS_CLIENT_ID` | Secret | OAuth client ID for a Databricks service principal |
| `DATABRICKS_CLIENT_SECRET` | Secret | OAuth client secret for that service principal |

Grant the service principal permission to use the selected cluster node type and
deploy jobs to the workspace. The workflow uses Databricks unified
authentication with the client ID and secret. Run `databricks bundle run
sales_mlops_pipeline -t prod` from a configured environment to start the
deployed job.

The job installs the ML dependencies on its cluster. Its generated model,
processed data, and metrics are stored on temporary cluster-local disk for the
duration of the demo run; they are not a persistent production artifact store.

## Demo walkthrough

1. Show `data/sales.csv` and the editable model/split settings in `params.yaml`.
2. Run `dvc repro` and walk through the ETL, training, and evaluation stages.
3. Inspect `data/processed_sales.csv` and `reports/metrics.json`.
4. Edit a model parameter in `params.yaml` and rerun `dvc repro` to demonstrate
   that DVC reruns the affected stages.
5. Run `python notebooks/sales_analysis.py` to view the sales breakdown and
   example prediction.
6. Configure the GitHub Actions settings above and push to `main` to demonstrate
   deployment of the Databricks job.

The dataset is intentionally small and synthetic for demonstrations; its
metrics are not intended to represent production performance.
