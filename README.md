# Online Retail Customer Segmentation

This project segments online retail customers by purchase behavior with K-Means, using exploratory notebooks, a reusable pipeline, MLflow tracking, and a Streamlit dashboard.

## Features

The model aggregates valid transactions into one record per customer:

| Feature | Definition |
| --- | --- |
| `RecencyDays` | Days since the customer's last valid invoice, relative to the latest valid invoice date in the input data. |
| `Frequency` | Number of distinct valid invoices for the customer. |
| `Monetary` | Total spend, calculated as `Quantity * UnitPrice`. |

Cancelled or adjustment invoices, non-positive quantities or prices, invalid dates, and rows without a customer ID are excluded. Features pass through `log1p` and `MinMaxScaler`. Yellowbrick selects `k`; silhouette score is the fallback if no elbow is found.

## Quick Start

The source workbook is `data/Online Retail.xlsx`. Python 3.14 was used for development.

```bash
python -m venv .venv
```

Activate the environment, then install the pinned dependencies:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the workflow from the repository root:

```bash
python -m src.train
python -m src.inference
streamlit run app.py
```

Training registers `customer-kmeans` with the `current` alias. Inference loads it and writes assignments to `outputs/customer_clusters.csv`, which the dashboard reads by default. A different predictions CSV can be uploaded in the sidebar.

## MLflow and Results

MLflow creates or reuses the `customer-segmentation-training` and `customer-segmentation-inference` experiments. Tracking metadata defaults to the project-root `mlflow.db` SQLite database, with artifacts stored locally. Set `MLFLOW_TRACKING_URI` to use another tracking store.

Training logs hyperparameters, internal clustering scores, profiles, a PCA plot, and the registered model. Inference logs batch-level scores and predictions. No holdout is used, so training scores are in-sample. Cluster IDs have no inherent meaning; compare feature profiles. Higher Silhouette/Calinski-Harabasz and lower Davies-Bouldin scores generally indicate better separation.

Recency uses the latest valid invoice date in each input batch. When comparing separate inference batches, consider that their reference dates may differ.

## Repository Layout

```text
app.py                 Streamlit results dashboard
src/                   Data, model, MLflow tracking, training, and inference modules
data/                  Online Retail source workbook
eda/                   Exploratory data analysis notebook
experiments/           Rule-based and K-Means notebooks
AGENTS.md              Contributor guidelines
```

Generated predictions, MLflow databases, and artifacts are local outputs excluded from version control. See [AGENTS.md](AGENTS.md) for contribution conventions and the [Streamlit](https://docs.streamlit.io/) and [MLflow](https://mlflow.org/docs/latest/) documentation for tool details.
