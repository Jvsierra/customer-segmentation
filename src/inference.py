"""Load the registered customer K-Means model and assign customer clusters."""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.data import FEATURE_COLUMNS, build_customer_features, load_transactions
from src.model import clustering_metrics
from src.tracking import PROJECT_ROOT, configure_tracking, get_or_create_experiment


def predict_customers(
    data_path: str | Path,
    output_path: str | Path,
    registered_model_name: str = "customer-kmeans",
    model_alias: str = "current",
    experiment_name: str = "customer-segmentation-inference",
    tracking_uri: str | None = None,
) -> tuple[str, pd.DataFrame]:
    """Load the registered model, predict clusters, and log the inference run.

    Args:
        data_path: Excel workbook containing transaction-level input data.
        output_path: Destination CSV for customer-level predictions.
        registered_model_name: Name of the model in the MLflow Registry.
        model_alias: Alias identifying the model version to load.
        experiment_name: MLflow experiment used for inference runs.
        tracking_uri: Optional MLflow tracking URI.

    Returns:
        The MLflow inference run ID and the customer prediction table.
    """
    data_path = Path(data_path).resolve()
    output_path = Path(output_path).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Transaction workbook not found: {data_path}")

    client = configure_tracking(tracking_uri)
    experiment_id = get_or_create_experiment(client, experiment_name)
    model_version = client.get_model_version_by_alias(registered_model_name, model_alias)
    source_training_run = client.get_run(model_version.run_id)
    model_uri = f"models:/{registered_model_name}@{model_alias}"
    model = mlflow.sklearn.load_model(model_uri)

    customer_features = build_customer_features(load_transactions(data_path))
    features = customer_features[FEATURE_COLUMNS].astype("float64")
    labels = np.asarray(model.predict(features), dtype=int)
    predictions = customer_features.copy()
    predictions["Cluster"] = labels
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)

    metrics = clustering_metrics(model[:-1].transform(features), labels)
    with mlflow.start_run(experiment_id=experiment_id, run_name="kmeans-inference") as active_run:
        run_id = active_run.info.run_id
        mlflow.set_tags(
            {
                "model_name": registered_model_name,
                "model_version": model_version.version,
                "model_alias": model_alias,
                "source_training_run_id": source_training_run.info.run_id,
                "data.source_path": str(data_path),
            }
        )
        mlflow.log_metric("inference_customer_count", len(predictions))
        mlflow.log_metric("inference_occupied_cluster_count", predictions["Cluster"].nunique())
        if metrics:
            mlflow.log_metrics({f"inference_{name}": value for name, value in metrics.items()})
            metric_status = "calculated"
        else:
            metric_status = "omitted: fewer than two clusters or one cluster per customer"
        mlflow.set_tag("inference_metrics_status", metric_status)

        dataset = mlflow.data.from_pandas(
            features,
            name="customer-rfm-features",
        )
        mlflow.log_input(dataset, context="inference")
        mlflow.log_artifact(str(output_path), artifact_path="predictions")

    loaded_run = MlflowClient(tracking_uri=mlflow.get_tracking_uri()).get_run(run_id)
    print(f"Inference run: {loaded_run.info.run_id}")
    return run_id, predictions


def main() -> None:
    """Parse CLI arguments and run customer cluster inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "Online Retail.xlsx",
        help="Path to the transaction Excel workbook.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "customer_clusters.csv",
        help="Path for the customer-level prediction CSV.",
    )
    parser.add_argument("--model-name", default="customer-kmeans")
    parser.add_argument("--model-alias", default="current")
    parser.add_argument("--experiment-name", default="customer-segmentation-inference")
    parser.add_argument("--tracking-uri", default=None)
    arguments = parser.parse_args()

    run_id, predictions = predict_customers(
        data_path=arguments.data_path,
        output_path=arguments.output_path,
        registered_model_name=arguments.model_name,
        model_alias=arguments.model_alias,
        experiment_name=arguments.experiment_name,
        tracking_uri=arguments.tracking_uri,
    )
    print(f"Inference run completed: {run_id}")
    print(f"Predictions saved for {len(predictions):,} customers: {arguments.output_path}")


if __name__ == "__main__":
    main()
