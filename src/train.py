"""Train and register the customer K-Means segmentation pipeline."""

import argparse
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from src.data import FEATURE_COLUMNS, build_customer_features, load_transactions
from src.model import (
    build_pipeline,
    clustering_metrics,
    save_cluster_profiles,
    save_pca_plot,
    select_optimal_k,
)
from src.tracking import PROJECT_ROOT, configure_tracking, get_or_create_experiment


def train_model(
    data_path: str | Path,
    experiment_name: str = "customer-segmentation-training",
    registered_model_name: str = "customer-kmeans",
    model_alias: str = "current",
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
    n_init: int = 20,
    tracking_uri: str | None = None,
) -> str:
    """Fit, evaluate, log, and register the customer segmentation model.

    Args:
        data_path: Excel workbook containing transaction-level input data.
        experiment_name: MLflow experiment used for training runs.
        registered_model_name: Stable name of the registered model.
        model_alias: Alias assigned to the newly registered model version.
        k_min: Smallest cluster count considered by Yellowbrick.
        k_max: Largest cluster count considered by Yellowbrick.
        random_state: Seed used to make K-Means reproducible.
        n_init: Number of K-Means initializations.
        tracking_uri: Optional MLflow tracking URI.

    Returns:
        The ID of the MLflow training run.
    """
    data_path = Path(data_path).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Transaction workbook not found: {data_path}")

    client = configure_tracking(tracking_uri)
    experiment_id = get_or_create_experiment(client, experiment_name)
    customer_features = build_customer_features(load_transactions(data_path))
    features = customer_features[FEATURE_COLUMNS].astype("float64")

    with tempfile.TemporaryDirectory(prefix="customer-segmentation-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        elbow_path = temporary_path / "elbow.png"
        selected_k, candidate_scores = select_optimal_k(
            features,
            k_min=k_min,
            k_max=k_max,
            random_state=random_state,
            n_init=n_init,
            elbow_plot_path=elbow_path,
        )

        pipeline = build_pipeline(selected_k, random_state=random_state, n_init=n_init)
        pipeline.fit(features)
        labels = pipeline.predict(features)
        scaled_features = pipeline[:-1].transform(features)
        metrics = clustering_metrics(scaled_features, labels)
        if not metrics:
            raise ValueError("The selected model did not produce valid clustering metrics.")

        profiles_path = temporary_path / "cluster_profiles.csv"
        pca_path = temporary_path / "clusters_pca.png"
        save_cluster_profiles(customer_features, labels, profiles_path)
        save_pca_plot(pipeline, features, labels, pca_path)
        if not elbow_path.exists():
            raise RuntimeError("Yellowbrick did not produce the expected elbow chart.")

        with mlflow.start_run(experiment_id=experiment_id, run_name="kmeans-training") as active_run:
            run_id = active_run.info.run_id
            mlflow.log_params(
                {
                    "n_clusters": selected_k,
                    "random_state": random_state,
                    "n_init": n_init,
                    "k_min": k_min,
                    "k_max": k_max,
                    "feature_columns": ",".join(FEATURE_COLUMNS),
                }
            )
            mlflow.log_metric("customer_count", len(customer_features))
            mlflow.log_metrics(metrics)
            mlflow.set_tag("data.source_path", str(data_path))
            if candidate_scores:
                mlflow.log_metrics(
                    {f"candidate_silhouette_k_{k}": score for k, score in candidate_scores.items()}
                )

            dataset = mlflow.data.from_pandas(
                features,
                name="customer-rfm-features",
            )
            mlflow.log_input(dataset, context="training")
            mlflow.log_artifact(str(profiles_path), artifact_path="analysis")
            mlflow.log_artifact(str(pca_path), artifact_path="analysis")
            mlflow.log_artifact(str(elbow_path), artifact_path="analysis")

            example = features.head(min(5, len(features)))
            signature = infer_signature(example, pipeline.predict(example))
            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="customer-kmeans-pipeline",
                signature=signature,
                input_example=example,
                registered_model_name=registered_model_name,
            )

        model_version = client.get_model_version(registered_model_name, model_info.registered_model_version)
        client.set_registered_model_alias(
            registered_model_name,
            model_alias,
            model_version.version,
        )
        return run_id


def main() -> None:
    """Parse CLI arguments and train the registered model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "Online Retail.xlsx",
        help="Path to the transaction Excel workbook.",
    )
    parser.add_argument("--experiment-name", default="customer-segmentation-training")
    parser.add_argument("--model-name", default="customer-kmeans")
    parser.add_argument("--model-alias", default="current")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--tracking-uri", default=None)
    arguments = parser.parse_args()

    run_id = train_model(
        data_path=arguments.data_path,
        experiment_name=arguments.experiment_name,
        registered_model_name=arguments.model_name,
        model_alias=arguments.model_alias,
        k_min=arguments.k_min,
        k_max=arguments.k_max,
        random_state=arguments.random_state,
        n_init=arguments.n_init,
        tracking_uri=arguments.tracking_uri,
    )
    print(f"Training run completed: {run_id}")


if __name__ == "__main__":
    main()
