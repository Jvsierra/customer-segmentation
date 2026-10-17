"""Small helpers for local MLflow tracking and experiment reuse."""

import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_tracking(tracking_uri: str | None = None) -> MlflowClient:
    """Configure MLflow to use the project-local SQLite database by default.

    Args:
        tracking_uri: Optional MLflow tracking URI. The
            ``MLFLOW_TRACKING_URI`` environment variable takes precedence when
            no value is passed.

    Returns:
        An MLflow client connected to the configured tracking store.
    """
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if uri is None:
        database_path = (PROJECT_ROOT / "mlflow.db").as_posix()
        uri = f"sqlite:///{database_path}"

    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def get_or_create_experiment(client: MlflowClient, name: str) -> str:
    """Return an experiment ID, creating the experiment when absent.

    Args:
        client: Configured MLflow tracking client.
        name: Case-sensitive experiment name.

    Returns:
        The existing or newly created experiment ID.
    """
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        return client.create_experiment(name)
    return experiment.experiment_id
