"""K-Means pipeline, cluster selection, metrics, and visualization."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer


class YellowbrickCompatibleKMeans(KMeans):
    """Expose the estimator type expected by Yellowbrick 1.5."""

    _estimator_type = "clusterer"


def build_pipeline(
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 20,
) -> Pipeline:
    """Build a reproducible preprocessing and K-Means pipeline.

    Args:
        n_clusters: Number of clusters.
        random_state: Seed used for K-Means initialization.
        n_init: Number of K-Means initializations.

    Returns:
        An unfitted scikit-learn pipeline.
    """
    return Pipeline(
        steps=[
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", MinMaxScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=n_clusters,
                    init="k-means++",
                    n_init=n_init,
                    random_state=random_state,
                ),
            ),
        ]
    )


def select_optimal_k(
    features: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
    n_init: int = 20,
    elbow_plot_path: str | Path | None = None,
) -> tuple[int, dict[int, float]]:
    """Select k with Yellowbrick, falling back to the best silhouette score.

    Args:
        features: Customer feature matrix in original, non-log scale.
        k_min: Smallest candidate cluster count, inclusive.
        k_max: Largest candidate cluster count, inclusive.
        random_state: Seed used for candidate K-Means estimators.
        n_init: Number of K-Means initializations.
        elbow_plot_path: Optional path for the Yellowbrick elbow chart.

    Returns:
        The selected k and candidate silhouette scores (empty when Yellowbrick
        identifies an elbow).
    """
    log_features = np.log1p(features)
    scaled_features = MinMaxScaler().fit_transform(log_features)
    max_valid_k = min(k_max, len(features) - 1)
    if k_min < 2 or max_valid_k < k_min:
        raise ValueError("At least three customers are required to select k.")

    visualizer = KElbowVisualizer(
        YellowbrickCompatibleKMeans(n_init=n_init, random_state=random_state),
        k=(k_min, max_valid_k + 1),
        metric="distortion",
        timings=False,
        locate_elbow=True,
    )
    visualizer.fit(scaled_features)
    if elbow_plot_path is not None:
        visualizer.show(outpath=str(elbow_plot_path), clear_figure=True)
    else:
        plt.close(visualizer.fig)

    if visualizer.elbow_value_ is not None:
        return int(visualizer.elbow_value_), {}

    candidate_scores: dict[int, float] = {}
    for candidate_k in range(k_min, max_valid_k + 1):
        candidate_model = KMeans(
            n_clusters=candidate_k,
            init="k-means++",
            n_init=n_init,
            random_state=random_state,
        )
        labels = candidate_model.fit_predict(scaled_features)
        candidate_scores[candidate_k] = float(silhouette_score(scaled_features, labels))

    return max(candidate_scores, key=candidate_scores.get), candidate_scores


def clustering_metrics(features: pd.DataFrame | np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Calculate internal clustering metrics when the labels permit it.

    Args:
        features: Feature matrix in the same scaled space used by K-Means.
        labels: Cluster assignment for each row.

    Returns:
        Available Silhouette, Davies-Bouldin, and Calinski-Harabasz scores.
    """
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return {}

    return {
        "silhouette_score": float(silhouette_score(features, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(features, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(features, labels)),
    }


def save_cluster_profiles(
    customer_features: pd.DataFrame,
    labels: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save original-scale mean and median feature profiles by cluster.

    Args:
        customer_features: Customer IDs and original-scale RFM features.
        labels: Cluster assignment for each customer.
        output_path: Destination CSV path.
    """
    clustered = customer_features.copy()
    clustered["Cluster"] = labels
    profiles = clustered.groupby("Cluster").agg(
        CustomerCount=("CustomerID", "nunique"),
        MeanRecencyDays=("RecencyDays", "mean"),
        MedianRecencyDays=("RecencyDays", "median"),
        MeanFrequency=("Frequency", "mean"),
        MedianFrequency=("Frequency", "median"),
        MeanMonetary=("Monetary", "mean"),
        MedianMonetary=("Monetary", "median"),
    )
    profiles.to_csv(output_path, index=True)


def save_pca_plot(
    pipeline: Pipeline,
    features: pd.DataFrame,
    labels: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save a two-dimensional PCA visualization of the fitted clusters.

    Args:
        pipeline: Fitted pipeline containing log1p, scaler, and K-Means steps.
        features: Customer feature matrix in original scale.
        labels: Cluster assignment for each customer.
        output_path: Destination image path.
    """
    scaled_features = pipeline[:-1].transform(features)
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_features)

    figure, axis = plt.subplots(figsize=(10, 7))
    points = axis.scatter(
        components[:, 0],
        components[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.7,
        s=28,
    )
    axis.set_title("Customer clusters projected onto the first two principal components")
    axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    axis.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    figure.colorbar(points, ax=axis, label="Cluster")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
