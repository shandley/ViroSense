"""Cluster quality metrics."""

import numpy as np


def silhouette_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score for clustering quality."""
    from sklearn.metrics import silhouette_score as sklearn_silhouette
    valid = labels >= 0
    if valid.sum() < 2:
        return 0.0
    return sklearn_silhouette(embeddings[valid], labels[valid])


def cluster_summary(labels: np.ndarray) -> dict:
    """Compute summary statistics for cluster assignments."""
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = int((labels == -1).sum())
    sizes = [int((labels == k).sum()) for k in unique_labels if k >= 0]

    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "n_total": len(labels),
        "cluster_sizes": sizes,
        "mean_cluster_size": float(np.mean(sizes)) if sizes else 0.0,
        "median_cluster_size": float(np.median(sizes)) if sizes else 0.0,
    }
