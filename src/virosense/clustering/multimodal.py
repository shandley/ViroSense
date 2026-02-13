"""Multi-modal embedding fusion and clustering."""

from dataclasses import dataclass

import numpy as np
from loguru import logger
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusterAssignment:
    """Cluster assignment for a single sequence."""

    sequence_id: str
    cluster_id: int  # -1 for noise
    is_representative: bool
    distance_to_centroid: float


def fuse_embeddings(
    dna_embeddings: np.ndarray,
    protein_embeddings: np.ndarray | None = None,
    mode: str = "multi",
) -> np.ndarray:
    """Fuse DNA and protein embeddings.

    Args:
        dna_embeddings: (N, dna_dim) Evo2 embeddings.
        protein_embeddings: (N, prot_dim) ProstT5 embeddings, optional.
        mode: "dna" (DNA only), "protein" (protein only), "multi" (concatenate).

    Returns:
        (N, fused_dim) fused embedding matrix.
    """
    if mode == "dna":
        return dna_embeddings
    elif mode == "protein":
        if protein_embeddings is None:
            raise ValueError("Protein embeddings required for mode='protein'")
        return protein_embeddings
    elif mode == "multi":
        if protein_embeddings is None:
            logger.warning("No protein embeddings provided, falling back to DNA only")
            return dna_embeddings
        return np.concatenate([dna_embeddings, protein_embeddings], axis=1)
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from: dna, protein, multi")


def cluster_sequences(
    embeddings: np.ndarray,
    sequence_ids: list[str],
    algorithm: str = "hdbscan",
    min_cluster_size: int = 5,
    n_clusters: int | None = None,
) -> list[ClusterAssignment]:
    """Cluster sequences based on fused embeddings.

    Scales embeddings to zero mean / unit variance before clustering,
    then assigns each sequence to a cluster and identifies representatives
    (closest to centroid).

    Args:
        embeddings: (N, dim) embedding matrix.
        sequence_ids: List of sequence identifiers.
        algorithm: Clustering algorithm (hdbscan, leiden, kmeans).
        min_cluster_size: Minimum cluster size (hdbscan/leiden).
        n_clusters: Number of clusters (kmeans only; estimated if None).

    Returns:
        List of ClusterAssignment for each sequence.
    """
    if len(sequence_ids) != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch: {len(sequence_ids)} sequence IDs vs "
            f"{embeddings.shape[0]} embeddings"
        )

    # Scale embeddings for better clustering
    scaled = StandardScaler().fit_transform(embeddings)

    if algorithm == "hdbscan":
        labels = _cluster_hdbscan(scaled, min_cluster_size)
    elif algorithm == "leiden":
        labels = _cluster_leiden(scaled, min_cluster_size)
    elif algorithm == "kmeans":
        labels = _cluster_kmeans(scaled, n_clusters or _estimate_k(scaled))
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. Choose from: hdbscan, leiden, kmeans"
        )

    return _build_assignments(embeddings, sequence_ids, labels)


def _cluster_hdbscan(embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Run HDBSCAN clustering."""
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        n_jobs=-1,
    )
    labels = model.fit_predict(embeddings)
    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    logger.info(
        f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points "
        f"(min_cluster_size={min_cluster_size})"
    )
    return labels


def _cluster_leiden(embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Run Leiden community detection on a k-NN graph."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        raise ImportError(
            "Leiden clustering requires 'leidenalg' and 'python-igraph'. "
            "Install with: pip install leidenalg python-igraph"
        )

    # Build k-NN graph
    k = min(15, len(embeddings) - 1)
    dists = pairwise_distances(embeddings, metric="euclidean")
    edges = []
    weights = []
    for i in range(len(embeddings)):
        neighbors = np.argsort(dists[i])[1 : k + 1]
        for j in neighbors:
            if i < j:
                edges.append((i, j))
                weights.append(1.0 / (1.0 + dists[i, j]))

    g = ig.Graph(n=len(embeddings), edges=edges, directed=False)
    g.es["weight"] = weights

    partition = leidenalg.find_partition(
        g, leidenalg.ModularityVertexPartition, weights="weight"
    )
    labels = np.array(partition.membership)

    # Merge small clusters into noise
    for cluster_id in set(labels):
        if (labels == cluster_id).sum() < min_cluster_size:
            labels[labels == cluster_id] = -1

    # Re-number clusters contiguously
    labels = _renumber_labels(labels)

    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    logger.info(f"Leiden: {n_clusters} clusters, {n_noise} noise points")
    return labels


def _cluster_kmeans(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """Run k-means clustering."""
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = model.fit_predict(embeddings)
    logger.info(f"KMeans: {n_clusters} clusters")
    return labels


def _estimate_k(embeddings: np.ndarray) -> int:
    """Estimate number of clusters using the elbow heuristic."""
    max_k = min(20, len(embeddings) // 3)
    if max_k < 2:
        return 2

    inertias = []
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, n_init=5, random_state=42, max_iter=100)
        model.fit(embeddings)
        inertias.append(model.inertia_)

    # Find elbow: largest drop in inertia
    diffs = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
    if not diffs:
        return 2
    best_k = np.argmax(diffs) + 2  # offset by starting k=2
    logger.info(f"Estimated k={best_k} clusters (elbow method)")
    return best_k


def _renumber_labels(labels: np.ndarray) -> np.ndarray:
    """Renumber cluster labels to be contiguous starting from 0, preserving -1."""
    result = np.full_like(labels, -1)
    new_id = 0
    for old_id in sorted(set(labels)):
        if old_id == -1:
            continue
        result[labels == old_id] = new_id
        new_id += 1
    return result


def _build_assignments(
    embeddings: np.ndarray,
    sequence_ids: list[str],
    labels: np.ndarray,
) -> list[ClusterAssignment]:
    """Build ClusterAssignment objects with centroid distances and representatives."""
    # Compute centroids per cluster
    centroids = {}
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue
        mask = labels == cluster_id
        centroids[cluster_id] = embeddings[mask].mean(axis=0)

    assignments = []
    # Track closest-to-centroid per cluster for representative flag
    closest = {}  # cluster_id -> (min_dist, index)

    for i, (seq_id, cluster_id) in enumerate(zip(sequence_ids, labels)):
        if cluster_id == -1:
            dist = 0.0
        else:
            diff = embeddings[i] - centroids[cluster_id]
            dist = float(np.linalg.norm(diff))
            if cluster_id not in closest or dist < closest[cluster_id][0]:
                closest[cluster_id] = (dist, i)

        assignments.append(
            ClusterAssignment(
                sequence_id=seq_id,
                cluster_id=int(cluster_id),
                is_representative=False,
                distance_to_centroid=dist,
            )
        )

    # Mark representatives
    for cluster_id, (_, idx) in closest.items():
        assignments[idx].is_representative = True

    return assignments
