"""Tests for clustering utilities."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from virosense.clustering.metrics import cluster_summary, silhouette_score
from virosense.clustering.multimodal import (
    ClusterAssignment,
    _build_assignments,
    _estimate_k,
    _reduce_pca,
    _renumber_labels,
    cluster_sequences,
    fuse_embeddings,
)


# --- Embedding fusion tests ---


def test_fuse_dna_only():
    """Test DNA-only mode returns DNA embeddings."""
    dna = np.random.randn(5, 4096).astype(np.float32)
    result = fuse_embeddings(dna, mode="dna")
    np.testing.assert_array_equal(result, dna)


def test_fuse_protein_only():
    """Test protein-only mode returns protein embeddings."""
    dna = np.random.randn(5, 4096).astype(np.float32)
    prot = np.random.randn(5, 1024).astype(np.float32)
    result = fuse_embeddings(dna, prot, mode="protein")
    np.testing.assert_array_equal(result, prot)


def test_fuse_protein_only_missing():
    """Test protein-only mode raises when no protein embeddings."""
    dna = np.random.randn(5, 4096).astype(np.float32)
    with pytest.raises(ValueError, match="Protein embeddings required"):
        fuse_embeddings(dna, None, mode="protein")


def test_fuse_multi():
    """Test multi-modal concatenation."""
    dna = np.random.randn(5, 4096).astype(np.float32)
    prot = np.random.randn(5, 1024).astype(np.float32)
    result = fuse_embeddings(dna, prot, mode="multi")
    assert result.shape == (5, 5120)


def test_fuse_multi_no_protein():
    """Test multi mode falls back to DNA when no protein embeddings."""
    dna = np.random.randn(5, 4096).astype(np.float32)
    result = fuse_embeddings(dna, None, mode="multi")
    np.testing.assert_array_equal(result, dna)


def test_fuse_invalid_mode():
    """Test invalid mode raises ValueError."""
    dna = np.random.randn(5, 4096).astype(np.float32)
    with pytest.raises(ValueError, match="Unknown mode"):
        fuse_embeddings(dna, mode="invalid")


# --- Clustering tests ---


def _make_clusterable_data(n_per_cluster=20, n_clusters=3, dim=64):
    """Generate synthetic data with clear cluster structure."""
    rng = np.random.RandomState(42)
    embeddings = []
    ids = []
    for c in range(n_clusters):
        center = rng.randn(dim) * 10
        points = center + rng.randn(n_per_cluster, dim) * 0.5
        embeddings.append(points)
        ids.extend([f"seq_{c}_{i}" for i in range(n_per_cluster)])
    return np.vstack(embeddings).astype(np.float32), ids


def test_cluster_hdbscan():
    """Test HDBSCAN clustering finds clusters in well-separated data."""
    embeddings, ids = _make_clusterable_data(n_per_cluster=20, n_clusters=3)
    assignments = cluster_sequences(
        embeddings, ids, algorithm="hdbscan", min_cluster_size=5
    )
    assert len(assignments) == 60
    assert all(isinstance(a, ClusterAssignment) for a in assignments)
    # Should find at least 2 clusters in well-separated data
    cluster_ids = {a.cluster_id for a in assignments if a.cluster_id >= 0}
    assert len(cluster_ids) >= 2


def test_cluster_kmeans():
    """Test KMeans clustering with explicit k."""
    embeddings, ids = _make_clusterable_data(n_per_cluster=15, n_clusters=3)
    assignments = cluster_sequences(
        embeddings, ids, algorithm="kmeans", n_clusters=3
    )
    assert len(assignments) == 45
    cluster_ids = {a.cluster_id for a in assignments}
    assert len(cluster_ids) == 3
    # KMeans never produces noise
    assert all(a.cluster_id >= 0 for a in assignments)


def test_cluster_kmeans_auto_k():
    """Test KMeans with automatic k estimation."""
    embeddings, ids = _make_clusterable_data(n_per_cluster=15, n_clusters=3)
    assignments = cluster_sequences(
        embeddings, ids, algorithm="kmeans"
    )
    assert len(assignments) == 45
    assert all(a.cluster_id >= 0 for a in assignments)


def test_cluster_invalid_algorithm():
    """Test invalid algorithm raises ValueError."""
    embeddings, ids = _make_clusterable_data()
    with pytest.raises(ValueError, match="Unknown algorithm"):
        cluster_sequences(embeddings, ids, algorithm="invalid")


def test_cluster_mismatched_ids():
    """Test mismatched sequence IDs and embeddings raises ValueError."""
    embeddings = np.random.randn(5, 64).astype(np.float32)
    ids = ["a", "b", "c"]  # only 3 IDs for 5 embeddings
    with pytest.raises(ValueError, match="Mismatch"):
        cluster_sequences(embeddings, ids)


def test_cluster_representatives():
    """Test that each cluster has exactly one representative."""
    embeddings, ids = _make_clusterable_data(n_per_cluster=15, n_clusters=3)
    assignments = cluster_sequences(
        embeddings, ids, algorithm="kmeans", n_clusters=3
    )
    # Each cluster should have exactly one representative
    for cid in range(3):
        reps = [a for a in assignments if a.cluster_id == cid and a.is_representative]
        assert len(reps) == 1


def test_cluster_centroid_distances():
    """Test that centroid distances are non-negative."""
    embeddings, ids = _make_clusterable_data()
    assignments = cluster_sequences(
        embeddings, ids, algorithm="kmeans", n_clusters=3
    )
    for a in assignments:
        assert a.distance_to_centroid >= 0.0


# --- PCA reduction tests ---


def test_cluster_with_pca_auto():
    """Test clustering with auto PCA (default, pca_dims=0)."""
    embeddings, ids = _make_clusterable_data(n_per_cluster=20, n_clusters=3, dim=512)
    assignments = cluster_sequences(
        embeddings, ids, algorithm="hdbscan", min_cluster_size=5, pca_dims=0
    )
    assert len(assignments) == 60
    cluster_ids = {a.cluster_id for a in assignments if a.cluster_id >= 0}
    assert len(cluster_ids) >= 2


def test_cluster_with_pca_explicit():
    """Test clustering with explicit PCA dimensions."""
    embeddings, ids = _make_clusterable_data(n_per_cluster=20, n_clusters=3, dim=512)
    assignments = cluster_sequences(
        embeddings, ids, algorithm="kmeans", n_clusters=3, pca_dims=10
    )
    assert len(assignments) == 60
    cluster_ids = {a.cluster_id for a in assignments}
    assert len(cluster_ids) == 3


def test_cluster_without_pca():
    """Test clustering with PCA disabled (pca_dims=None)."""
    embeddings, ids = _make_clusterable_data(n_per_cluster=20, n_clusters=3, dim=64)
    assignments = cluster_sequences(
        embeddings, ids, algorithm="hdbscan", min_cluster_size=5, pca_dims=None
    )
    assert len(assignments) == 60
    cluster_ids = {a.cluster_id for a in assignments if a.cluster_id >= 0}
    assert len(cluster_ids) >= 2


def test_reduce_pca_auto():
    """Test auto PCA selects components for 90% variance."""
    rng = np.random.RandomState(42)
    # Create data with clear low-rank structure
    n, dim = 50, 200
    low_rank = rng.randn(n, 5) @ rng.randn(5, dim)
    noise = rng.randn(n, dim) * 0.1
    data = StandardScaler().fit_transform(low_rank + noise)
    reduced = _reduce_pca(data, n_components=0)
    # Should select a small number of components
    assert reduced.shape[0] == n
    assert reduced.shape[1] <= 20


def test_reduce_pca_explicit():
    """Test explicit PCA dimensions."""
    rng = np.random.RandomState(42)
    data = StandardScaler().fit_transform(rng.randn(30, 100))
    reduced = _reduce_pca(data, n_components=10)
    assert reduced.shape == (30, 10)


def test_reduce_pca_caps_at_samples():
    """Test PCA caps components at n_samples when n_samples < n_features."""
    rng = np.random.RandomState(42)
    data = StandardScaler().fit_transform(rng.randn(10, 500))
    reduced = _reduce_pca(data, n_components=50)
    assert reduced.shape[1] <= 10


# --- Helper function tests ---


def test_renumber_labels():
    """Test label renumbering preserves -1 and makes contiguous."""
    labels = np.array([5, 5, -1, 10, 10, 10, -1, 3])
    result = _renumber_labels(labels)
    assert set(result[result >= 0]) == {0, 1, 2}
    assert (result == -1).sum() == 2


def test_estimate_k():
    """Test k estimation returns reasonable value."""
    embeddings, _ = _make_clusterable_data(n_per_cluster=15, n_clusters=3)
    k = _estimate_k(embeddings)
    assert 2 <= k <= 10  # Should be near 3 for well-separated data


def test_estimate_k_small():
    """Test k estimation with very few points."""
    embeddings = np.random.randn(4, 10).astype(np.float32)
    k = _estimate_k(embeddings)
    assert k == 2  # Should return minimum


def test_build_assignments_noise():
    """Test build_assignments handles noise points correctly."""
    embeddings = np.random.randn(5, 10).astype(np.float32)
    ids = ["a", "b", "c", "d", "e"]
    labels = np.array([0, 0, -1, 1, 1])
    assignments = _build_assignments(embeddings, ids, labels)
    noise = [a for a in assignments if a.cluster_id == -1]
    assert len(noise) == 1
    assert noise[0].sequence_id == "c"
    assert noise[0].is_representative is False
    assert noise[0].distance_to_centroid == 0.0


# --- Metrics tests ---


def test_cluster_summary():
    """Test cluster summary statistics."""
    labels = np.array([0, 0, 0, 1, 1, -1, -1])
    summary = cluster_summary(labels)
    assert summary["n_clusters"] == 2
    assert summary["n_noise"] == 2
    assert summary["n_total"] == 7
    assert sorted(summary["cluster_sizes"]) == [2, 3]


def test_silhouette_score_valid():
    """Test silhouette score with valid clusters."""
    embeddings, _ = _make_clusterable_data(n_per_cluster=15, n_clusters=3)
    labels = np.array([0] * 15 + [1] * 15 + [2] * 15)
    score = silhouette_score(embeddings, labels)
    assert -1.0 <= score <= 1.0
    # Well-separated data should have high silhouette
    assert score > 0.5


def test_silhouette_score_with_noise():
    """Test silhouette score excludes noise points."""
    embeddings, _ = _make_clusterable_data(n_per_cluster=15, n_clusters=2)
    labels = np.array([0] * 15 + [1] * 15)
    labels[0] = -1  # Mark one as noise
    score = silhouette_score(embeddings, labels)
    assert -1.0 <= score <= 1.0


def test_silhouette_score_insufficient():
    """Test silhouette score returns 0 with too few valid points."""
    embeddings = np.random.randn(3, 10).astype(np.float32)
    labels = np.array([-1, -1, 0])
    score = silhouette_score(embeddings, labels)
    assert score == 0.0
