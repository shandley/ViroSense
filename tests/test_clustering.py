"""Tests for clustering utilities."""

import numpy as np

from virosense.clustering.multimodal import fuse_embeddings
from virosense.clustering.metrics import cluster_summary


def test_fuse_dna_only():
    """Test DNA-only mode returns DNA embeddings."""
    dna = np.random.randn(5, 4096).astype(np.float32)
    result = fuse_embeddings(dna, mode="dna")
    np.testing.assert_array_equal(result, dna)


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


def test_cluster_summary():
    """Test cluster summary statistics."""
    labels = np.array([0, 0, 0, 1, 1, -1, -1])
    summary = cluster_summary(labels)
    assert summary["n_clusters"] == 2
    assert summary["n_noise"] == 2
    assert summary["n_total"] == 7
    assert sorted(summary["cluster_sizes"]) == [2, 3]
