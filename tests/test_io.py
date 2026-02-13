"""Tests for I/O utilities."""

import numpy as np

from virosense.io.fasta import read_fasta, filter_by_length
from virosense.io.embeddings import save_embeddings, load_embeddings


def test_read_fasta(small_contigs_fasta):
    """Test reading a FASTA file."""
    sequences = read_fasta(small_contigs_fasta)
    assert len(sequences) == 3
    assert "contig_1" in sequences
    assert "contig_2" in sequences
    assert "contig_short" in sequences


def test_filter_by_length(small_contigs_fasta):
    """Test filtering sequences by minimum length."""
    sequences = read_fasta(small_contigs_fasta)
    filtered = filter_by_length(sequences, min_length=500)
    assert "contig_short" not in filtered
    assert len(filtered) == 2


def test_embeddings_roundtrip(tmp_output):
    """Test saving and loading embeddings."""
    ids = ["seq_a", "seq_b", "seq_c"]
    embeddings = np.random.randn(3, 4096).astype(np.float32)

    path = tmp_output / "test_embeddings.npz"
    save_embeddings(path, ids, embeddings)

    loaded_ids, loaded_embeddings = load_embeddings(path)
    assert loaded_ids == ids
    np.testing.assert_array_almost_equal(loaded_embeddings, embeddings)
