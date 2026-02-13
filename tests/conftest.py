"""Shared test fixtures for virosense."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from virosense.backends.base import EmbeddingRequest, EmbeddingResult, Evo2Backend


TEST_DATA_DIR = Path(__file__).parent / "test_data"


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return TEST_DATA_DIR


@pytest.fixture
def small_contigs_fasta(test_data_dir):
    """Path to small test contigs FASTA."""
    return test_data_dir / "small_contigs.fasta"


@pytest.fixture
def mock_backend():
    """Mock Evo2 backend that returns random embeddings."""
    backend = MagicMock(spec=Evo2Backend)
    backend.is_available.return_value = True
    backend.max_context_length.return_value = 1_000_000

    def mock_extract(request: EmbeddingRequest) -> EmbeddingResult:
        n = len(request.sequences)
        return EmbeddingResult(
            sequence_ids=list(request.sequences.keys()),
            embeddings=np.random.randn(n, 4096).astype(np.float32),
            layer=request.layer,
            model=request.model,
        )

    backend.extract_embeddings.side_effect = mock_extract
    return backend


@pytest.fixture
def mock_embeddings():
    """Generate mock embeddings for testing."""
    rng = np.random.default_rng(42)
    ids = ["seq_1", "seq_2", "seq_3", "seq_4", "seq_5"]
    embeddings = rng.standard_normal((5, 4096)).astype(np.float32)
    return ids, embeddings


@pytest.fixture
def tmp_output(tmp_path):
    """Temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out
