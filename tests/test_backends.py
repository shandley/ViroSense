"""Tests for Evo2 backends."""

from virosense.backends.base import (
    EmbeddingRequest,
    EmbeddingResult,
    Evo2Backend,
    get_backend,
)
from virosense.backends.nim import NIMBackend
from virosense.backends.local import LocalBackend
from virosense.backends.modal import ModalBackend


def test_get_backend_nim():
    """Test factory returns NIMBackend."""
    backend = get_backend("nim")
    assert isinstance(backend, NIMBackend)


def test_get_backend_local():
    """Test factory returns LocalBackend."""
    backend = get_backend("local")
    assert isinstance(backend, LocalBackend)


def test_get_backend_modal():
    """Test factory returns ModalBackend."""
    backend = get_backend("modal")
    assert isinstance(backend, ModalBackend)


def test_nim_max_context():
    """Test NIM backend max context length."""
    backend = NIMBackend(api_key="test", model="evo2_7b")
    assert backend.max_context_length() == 1_000_000


def test_nim_availability_with_key():
    """Test NIM backend availability check."""
    backend = NIMBackend(api_key="test_key")
    assert backend.is_available() is True


def test_nim_availability_without_key():
    """Test NIM backend unavailable without API key."""
    backend = NIMBackend(api_key=None)
    assert backend.is_available() is False


def test_embedding_request():
    """Test EmbeddingRequest creation."""
    req = EmbeddingRequest(
        sequences={"seq1": "ATGC", "seq2": "GCTA"},
        layer="blocks.28.mlp.l3",
        model="evo2_7b",
    )
    assert len(req.sequences) == 2
    assert req.layer == "blocks.28.mlp.l3"


def test_mock_backend(mock_backend):
    """Test mock backend from fixtures."""
    assert mock_backend.is_available() is True

    req = EmbeddingRequest(sequences={"s1": "ATGC", "s2": "GCTA"})
    result = mock_backend.extract_embeddings(req)
    assert len(result.sequence_ids) == 2
    assert result.embeddings.shape == (2, 4096)
