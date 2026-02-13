"""Abstract base class for Evo2 inference backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EmbeddingRequest:
    """Request for Evo2 embedding extraction."""

    sequences: dict[str, str]  # id -> DNA sequence
    layer: str = "blocks.28.mlp.l3"
    model: str = "evo2_7b"


@dataclass
class EmbeddingResult:
    """Result from Evo2 embedding extraction."""

    sequence_ids: list[str]
    embeddings: np.ndarray  # (N, embed_dim) float32
    layer: str = "blocks.28.mlp.l3"
    model: str = "evo2_7b"


class Evo2Backend(ABC):
    """Abstract base class for Evo2 inference backends."""

    @abstractmethod
    def extract_embeddings(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Extract embeddings for a batch of DNA sequences."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        ...

    @abstractmethod
    def max_context_length(self) -> int:
        """Return maximum supported context length in nucleotides."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(available={self.is_available()})"


def get_backend(name: str, **kwargs) -> Evo2Backend:
    """Factory function to get an Evo2 backend by name."""
    if name == "nim":
        from virosense.backends.nim import NIMBackend
        return NIMBackend(**kwargs)
    elif name == "local":
        from virosense.backends.local import LocalBackend
        return LocalBackend(**kwargs)
    elif name == "modal":
        from virosense.backends.modal import ModalBackend
        return ModalBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {name}. Choose from: nim, local, modal")
