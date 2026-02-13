"""Local GPU backend for Evo2 inference."""

from virosense.backends.base import EmbeddingRequest, EmbeddingResult, Evo2Backend
from virosense.utils.constants import EVO2_MODELS


class LocalBackend(Evo2Backend):
    """Evo2 inference on local GPU.

    Requires NVIDIA GPU (H100/Ada+) with CUDA 12.1+ and FP8 support.
    Uses the evo2 Python package directly.
    """

    def __init__(self, model: str = "evo2_7b", device: str = "cuda:0"):
        self.model = model
        self.device = device

    def extract_embeddings(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Extract embeddings using local GPU."""
        raise NotImplementedError(
            "Local backend requires NVIDIA GPU with CUDA 12.1+. "
            "Use --backend nim for API-based inference."
        )

    def is_available(self) -> bool:
        """Check if local GPU with evo2 is available."""
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            import evo2  # noqa: F401
            return True
        except ImportError:
            return False

    def max_context_length(self) -> int:
        """Return max context length for the configured model."""
        return EVO2_MODELS.get(self.model, {}).get("max_context", 8192)
