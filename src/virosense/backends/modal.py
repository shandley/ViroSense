"""Modal.com serverless GPU backend for Evo2 inference."""

from virosense.backends.base import EmbeddingRequest, EmbeddingResult, Evo2Backend
from virosense.utils.constants import EVO2_MODELS


class ModalBackend(Evo2Backend):
    """Evo2 inference via Modal.com serverless GPUs.

    Provides on-demand GPU access without managing infrastructure.
    Requires Modal account and authentication.
    """

    def __init__(self, model: str = "evo2_7b"):
        self.model = model

    def extract_embeddings(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Extract embeddings via Modal serverless GPU."""
        raise NotImplementedError(
            "Modal backend not yet implemented. See Phase 2 in the plan."
        )

    def is_available(self) -> bool:
        """Check if Modal is configured."""
        try:
            import modal  # noqa: F401
            return True
        except ImportError:
            return False

    def max_context_length(self) -> int:
        """Return max context length for the configured model."""
        return EVO2_MODELS.get(self.model, {}).get("max_context", 8192)
