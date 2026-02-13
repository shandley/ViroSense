"""NVIDIA NIM API backend for Evo2 inference."""

import numpy as np
from loguru import logger

from virosense.backends.base import EmbeddingRequest, EmbeddingResult, Evo2Backend
from virosense.utils.constants import (
    EVO2_MODELS,
    NIM_BASE_URL,
    NIM_FORWARD_ENDPOINT,
    get_nvidia_api_key,
)


class NIMBackend(Evo2Backend):
    """Evo2 inference via NVIDIA NIM API.

    Default backend — works on any machine with internet access.
    Requires NVIDIA_API_KEY environment variable.
    """

    def __init__(self, api_key: str | None = None, model: str = "evo2_7b"):
        self.api_key = api_key or get_nvidia_api_key()
        self.model = model
        self._base_url = NIM_BASE_URL

    def extract_embeddings(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Extract embeddings via NIM API."""
        raise NotImplementedError(
            "NIM backend not yet implemented. See Phase 2 in the plan."
        )

    def is_available(self) -> bool:
        """Check if NIM API is accessible."""
        return self.api_key is not None

    def max_context_length(self) -> int:
        """Return max context length for the configured model."""
        return EVO2_MODELS.get(self.model, {}).get("max_context", 8192)
