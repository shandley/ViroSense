"""MLX backend for Evo2 inference on Apple Silicon.

Runs the Evo2 7B model locally using Apple's MLX framework.
No NVIDIA GPU required — uses unified memory on M-series chips.

Requires:
  - Apple Silicon Mac (M1+)
  - mlx>=0.20, safetensors>=0.4
  - Model weights: run `python scripts/download_evo2_weights.py`
"""

from __future__ import annotations

import random

import numpy as np
from loguru import logger

from virosense.backends.base import EmbeddingRequest, EmbeddingResult, Evo2Backend
from virosense.utils.constants import (
    MLX_MAX_CONTEXT_LENGTH,
    MLX_MODEL_DIR,
)


class MLXBackend(Evo2Backend):
    """Evo2 inference on Apple Silicon via MLX.

    Loads the Evo2 7B model in float16 and extracts mean-pooled
    embeddings from a specified hidden layer. Model is loaded lazily
    on the first call to extract_embeddings().
    """

    def __init__(self, model: str = "evo2_7b", model_dir: str | None = None):
        self.model_name = model
        self._model_dir = model_dir or str(MLX_MODEL_DIR)
        self._model = None  # lazy load

    def _ensure_model(self) -> None:
        """Load model weights if not already loaded."""
        if self._model is not None:
            return

        import mlx.core as mx

        from virosense.backends.mlx_model import Evo2Model, load_weights

        logger.info("Loading Evo2 7B model into MLX...")
        self._model = Evo2Model()
        load_weights(self._model, self._model_dir)
        mx.eval(self._model.parameters())  # materialize all weights
        logger.info("Evo2 7B model loaded successfully")

    def extract_embeddings(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Extract embeddings using local MLX inference.

        Args:
            request: EmbeddingRequest with sequences and layer specification.

        Returns:
            EmbeddingResult with (N, embed_dim) mean-pooled embeddings.
        """
        from virosense.backends.mlx_model import tokenize_dna

        self._ensure_model()

        sanitized = self._sanitize_sequences(request.sequences)
        layer = request.layer
        n_seqs = len(sanitized)

        logger.info(
            f"Extracting embeddings for {n_seqs} sequences "
            f"via MLX (layer: {layer})"
        )

        embeddings = []
        for i, (_seq_id, sequence) in enumerate(sanitized.items(), 1):
            tokens = tokenize_dna(sequence)
            embedding = self._model(tokens, extract_layer=layer)
            embeddings.append(np.array(embedding))

            if i % 10 == 0 or i == n_seqs:
                logger.info(f"  Progress: {i}/{n_seqs} sequences")

        sequence_ids = list(sanitized.keys())
        embeddings_matrix = np.stack(embeddings).astype(np.float32)

        logger.info(
            f"Extracted embeddings: {embeddings_matrix.shape} "
            f"({len(sequence_ids)} sequences)"
        )

        return EmbeddingResult(
            sequence_ids=sequence_ids,
            embeddings=embeddings_matrix,
            layer=layer,
            model=request.model,
        )

    @staticmethod
    def _sanitize_sequences(sequences: dict[str, str]) -> dict[str, str]:
        """Validate and sanitize DNA sequences.

        Same logic as NIM backend: uppercase, validate ACGTN, replace N
        with random bases. No length limit enforced (MLX handles long seqs
        up to memory limits).
        """
        valid_bases = set("ACGTN")
        sanitized = {}
        for seq_id, seq in sequences.items():
            seq = seq.upper()
            if len(seq) == 0:
                raise ValueError(f"Sequence {seq_id} is empty.")
            invalid = set(seq) - valid_bases
            if invalid:
                raise ValueError(
                    f"Sequence {seq_id} contains invalid characters: "
                    f"{invalid}. Only A, C, G, T, N are allowed."
                )
            n_count = seq.count("N")
            if n_count > 0:
                logger.debug(
                    f"Replacing {n_count} N bases in {seq_id} "
                    f"({100 * n_count / len(seq):.1f}%)"
                )
                seq = "".join(
                    random.choice("ACGT") if c == "N" else c for c in seq
                )
            sanitized[seq_id] = seq
        return sanitized

    def is_available(self) -> bool:
        """Check if MLX and model weights are available."""
        try:
            import mlx  # noqa: F401
        except ImportError:
            return False

        from pathlib import Path
        weight_path = Path(self._model_dir) / "model.safetensors"
        return weight_path.exists()

    def max_context_length(self) -> int:
        """Return max context length (memory-limited, not architecture-limited)."""
        return MLX_MAX_CONTEXT_LENGTH
