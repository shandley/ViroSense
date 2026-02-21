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

    Loads the Evo2 7B model and extracts mean-pooled embeddings from
    a specified hidden layer. Model is loaded lazily on the first call
    to extract_embeddings().

    Args:
        model: Model name (currently only "evo2_7b" supported).
        model_dir: Path to directory containing model.safetensors.
        quantize: Bit width for weight quantization (4 or 8). None for fp16.
            4-bit gives ~1.3x speedup and 4x memory reduction with 0.96 cosine
            similarity to fp16. 8-bit is near-lossless with 2x memory reduction.
    """

    def __init__(
        self,
        model: str = "evo2_7b",
        model_dir: str | None = None,
        quantize: int | None = 4,
    ):
        self.model_name = model
        self._model_dir = model_dir or str(MLX_MODEL_DIR)
        self._quantize = quantize
        self._model = None  # lazy load

    def _ensure_model(self) -> None:
        """Load model weights if not already loaded."""
        if self._model is not None:
            return

        import mlx.core as mx
        import mlx.nn as nn

        from virosense.backends.mlx_model import Evo2Model, load_weights

        logger.info("Loading Evo2 7B model into MLX...")
        self._model = Evo2Model()
        load_weights(self._model, self._model_dir)

        if self._quantize is not None:
            logger.info(f"Quantizing model to {self._quantize}-bit...")
            nn.quantize(self._model, bits=self._quantize, group_size=64)

        mx.eval(self._model.parameters())  # materialize all weights
        logger.info("Evo2 7B model loaded successfully")

    def extract_embeddings(
        self, request: EmbeddingRequest, batch_size: int = 4
    ) -> EmbeddingResult:
        """Extract embeddings using local MLX inference.

        Groups sequences by length and processes same-length sequences
        in batches for improved throughput. Variable-length sequences
        are processed individually.

        Args:
            request: EmbeddingRequest with sequences and layer specification.
            batch_size: Max sequences per batch (same-length only).

        Returns:
            EmbeddingResult with (N, embed_dim) mean-pooled embeddings.
        """
        import mlx.core as mx

        from virosense.backends.mlx_model import tokenize_dna

        self._ensure_model()

        sanitized = self._sanitize_sequences(request.sequences)
        layer = request.layer
        n_seqs = len(sanitized)

        logger.info(
            f"Extracting embeddings for {n_seqs} sequences "
            f"via MLX (layer: {layer}, batch_size: {batch_size})"
        )

        # Group sequences by length for batched inference
        length_groups: dict[int, list[tuple[str, str]]] = {}
        for seq_id, sequence in sanitized.items():
            seq_len = len(sequence)
            length_groups.setdefault(seq_len, []).append((seq_id, sequence))

        n_groups = len(length_groups)
        if n_groups < n_seqs:
            logger.info(
                f"  Grouped into {n_groups} length buckets "
                f"(largest batch: {max(len(g) for g in length_groups.values())})"
            )

        # Process each length group in batches
        result_map: dict[str, np.ndarray] = {}
        completed = 0

        for _seq_len, group in length_groups.items():
            for batch_start in range(0, len(group), batch_size):
                batch = group[batch_start:batch_start + batch_size]

                if len(batch) == 1:
                    # Single sequence — no batching overhead
                    seq_id, sequence = batch[0]
                    tokens = tokenize_dna(sequence)
                    embedding = self._model(tokens, extract_layer=layer)
                    result_map[seq_id] = np.array(embedding)
                else:
                    # Batch: stack tokens into (B, L)
                    token_list = [tokenize_dna(seq) for _, seq in batch]
                    tokens_batched = mx.stack(token_list)  # (B, L)
                    embeddings_batch = self._model(
                        tokens_batched, extract_layer=layer
                    )  # (B, D)
                    embeddings_np = np.array(embeddings_batch)
                    for j, (seq_id, _) in enumerate(batch):
                        result_map[seq_id] = embeddings_np[j]

                completed += len(batch)
                if completed % 10 == 0 or completed == n_seqs:
                    logger.info(f"  Progress: {completed}/{n_seqs} sequences")

        # Preserve original order
        sequence_ids = list(sanitized.keys())
        embeddings_matrix = np.stack(
            [result_map[sid] for sid in sequence_ids]
        ).astype(np.float32)

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
                bases = b"ACGT"
                buf = bytearray(seq, "ascii")
                for i in range(len(buf)):
                    if buf[i] == 78:  # ord('N')
                        buf[i] = bases[random.randint(0, 3)]
                seq = buf.decode("ascii")
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
