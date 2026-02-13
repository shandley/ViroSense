"""Embedding cache I/O (NPZ format)."""

from pathlib import Path

import numpy as np
from loguru import logger


def save_embeddings(
    path: Path,
    sequence_ids: list[str],
    embeddings: np.ndarray,
    metadata: dict | None = None,
) -> None:
    """Save embeddings to NPZ file.

    Args:
        path: Output path (.npz).
        sequence_ids: List of sequence identifiers.
        embeddings: (N, dim) embedding matrix.
        metadata: Optional metadata dict to store.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "sequence_ids": np.array(sequence_ids),
        "embeddings": embeddings,
    }
    if metadata:
        save_dict["metadata"] = np.array([metadata])

    np.savez(path, **save_dict)
    logger.info(f"Saved {len(sequence_ids)} embeddings to {path}")


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    """Load embeddings from NPZ file.

    Args:
        path: Path to NPZ file.

    Returns:
        Tuple of (sequence_ids, embeddings_matrix).
    """
    data = np.load(path, allow_pickle=True)
    sequence_ids = list(data["sequence_ids"])
    embeddings = data["embeddings"]
    logger.info(f"Loaded {len(sequence_ids)} embeddings from {path}")
    return sequence_ids, embeddings
