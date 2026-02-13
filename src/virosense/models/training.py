"""Training loop for classifier heads on Evo2 embeddings."""

from pathlib import Path

import numpy as np
from loguru import logger


def train_classifier(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    epochs: int = 50,
    lr: float = 1e-3,
    val_split: float = 0.2,
    task: str = "viral_vs_cellular",
) -> dict:
    """Train a classification head on frozen embeddings.

    Args:
        embeddings: (N, embed_dim) training embeddings.
        labels: (N,) integer labels.
        output_dir: Directory to save model and metrics.
        epochs: Number of training epochs.
        lr: Learning rate.
        val_split: Fraction for validation set.
        task: Classification task name.

    Returns:
        Dict with training metrics (accuracy, loss, etc.).
    """
    raise NotImplementedError(
        "Training loop not yet implemented. See Phase 8 in the plan."
    )
