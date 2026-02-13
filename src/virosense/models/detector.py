"""Viral vs cellular classifier head on frozen Evo2 embeddings."""

from dataclasses import dataclass, field

import numpy as np
from loguru import logger


@dataclass
class ClassifierConfig:
    """Configuration for the classification head."""

    input_dim: int = 4096  # Evo2 7B embedding dim
    hidden_dims: list[int] = field(default_factory=lambda: [512, 128])
    num_classes: int = 2
    dropout: float = 0.1


@dataclass
class DetectionResult:
    """Result for a single contig classification."""

    contig_id: str
    contig_length: int
    viral_score: float  # 0.0-1.0
    classification: str  # "viral", "cellular", "ambiguous"


def classify_contigs(
    embeddings: np.ndarray,
    sequence_ids: list[str],
    sequence_lengths: list[int],
    threshold: float = 0.5,
) -> list[DetectionResult]:
    """Classify contigs as viral or cellular using embeddings.

    Uses scikit-learn classifier on frozen Evo2 embeddings.

    Args:
        embeddings: (N, embed_dim) embedding matrix.
        sequence_ids: List of contig identifiers.
        sequence_lengths: List of contig lengths in bp.
        threshold: Score threshold for viral classification.

    Returns:
        List of DetectionResult for each contig.
    """
    raise NotImplementedError(
        "Classifier not yet implemented. See Phase 4 in the plan."
    )
