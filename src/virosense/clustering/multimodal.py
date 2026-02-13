"""Multi-modal embedding fusion and clustering."""

from dataclasses import dataclass

import numpy as np
from loguru import logger


@dataclass
class ClusterAssignment:
    """Cluster assignment for a single sequence."""

    sequence_id: str
    cluster_id: int  # -1 for noise
    is_representative: bool
    distance_to_centroid: float


def fuse_embeddings(
    dna_embeddings: np.ndarray,
    protein_embeddings: np.ndarray | None = None,
    mode: str = "multi",
) -> np.ndarray:
    """Fuse DNA and protein embeddings.

    Args:
        dna_embeddings: (N, dna_dim) Evo2 embeddings.
        protein_embeddings: (N, prot_dim) ProstT5 embeddings, optional.
        mode: "dna" (DNA only), "protein" (protein only), "multi" (concatenate).

    Returns:
        (N, fused_dim) fused embedding matrix.
    """
    if mode == "dna":
        return dna_embeddings
    elif mode == "protein":
        if protein_embeddings is None:
            raise ValueError("Protein embeddings required for mode='protein'")
        return protein_embeddings
    elif mode == "multi":
        if protein_embeddings is None:
            logger.warning("No protein embeddings provided, falling back to DNA only")
            return dna_embeddings
        return np.concatenate([dna_embeddings, protein_embeddings], axis=1)
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from: dna, protein, multi")


def cluster_sequences(
    embeddings: np.ndarray,
    sequence_ids: list[str],
    algorithm: str = "hdbscan",
    min_cluster_size: int = 5,
) -> list[ClusterAssignment]:
    """Cluster sequences based on fused embeddings.

    Args:
        embeddings: (N, dim) embedding matrix.
        sequence_ids: List of sequence identifiers.
        algorithm: Clustering algorithm (hdbscan, leiden, kmeans).
        min_cluster_size: Minimum cluster size.

    Returns:
        List of ClusterAssignment for each sequence.
    """
    raise NotImplementedError(
        "Clustering not yet implemented. See Phase 6 in the plan."
    )
