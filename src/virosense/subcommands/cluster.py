"""Cluster unclassified viral sequences using multi-modal embeddings."""

from pathlib import Path

from loguru import logger


def run_cluster(
    input_file: str,
    output_dir: str,
    backend: str = "nim",
    model: str = "evo2_7b",
    mode: str = "multi",
    algorithm: str = "hdbscan",
    min_cluster_size: int = 5,
    threads: int = 4,
    vhold_embeddings: str | None = None,
) -> None:
    """Run multi-modal clustering pipeline.

    1. Read unclassified viral sequences
    2. Extract DNA embeddings (Evo2) and optionally protein embeddings (ProstT5)
    3. Fuse embeddings based on selected mode
    4. Cluster using selected algorithm
    5. Write cluster assignments and quality metrics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Clustering viral sequences from {input_file}")
    logger.info(f"Mode: {mode}, Algorithm: {algorithm}")
    logger.info(f"Backend: {backend}, Min cluster size: {min_cluster_size}")

    raise NotImplementedError(
        "cluster pipeline not yet implemented. See Phase 6 in the plan."
    )
