"""Annotate ORFs with genomic context from Evo2 embeddings."""

from pathlib import Path

from loguru import logger


def run_context(
    input_file: str,
    orfs_file: str,
    output_dir: str,
    backend: str = "nim",
    model: str = "evo2_7b",
    window_size: int = 2000,
    vhold_output: str | None = None,
    threads: int = 4,
) -> None:
    """Run genomic context annotation pipeline.

    1. Read viral contigs and ORF predictions
    2. Extract Evo2 embeddings for genomic windows around each ORF
    3. Optionally merge with vHold protein-level annotations
    4. Write enhanced annotation results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Annotating ORFs with genomic context from {input_file}")
    logger.info(f"ORF predictions: {orfs_file}")
    logger.info(f"Backend: {backend}, Window size: {window_size}")

    raise NotImplementedError(
        "context pipeline not yet implemented. See Phase 7 in the plan."
    )
