"""Detect viral sequences in metagenomic contigs."""

from pathlib import Path

from loguru import logger


def run_detect(
    input_file: str,
    output_dir: str,
    backend: str = "nim",
    model: str = "evo2_7b",
    threshold: float = 0.5,
    min_length: int = 500,
    batch_size: int = 16,
    threads: int = 4,
    layer: str = "blocks.28.mlp.l3",
    cache_dir: str | None = None,
) -> None:
    """Run viral detection pipeline.

    1. Read metagenomic contigs from FASTA
    2. Filter by minimum length
    3. Extract Evo2 embeddings via selected backend
    4. Classify contigs as viral/cellular using trained classifier
    5. Write results TSV
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Detecting viral sequences in {input_file}")
    logger.info(f"Backend: {backend}, Model: {model}")
    logger.info(f"Threshold: {threshold}, Min length: {min_length}")

    raise NotImplementedError(
        "detect pipeline not yet implemented. See Phase 4 in the plan."
    )
