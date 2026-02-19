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
    nim_url: str | None = None,
) -> None:
    """Run viral detection pipeline.

    1. Read metagenomic contigs from FASTA
    2. Filter by minimum length
    3. Extract Evo2 embeddings via selected backend
    4. Classify contigs as viral/cellular using trained classifier
    5. Write results TSV
    """
    from virosense.backends.base import get_backend
    from virosense.features.evo2_embeddings import extract_embeddings
    from virosense.io.fasta import filter_by_length, read_fasta
    from virosense.io.results import write_tsv
    from virosense.models.detector import (
        ViralClassifier,
        classify_contigs,
        get_default_model_path,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Detecting viral sequences in {input_file}")
    logger.info(f"Backend: {backend}, Model: {model}")
    logger.info(f"Threshold: {threshold}, Min length: {min_length}")

    # 1. Read and filter sequences
    sequences = read_fasta(input_file)
    sequences = filter_by_length(sequences, min_length)
    if not sequences:
        logger.warning("No sequences remaining after length filter.")
        return

    # 2. Load classifier
    classifier = _load_classifier(None)

    # 3. Get backend and extract embeddings
    evo2_backend = get_backend(backend, api_key=None, model=model, nim_url=nim_url)
    if not evo2_backend.is_available():
        raise RuntimeError(
            f"Backend {backend!r} is not available. "
            "Check your configuration (e.g., NVIDIA_API_KEY for nim)."
        )

    cache_path = Path(cache_dir) if cache_dir else None
    result = extract_embeddings(
        sequences=sequences,
        backend=evo2_backend,
        layer=layer,
        model=model,
        batch_size=batch_size,
        cache_dir=cache_path,
    )

    # 4. Classify contigs
    sequence_lengths = [len(sequences[sid]) for sid in result.sequence_ids]
    detection_results = classify_contigs(
        embeddings=result.embeddings,
        sequence_ids=result.sequence_ids,
        sequence_lengths=sequence_lengths,
        classifier=classifier,
        threshold=threshold,
    )

    # 5. Write results
    write_tsv(detection_results, output_path, "detection_results.tsv")
    logger.info(f"Detection complete. Results written to {output_path}")


def _load_classifier(model_path: str | None) -> "ViralClassifier":
    """Load classifier from specified path or default location."""
    from virosense.models.detector import ViralClassifier, get_default_model_path

    if model_path:
        return ViralClassifier.load(model_path)

    default_path = get_default_model_path()
    if default_path.exists():
        return ViralClassifier.load(default_path)

    raise FileNotFoundError(
        f"No classifier model found at {default_path}. "
        "Train a reference model first with:\n"
        "  virosense build-reference -i sequences.fasta --labels labels.tsv -o model_dir/\n"
        "Then copy the model to the default location:\n"
        f"  cp model_dir/classifier.joblib {default_path}"
    )
