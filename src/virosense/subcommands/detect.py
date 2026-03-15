"""Detect viral sequences in metagenomic contigs."""

from pathlib import Path

import numpy as np
from loguru import logger


def run_detect(
    input_file: str,
    output_dir: str,
    backend: str = "nim",
    model: str = "evo2_7b",
    threshold: float = 0.5,
    min_length: int = 500,
    batch_size: int = 16,
    layer: str = "blocks.28.mlp.l3",
    cache_dir: str | None = None,
    nim_url: str | None = None,
    max_concurrent: int | None = None,
    classifier_model: str | None = None,
) -> None:
    """Run viral detection pipeline.

    1. Read metagenomic contigs from FASTA
    2. Filter by minimum length
    3. Extract Evo2 embeddings via selected backend
    4. Classify contigs as viral/cellular using trained classifier
    5. Write results TSV, summary JSON, filtered FASTA, and HTML report
    """
    from collections import Counter

    from virosense.backends.base import get_backend
    from virosense.features.evo2_embeddings import extract_embeddings
    from virosense.io.fasta import filter_by_length, read_fasta, write_fasta
    from virosense.io.report import generate_detect_report
    from virosense.io.results import write_json, write_tsv
    from virosense.models.detector import (
        ViralClassifier,
        classify_contigs,
        get_default_model_path,
    )

    # Validate threshold
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"Threshold must be between 0.0 and 1.0, got {threshold}"
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
    classifier = _load_classifier(classifier_model)

    # 3. Get backend and extract embeddings
    evo2_backend = get_backend(
        backend, api_key=None, model=model, nim_url=nim_url,
        max_concurrent=max_concurrent,
    )
    model = evo2_backend.model  # Use backend's (possibly auto-corrected) model name
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

    # 4. Validate embedding dimensions match classifier
    expected_dim = classifier.metadata.get("input_dim")
    actual_dim = result.embeddings.shape[1]
    if expected_dim and actual_dim != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: classifier expects {expected_dim}-D "
            f"but backend produced {actual_dim}-D. "
            f"This typically means the classifier was trained with a different "
            f"Evo2 model (e.g., 40B produces 8192-D, 7B produces 4096-D). "
            f"Retrain the classifier with: virosense build-reference --model {model}"
        )

    # 5. Classify contigs
    sequence_lengths = [len(sequences[sid]) for sid in result.sequence_ids]
    detection_results = classify_contigs(
        embeddings=result.embeddings,
        sequence_ids=result.sequence_ids,
        sequence_lengths=sequence_lengths,
        classifier=classifier,
        threshold=threshold,
    )

    # 6. Write results TSV
    write_tsv(detection_results, output_path, "detection_results.tsv")

    # 7. Write filtered viral contigs FASTA
    _NON_VIRAL = {"cellular", "chromosome", "plasmid", "ambiguous"}
    viral_ids = {r.contig_id for r in detection_results if r.classification not in _NON_VIRAL}
    if viral_ids:
        viral_seqs = {sid: sequences[sid] for sid in result.sequence_ids if sid in viral_ids}
        write_fasta(viral_seqs, output_path / "viral_contigs.fasta")

    # 8. Build and write summary JSON
    scores = [r.viral_score for r in detection_results]
    counts = Counter(r.classification for r in detection_results)

    n_viral = sum(1 for r in detection_results if r.classification not in _NON_VIRAL)
    n_cellular = sum(1 for r in detection_results if r.classification in {"cellular", "chromosome", "plasmid"})
    n_ambiguous = counts.get("ambiguous", 0)

    summary = {
        "n_sequences": len(detection_results),
        "n_viral": n_viral,
        "n_cellular": n_cellular,
        "n_ambiguous": n_ambiguous,
        "classification_counts": dict(counts),
        "score_distribution": {
            "mean": round(float(np.mean(scores)), 4),
            "median": round(float(np.median(scores)), 4),
            "min": round(float(np.min(scores)), 4),
            "max": round(float(np.max(scores)), 4),
            "above_0.9": sum(1 for s in scores if s >= 0.9),
            "between_0.5_0.9": sum(1 for s in scores if 0.5 <= s < 0.9),
            "below_0.5": sum(1 for s in scores if s < 0.5),
        },
        "parameters": {
            "threshold": threshold,
            "min_length": min_length,
            "backend": backend,
            "model": model,
            "layer": layer,
        },
        "classifier": {
            "model_path": str(classifier_model) if classifier_model else "default",
            "input_dim": classifier.metadata.get("input_dim"),
            "n_classes": classifier.metadata.get("n_classes"),
            "class_names": classifier.metadata.get("class_names"),
            "calibrated": classifier.metadata.get("calibrated"),
        },
    }
    write_json(summary, output_path, "detection_summary.json")

    # 9. Generate interactive HTML report
    generate_detect_report(detection_results, sequences, summary, output_path)

    logger.info(
        f"Detection complete: {n_viral} viral, "
        f"{n_cellular} cellular, {n_ambiguous} ambiguous. "
        f"Results in {output_path}"
    )


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
