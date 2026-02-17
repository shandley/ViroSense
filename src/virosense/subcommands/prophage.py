"""Detect prophage regions in bacterial chromosomes via sliding window analysis."""

from pathlib import Path

from loguru import logger


def run_prophage(
    input_file: str,
    output_dir: str,
    backend: str = "nim",
    model: str = "evo2_7b",
    threshold: float = 0.5,
    window_size: int = 5000,
    step_size: int = 2000,
    min_region_length: int = 5000,
    merge_gap: int = 3000,
    batch_size: int = 16,
    layer: str = "blocks.28.mlp.l3",
    cache_dir: str | None = None,
    classifier_model: str | None = None,
) -> None:
    """Run prophage detection pipeline.

    1. Read bacterial chromosomes from FASTA
    2. Generate overlapping sliding windows
    3. Extract Evo2 embeddings for each window
    4. Score windows with trained viral classifier
    5. Merge consecutive high-scoring windows into prophage regions
    6. Write results (TSV + BED)
    """
    from virosense.backends.base import get_backend
    from virosense.features.evo2_embeddings import extract_embeddings
    from virosense.io.fasta import read_fasta
    from virosense.io.results import write_bed, write_json, write_tsv
    from virosense.models.prophage import (
        generate_windows,
        merge_prophage_regions,
        score_windows,
    )
    from virosense.subcommands.detect import _load_classifier

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning for prophage regions in {input_file}")
    logger.info(
        f"Window: {window_size} bp, Step: {step_size} bp, "
        f"Threshold: {threshold}"
    )
    logger.info(f"Backend: {backend}, Model: {model}")

    # 1. Read chromosomes
    chromosomes = read_fasta(input_file)
    if not chromosomes:
        logger.warning("No sequences found in input file.")
        return

    total_bp = sum(len(s) for s in chromosomes.values())
    logger.info(
        f"Loaded {len(chromosomes)} sequence(s), "
        f"{total_bp:,} bp total"
    )

    # 2. Clamp window size to backend limit
    evo2_backend = get_backend(backend, api_key=None, model=model)
    if not evo2_backend.is_available():
        raise RuntimeError(
            f"Backend {backend!r} is not available. "
            "Check your configuration (e.g., NVIDIA_API_KEY for nim)."
        )

    max_ctx = evo2_backend.max_context_length()
    if window_size > max_ctx:
        logger.warning(
            f"Window size {window_size} exceeds backend limit "
            f"({max_ctx}), clamping to {max_ctx}"
        )
        window_size = max_ctx

    # 3. Generate windows
    window_sequences, window_metadata = generate_windows(
        chromosomes, window_size=window_size, step_size=step_size
    )

    if not window_sequences:
        logger.warning("No windows generated.")
        return

    # 4. Load classifier
    classifier = _load_classifier(classifier_model)

    # 5. Extract embeddings
    cache_path = Path(cache_dir) if cache_dir else None
    result = extract_embeddings(
        sequences=window_sequences,
        backend=evo2_backend,
        layer=layer,
        model=model,
        batch_size=batch_size,
        cache_dir=cache_path,
    )

    # 6. Score windows
    window_results = score_windows(
        embeddings=result.embeddings,
        sequence_ids=result.sequence_ids,
        window_metadata=window_metadata,
        classifier=classifier,
        threshold=threshold,
    )

    # 7. Merge into prophage regions
    regions = merge_prophage_regions(
        window_results,
        threshold=threshold,
        min_region_length=min_region_length,
        merge_gap=merge_gap,
    )

    # 8. Write results
    write_tsv(window_results, output_path, "prophage_windows.tsv")
    write_tsv(regions, output_path, "prophage_regions.tsv")
    write_bed(regions, output_path, "prophage_regions.bed")

    summary = {
        "n_chromosomes": len(chromosomes),
        "total_bp": total_bp,
        "n_windows": len(window_results),
        "n_viral_windows": sum(
            1 for w in window_results if w.classification == "viral"
        ),
        "n_regions": len(regions),
        "regions": [
            {
                "region_id": r.region_id,
                "chromosome_id": r.chromosome_id,
                "start": r.start,
                "end": r.end,
                "length": r.length,
                "mean_score": r.mean_score,
                "max_score": r.max_score,
            }
            for r in regions
        ],
        "parameters": {
            "window_size": window_size,
            "step_size": step_size,
            "threshold": threshold,
            "min_region_length": min_region_length,
            "merge_gap": merge_gap,
        },
    }
    write_json(summary, output_path, "prophage_summary.json")

    logger.info(
        f"Prophage detection complete: {len(regions)} region(s) found. "
        f"Results written to {output_path}"
    )
