"""Detect prophage regions in bacterial chromosomes via sliding window analysis."""

from pathlib import Path

from loguru import logger

from virosense.utils.constants import ADAPTIVE_AUTO_BYPASS_THRESHOLD


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
    nim_url: str | None = None,
    scan_mode: str = "adaptive",
    coarse_window_size: int = 15000,
    coarse_step_size: int = 10000,
    coarse_threshold: float = 0.3,
    margin: int = 20000,
) -> None:
    """Run prophage detection pipeline.

    Supports two scan modes:
    - "full": Single-pass scan at fine resolution (original behavior).
    - "adaptive": Two-pass scan — coarse pass to identify candidate regions,
      then fine pass only on candidates. ~5x fewer API calls for typical
      bacterial chromosomes.

    Auto-bypasses to full mode when total fine windows < threshold.
    """
    from virosense.backends.base import get_backend
    from virosense.features.evo2_embeddings import extract_embeddings
    from virosense.io.fasta import read_fasta
    from virosense.io.results import write_bed, write_json, write_tsv
    from virosense.models.prophage import (
        generate_windows,
        generate_windows_for_regions,
        identify_candidate_regions,
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

    # 2. Clamp window sizes to backend limit
    evo2_backend = get_backend(backend, api_key=None, model=model, nim_url=nim_url)
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
    if coarse_window_size > max_ctx:
        logger.warning(
            f"Coarse window size {coarse_window_size} exceeds backend limit "
            f"({max_ctx}), clamping to {max_ctx}"
        )
        coarse_window_size = max_ctx

    # 3. Auto-bypass: skip coarse pass if input is small
    actual_scan_mode = scan_mode
    if scan_mode == "adaptive":
        total_fine_windows = sum(
            max(1, (len(seq) - window_size) // step_size + 1)
            for seq in chromosomes.values()
        )
        if total_fine_windows < ADAPTIVE_AUTO_BYPASS_THRESHOLD:
            logger.info(
                f"Only {total_fine_windows} windows at fine resolution — "
                f"skipping coarse pass, running full scan"
            )
            actual_scan_mode = "full"
        else:
            logger.info(
                f"Adaptive scan: {total_fine_windows} fine windows estimated, "
                f"running coarse pass first"
            )

    # 4. Load classifier
    classifier = _load_classifier(classifier_model)

    cache_path = Path(cache_dir) if cache_dir else None
    adaptive_stats = None

    if actual_scan_mode == "adaptive":
        # === COARSE PASS ===
        coarse_seqs, coarse_meta = generate_windows(
            chromosomes,
            window_size=coarse_window_size,
            step_size=coarse_step_size,
        )

        if not coarse_seqs:
            logger.warning("No coarse windows generated.")
            return

        coarse_result = extract_embeddings(
            sequences=coarse_seqs,
            backend=evo2_backend,
            layer=layer,
            model=model,
            batch_size=batch_size,
            cache_dir=cache_path,
        )

        coarse_window_results = score_windows(
            embeddings=coarse_result.embeddings,
            sequence_ids=coarse_result.sequence_ids,
            window_metadata=coarse_meta,
            classifier=classifier,
            threshold=coarse_threshold,
        )

        chrom_lengths = {cid: len(seq) for cid, seq in chromosomes.items()}
        candidates = identify_candidate_regions(
            coarse_window_results,
            coarse_threshold=coarse_threshold,
            margin=margin,
            chromosome_lengths=chrom_lengths,
        )

        if not candidates:
            logger.info("No candidate prophage regions found in coarse pass.")
            # Write empty results
            write_tsv([], output_path, "prophage_windows.tsv")
            write_tsv([], output_path, "prophage_regions.tsv")
            write_bed([], output_path, "prophage_regions.bed")
            summary = {
                "n_chromosomes": len(chromosomes),
                "total_bp": total_bp,
                "n_windows": 0,
                "n_viral_windows": 0,
                "n_regions": 0,
                "regions": [],
                "parameters": {
                    "scan_mode": "adaptive",
                    "window_size": window_size,
                    "step_size": step_size,
                    "threshold": threshold,
                    "min_region_length": min_region_length,
                    "merge_gap": merge_gap,
                },
                "adaptive": {
                    "coarse_window_size": coarse_window_size,
                    "coarse_step_size": coarse_step_size,
                    "coarse_threshold": coarse_threshold,
                    "margin": margin,
                    "n_coarse_windows": len(coarse_window_results),
                    "n_coarse_hits": 0,
                    "n_candidate_regions": 0,
                    "n_fine_windows": 0,
                },
            }
            write_json(summary, output_path, "prophage_summary.json")
            logger.info(
                "Prophage detection complete: 0 region(s) found. "
                f"Results written to {output_path}"
            )
            return

        # === FINE PASS ===
        fine_seqs, fine_meta = generate_windows_for_regions(
            chromosomes,
            regions=candidates,
            window_size=window_size,
            step_size=step_size,
        )

        fine_result = extract_embeddings(
            sequences=fine_seqs,
            backend=evo2_backend,
            layer=layer,
            model=model,
            batch_size=batch_size,
            cache_dir=cache_path,
        )

        window_results = score_windows(
            embeddings=fine_result.embeddings,
            sequence_ids=fine_result.sequence_ids,
            window_metadata=fine_meta,
            classifier=classifier,
            threshold=threshold,
        )

        n_coarse_hits = sum(
            1 for w in coarse_window_results if w.viral_score >= coarse_threshold
        )
        adaptive_stats = {
            "coarse_window_size": coarse_window_size,
            "coarse_step_size": coarse_step_size,
            "coarse_threshold": coarse_threshold,
            "margin": margin,
            "n_coarse_windows": len(coarse_window_results),
            "n_coarse_hits": n_coarse_hits,
            "n_candidate_regions": len(candidates),
            "n_fine_windows": len(window_results),
        }

    else:
        # === FULL SCAN (single pass) ===
        window_sequences, window_metadata = generate_windows(
            chromosomes, window_size=window_size, step_size=step_size
        )

        if not window_sequences:
            logger.warning("No windows generated.")
            return

        result = extract_embeddings(
            sequences=window_sequences,
            backend=evo2_backend,
            layer=layer,
            model=model,
            batch_size=batch_size,
            cache_dir=cache_path,
        )

        window_results = score_windows(
            embeddings=result.embeddings,
            sequence_ids=result.sequence_ids,
            window_metadata=window_metadata,
            classifier=classifier,
            threshold=threshold,
        )

    # 5. Merge into prophage regions
    regions = merge_prophage_regions(
        window_results,
        threshold=threshold,
        min_region_length=min_region_length,
        merge_gap=merge_gap,
    )

    # 6. Write results
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
            "scan_mode": actual_scan_mode,
            "window_size": window_size,
            "step_size": step_size,
            "threshold": threshold,
            "min_region_length": min_region_length,
            "merge_gap": merge_gap,
        },
    }
    if adaptive_stats:
        summary["adaptive"] = adaptive_stats

    write_json(summary, output_path, "prophage_summary.json")

    logger.info(
        f"Prophage detection complete: {len(regions)} region(s) found. "
        f"Results written to {output_path}"
    )
