"""Annotate ORFs with genomic context from Evo2 embeddings."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from virosense.io.orfs import ORF


@dataclass
class ContextAnnotation:
    """Enhanced ORF annotation with genomic context."""

    orf_id: str
    contig_id: str
    start: int
    end: int
    strand: str
    window_embedding_norm: float
    vhold_annotation: str | None = None
    vhold_score: float | None = None


def run_context(
    input_file: str,
    orfs_file: str,
    output_dir: str,
    backend: str = "nim",
    model: str = "evo2_7b",
    window_size: int = 2000,
    vhold_output: str | None = None,
    threads: int = 4,
    layer: str = "blocks.28.mlp.l3",
    cache_dir: str | None = None,
) -> None:
    """Run genomic context annotation pipeline.

    1. Read viral contigs and ORF predictions
    2. Extract Evo2 embeddings for genomic windows around each ORF
    3. Optionally merge with vHold protein-level annotations
    4. Write enhanced annotation results
    """
    from virosense.backends.base import get_backend
    from virosense.features.evo2_embeddings import extract_embeddings
    from virosense.io.fasta import read_fasta
    from virosense.io.orfs import parse_orfs
    from virosense.io.results import write_json, write_tsv

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Annotating ORFs with genomic context from {input_file}")
    logger.info(f"ORF predictions: {orfs_file}")
    logger.info(f"Backend: {backend}, Window size: {window_size}")

    # 1. Read contigs and ORFs
    contigs = read_fasta(input_file)
    orfs = parse_orfs(orfs_file)

    if not orfs:
        logger.warning("No ORFs found in input file.")
        return

    # Filter ORFs to those whose contig exists in the FASTA
    valid_orfs = [o for o in orfs if o.contig_id in contigs]
    if len(valid_orfs) < len(orfs):
        logger.warning(
            f"Skipped {len(orfs) - len(valid_orfs)} ORFs "
            f"(contig not found in FASTA)"
        )
    orfs = valid_orfs

    if not orfs:
        logger.warning("No ORFs remaining after contig matching.")
        return

    # 2. Extract genomic windows around each ORF
    windows = extract_orf_windows(contigs, orfs, window_size)
    logger.info(f"Extracted {len(windows)} genomic windows")

    # 3. Get Evo2 embeddings for windows
    evo2_backend = get_backend(backend, model=model)
    if not evo2_backend.is_available():
        raise RuntimeError(
            f"Backend {backend!r} is not available. "
            "Check your configuration (e.g., NVIDIA_API_KEY for nim)."
        )

    cache_path = Path(cache_dir) if cache_dir else None
    result = extract_embeddings(
        sequences=windows,
        backend=evo2_backend,
        layer=layer,
        model=model,
        cache_dir=cache_path,
    )

    # Build embedding lookup
    embed_lookup = dict(zip(result.sequence_ids, result.embeddings))

    # 4. Optionally load vHold annotations
    vhold_annotations = {}
    if vhold_output:
        from virosense.features.prostt5_bridge import load_vhold_annotations

        vhold_annotations = load_vhold_annotations(vhold_output)
        logger.info(f"Loaded {len(vhold_annotations)} vHold annotations")

    # 5. Build context annotations
    annotations = []
    for orf in orfs:
        window_id = f"{orf.orf_id}_w{window_size}"
        embedding = embed_lookup.get(window_id)
        embed_norm = float(np.linalg.norm(embedding)) if embedding is not None else 0.0

        # Look up vHold annotation by ORF ID
        vhold_info = vhold_annotations.get(orf.orf_id, {})
        vhold_annot = None
        vhold_score_val = None
        if vhold_info:
            # vHold TSV typically has 'annotation' and 'score' columns
            vhold_annot = str(
                vhold_info.get("annotation", vhold_info.get("function", None))
            )
            vhold_score_val = vhold_info.get("score", vhold_info.get("evalue", None))
            if vhold_score_val is not None:
                vhold_score_val = float(vhold_score_val)

        annotations.append(
            ContextAnnotation(
                orf_id=orf.orf_id,
                contig_id=orf.contig_id,
                start=orf.start,
                end=orf.end,
                strand=orf.strand,
                window_embedding_norm=round(embed_norm, 4),
                vhold_annotation=vhold_annot,
                vhold_score=vhold_score_val,
            )
        )

    # 6. Write results
    write_tsv(annotations, output_path, "context_annotations.tsv")

    summary = {
        "n_orfs": len(annotations),
        "n_contigs": len(set(a.contig_id for a in annotations)),
        "window_size": window_size,
        "n_with_vhold": sum(1 for a in annotations if a.vhold_annotation is not None),
        "mean_embedding_norm": float(
            np.mean([a.window_embedding_norm for a in annotations])
        ),
    }
    write_json(summary, output_path, "context_summary.json")

    logger.info(
        f"Context annotation complete: {summary['n_orfs']} ORFs, "
        f"{summary['n_with_vhold']} with vHold annotations"
    )


def extract_orf_windows(
    contigs: dict[str, str],
    orfs: list[ORF],
    window_size: int = 2000,
) -> dict[str, str]:
    """Extract DNA windows centered on each ORF.

    Each window extends ±window_size/2 from the ORF midpoint,
    clipped to contig bounds. Minimum window length is 100 bp.

    Args:
        contigs: Dict of contig_id -> DNA sequence.
        orfs: List of ORF objects.
        window_size: Total window size in bp.

    Returns:
        Dict of window_id -> DNA sequence.
    """
    half = window_size // 2
    windows = {}

    for orf in orfs:
        contig_seq = contigs.get(orf.contig_id)
        if contig_seq is None:
            continue

        contig_len = len(contig_seq)
        midpoint = (orf.start + orf.end) // 2

        win_start = max(0, midpoint - half)
        win_end = min(contig_len, midpoint + half)

        window_seq = contig_seq[win_start:win_end]
        if len(window_seq) < 100:
            continue

        window_id = f"{orf.orf_id}_w{window_size}"
        windows[window_id] = window_seq

    return windows
