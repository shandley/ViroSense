"""Prophage detection via sliding window analysis of bacterial chromosomes."""

from dataclasses import dataclass

import numpy as np
from loguru import logger


@dataclass
class WindowResult:
    """Classification result for a single genomic window."""

    window_id: str
    chromosome_id: str
    start: int
    end: int
    viral_score: float
    classification: str  # "viral", "cellular", "ambiguous"


@dataclass
class CandidateRegion:
    """A genomic interval identified by coarse scanning for fine-resolution follow-up."""

    chromosome_id: str
    start: int
    end: int


@dataclass
class ProphageRegion:
    """A merged prophage region from consecutive high-scoring windows."""

    region_id: str
    chromosome_id: str
    start: int
    end: int
    length: int
    n_windows: int
    mean_score: float
    max_score: float


def generate_windows(
    chromosomes: dict[str, str],
    window_size: int = 5000,
    step_size: int = 2000,
) -> tuple[dict[str, str], list[dict]]:
    """Tile chromosomes into overlapping windows for embedding extraction.

    Args:
        chromosomes: Dict of chromosome_id -> DNA sequence.
        window_size: Window size in bp.
        step_size: Step size in bp between window starts.

    Returns:
        Tuple of (window_sequences, window_metadata):
        - window_sequences: dict of window_id -> DNA sequence
        - window_metadata: list of dicts with chromosome_id, start, end
    """
    window_sequences = {}
    window_metadata = []

    for chrom_id, seq in chromosomes.items():
        chrom_len = len(seq)

        if chrom_len <= window_size:
            # Chromosome fits in a single window
            wid = f"{chrom_id}:0:{chrom_len}"
            window_sequences[wid] = seq
            window_metadata.append({
                "window_id": wid,
                "chromosome_id": chrom_id,
                "start": 0,
                "end": chrom_len,
            })
            continue

        for start in range(0, chrom_len - window_size + 1, step_size):
            end = start + window_size
            wid = f"{chrom_id}:{start}:{end}"
            window_sequences[wid] = seq[start:end]
            window_metadata.append({
                "window_id": wid,
                "chromosome_id": chrom_id,
                "start": start,
                "end": end,
            })

        # Ensure the last window covers the chromosome end
        last_start = chrom_len - window_size
        if last_start % step_size != 0:
            wid = f"{chrom_id}:{last_start}:{chrom_len}"
            if wid not in window_sequences:
                window_sequences[wid] = seq[last_start:chrom_len]
                window_metadata.append({
                    "window_id": wid,
                    "chromosome_id": chrom_id,
                    "start": last_start,
                    "end": chrom_len,
                })

    logger.info(
        f"Generated {len(window_sequences)} windows from "
        f"{len(chromosomes)} chromosome(s) "
        f"(window={window_size}, step={step_size})"
    )
    return window_sequences, window_metadata


def identify_candidate_regions(
    coarse_results: list[WindowResult],
    coarse_threshold: float = 0.3,
    margin: int = 20_000,
    chromosome_lengths: dict[str, int] | None = None,
) -> list[CandidateRegion]:
    """Identify candidate prophage regions from coarse-pass window scores.

    Any coarse window scoring >= coarse_threshold is flagged as a hit.
    Hits are expanded by ±margin bp, then overlapping intervals on the
    same chromosome are merged into contiguous CandidateRegions.

    Args:
        coarse_results: WindowResult list from the coarse scoring pass.
        coarse_threshold: Score threshold for flagging a coarse window as a hit.
        margin: Basepairs to add on each side of every hit.
        chromosome_lengths: Dict of chromosome_id -> length to clamp coordinates.

    Returns:
        List of CandidateRegion intervals for fine-resolution scanning.
    """
    # Collect hit intervals per chromosome
    hits_by_chrom: dict[str, list[tuple[int, int]]] = {}
    for w in coarse_results:
        if w.viral_score >= coarse_threshold:
            chrom_len = (
                chromosome_lengths.get(w.chromosome_id) if chromosome_lengths else None
            )
            start = max(0, w.start - margin)
            end = w.end + margin
            if chrom_len is not None:
                end = min(end, chrom_len)
            hits_by_chrom.setdefault(w.chromosome_id, []).append((start, end))

    # Merge overlapping intervals per chromosome
    candidates: list[CandidateRegion] = []
    for chrom_id in sorted(hits_by_chrom.keys()):
        intervals = sorted(hits_by_chrom[chrom_id])
        merged_start, merged_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= merged_end:
                merged_end = max(merged_end, end)
            else:
                candidates.append(
                    CandidateRegion(chrom_id, merged_start, merged_end)
                )
                merged_start, merged_end = start, end
        candidates.append(CandidateRegion(chrom_id, merged_start, merged_end))

    n_hits = sum(len(v) for v in hits_by_chrom.values())
    logger.info(
        f"Identified {len(candidates)} candidate region(s) on "
        f"{len(hits_by_chrom)} chromosome(s) from {n_hits} coarse hit(s)"
    )
    return candidates


def generate_windows_for_regions(
    chromosomes: dict[str, str],
    regions: list[CandidateRegion],
    window_size: int = 5000,
    step_size: int = 2000,
) -> tuple[dict[str, str], list[dict]]:
    """Generate fine-resolution windows only within candidate regions.

    Like generate_windows(), but restricted to coordinate intervals
    defined by candidate regions. Window IDs use global chromosome
    coordinates for compatibility with downstream functions.

    Args:
        chromosomes: Dict of chromosome_id -> full DNA sequence.
        regions: CandidateRegion intervals from identify_candidate_regions().
        window_size: Fine-resolution window size in bp.
        step_size: Fine-resolution step size in bp.

    Returns:
        Same format as generate_windows(): (window_sequences, window_metadata).
    """
    window_sequences: dict[str, str] = {}
    window_metadata: list[dict] = []
    seen_wids: set[str] = set()

    for region in regions:
        seq = chromosomes.get(region.chromosome_id)
        if seq is None:
            continue

        region_len = region.end - region.start
        if region_len <= window_size:
            # Region fits in a single window
            wid = f"{region.chromosome_id}:{region.start}:{region.end}"
            if wid not in seen_wids:
                seen_wids.add(wid)
                window_sequences[wid] = seq[region.start : region.end]
                window_metadata.append({
                    "window_id": wid,
                    "chromosome_id": region.chromosome_id,
                    "start": region.start,
                    "end": region.end,
                })
            continue

        for offset in range(0, region_len - window_size + 1, step_size):
            start = region.start + offset
            end = start + window_size
            wid = f"{region.chromosome_id}:{start}:{end}"
            if wid not in seen_wids:
                seen_wids.add(wid)
                window_sequences[wid] = seq[start:end]
                window_metadata.append({
                    "window_id": wid,
                    "chromosome_id": region.chromosome_id,
                    "start": start,
                    "end": end,
                })

        # Ensure the last window covers the region end
        last_start = region.end - window_size
        if last_start > region.start:
            wid = f"{region.chromosome_id}:{last_start}:{region.end}"
            if wid not in seen_wids:
                seen_wids.add(wid)
                window_sequences[wid] = seq[last_start : region.end]
                window_metadata.append({
                    "window_id": wid,
                    "chromosome_id": region.chromosome_id,
                    "start": last_start,
                    "end": region.end,
                })

    logger.info(
        f"Generated {len(window_sequences)} fine-resolution windows "
        f"across {len(regions)} candidate region(s)"
    )
    return window_sequences, window_metadata


def score_windows(
    embeddings: np.ndarray,
    sequence_ids: list[str],
    window_metadata: list[dict],
    classifier,
    threshold: float = 0.5,
) -> list[WindowResult]:
    """Score genomic windows using a trained viral classifier.

    Args:
        embeddings: (N, embed_dim) embedding matrix.
        sequence_ids: Ordered sequence IDs from embedding extraction.
        window_metadata: Window coordinate metadata from generate_windows().
        classifier: Trained ViralClassifier instance.
        threshold: Score threshold for viral classification.

    Returns:
        List of WindowResult for each window.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
        probas = classifier.predict_proba(embeddings)

    # Viral class is the last class (index 1 for binary)
    viral_idx = probas.shape[1] - 1
    viral_scores = probas[:, viral_idx]

    # Build lookup from window_id -> metadata
    meta_lookup = {m["window_id"]: m for m in window_metadata}

    results = []
    for i, seq_id in enumerate(sequence_ids):
        score = float(viral_scores[i])
        if score >= threshold:
            classification = "viral"
        elif score <= (1.0 - threshold):
            classification = "cellular"
        else:
            classification = "ambiguous"

        meta = meta_lookup[seq_id]
        results.append(WindowResult(
            window_id=seq_id,
            chromosome_id=meta["chromosome_id"],
            start=meta["start"],
            end=meta["end"],
            viral_score=round(score, 4),
            classification=classification,
        ))

    n_viral = sum(1 for r in results if r.classification == "viral")
    n_cellular = sum(1 for r in results if r.classification == "cellular")
    n_ambiguous = sum(1 for r in results if r.classification == "ambiguous")
    logger.info(
        f"Window scoring: {n_viral} viral, {n_cellular} cellular, "
        f"{n_ambiguous} ambiguous (threshold={threshold})"
    )
    return results


def merge_prophage_regions(
    window_results: list[WindowResult],
    threshold: float = 0.5,
    min_region_length: int = 5000,
    merge_gap: int = 3000,
) -> list[ProphageRegion]:
    """Merge consecutive high-scoring windows into prophage regions.

    Groups windows by chromosome, sorts by position, and merges
    consecutive viral-scoring windows. Regions separated by gaps
    <= merge_gap bp are merged together. Regions shorter than
    min_region_length are filtered out.

    Args:
        window_results: Per-window scoring results.
        threshold: Score threshold for viral classification.
        min_region_length: Minimum prophage region length in bp.
        merge_gap: Maximum gap (bp) between viral windows to merge.

    Returns:
        List of ProphageRegion for detected prophage regions.
    """
    # Group windows by chromosome
    by_chrom: dict[str, list[WindowResult]] = {}
    for w in window_results:
        by_chrom.setdefault(w.chromosome_id, []).append(w)

    regions = []
    region_counter = 0

    for chrom_id in sorted(by_chrom.keys()):
        windows = sorted(by_chrom[chrom_id], key=lambda w: w.start)

        # Find runs of viral windows, allowing gaps <= merge_gap
        current_region_windows: list[WindowResult] = []
        current_region_end = -1

        for w in windows:
            is_viral = w.viral_score >= threshold

            if is_viral:
                if not current_region_windows:
                    # Start new region
                    current_region_windows = [w]
                    current_region_end = w.end
                elif w.start - current_region_end <= merge_gap:
                    # Extend or merge into current region
                    current_region_windows.append(w)
                    current_region_end = max(current_region_end, w.end)
                else:
                    # Gap too large — emit current region, start new one
                    region = _finalize_region(
                        current_region_windows, chrom_id, region_counter
                    )
                    if region and region.length >= min_region_length:
                        regions.append(region)
                        region_counter += 1
                    current_region_windows = [w]
                    current_region_end = w.end
            else:
                # Non-viral window — check if gap is still within tolerance
                if current_region_windows and w.start - current_region_end <= merge_gap:
                    # Keep region open (gap tolerance), but don't add this
                    # window to the region's viral windows
                    pass
                elif current_region_windows:
                    # Gap exceeded — emit current region
                    region = _finalize_region(
                        current_region_windows, chrom_id, region_counter
                    )
                    if region and region.length >= min_region_length:
                        regions.append(region)
                        region_counter += 1
                    current_region_windows = []
                    current_region_end = -1

        # Emit any remaining region
        if current_region_windows:
            region = _finalize_region(
                current_region_windows, chrom_id, region_counter
            )
            if region and region.length >= min_region_length:
                regions.append(region)
                region_counter += 1

    logger.info(
        f"Merged into {len(regions)} prophage region(s) "
        f"(min_length={min_region_length}, merge_gap={merge_gap})"
    )
    return regions


def _finalize_region(
    windows: list[WindowResult],
    chrom_id: str,
    region_idx: int,
) -> ProphageRegion:
    """Create a ProphageRegion from a list of viral windows."""
    start = min(w.start for w in windows)
    end = max(w.end for w in windows)
    scores = [w.viral_score for w in windows]

    return ProphageRegion(
        region_id=f"prophage_{region_idx}",
        chromosome_id=chrom_id,
        start=start,
        end=end,
        length=end - start,
        n_windows=len(windows),
        mean_score=round(float(np.mean(scores)), 4),
        max_score=round(float(np.max(scores)), 4),
    )
