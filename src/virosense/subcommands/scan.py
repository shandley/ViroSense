"""Per-position embedding analysis: gene boundaries, codon periodicity, coding detection."""

import json
import warnings
from pathlib import Path

import numpy as np
from loguru import logger


def run_scan(
    input_file: str,
    output_dir: str,
    cache_dir: str | None = None,
    coding: bool = True,
    periodicity: bool = True,
    boundaries: bool = False,
    window: int = 30,
) -> None:
    """Analyze per-position Evo2 embeddings for gene structure.

    Requires per-position embeddings in cache_dir/per_position/*.npy
    (produced by virosense embed --per-position).

    Args:
        input_file: Input FASTA (to get sequence IDs and gene calls).
        output_dir: Output directory for results.
        cache_dir: Directory with per-position embedding .npy files.
        coding: Predict coding vs non-coding regions.
        periodicity: Compute codon periodicity features.
        boundaries: Detect gene boundaries via norm derivative peaks.
        window: Smoothing window size in bp.
    """
    from Bio import SeqIO
    from scipy.ndimage import uniform_filter1d

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find per-position embeddings — check both per_position/ subdirectory
    # and the cache directory root (for backward compatibility with scripts)
    pp_dir = None
    search_dirs = []
    if cache_dir:
        search_dirs.extend([Path(cache_dir) / "per_position", Path(cache_dir)])
    search_dirs.extend([Path(output_dir) / "per_position", Path(output_dir)])

    for candidate in search_dirs:
        if candidate.exists() and list(candidate.glob("*.npy")):
            pp_dir = candidate
            break

    if pp_dir is None:
        raise FileNotFoundError(
            "Per-position embeddings (.npy files) not found. "
            "Run 'virosense embed --per-position' first."
        )

    npy_files = {f.stem: f for f in pp_dir.glob("*.npy")}
    logger.info(f"Found {len(npy_files)} per-position embedding files in {pp_dir}")

    # Load sequences for gene calling
    records = {rec.description: rec for rec in SeqIO.parse(input_file, "fasta")}
    logger.info(f"Loaded {len(records)} sequences from {input_file}")

    # Gene caller
    try:
        import pyrodigal_gv
        gene_finder = pyrodigal_gv.ViralGeneFinder(meta=True)
    except ImportError:
        import pyrodigal
        gene_finder = pyrodigal.GeneFinder(meta=True)

    all_results = []

    for seq_id, record in records.items():
        # Find matching .npy file
        safe_name = seq_id.replace("/", "_")[:80]
        npy_path = npy_files.get(safe_name)
        if npy_path is None:
            # Try partial matching
            for key, path in npy_files.items():
                if key[:40] == safe_name[:40]:
                    npy_path = path
                    break
        if npy_path is None:
            continue

        per_position = np.load(npy_path)
        sequence = str(record.seq).upper()
        seq_len = len(sequence)

        if per_position.shape[0] != seq_len:
            logger.warning(
                f"Length mismatch for {seq_id[:50]}: "
                f"FASTA={seq_len}, embeddings={per_position.shape[0]}"
            )
            continue

        # Gene calls for ground truth / coding mask
        genes = [
            {"start": g.begin, "end": g.end, "strand": g.strand}
            for g in gene_finder.find_genes(sequence.encode())
        ]

        norms = np.linalg.norm(per_position, axis=1)
        result = {
            "sequence_id": seq_id,
            "length": seq_len,
            "n_genes": len(genes),
        }

        # Build coding mask
        coding_mask = np.zeros(seq_len, dtype=bool)
        for g in genes:
            s = min(g["start"], g["end"]) - 1
            e = max(g["start"], g["end"])
            coding_mask[s:e] = True
        result["coding_fraction"] = round(float(coding_mask.mean()), 3)

        # Normalized embeddings for cosine computations
        norm_mat = np.linalg.norm(per_position, axis=1, keepdims=True)
        norm_mat[norm_mat == 0] = 1
        normalized = per_position / norm_mat

        # --- Coding prediction ---
        if coding and coding_mask.sum() > 50 and (~coding_mask).sum() > 20:
            coding_norm = norms[coding_mask].mean()
            inter_norm = norms[~coding_mask].mean()
            result["coding_norm_mean"] = round(float(coding_norm), 1)
            result["intergenic_norm_mean"] = round(float(inter_norm), 1)
            result["norm_ratio"] = round(float(coding_norm / inter_norm), 3)

            # Norm-threshold accuracy
            smooth_norms = uniform_filter1d(norms, size=window)
            threshold = (coding_norm + inter_norm) / 2
            pred_norm = smooth_norms >= threshold
            result["norm_coding_accuracy"] = round(
                float((pred_norm == coding_mask).mean()), 3
            )

        # --- Periodicity ---
        if periodicity and genes:
            largest = max(genes, key=lambda g: abs(g["end"] - g["start"]))
            gs = min(largest["start"], largest["end"]) - 1
            ge = max(largest["start"], largest["end"])

            if ge - gs >= 150:
                gene_norms = norms[gs:ge]
                nc = gene_norms - gene_norms.mean()
                autocorr = np.correlate(nc, nc, mode="full")
                autocorr = autocorr[len(autocorr) // 2:]
                autocorr /= autocorr[0] if autocorr[0] > 0 else 1

                result["lag1_autocorr"] = round(float(autocorr[1]), 4) if len(autocorr) > 1 else None
                result["lag2_autocorr"] = round(float(autocorr[2]), 4) if len(autocorr) > 2 else None
                result["lag3_autocorr"] = round(float(autocorr[3]), 4) if len(autocorr) > 3 else None

                # FFT dominant period
                fft = np.abs(np.fft.rfft(nc))
                freqs = np.fft.rfftfreq(len(nc))
                top_idx = np.argmax(fft[1:]) + 1
                result["dominant_fft_period"] = round(1.0 / freqs[top_idx], 1) if freqs[top_idx] > 0 else None

                # Offset cosine inversion
                cos1 = np.sum(normalized[:-1] * normalized[1:], axis=1)
                cos3_raw = np.sum(normalized[:-3] * normalized[3:], axis=1)

                if coding_mask.sum() > 10 and (~coding_mask).sum() > 10:
                    result["cos1_coding"] = round(float(cos1[coding_mask[:-1]].mean()), 4)
                    result["cos3_coding"] = round(float(cos3_raw[coding_mask[:-3]].mean()), 4)
                    result["cos1_intergenic"] = round(float(cos1[~coding_mask[:-1]].mean()), 4)
                    result["cos3_intergenic"] = round(float(cos3_raw[~coding_mask[:-3]].mean()), 4)
                    result["offset3_inversion"] = result["cos3_coding"] > result["cos1_coding"]

                    # Inversion-based coding accuracy
                    cos3_padded = np.zeros(len(cos1))
                    cos3_padded[:len(cos3_raw)] = cos3_raw
                    smooth_cos1 = uniform_filter1d(cos1, size=60)
                    smooth_cos3 = uniform_filter1d(cos3_padded, size=60)
                    inv_pred = smooth_cos3 > smooth_cos1
                    result["inversion_coding_accuracy"] = round(
                        float((inv_pred[:seq_len - 1] == coding_mask[:seq_len - 1]).mean()), 3
                    )

        # --- Boundaries ---
        if boundaries and coding_mask.sum() > 50 and (~coding_mask).sum() > 20:
            smooth_norms = uniform_filter1d(norms, size=window)
            norm_deriv = np.abs(np.diff(smooth_norms))
            smooth_deriv = uniform_filter1d(norm_deriv, size=20)
            deriv_threshold = smooth_deriv.mean() + 1.5 * smooth_deriv.std()

            peaks = []
            for i in range(1, len(smooth_deriv) - 1):
                if (smooth_deriv[i] > deriv_threshold
                        and smooth_deriv[i] > smooth_deriv[i - 1]
                        and smooth_deriv[i] > smooth_deriv[i + 1]):
                    peaks.append(i)

            # Merge nearby
            merged = []
            for p in peaks:
                if not merged or p - merged[-1] > 30:
                    merged.append(p)
                elif smooth_deriv[p] > smooth_deriv[merged[-1]]:
                    merged[-1] = p

            actual = sorted(
                {min(g["start"], g["end"]) - 1 for g in genes}
                | {max(g["start"], g["end"]) - 1 for g in genes}
            )

            tolerance = 50
            hits = sum(
                1 for ab in actual
                if merged and min(abs(p - ab) for p in merged) <= tolerance
            )

            result["n_boundaries"] = len(actual)
            result["n_peaks"] = len(merged)
            result["boundary_hits"] = hits
            result["boundary_recall"] = round(hits / len(actual), 3) if actual else 0

        all_results.append(result)

    # Write results
    import pandas as pd

    df = pd.DataFrame(all_results)
    df.to_csv(output_path / "scan_results.tsv", sep="\t", index=False)

    with open(output_path / "scan_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    if len(df) > 0:
        logger.info(f"Scanned {len(df)} sequences")

        if "norm_ratio" in df.columns:
            valid = df["norm_ratio"].dropna()
            if len(valid) > 0:
                logger.info(f"Norm ratio: {valid.mean():.3f} ± {valid.std():.3f}")

        if "lag3_autocorr" in df.columns:
            valid = df["lag3_autocorr"].dropna()
            if len(valid) > 0:
                logger.info(f"Lag-3 autocorrelation: {valid.mean():.3f} ± {valid.std():.3f}")

        if "inversion_coding_accuracy" in df.columns:
            valid = df["inversion_coding_accuracy"].dropna()
            if len(valid) > 0:
                logger.info(f"Inversion coding accuracy: {valid.mean():.1%} ± {valid.std():.1%}")

    logger.info(f"Results written to {output_path}/")
