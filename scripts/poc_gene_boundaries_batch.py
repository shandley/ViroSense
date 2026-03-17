#!/usr/bin/env python3
"""Batch gene boundary detection across diverse sequences.

Extracts per-position embeddings for each sequence, runs Pyrodigal-gv gene
calling, and computes coding/intergenic embedding statistics.

Usage:
    export NVIDIA_API_KEY=...
    uv run python scripts/poc_gene_boundaries_batch.py \
        --fasta results/poc_gene_boundaries_expanded/samples.fasta \
        --sample-list results/poc_gene_boundaries_expanded/sample_list.csv \
        --output results/poc_gene_boundaries_expanded/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from poc_gene_boundaries import (
    extract_per_position_embeddings,
    call_genes_pyrodigal,
)


def analyze_one(
    sequence: str,
    seq_id: str,
    per_position: np.ndarray,
    window: int = 30,
) -> dict:
    """Compute coding/intergenic statistics for one sequence."""
    seq_len = per_position.shape[0]

    # Gene calling
    genes = call_genes_pyrodigal(sequence)
    if not genes:
        return {"seq_id": seq_id, "seq_len": seq_len, "n_genes": 0, "error": "no genes found"}

    # Position-wise norms
    norms = np.linalg.norm(per_position, axis=1)

    # Adjacent cosine similarity
    norm_mat = np.linalg.norm(per_position, axis=1, keepdims=True)
    norm_mat[norm_mat == 0] = 1
    normalized = per_position / norm_mat
    adj_cosine = np.sum(normalized[:-1] * normalized[1:], axis=1)

    # Coding mask
    coding_mask = np.zeros(seq_len, dtype=bool)
    boundary_mask = np.zeros(seq_len, dtype=bool)
    boundary_width = 20

    for gene in genes:
        s = min(gene["start"], gene["end"]) - 1
        e = max(gene["start"], gene["end"])
        coding_mask[s:e] = True
        for bp in [s, e]:
            lo = max(0, bp - boundary_width)
            hi = min(seq_len, bp + boundary_width)
            boundary_mask[lo:hi] = True

    intergenic_mask = ~coding_mask
    coding_frac = coding_mask.sum() / seq_len

    result = {
        "seq_id": seq_id,
        "seq_len": seq_len,
        "n_genes": len(genes),
        "coding_fraction": round(float(coding_frac), 3),
    }

    if coding_mask.sum() > 0 and intergenic_mask.sum() > 10:
        result["coding_norm_mean"] = round(float(norms[coding_mask].mean()), 1)
        result["intergenic_norm_mean"] = round(float(norms[intergenic_mask].mean()), 1)
        result["norm_ratio"] = round(float(norms[coding_mask].mean() / norms[intergenic_mask].mean()), 3)

        result["coding_cosine_mean"] = round(float(adj_cosine[coding_mask[:-1]].mean()), 4)
        result["intergenic_cosine_mean"] = round(float(adj_cosine[intergenic_mask[:-1]].mean()), 4)

        # Boundary cosine
        boundary_positions = np.where(boundary_mask[:-1])[0]
        if len(boundary_positions) > 0:
            result["boundary_cosine_mean"] = round(float(adj_cosine[boundary_positions].mean()), 4)

        # Norm-threshold coding prediction accuracy
        from scipy.ndimage import uniform_filter1d
        smooth_norms = uniform_filter1d(norms, size=window)
        threshold = (norms[coding_mask].mean() + norms[intergenic_mask].mean()) / 2
        pred_coding = smooth_norms >= threshold
        accuracy = float((pred_coding == coding_mask).mean())
        result["coding_prediction_accuracy"] = round(accuracy, 3)

        # Boundary detection via norm derivative
        norm_deriv = np.abs(np.diff(smooth_norms))
        smooth_deriv = uniform_filter1d(norm_deriv, size=20)
        deriv_threshold = smooth_deriv.mean() + 1.5 * smooth_deriv.std()

        # Find peaks
        peaks = []
        for i in range(1, len(smooth_deriv) - 1):
            if (smooth_deriv[i] > deriv_threshold
                    and smooth_deriv[i] > smooth_deriv[i - 1]
                    and smooth_deriv[i] > smooth_deriv[i + 1]):
                peaks.append(i)

        # Merge nearby peaks
        merged = []
        for p in peaks:
            if not merged or p - merged[-1] > 30:
                merged.append(p)
            elif smooth_deriv[p] > smooth_deriv[merged[-1]]:
                merged[-1] = p

        # Compare to actual boundaries
        boundaries = sorted({min(g["start"], g["end"]) - 1 for g in genes}
                            | {max(g["start"], g["end"]) - 1 for g in genes})

        tolerance = 50
        hits = 0
        for ab in boundaries:
            if merged and min(abs(p - ab) for p in merged) <= tolerance:
                hits += 1

        result["n_boundaries"] = len(boundaries)
        result["n_peaks"] = len(merged)
        result["boundary_hits"] = hits
        result["boundary_recall"] = round(hits / len(boundaries), 3) if boundaries else 0
        result["boundary_precision"] = round(hits / len(merged), 3) if merged else 0

    elif intergenic_mask.sum() <= 10:
        # Almost entirely coding — can't assess intergenic
        result["coding_norm_mean"] = round(float(norms[coding_mask].mean()), 1)
        result["note"] = "nearly_all_coding"

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip sequences with existing per-position .npy files")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_df = pd.read_csv(args.sample_list)
    records = {rec.description: rec for rec in SeqIO.parse(args.fasta, "fasta")}
    logger.info(f"Loaded {len(records)} sequences, {len(sample_df)} in sample list")

    all_results = []

    for i, (_, row) in enumerate(sample_df.iterrows()):
        sid = row["sequence_id"]
        if sid not in records:
            logger.warning(f"Sequence {sid} not found in FASTA, skipping")
            continue

        seq = str(records[sid].seq).upper()
        npy_file = output_dir / f"{sid.replace('/', '_')[:80]}.npy"

        logger.info(f"[{i + 1}/{len(sample_df)}] {sid[:60]}... ({len(seq)} bp)")

        # Extract or load per-position embeddings
        if args.skip_existing and npy_file.exists():
            per_position = np.load(npy_file)
            logger.info(f"  Loaded cached: {per_position.shape}")
        else:
            try:
                per_position = extract_per_position_embeddings(seq, sid)
                np.save(npy_file, per_position)
            except Exception as e:
                logger.error(f"  Failed: {e}")
                all_results.append({"seq_id": sid, "error": str(e)})
                continue

        # Analyze
        result = analyze_one(seq, sid, per_position)
        result["category"] = row.get("category", "unknown")
        result["sample_group"] = row.get("sample_group", "unknown")
        result["length"] = len(seq)
        all_results.append(result)

        # Progress
        if "norm_ratio" in result:
            logger.info(
                f"  genes={result['n_genes']}, norm_ratio={result['norm_ratio']:.3f}, "
                f"coding_acc={result.get('coding_prediction_accuracy', 'N/A')}, "
                f"boundary_recall={result.get('boundary_recall', 'N/A')}"
            )

    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "batch_results.csv", index=False)

    with open(output_dir / "batch_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    valid = results_df[results_df["norm_ratio"].notna()]
    print("\n" + "=" * 80)
    print(f"GENE BOUNDARY DETECTION — {len(valid)}/{len(results_df)} sequences analyzed")
    print("=" * 80)

    print(f"\n--- Norm ratio (coding/intergenic) by category ---")
    for cat in sorted(valid["category"].unique()):
        cat_df = valid[valid["category"] == cat]
        ratios = cat_df["norm_ratio"]
        print(f"  {cat:<15}: {ratios.mean():.3f} ± {ratios.std():.3f} (n={len(cat_df)})")

    print(f"\n--- Coding prediction accuracy by category ---")
    acc_col = "coding_prediction_accuracy"
    for cat in sorted(valid["category"].unique()):
        cat_df = valid[valid["category"] == cat]
        if acc_col in cat_df.columns:
            accs = cat_df[acc_col].dropna()
            if len(accs) > 0:
                print(f"  {cat:<15}: {accs.mean():.1%} ± {accs.std():.1%} (n={len(accs)})")

    print(f"\n--- Boundary recall by category ---")
    for cat in sorted(valid["category"].unique()):
        cat_df = valid[valid["category"] == cat]
        if "boundary_recall" in cat_df.columns:
            recalls = cat_df["boundary_recall"].dropna()
            if len(recalls) > 0:
                print(f"  {cat:<15}: {recalls.mean():.1%} ± {recalls.std():.1%} (n={len(recalls)})")

    print(f"\n--- Overall ---")
    if len(valid) > 0:
        print(f"  Norm ratio:     {valid['norm_ratio'].mean():.3f} ± {valid['norm_ratio'].std():.3f}")
        if acc_col in valid.columns:
            accs = valid[acc_col].dropna()
            print(f"  Coding accuracy: {accs.mean():.1%} ± {accs.std():.1%}")
        if "boundary_recall" in valid.columns:
            recalls = valid["boundary_recall"].dropna()
            print(f"  Boundary recall: {recalls.mean():.1%} ± {recalls.std():.1%}")

    logger.info(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
