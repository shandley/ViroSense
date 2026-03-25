#!/usr/bin/env python3
"""
Validate Evo2 false positive predictions against RNA-seq coverage.

Compares per-position RNA-seq read depth at:
1. True positive positions (predicted coding, annotated coding)
2. False positive positions (predicted coding, annotated non-coding)
3. True negative positions (predicted non-coding, annotated non-coding)
4. False negative positions (predicted non-coding, annotated coding)

If FP regions show significantly higher coverage than TN regions,
this validates that our "false positives" correspond to real transcription.

Usage:
    uv run python scripts/validate_fp_rnaseq.py \
        --coverage /path/to/coverage_region.tsv \
        --organism physcomitrium_patens
"""

import argparse
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path("results/experiments/nonmodel_genome")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", type=str, required=True,
                        help="samtools depth TSV (chrom, pos, depth)")
    parser.add_argument("--organism", type=str, default="physcomitrium_patens")
    parser.add_argument("--region-start", type=int, default=16_665_000)
    args = parser.parse_args()

    prefix = f"{args.organism}_{args.region_start}_{args.region_start + 500_000}"

    # Load coverage
    print("Loading RNA-seq coverage...")
    coverage = np.zeros(500_001)
    with open(args.coverage) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            pos = int(parts[1]) - args.region_start  # convert to local coords
            depth = int(parts[2])
            if 0 <= pos < len(coverage):
                coverage[pos] = depth

    covered_positions = (coverage > 0).sum()
    print(f"  {covered_positions:,} / {len(coverage):,} positions with coverage "
          f"({covered_positions/len(coverage):.1%})")
    print(f"  Mean depth: {coverage.mean():.1f}, median: {np.median(coverage):.0f}")

    # Load our predictions
    print("\nLoading Evo2 predictions...")
    cached = np.load(DATA_DIR / f"{prefix}_cosines.npz")
    cos1, cos3 = cached["cos1"], cached["cos3"]
    with open(DATA_DIR / f"{prefix}_genes.json") as f:
        ann = json.load(f)
    with open(DATA_DIR / f"{prefix}.fasta") as f:
        seq = "".join(l.strip() for l in f if not l.startswith(">"))
    seq_len = len(seq)

    kernel = np.ones(200) / 200
    cos1_s = np.convolve(cos1[:seq_len], kernel, mode="same")
    cos3_s = np.convolve(cos3[:seq_len], kernel, mode="same")
    inversion = cos3_s - cos1_s
    predicted = (inversion > 0).astype(int)

    gene_track = np.zeros(seq_len)
    for cds in ann.get("cds", []):
        s, e = max(0, cds["start"]), min(cds["end"], seq_len)
        gene_track[s:e] = 1

    # Classify positions
    tp_mask = (predicted == 1) & (gene_track == 1)
    fp_mask = (predicted == 1) & (gene_track == 0)
    tn_mask = (predicted == 0) & (gene_track == 0)
    fn_mask = (predicted == 0) & (gene_track == 1)

    # Coverage statistics per category
    cov = coverage[:seq_len]
    categories = [
        ("True Positive (coding, predicted coding)", tp_mask),
        ("False Positive (non-coding, predicted coding)", fp_mask),
        ("True Negative (non-coding, predicted non-coding)", tn_mask),
        ("False Negative (coding, predicted non-coding)", fn_mask),
    ]

    print(f"\n{'='*80}")
    print("RNA-seq coverage by prediction category")
    print(f"{'='*80}")
    print(f"{'Category':<50s} {'N positions':>12s} {'Mean depth':>11s} "
          f"{'Median':>8s} {'% covered':>10s}")
    print(f"{'-'*80}")

    results = {}
    for name, mask in categories:
        positions = cov[mask]
        if len(positions) == 0:
            continue
        mean_d = positions.mean()
        median_d = np.median(positions)
        pct_covered = (positions > 0).sum() / len(positions) * 100
        short = name.split("(")[0].strip()
        results[short] = {
            "n": int(len(positions)),
            "mean_depth": float(mean_d),
            "median_depth": float(median_d),
            "pct_covered": float(pct_covered),
        }
        print(f"{name:<50s} {len(positions):>12,} {mean_d:>11.1f} "
              f"{median_d:>8.0f} {pct_covered:>9.1f}%")

    # Key test: FP coverage vs TN coverage
    fp_cov = cov[fp_mask]
    tn_cov = cov[tn_mask]

    if len(fp_cov) > 0 and len(tn_cov) > 0:
        fp_pct = (fp_cov > 0).sum() / len(fp_cov) * 100
        tn_pct = (tn_cov > 0).sum() / len(tn_cov) * 100
        enrichment = fp_pct / tn_pct if tn_pct > 0 else float("inf")

        print(f"\n{'='*80}")
        print("KEY RESULT: False positive transcription enrichment")
        print(f"{'='*80}")
        print(f"FP positions with RNA-seq coverage: {fp_pct:.1f}%")
        print(f"TN positions with RNA-seq coverage: {tn_pct:.1f}%")
        print(f"Enrichment (FP / TN): {enrichment:.2f}×")
        print(f"FP mean depth: {fp_cov.mean():.1f} vs TN mean depth: {tn_cov.mean():.1f}")

        # Mann-Whitney U test
        from scipy.stats import mannwhitneyu
        # Sample to avoid memory issues
        n_sample = min(50000, len(fp_cov), len(tn_cov))
        fp_sample = np.random.choice(fp_cov, n_sample, replace=False)
        tn_sample = np.random.choice(tn_cov, n_sample, replace=False)
        stat, pvalue = mannwhitneyu(fp_sample, tn_sample, alternative="greater")
        print(f"Mann-Whitney U (FP > TN): p = {pvalue:.2e}")

        if enrichment > 1.5 and pvalue < 0.001:
            print(f"\n*** VALIDATED: FP regions show {enrichment:.1f}× more transcription ***")
            print("*** This supports gene discovery — 'false positives' are transcribed ***")
        elif enrichment > 1.2:
            print(f"\n* Modest enrichment ({enrichment:.1f}×) — suggestive but not definitive *")
        else:
            print(f"\n  No enrichment — FPs do not show elevated transcription")

    # Contiguous FP run analysis
    print(f"\n{'='*80}")
    print("Contiguous FP run transcription")
    print(f"{'='*80}")

    fp_runs = []
    in_run = False
    run_start = 0
    for i in range(seq_len):
        if fp_mask[i] and not in_run:
            run_start = i
            in_run = True
        elif not fp_mask[i] and in_run:
            if i - run_start >= 300:
                fp_runs.append((run_start, i))
            in_run = False

    transcribed_runs = 0
    for rs, re in fp_runs:
        run_cov = cov[rs:re]
        pct = (run_cov > 0).sum() / len(run_cov) * 100
        if pct > 50:
            transcribed_runs += 1

    print(f"FP runs >= 300bp: {len(fp_runs)}")
    print(f"Runs with >50% positions covered by RNA-seq: {transcribed_runs} "
          f"({transcribed_runs/len(fp_runs):.0%})")

    # Save results
    output = {
        "organism": args.organism,
        "region_start": args.region_start,
        "categories": results,
        "fp_transcription_enrichment": enrichment if len(fp_cov) > 0 else None,
        "fp_runs_total": len(fp_runs),
        "fp_runs_transcribed": transcribed_runs,
    }
    out_path = DATA_DIR / f"{args.organism}_rnaseq_validation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
