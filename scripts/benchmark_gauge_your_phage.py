#!/usr/bin/env python3
"""Benchmark ViroSense on the Gauge Your Phage dataset.

Gauge Your Phage (GYP) is a community benchmark for phage detection tools.
It provides pre-fragmented 1-15 kbp contigs from phage genomes, bacterial
chromosomes, and plasmids.

This script:
1. Samples a stratified subset from each class (length-balanced)
2. Extracts Evo2 embeddings via NIM API
3. Classifies using the reference model
4. Reports metrics comparable to published GYP results

Usage:
    python scripts/benchmark_gauge_your_phage.py \
        --phage data/benchmarks/gauge_your_phage/phage_fragment_set.fasta \
        --chromosome data/benchmarks/gauge_your_phage/host_chr_pvog_fragments.fasta \
        --plasmid data/benchmarks/gauge_your_phage/host_plasmid_pvog_fragments.fasta \
        --model ~/.virosense/models/reference_classifier.joblib \
        --output data/validation/gauge_your_phage/ \
        --n-phage 500 --n-chromosome 500 --n-plasmid 200 \
        --cache-dir data/validation/gauge_your_phage/cache
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# IUPAC ambiguity codes beyond A/C/G/T/N
_IUPAC_AMBIG = re.compile(r"[^ACGTN]")

# GYP length bins
LENGTH_BINS = [(1000, 3000), (3000, 5000), (5000, 10000), (10000, 15001)]
BIN_NAMES = ["1-3kb", "3-5kb", "5-10kb", "10-15kb"]


def _sanitize_iupac(seq: str) -> str:
    """Replace IUPAC ambiguity codes with N."""
    return _IUPAC_AMBIG.sub("N", seq.upper())


def sample_sequences(
    fasta_path: Path,
    n_samples: int,
    seed: int = 42,
) -> dict[str, str]:
    """Sample sequences stratified by length bin."""
    rng = np.random.default_rng(seed)

    # Read all sequences
    all_seqs = {}
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        all_seqs[rec.id] = _sanitize_iupac(str(rec.seq))

    if len(all_seqs) <= n_samples:
        logger.info(f"Using all {len(all_seqs)} sequences from {fasta_path.name}")
        return all_seqs

    # Bin by length
    binned: dict[str, list[str]] = {name: [] for name in BIN_NAMES}
    for sid, seq in all_seqs.items():
        slen = len(seq)
        for (lo, hi), name in zip(LENGTH_BINS, BIN_NAMES):
            if lo <= slen < hi:
                binned[name].append(sid)
                break

    # Sample equally per bin
    per_bin = n_samples // len(BIN_NAMES)
    selected = {}
    for name, ids in binned.items():
        n = min(per_bin, len(ids))
        if n == 0:
            continue
        chosen = rng.choice(ids, size=n, replace=False)
        for sid in chosen:
            selected[sid] = all_seqs[sid]
        logger.info(f"  {name}: {n}/{len(ids)} sequences")

    # Fill remaining from largest bins
    remaining = n_samples - len(selected)
    if remaining > 0:
        pool = [sid for sid in all_seqs if sid not in selected]
        if pool:
            extra = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
            for sid in extra:
                selected[sid] = all_seqs[sid]

    logger.info(f"Selected {len(selected)} from {fasta_path.name}")
    return selected


def run_benchmark(
    phage_fasta: Path,
    chromosome_fasta: Path,
    plasmid_fasta: Path,
    model_path: Path,
    output_dir: Path,
    n_phage: int = 500,
    n_chromosome: int = 500,
    n_plasmid: int = 200,
    cache_dir: Path | None = None,
    nim_url: str | None = None,
    batch_size: int = 16,
    max_concurrent: int | None = None,
) -> dict:
    """Run the Gauge Your Phage benchmark."""
    import os

    from virosense.backends.base import get_backend
    from virosense.features.evo2_embeddings import extract_embeddings
    from virosense.models.detector import ViralClassifier, classify_contigs

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sample sequences
    logger.info("=== Sampling sequences ===")
    logger.info(f"Phage ({n_phage}):")
    phage_seqs = sample_sequences(phage_fasta, n_phage)
    logger.info(f"Chromosome ({n_chromosome}):")
    chr_seqs = sample_sequences(chromosome_fasta, n_chromosome)
    logger.info(f"Plasmid ({n_plasmid}):")
    plas_seqs = sample_sequences(plasmid_fasta, n_plasmid)

    # Combine
    all_sequences = {}
    ground_truth = {}  # sid -> class name

    for sid, seq in phage_seqs.items():
        all_sequences[sid] = seq
        ground_truth[sid] = "phage"
    for sid, seq in chr_seqs.items():
        all_sequences[sid] = seq
        ground_truth[sid] = "chromosome"
    for sid, seq in plas_seqs.items():
        all_sequences[sid] = seq
        ground_truth[sid] = "plasmid"

    total = len(all_sequences)
    logger.info(
        f"Benchmark set: {len(phage_seqs)} phage + {len(chr_seqs)} chromosome "
        f"+ {len(plas_seqs)} plasmid = {total} total"
    )

    # Save manifest
    manifest = []
    for sid, seq in all_sequences.items():
        manifest.append({
            "sequence_id": sid,
            "length": len(seq),
            "category": ground_truth[sid],
        })
    pd.DataFrame(manifest).to_csv(
        output_dir / "benchmark_manifest.tsv", sep="\t", index=False
    )

    # 2. Extract embeddings
    logger.info("=== Extracting Evo2 embeddings ===")
    api_key = os.environ.get("NVIDIA_API_KEY")
    backend = get_backend(
        "nim", api_key=api_key, nim_url=nim_url, max_concurrent=max_concurrent,
    )

    if not backend.is_available():
        raise RuntimeError(
            "NIM backend not available. Set NVIDIA_API_KEY or use --nim-url."
        )

    cache_path = cache_dir or output_dir / "cache"
    result = extract_embeddings(
        sequences=all_sequences,
        backend=backend,
        layer="blocks.28.mlp.l3",
        model="evo2_7b",
        batch_size=batch_size,
        cache_dir=cache_path,
        checkpoint_every=25,
    )

    # 3. Classify
    logger.info("=== Running classification ===")
    classifier = ViralClassifier.load(model_path)
    sequence_lengths = [len(all_sequences[sid]) for sid in result.sequence_ids]
    detection_results = classify_contigs(
        embeddings=result.embeddings,
        sequence_ids=result.sequence_ids,
        sequence_lengths=sequence_lengths,
        classifier=classifier,
        threshold=0.5,
    )

    # Detect 3-class model
    n_classes = classifier.metadata.get("n_classes", 2)
    is_3class = n_classes == 3

    # 4. Build results dataframe
    rows = []
    for dr in detection_results:
        cat = ground_truth[dr.contig_id]
        row = {
            "sequence_id": dr.contig_id,
            "contig_length": dr.contig_length,
            "viral_score": dr.viral_score,
            "classification": dr.classification,
            "true_category": cat,
            # Binary: phage=viral, chromosome+plasmid=non-viral
            "true_viral": 1 if cat == "phage" else 0,
            "pred_viral": 1 if dr.classification == "viral" else 0,
        }
        if is_3class:
            row["chromosome_score"] = dr.chromosome_score
            row["plasmid_score"] = dr.plasmid_score
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "detailed_results.tsv", sep="\t", index=False)

    # 5. Compute metrics
    logger.info("=== Computing metrics ===")
    results = _compute_metrics(df, output_dir)

    return results


def _compute_metrics(df: pd.DataFrame, output_dir: Path) -> dict:
    """Compute comprehensive benchmark metrics."""
    y_true = df["true_viral"].values
    y_pred = df["pred_viral"].values
    y_score = df["viral_score"].values

    # --- Overall binary metrics ---
    overall = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "specificity": float(
            (df[df["true_viral"] == 0]["pred_viral"] == 0).mean()
        ),
    }
    try:
        overall["auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        overall["auc"] = None

    # Per-class stats
    for cat in ["phage", "chromosome", "plasmid"]:
        subset = df[df["true_category"] == cat]
        if len(subset) == 0:
            continue
        is_viral = cat == "phage"
        if is_viral:
            # Sensitivity = correctly called viral
            overall[f"{cat}_sensitivity"] = float(
                (subset["pred_viral"] == 1).mean()
            )
        else:
            # Specificity = correctly called non-viral
            overall[f"{cat}_specificity"] = float(
                (subset["pred_viral"] == 0).mean()
            )
        overall[f"{cat}_mean_score"] = float(subset["viral_score"].mean())
        overall[f"{cat}_n"] = int(len(subset))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    overall["confusion_matrix"] = cm.tolist()

    # --- Per-length-bin metrics ---
    length_metrics = {}
    for (lo, hi), bin_name in zip(LENGTH_BINS, BIN_NAMES):
        mask = (df["contig_length"] >= lo) & (df["contig_length"] < hi)
        bin_df = df[mask]
        if len(bin_df) == 0:
            continue

        bm = {"n_total": len(bin_df)}
        for cat in ["phage", "chromosome", "plasmid"]:
            cat_df = bin_df[bin_df["true_category"] == cat]
            bm[f"n_{cat}"] = len(cat_df)
            if len(cat_df) == 0:
                continue
            if cat == "phage":
                bm["phage_sensitivity"] = float(
                    (cat_df["pred_viral"] == 1).mean()
                )
                bm["phage_mean_score"] = float(cat_df["viral_score"].mean())
            else:
                bm[f"{cat}_specificity"] = float(
                    (cat_df["pred_viral"] == 0).mean()
                )

        if len(bin_df["true_viral"].unique()) > 1:
            bm["accuracy"] = float(
                accuracy_score(bin_df["true_viral"], bin_df["pred_viral"])
            )
            bm["f1"] = float(
                f1_score(bin_df["true_viral"], bin_df["pred_viral"])
            )

        length_metrics[bin_name] = bm

    # --- Threshold sweep ---
    thresholds = {}
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred_t = (y_score >= t).astype(int)
        thresholds[str(t)] = {
            "sensitivity": float(recall_score(y_true, pred_t)),
            "specificity": float(
                ((y_true == 0) & (pred_t == 0)).sum() / (y_true == 0).sum()
            ),
            "precision": float(
                precision_score(y_true, pred_t, zero_division=0)
            ),
            "f1": float(f1_score(y_true, pred_t)),
        }

    # --- Plasmid vs phage analysis ---
    phage_plasmid = df[df["true_category"].isin(["phage", "plasmid"])]
    plasmid_analysis = {}
    if len(phage_plasmid) > 0:
        plas = df[df["true_category"] == "plasmid"]
        plasmid_analysis = {
            "n_plasmid": len(plas),
            "plasmid_called_viral": int((plas["pred_viral"] == 1).sum()),
            "plasmid_fp_rate": float((plas["pred_viral"] == 1).mean()),
            "plasmid_mean_score": float(plas["viral_score"].mean()),
            "plasmid_max_score": float(plas["viral_score"].max()),
        }

    # --- 3-class evaluation (if model supports it) ---
    threeclass_metrics = None
    if "chromosome_score" in df.columns and df["chromosome_score"].notna().any():
        # Map true category to predicted class from model
        pred_class = df["classification"].values
        true_class = df["true_category"].values

        # Normalize: GYP uses "phage" but model uses "viral"
        true_class_normalized = np.array([
            "viral" if c == "phage" else c for c in true_class
        ])

        # 3-class confusion matrix and report
        all_labels = sorted(set(true_class_normalized) | set(pred_class))
        threeclass_metrics = {
            "confusion_matrix_3class": confusion_matrix(
                true_class_normalized, pred_class, labels=all_labels
            ).tolist(),
            "labels": all_labels,
            "classification_report_3class": classification_report(
                true_class_normalized, pred_class,
                labels=all_labels,
                output_dict=True,
                zero_division=0,
            ),
        }

        # Key metric: what fraction of plasmids are correctly classified as plasmid?
        plas_mask = true_class_normalized == "plasmid"
        if plas_mask.sum() > 0:
            plas_pred = pred_class[plas_mask]
            threeclass_metrics["plasmid_as_plasmid"] = float(
                (plas_pred == "plasmid").mean()
            )
            threeclass_metrics["plasmid_as_viral"] = float(
                (plas_pred == "viral").mean()
            )
            threeclass_metrics["plasmid_as_chromosome"] = float(
                (plas_pred == "chromosome").mean()
            )

    # --- Assemble and save ---
    benchmark_results = {
        "experiment": "gauge_your_phage_benchmark",
        "description": (
            "ViroSense reference classifier on Gauge Your Phage dataset "
            "(stratified sample)"
        ),
        "overall_metrics": overall,
        "length_bin_metrics": length_metrics,
        "threshold_sweep": thresholds,
        "plasmid_analysis": plasmid_analysis,
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=["non-viral", "viral"],
            output_dict=True,
        ),
    }
    if threeclass_metrics:
        benchmark_results["threeclass_metrics"] = threeclass_metrics

    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)

    # --- Print summary ---
    logger.info("=" * 65)
    logger.info("GAUGE YOUR PHAGE BENCHMARK RESULTS")
    logger.info("=" * 65)
    logger.info(f"Phage: {overall.get('phage_n', 0)}, "
                f"Chromosome: {overall.get('chromosome_n', 0)}, "
                f"Plasmid: {overall.get('plasmid_n', 0)}")
    logger.info(f"Overall accuracy:         {overall['accuracy']:.3f}")
    logger.info(f"Phage sensitivity:        {overall.get('phage_sensitivity', 'N/A')}")
    logger.info(f"Chromosome specificity:   {overall.get('chromosome_specificity', 'N/A')}")
    logger.info(f"Plasmid specificity:      {overall.get('plasmid_specificity', 'N/A')}")
    logger.info(f"Precision:                {overall['precision']:.3f}")
    logger.info(f"F1 score:                 {overall['f1']:.3f}")
    logger.info(f"AUC:                      {overall.get('auc', 'N/A')}")
    logger.info("")
    logger.info("Per-length-bin phage sensitivity:")
    for bin_name, bm in length_metrics.items():
        sens = bm.get("phage_sensitivity", "N/A")
        n = bm.get("n_phage", 0)
        if isinstance(sens, float):
            logger.info(f"  {bin_name:>8}: {sens:.3f} ({n} phage seqs)")
        else:
            logger.info(f"  {bin_name:>8}: {sens} ({n} phage seqs)")
    logger.info("")
    logger.info("Per-length-bin chromosome specificity:")
    for bin_name, bm in length_metrics.items():
        spec = bm.get("chromosome_specificity", "N/A")
        n = bm.get("n_chromosome", 0)
        if isinstance(spec, float):
            logger.info(f"  {bin_name:>8}: {spec:.3f} ({n} chr seqs)")
        else:
            logger.info(f"  {bin_name:>8}: {spec} ({n} chr seqs)")

    if plasmid_analysis:
        logger.info("")
        logger.info(f"Plasmid false positive rate: "
                    f"{plasmid_analysis['plasmid_fp_rate']:.3f} "
                    f"({plasmid_analysis['plasmid_called_viral']}/"
                    f"{plasmid_analysis['n_plasmid']})")
        logger.info(f"Plasmid mean score:         "
                    f"{plasmid_analysis['plasmid_mean_score']:.4f}")

    # Threshold sweep summary
    logger.info("")
    logger.info("Threshold sweep:")
    logger.info(f"  {'Threshold':>9}  {'Sens':>6}  {'Spec':>6}  {'Prec':>6}  {'F1':>6}")
    for t, tm in thresholds.items():
        logger.info(f"  {t:>9}  {tm['sensitivity']:>6.3f}  {tm['specificity']:>6.3f}  "
                    f"{tm['precision']:>6.3f}  {tm['f1']:>6.3f}")

    logger.info("=" * 65)

    # 3-class results summary
    if threeclass_metrics:
        logger.info("")
        logger.info("3-CLASS MODEL RESULTS:")
        if "plasmid_as_plasmid" in threeclass_metrics:
            logger.info(f"  Plasmid → plasmid:    {threeclass_metrics['plasmid_as_plasmid']:.3f}")
            logger.info(f"  Plasmid → viral:      {threeclass_metrics['plasmid_as_viral']:.3f}")
            logger.info(f"  Plasmid → chromosome: {threeclass_metrics['plasmid_as_chromosome']:.3f}")
        report = threeclass_metrics.get("classification_report_3class", {})
        for cls in threeclass_metrics.get("labels", []):
            if cls in report:
                m = report[cls]
                logger.info(f"  {cls:>12}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1-score']:.3f}")

    # Print misclassified phages
    fn = df[(df["true_category"] == "phage") & (df["pred_viral"] == 0)]
    if len(fn) > 0:
        logger.info(f"\nFalse negatives ({len(fn)} phages missed):")
        for _, row in fn.sort_values("viral_score").head(20).iterrows():
            logger.info(f"  {row['sequence_id'][:60]}: "
                        f"score={row['viral_score']:.4f}, "
                        f"length={row['contig_length']}")

    # Print false positive chromosomes/plasmids
    fp = df[(df["true_viral"] == 0) & (df["pred_viral"] == 1)]
    if len(fp) > 0:
        logger.info(f"\nFalse positives ({len(fp)} non-viral called viral):")
        for _, row in fp.sort_values("viral_score", ascending=False).head(20).iterrows():
            logger.info(f"  {row['sequence_id'][:60]}: "
                        f"score={row['viral_score']:.4f}, "
                        f"length={row['contig_length']}, "
                        f"type={row['true_category']}")

    return benchmark_results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ViroSense on Gauge Your Phage dataset"
    )
    parser.add_argument(
        "--phage", required=True, type=Path,
        help="Phage fragment FASTA"
    )
    parser.add_argument(
        "--chromosome", required=True, type=Path,
        help="Chromosome fragment FASTA"
    )
    parser.add_argument(
        "--plasmid", required=True, type=Path,
        help="Plasmid fragment FASTA"
    )
    parser.add_argument(
        "--model", type=Path,
        default=Path.home() / ".virosense" / "models" / "reference_classifier.joblib",
        help="Path to trained classifier model"
    )
    parser.add_argument(
        "--output", "-o", required=True, type=Path,
        help="Output directory"
    )
    parser.add_argument("--n-phage", type=int, default=500)
    parser.add_argument("--n-chromosome", type=int, default=500)
    parser.add_argument("--n-plasmid", type=int, default=200)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--nim-url", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-concurrent", type=int, default=None,
                        help="Max concurrent NIM requests")
    args = parser.parse_args()

    run_benchmark(
        phage_fasta=args.phage,
        chromosome_fasta=args.chromosome,
        plasmid_fasta=args.plasmid,
        model_path=args.model,
        output_dir=args.output,
        n_phage=args.n_phage,
        n_chromosome=args.n_chromosome,
        n_plasmid=args.n_plasmid,
        cache_dir=args.cache_dir,
        nim_url=args.nim_url,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
    )


if __name__ == "__main__":
    main()
