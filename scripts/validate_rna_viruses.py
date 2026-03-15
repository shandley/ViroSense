#!/usr/bin/env python3
"""Validate ViroSense reference classifier on RNA virus sequences.

Tests zero-shot generalization: the reference classifier was trained only
on prokaryotic phages + bacteria. This script evaluates whether it can
correctly detect eukaryotic RNA viruses (as cDNA) that Evo2 was never
trained on (biosecurity exclusion).

The validation creates a balanced test set:
- Positive class: RNA virus sequences (NC_ RefSeq, complete genomes)
- Negative class: Bacterial sequences (from reference training data)

All sequences are sent through the NIM API for Evo2 embedding extraction,
then classified by the existing reference model.

Usage:
    python scripts/validate_rna_viruses.py \
        --rna-fasta data/reference/rna_viruses/RNA_virus_database.fasta \
        --cellular-fasta data/reference/cleaned/sequences.fasta \
        --cellular-labels data/reference/cleaned/labels.tsv \
        --model ~/.virosense/models/reference_classifier.joblib \
        --output data/validation/rna_virus/ \
        --n-samples 200 \
        --cache-dir data/validation/rna_virus/cache
"""

import argparse
import json
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


import re

# IUPAC ambiguity codes beyond A/C/G/T/N
_IUPAC_AMBIG = re.compile(r"[^ACGTN]")


def _sanitize_iupac(seq: str) -> str:
    """Replace IUPAC ambiguity codes (R, Y, W, S, M, K, etc.) with N."""
    return _IUPAC_AMBIG.sub("N", seq.upper())


def select_rna_virus_sequences(
    fasta_path: Path,
    n_samples: int,
    min_length: int = 500,
    max_length: int = 16000,
    seed: int = 42,
) -> dict[str, str]:
    """Select a stratified sample of NC_ RNA virus sequences.

    Filters to RefSeq complete genomes (NC_ accessions) within the NIM
    context window, then samples uniformly across length bins.

    Returns:
        Dict of sequence_id -> sequence.
    """
    rng = np.random.default_rng(seed)

    # Collect all NC_ sequences in range
    candidates = {}
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        if not rec.id.startswith("NC_"):
            continue
        seq = _sanitize_iupac(str(rec.seq))
        if min_length <= len(seq) <= max_length:
            candidates[rec.id] = seq

    logger.info(
        f"Found {len(candidates)} NC_ sequences in "
        f"{min_length}-{max_length} bp range"
    )

    if len(candidates) <= n_samples:
        logger.info(f"Using all {len(candidates)} candidates (< {n_samples} requested)")
        return candidates

    # Stratified sampling by length bin
    bins = [(500, 1000), (1000, 3000), (3000, 5000), (5000, 10000), (10000, 16000)]
    binned: dict[str, list[str]] = {f"{lo}-{hi}": [] for lo, hi in bins}

    for sid, seq in candidates.items():
        slen = len(seq)
        for lo, hi in bins:
            if lo <= slen < hi or (hi == 16000 and slen == hi):
                binned[f"{lo}-{hi}"].append(sid)
                break

    # Sample proportionally from each bin
    per_bin = max(1, n_samples // len(bins))
    selected = {}
    for bin_name, ids in binned.items():
        n = min(per_bin, len(ids))
        if n == 0:
            continue
        chosen = rng.choice(ids, size=n, replace=False)
        for sid in chosen:
            selected[sid] = candidates[sid]
        logger.info(f"  Bin {bin_name}: {n}/{len(ids)} sequences")

    # Fill remaining slots from largest bins
    remaining = n_samples - len(selected)
    if remaining > 0:
        pool = [sid for sid in candidates if sid not in selected]
        extra = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
        for sid in extra:
            selected[sid] = candidates[sid]

    logger.info(f"Selected {len(selected)} RNA virus sequences")
    return selected


def select_cellular_sequences(
    fasta_path: Path,
    labels_path: Path,
    n_samples: int,
    seed: int = 42,
) -> dict[str, str]:
    """Select cellular (label=0) sequences from reference training data.

    Returns:
        Dict of sequence_id -> sequence.
    """
    rng = np.random.default_rng(seed)

    # Read labels to find cellular sequences
    labels_df = pd.read_csv(labels_path, sep="\t")
    cellular_ids = set(
        labels_df[labels_df["label"] == 0]["sequence_id"].values
    )

    # Read sequences
    cellular = {}
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        if rec.id in cellular_ids:
            cellular[rec.id] = str(rec.seq).upper()

    logger.info(f"Found {len(cellular)} cellular sequences in reference data")

    if len(cellular) <= n_samples:
        return cellular

    chosen_ids = rng.choice(list(cellular.keys()), size=n_samples, replace=False)
    selected = {sid: cellular[sid] for sid in chosen_ids}
    logger.info(f"Selected {len(selected)} cellular sequences")
    return selected


def run_validation(
    rna_fasta: Path,
    cellular_fasta: Path,
    cellular_labels: Path,
    model_path: Path,
    output_dir: Path,
    n_samples: int = 200,
    cache_dir: Path | None = None,
    nim_url: str | None = None,
    batch_size: int = 16,
) -> dict:
    """Run the RNA virus validation experiment.

    1. Select balanced RNA virus + cellular test set
    2. Extract Evo2 embeddings via NIM
    3. Classify with reference model
    4. Compute metrics by category and length bin

    Returns:
        Dict with validation metrics.
    """
    from virosense.backends.base import get_backend
    from virosense.features.evo2_embeddings import extract_embeddings
    from virosense.models.detector import ViralClassifier, classify_contigs

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Select test sequences
    logger.info("=== Selecting test sequences ===")
    rna_seqs = select_rna_virus_sequences(rna_fasta, n_samples)
    cellular_seqs = select_cellular_sequences(
        cellular_fasta, cellular_labels, n_samples
    )

    # Combine into unified test set
    all_sequences = {}
    ground_truth = {}  # seq_id -> "rna_virus" or "cellular"

    for sid, seq in rna_seqs.items():
        all_sequences[sid] = seq
        ground_truth[sid] = "rna_virus"

    for sid, seq in cellular_seqs.items():
        all_sequences[sid] = seq
        ground_truth[sid] = "cellular"

    logger.info(
        f"Test set: {len(rna_seqs)} RNA viruses + "
        f"{len(cellular_seqs)} cellular = {len(all_sequences)} total"
    )

    # Save test set composition
    test_manifest = []
    for sid, seq in all_sequences.items():
        test_manifest.append({
            "sequence_id": sid,
            "length": len(seq),
            "category": ground_truth[sid],
        })
    manifest_df = pd.DataFrame(test_manifest)
    manifest_df.to_csv(output_dir / "test_manifest.tsv", sep="\t", index=False)

    # 2. Extract embeddings
    logger.info("=== Extracting Evo2 embeddings ===")
    import os
    api_key = os.environ.get("NVIDIA_API_KEY")
    backend = get_backend("nim", api_key=api_key, nim_url=nim_url)

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

    # 3. Load classifier and classify
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

    # 4. Build results dataframe
    rows = []
    for dr in detection_results:
        rows.append({
            "sequence_id": dr.contig_id,
            "contig_length": dr.contig_length,
            "viral_score": dr.viral_score,
            "classification": dr.classification,
            "true_category": ground_truth[dr.contig_id],
            "true_label": 1 if ground_truth[dr.contig_id] == "rna_virus" else 0,
            "pred_label": 1 if dr.classification == "viral" else 0,
        })
    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "detailed_results.tsv", sep="\t", index=False)

    # 5. Compute metrics
    logger.info("=== Computing metrics ===")
    y_true = results_df["true_label"].values
    y_pred = results_df["pred_label"].values
    y_score = results_df["viral_score"].values

    overall = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_score)),
        "n_rna_virus": int((y_true == 1).sum()),
        "n_cellular": int((y_true == 0).sum()),
        "n_total": len(y_true),
    }

    # Sensitivity on RNA viruses (recall for viral class)
    rna_mask = results_df["true_category"] == "rna_virus"
    rna_results = results_df[rna_mask]
    overall["rna_virus_sensitivity"] = float(
        (rna_results["pred_label"] == 1).mean()
    )
    overall["rna_virus_mean_score"] = float(rna_results["viral_score"].mean())
    overall["rna_virus_min_score"] = float(rna_results["viral_score"].min())

    # Specificity on cellular (true negative rate)
    cell_mask = results_df["true_category"] == "cellular"
    cell_results = results_df[cell_mask]
    overall["cellular_specificity"] = float(
        (cell_results["pred_label"] == 0).mean()
    )
    overall["cellular_mean_score"] = float(cell_results["viral_score"].mean())
    overall["cellular_max_score"] = float(cell_results["viral_score"].max())

    # Per-length-bin metrics
    bins = [(500, 1000), (1000, 3000), (3000, 5000), (5000, 10000), (10000, 16000)]
    length_metrics = {}
    for lo, hi in bins:
        bin_name = f"{lo}-{hi}"
        mask = (results_df["contig_length"] >= lo) & (results_df["contig_length"] < hi)
        if hi == 16000:
            mask = mask | (results_df["contig_length"] == hi)
        bin_df = results_df[mask]
        if len(bin_df) == 0:
            continue

        bin_rna = bin_df[bin_df["true_category"] == "rna_virus"]
        bin_cell = bin_df[bin_df["true_category"] == "cellular"]

        bin_metrics = {
            "n_total": len(bin_df),
            "n_rna_virus": len(bin_rna),
            "n_cellular": len(bin_cell),
        }
        if len(bin_rna) > 0:
            bin_metrics["rna_sensitivity"] = float(
                (bin_rna["pred_label"] == 1).mean()
            )
            bin_metrics["rna_mean_score"] = float(bin_rna["viral_score"].mean())
        if len(bin_cell) > 0:
            bin_metrics["cell_specificity"] = float(
                (bin_cell["pred_label"] == 0).mean()
            )
        if len(bin_df["true_label"].unique()) > 1:
            bin_metrics["accuracy"] = float(
                accuracy_score(bin_df["true_label"], bin_df["pred_label"])
            )
        length_metrics[bin_name] = bin_metrics

    # Misclassified sequences
    misclassified = results_df[results_df["true_label"] != results_df["pred_label"]]
    false_negatives = misclassified[misclassified["true_category"] == "rna_virus"]
    false_positives = misclassified[misclassified["true_category"] == "cellular"]

    overall["n_false_negatives"] = len(false_negatives)
    overall["n_false_positives"] = len(false_positives)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    overall["confusion_matrix"] = cm.tolist()

    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=["cellular", "viral"],
        output_dict=True,
    )

    # Full results dict
    validation_results = {
        "experiment": "rna_virus_zero_shot_validation",
        "description": (
            "Testing reference classifier (trained on phages + bacteria) "
            "on eukaryotic RNA virus cDNA sequences"
        ),
        "overall_metrics": overall,
        "length_bin_metrics": length_metrics,
        "classification_report": report,
    }

    # Save results
    with open(output_dir / "validation_results.json", "w") as f:
        json.dump(validation_results, f, indent=2)

    # Print summary
    logger.info("=" * 60)
    logger.info("RNA VIRUS VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Test set: {overall['n_rna_virus']} RNA viruses + "
                f"{overall['n_cellular']} cellular")
    logger.info(f"Overall accuracy:        {overall['accuracy']:.3f}")
    logger.info(f"RNA virus sensitivity:   {overall['rna_virus_sensitivity']:.3f}")
    logger.info(f"Cellular specificity:    {overall['cellular_specificity']:.3f}")
    logger.info(f"F1 score:                {overall['f1']:.3f}")
    logger.info(f"AUC:                     {overall['auc']:.3f}")
    logger.info(f"RNA virus mean score:    {overall['rna_virus_mean_score']:.4f}")
    logger.info(f"RNA virus min score:     {overall['rna_virus_min_score']:.4f}")
    logger.info(f"Cellular mean score:     {overall['cellular_mean_score']:.4f}")
    logger.info(f"Cellular max score:      {overall['cellular_max_score']:.4f}")
    logger.info(f"False negatives (missed viruses): {overall['n_false_negatives']}")
    logger.info(f"False positives (false alarms):   {overall['n_false_positives']}")
    logger.info("")
    logger.info("Per-length-bin sensitivity (RNA viruses):")
    for bin_name, bm in length_metrics.items():
        if "rna_sensitivity" in bm:
            logger.info(f"  {bin_name:>12} bp: {bm['rna_sensitivity']:.3f} "
                        f"({bm['n_rna_virus']} seqs)")
    logger.info("=" * 60)

    if len(false_negatives) > 0:
        logger.info("\nFalse negatives (missed RNA viruses):")
        for _, row in false_negatives.iterrows():
            logger.info(f"  {row['sequence_id']}: score={row['viral_score']:.4f}, "
                        f"length={row['contig_length']}")

    if len(false_positives) > 0:
        logger.info("\nFalse positives (cellular called viral):")
        for _, row in false_positives.head(10).iterrows():
            logger.info(f"  {row['sequence_id']}: score={row['viral_score']:.4f}, "
                        f"length={row['contig_length']}")

    return validation_results


def main():
    parser = argparse.ArgumentParser(
        description="Validate ViroSense on RNA virus sequences (zero-shot)"
    )
    parser.add_argument(
        "--rna-fasta", required=True, type=Path,
        help="RNA virus database FASTA (e.g., RNA_virus_database.fasta)"
    )
    parser.add_argument(
        "--cellular-fasta", required=True, type=Path,
        help="Cellular sequences FASTA (from reference training data)"
    )
    parser.add_argument(
        "--cellular-labels", required=True, type=Path,
        help="Labels TSV for cellular FASTA (sequence_id, label columns)"
    )
    parser.add_argument(
        "--model", type=Path,
        default=Path.home() / ".virosense" / "models" / "reference_classifier.joblib",
        help="Path to trained classifier model"
    )
    parser.add_argument(
        "--output", "-o", required=True, type=Path,
        help="Output directory for validation results"
    )
    parser.add_argument(
        "--n-samples", type=int, default=200,
        help="Number of sequences per class (default: 200)"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Embedding cache directory (enables resume)"
    )
    parser.add_argument(
        "--nim-url", default=None,
        help="Self-hosted NIM URL (e.g., http://localhost:8000)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for embedding extraction"
    )
    args = parser.parse_args()

    run_validation(
        rna_fasta=args.rna_fasta,
        cellular_fasta=args.cellular_fasta,
        cellular_labels=args.cellular_labels,
        model_path=args.model,
        output_dir=args.output,
        n_samples=args.n_samples,
        cache_dir=args.cache_dir,
        nim_url=args.nim_url,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
