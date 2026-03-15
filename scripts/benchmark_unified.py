"""Unified benchmark for ViroSense 40B vs 7B head-to-head comparison.

Three subcommands:
  manifest  Generate stratified benchmark samples (shared between tiers)
  run       Extract embeddings, classify, compute metrics for one tier
  compare   Side-by-side comparison of two tier results

Example workflow:
    # Step 1: Generate shared manifest (run once, both tiers use same seqs)
    python scripts/benchmark_unified.py manifest \
        --gyp-phage data/benchmarks/gauge_your_phage/phage_fragment_set.fasta \
        --gyp-chromosome data/benchmarks/gauge_your_phage/host_chr_pvog_fragments.fasta \
        --gyp-plasmid data/benchmarks/gauge_your_phage/host_plasmid_pvog_fragments.fasta \
        --rna-fasta data/reference/rna_viruses/RNA_virus_database.fasta \
        --cellular-fasta data/reference/3class/sequences.fasta \
        --cellular-labels data/reference/3class/labels.tsv \
        --output results/benchmark/manifest/

    # Step 2a: Run 40B benchmark (cloud NIM)
    python scripts/benchmark_unified.py run \
        --manifest results/benchmark/manifest/ \
        --classifier data/reference/model/classifier.joblib \
        --backend nim --model evo2_40b \
        --cache-dir results/benchmark/cache_40b/ \
        --output results/benchmark/40b/

    # Step 2b: Run 7B benchmark (HTCF NIM)
    python scripts/benchmark_unified.py run \
        --manifest results/benchmark/manifest/ \
        --classifier model_7b/classifier.joblib \
        --backend nim --nim-url http://n099:8000 --model evo2_7b \
        --cache-dir results/benchmark/cache_7b/ \
        --output results/benchmark/7b/

    # Step 3: Compare tiers
    python scripts/benchmark_unified.py compare \
        --tier-a results/benchmark/40b/ --label-a "Evo2 40B" \
        --tier-b results/benchmark/7b/ --label-b "Evo2 7B" \
        --output results/benchmark/comparison/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GYP_BINS = [(1000, 3000), (3000, 5000), (5000, 10000), (10000, 15000)]
GYP_BIN_NAMES = ["1-3kb", "3-5kb", "5-10kb", "10-15kb"]
RNA_BINS = [(500, 1000), (1000, 3000), (3000, 5000), (5000, 10000), (10000, 16000)]
RNA_BIN_NAMES = ["500-1kb", "1-3kb", "3-5kb", "5-10kb", "10-16kb"]

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _read_fasta(path: str | Path) -> dict[str, str]:
    """Read FASTA file into {id: sequence} dict."""
    from Bio import SeqIO
    seqs = {}
    for rec in SeqIO.parse(str(path), "fasta"):
        seqs[rec.id] = str(rec.seq).upper()
    logger.info(f"Read {len(seqs)} sequences from {Path(path).name}")
    return seqs


def _bin_sequences(
    seqs: dict[str, str], bins: list[tuple[int, int]]
) -> dict[str, list[str]]:
    """Bin sequence IDs by length. Returns {bin_label: [seq_ids]}."""
    binned: dict[str, list[str]] = {}
    for lo, hi in bins:
        label = f"{lo // 1000}-{hi // 1000}kb" if lo >= 1000 else f"{lo}-{hi // 1000}kb"
        binned[label] = [sid for sid, seq in seqs.items() if lo <= len(seq) < hi]
    return binned


def _stratified_sample(
    binned: dict[str, list[str]],
    n_total: int,
    rng: np.random.Generator,
) -> list[str]:
    """Stratified sample across length bins. 0 = take all."""
    if n_total == 0:
        return [sid for ids in binned.values() for sid in ids]

    n_bins = len(binned)
    per_bin = n_total // n_bins
    remainder = n_total % n_bins
    selected: list[str] = []
    overflow: list[str] = []

    for i, (label, ids) in enumerate(binned.items()):
        target = per_bin + (1 if i < remainder else 0)
        if len(ids) <= target:
            selected.extend(ids)
            shortfall = target - len(ids)
            if shortfall > 0:
                overflow.append(f"{label}:{shortfall}")
        else:
            chosen = rng.choice(ids, size=target, replace=False).tolist()
            selected.extend(chosen)

    # If any bins had too few, fill from remaining unchosen sequences
    if len(selected) < n_total:
        all_ids = [sid for ids in binned.values() for sid in ids]
        remaining = [sid for sid in all_ids if sid not in set(selected)]
        needed = n_total - len(selected)
        if remaining and needed > 0:
            extra = rng.choice(remaining, size=min(needed, len(remaining)), replace=False)
            selected.extend(extra.tolist())

    return selected


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    pos_label: int = 1,
) -> dict:
    """Compute binary classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        metrics["auc"] = None
    return metrics


def _threshold_sweep(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    thresholds: list[float],
) -> list[dict]:
    """Compute metrics at multiple decision thresholds."""
    results = []
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        m = _compute_metrics(y_true, y_pred, y_scores)
        m["threshold"] = t
        results.append(m)
    return results


# ---------------------------------------------------------------------------
# MANIFEST subcommand
# ---------------------------------------------------------------------------

def cmd_manifest(args: argparse.Namespace) -> None:
    """Generate stratified benchmark manifest shared between tiers."""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=args.seed)

    manifest_rows: list[dict] = []
    all_sequences: dict[str, str] = {}

    # --- GYP datasets ---
    if args.gyp_phage:
        logger.info("Loading GYP phage sequences...")
        phage_seqs = _read_fasta(args.gyp_phage)
        phage_binned = _bin_sequences(phage_seqs, GYP_BINS)
        phage_ids = _stratified_sample(phage_binned, args.n_phage, rng)
        for sid in phage_ids:
            manifest_rows.append({
                "sequence_id": sid, "length": len(phage_seqs[sid]),
                "category": "phage", "dataset": "gyp",
                "length_bin": _get_bin_label(len(phage_seqs[sid]), GYP_BINS),
            })
            all_sequences[sid] = phage_seqs[sid]
        logger.info(f"  Selected {len(phage_ids)}/{len(phage_seqs)} phage")

    if args.gyp_chromosome:
        logger.info("Loading GYP chromosome sequences...")
        chr_seqs = _read_fasta(args.gyp_chromosome)
        chr_binned = _bin_sequences(chr_seqs, GYP_BINS)
        chr_ids = _stratified_sample(chr_binned, args.n_chromosome, rng)
        for sid in chr_ids:
            manifest_rows.append({
                "sequence_id": sid, "length": len(chr_seqs[sid]),
                "category": "chromosome", "dataset": "gyp",
                "length_bin": _get_bin_label(len(chr_seqs[sid]), GYP_BINS),
            })
            all_sequences[sid] = chr_seqs[sid]
        logger.info(f"  Selected {len(chr_ids)}/{len(chr_seqs)} chromosome")

    if args.gyp_plasmid:
        logger.info("Loading GYP plasmid sequences...")
        plas_seqs = _read_fasta(args.gyp_plasmid)
        plas_binned = _bin_sequences(plas_seqs, GYP_BINS)
        plas_ids = _stratified_sample(plas_binned, args.n_plasmid, rng)
        for sid in plas_ids:
            manifest_rows.append({
                "sequence_id": sid, "length": len(plas_seqs[sid]),
                "category": "plasmid", "dataset": "gyp",
                "length_bin": _get_bin_label(len(plas_seqs[sid]), GYP_BINS),
            })
            all_sequences[sid] = plas_seqs[sid]
        logger.info(f"  Selected {len(plas_ids)}/{len(plas_seqs)} plasmid")

    # --- RNA virus dataset ---
    if args.rna_fasta:
        logger.info("Loading RNA virus database...")
        rna_raw = _read_fasta(args.rna_fasta)
        # Filter: NC_ accessions only, 500-16000 bp, convert U→T
        rna_seqs = {}
        for sid, seq in rna_raw.items():
            if not sid.startswith("NC_"):
                continue
            seq = seq.replace("U", "T")
            if 500 <= len(seq) <= 16000:
                rna_seqs[sid] = seq
        logger.info(f"  Filtered to {len(rna_seqs)} NC_ RefSeq sequences (500-16kb)")

        rna_binned = _bin_sequences(rna_seqs, RNA_BINS)
        rna_ids = _stratified_sample(rna_binned, args.n_rna_virus, rng)
        for sid in rna_ids:
            manifest_rows.append({
                "sequence_id": sid, "length": len(rna_seqs[sid]),
                "category": "rna_virus", "dataset": "rna_virus",
                "length_bin": _get_bin_label(len(rna_seqs[sid]), RNA_BINS),
            })
            all_sequences[sid] = rna_seqs[sid]
        logger.info(f"  Selected {len(rna_ids)} RNA viruses")

    # --- Cellular control (for RNA virus comparison) ---
    if args.cellular_fasta and args.cellular_labels:
        logger.info("Loading cellular control sequences...")
        cell_all = _read_fasta(args.cellular_fasta)
        labels_df = pd.read_csv(args.cellular_labels, sep="\t")
        chr_label_ids = set(labels_df[labels_df["label"] == "chromosome"]["sequence_id"])
        cell_seqs = {sid: seq for sid, seq in cell_all.items() if sid in chr_label_ids}
        cell_seqs = {sid: seq for sid, seq in cell_seqs.items() if 500 <= len(seq) <= 16000}
        logger.info(f"  {len(cell_seqs)} chromosome sequences available (500-16kb)")

        cell_binned = _bin_sequences(cell_seqs, RNA_BINS)
        cell_ids = _stratified_sample(cell_binned, args.n_cellular, rng)
        for sid in cell_ids:
            manifest_rows.append({
                "sequence_id": sid, "length": len(cell_seqs[sid]),
                "category": "cellular", "dataset": "rna_cellular",
                "length_bin": _get_bin_label(len(cell_seqs[sid]), RNA_BINS),
            })
            all_sequences[sid] = cell_seqs[sid]
        logger.info(f"  Selected {len(cell_ids)} cellular controls")

    # --- Write outputs ---
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output / "manifest.tsv"
    manifest_df.to_csv(manifest_path, sep="\t", index=False)
    logger.info(f"Wrote manifest: {manifest_path} ({len(manifest_df)} sequences)")

    # Write combined FASTA
    fasta_path = output / "sequences.fasta"
    with open(fasta_path, "w") as f:
        for sid in manifest_df["sequence_id"]:
            f.write(f">{sid}\n{all_sequences[sid]}\n")
    logger.info(f"Wrote sequences: {fasta_path}")

    # Summary
    summary = {
        "seed": args.seed,
        "total_sequences": len(manifest_df),
        "datasets": {},
    }
    for dataset in manifest_df["dataset"].unique():
        ds_df = manifest_df[manifest_df["dataset"] == dataset]
        summary["datasets"][dataset] = {
            "total": len(ds_df),
            "categories": dict(Counter(ds_df["category"])),
            "length_bins": dict(Counter(ds_df["length_bin"])),
        }

    summary_path = output / "manifest_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote summary: {summary_path}")

    # Print overview
    logger.info("=== Manifest Summary ===")
    for dataset, info in summary["datasets"].items():
        logger.info(f"  {dataset}: {info['total']} sequences")
        for cat, n in info["categories"].items():
            logger.info(f"    {cat}: {n}")


def _get_bin_label(length: int, bins: list[tuple[int, int]]) -> str:
    """Get bin label for a sequence length."""
    for lo, hi in bins:
        if lo <= length < hi:
            if lo >= 1000:
                return f"{lo // 1000}-{hi // 1000}kb"
            return f"{lo}-{hi // 1000}kb"
    return "other"


# ---------------------------------------------------------------------------
# RUN subcommand
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    """Run benchmark: embeddings → classification → metrics."""
    manifest_dir = Path(args.manifest)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Load manifest and sequences
    manifest = pd.read_csv(manifest_dir / "manifest.tsv", sep="\t")
    logger.info(f"Loaded manifest: {len(manifest)} sequences")

    fasta_path = manifest_dir / "sequences.fasta"
    sequences = _read_fasta(fasta_path)

    # Ensure manifest sequences match FASTA
    manifest_ids = set(manifest["sequence_id"])
    fasta_ids = set(sequences.keys())
    missing = manifest_ids - fasta_ids
    if missing:
        logger.warning(f"{len(missing)} manifest sequences not in FASTA, skipping")
        manifest = manifest[manifest["sequence_id"].isin(fasta_ids)]

    # --- Extract embeddings ---
    from virosense.backends.base import get_backend
    from virosense.features.evo2_embeddings import extract_embeddings

    backend_kwargs = {"model": args.model}
    if args.nim_url:
        backend_kwargs["nim_url"] = args.nim_url

    backend = get_backend(args.backend, **backend_kwargs)
    model = backend.model  # Use auto-corrected model name

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    layer = args.layer

    logger.info(f"Backend: {args.backend}, model: {model}, layer: {layer}")
    logger.info(f"Extracting embeddings for {len(manifest)} sequences...")

    seqs_for_embed = {sid: sequences[sid] for sid in manifest["sequence_id"]}
    embed_result = extract_embeddings(
        sequences=seqs_for_embed,
        backend=backend,
        layer=layer,
        model=model,
        cache_dir=cache_dir,
        checkpoint_every=25,
    )

    # Build ordered embedding matrix matching manifest order
    # Some sequences may have failed extraction — filter to those we have
    embed_map = dict(zip(embed_result.sequence_ids, embed_result.embeddings))
    all_ids = list(manifest["sequence_id"])
    available_mask = [sid in embed_map for sid in all_ids]
    n_missing = sum(1 for m in available_mask if not m)
    if n_missing > 0:
        logger.warning(
            f"{n_missing}/{len(all_ids)} sequences missing embeddings — "
            f"excluding from benchmark"
        )
        manifest = manifest[available_mask].reset_index(drop=True)
    ordered_ids = list(manifest["sequence_id"])
    embeddings = np.stack([embed_map[sid] for sid in ordered_ids])
    lengths = list(manifest["length"])

    # --- Classification ---
    from virosense.models.detector import ViralClassifier, classify_contigs

    classifier = ViralClassifier.load(args.classifier)
    logger.info(f"Loaded classifier: {classifier.metadata}")

    results = classify_contigs(
        embeddings=embeddings,
        sequence_ids=ordered_ids,
        sequence_lengths=lengths,
        classifier=classifier,
        threshold=0.5,
    )

    # --- Build results DataFrame ---
    rows = []
    for r, (_, mrow) in zip(results, manifest.iterrows()):
        rows.append({
            "sequence_id": r.contig_id,
            "length": r.contig_length,
            "true_category": mrow["category"],
            "dataset": mrow["dataset"],
            "length_bin": mrow["length_bin"],
            "viral_score": r.viral_score,
            "classification": r.classification,
            "chromosome_score": r.chromosome_score,
            "plasmid_score": r.plasmid_score,
            "phage_score": r.phage_score,
            "rna_virus_score": r.rna_virus_score,
        })
    results_df = pd.DataFrame(rows)
    results_df.to_csv(output / "detailed_results.tsv", sep="\t", index=False)

    # --- Compute metrics per dataset ---
    all_metrics = {
        "model": model,
        "backend": args.backend,
        "layer": layer,
        "classifier": str(args.classifier),
        "total_sequences": len(results_df),
        "datasets": {},
    }

    # GYP metrics (phage vs non-viral)
    gyp_df = results_df[results_df["dataset"] == "gyp"]
    if len(gyp_df) > 0:
        all_metrics["datasets"]["gyp"] = _compute_gyp_metrics(gyp_df)

    # RNA virus metrics (rna_virus vs cellular)
    rna_df = results_df[results_df["dataset"].isin(["rna_virus", "rna_cellular"])]
    if len(rna_df) > 0:
        all_metrics["datasets"]["rna_virus"] = _compute_rna_metrics(rna_df)

    # Overall metrics (all viral vs all non-viral)
    all_metrics["overall"] = _compute_overall_metrics(results_df)

    # Write metrics
    metrics_path = output / "benchmark_results.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logger.info(f"Wrote metrics: {metrics_path}")

    # Print summary
    _print_run_summary(all_metrics, results_df)


def _compute_gyp_metrics(df: pd.DataFrame) -> dict:
    """Compute GYP benchmark metrics (phage detection)."""
    # Binary: phage=viral, chromosome/plasmid=non-viral
    y_true = (df["true_category"] == "phage").astype(int).values
    y_scores = df["viral_score"].values
    y_pred = (y_scores >= 0.5).astype(int)

    metrics = _compute_metrics(y_true, y_pred, y_scores)

    # Per-category breakdown
    categories = {}
    for cat in ["phage", "chromosome", "plasmid"]:
        cat_df = df[df["true_category"] == cat]
        if len(cat_df) == 0:
            continue
        is_viral = cat == "phage"
        scores = cat_df["viral_score"].values
        correct = (scores >= 0.5) == is_viral
        categories[cat] = {
            "n": len(cat_df),
            "correct": int(correct.sum()),
            "rate": float(correct.mean()),
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std()),
            "min_score": float(scores.min()),
            "max_score": float(scores.max()),
        }
    metrics["categories"] = categories

    # Per-length-bin phage sensitivity
    phage_df = df[df["true_category"] == "phage"]
    bin_metrics = {}
    for bin_name in GYP_BIN_NAMES:
        bin_df = phage_df[phage_df["length_bin"] == bin_name]
        if len(bin_df) == 0:
            continue
        scores = bin_df["viral_score"].values
        bin_metrics[bin_name] = {
            "n": len(bin_df),
            "sensitivity": float((scores >= 0.5).mean()),
            "mean_score": float(scores.mean()),
        }
    metrics["phage_by_length_bin"] = bin_metrics

    # Threshold sweep
    metrics["threshold_sweep"] = _threshold_sweep(y_true, y_scores, THRESHOLDS)

    # Misclassification details
    fn_df = df[(df["true_category"] == "phage") & (df["viral_score"] < 0.5)]
    metrics["false_negatives"] = fn_df[["sequence_id", "length", "viral_score"]].to_dict("records")

    fp_df = df[(df["true_category"] != "phage") & (df["viral_score"] >= 0.5)]
    metrics["n_false_positives"] = len(fp_df)
    metrics["fp_by_category"] = dict(Counter(fp_df["true_category"]))

    return metrics


def _compute_rna_metrics(df: pd.DataFrame) -> dict:
    """Compute RNA virus benchmark metrics (zero-shot generalization)."""
    y_true = df["true_category"].isin(["rna_virus"]).astype(int).values
    y_scores = df["viral_score"].values
    y_pred = (y_scores >= 0.5).astype(int)

    metrics = _compute_metrics(y_true, y_pred, y_scores)

    # RNA virus sensitivity by length bin
    rna_df = df[df["true_category"] == "rna_virus"]
    bin_metrics = {}
    for bin_name in RNA_BIN_NAMES:
        bin_df = rna_df[rna_df["length_bin"] == bin_name]
        if len(bin_df) == 0:
            continue
        scores = bin_df["viral_score"].values
        bin_metrics[bin_name] = {
            "n": len(bin_df),
            "sensitivity": float((scores >= 0.5).mean()),
            "mean_score": float(scores.mean()),
        }
    metrics["rna_by_length_bin"] = bin_metrics

    # Cellular specificity
    cell_df = df[df["true_category"] == "cellular"]
    if len(cell_df) > 0:
        cell_scores = cell_df["viral_score"].values
        metrics["cellular_specificity"] = float((cell_scores < 0.5).mean())
        metrics["cellular_mean_score"] = float(cell_scores.mean())

    # Threshold sweep
    metrics["threshold_sweep"] = _threshold_sweep(y_true, y_scores, THRESHOLDS)

    # False negatives
    fn_df = df[(df["true_category"] == "rna_virus") & (df["viral_score"] < 0.5)]
    metrics["false_negatives"] = fn_df[["sequence_id", "length", "viral_score"]].to_dict("records")

    return metrics


def _compute_overall_metrics(df: pd.DataFrame) -> dict:
    """Compute overall viral vs non-viral metrics across all datasets."""
    viral_cats = {"phage", "rna_virus"}
    y_true = df["true_category"].isin(viral_cats).astype(int).values
    y_scores = df["viral_score"].values
    y_pred = (y_scores >= 0.5).astype(int)
    return _compute_metrics(y_true, y_pred, y_scores)


def _print_run_summary(metrics: dict, df: pd.DataFrame) -> None:
    """Print benchmark summary to console."""
    logger.info("=" * 60)
    logger.info(f"BENCHMARK RESULTS — {metrics['model']}")
    logger.info("=" * 60)

    if "gyp" in metrics.get("datasets", {}):
        gyp = metrics["datasets"]["gyp"]
        logger.info(f"\nGYP Benchmark:")
        logger.info(f"  Accuracy: {gyp['accuracy']:.3f}")
        logger.info(f"  F1:       {gyp['f1']:.3f}")
        logger.info(f"  AUC:      {gyp.get('auc', 'N/A')}")
        for cat, info in gyp.get("categories", {}).items():
            logger.info(f"  {cat}: {info['rate']:.3f} ({info['correct']}/{info['n']})")
        logger.info(f"  Phage by length:")
        for b, info in gyp.get("phage_by_length_bin", {}).items():
            logger.info(f"    {b}: {info['sensitivity']:.3f} (n={info['n']})")

    if "rna_virus" in metrics.get("datasets", {}):
        rna = metrics["datasets"]["rna_virus"]
        logger.info(f"\nRNA Virus Benchmark (zero-shot):")
        logger.info(f"  Accuracy: {rna['accuracy']:.3f}")
        logger.info(f"  F1:       {rna['f1']:.3f}")
        logger.info(f"  AUC:      {rna.get('auc', 'N/A')}")
        logger.info(f"  Sensitivity by length:")
        for b, info in rna.get("rna_by_length_bin", {}).items():
            logger.info(f"    {b}: {info['sensitivity']:.3f} (n={info['n']})")

    overall = metrics.get("overall", {})
    logger.info(f"\nOverall (all datasets):")
    logger.info(f"  Accuracy: {overall.get('accuracy', 'N/A')}")
    logger.info(f"  F1:       {overall.get('f1', 'N/A')}")
    logger.info(f"  AUC:      {overall.get('auc', 'N/A')}")


# ---------------------------------------------------------------------------
# COMPARE subcommand
# ---------------------------------------------------------------------------

def cmd_compare(args: argparse.Namespace) -> None:
    """Compare results from two tiers."""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Load results
    a_results = pd.read_csv(Path(args.tier_a) / "detailed_results.tsv", sep="\t")
    b_results = pd.read_csv(Path(args.tier_b) / "detailed_results.tsv", sep="\t")
    a_metrics = json.loads((Path(args.tier_a) / "benchmark_results.json").read_text())
    b_metrics = json.loads((Path(args.tier_b) / "benchmark_results.json").read_text())

    label_a = args.label_a or a_metrics.get("model", "Tier A")
    label_b = args.label_b or b_metrics.get("model", "Tier B")

    # Merge on sequence_id
    merged = a_results.merge(
        b_results[["sequence_id", "viral_score", "classification"]],
        on="sequence_id",
        suffixes=("_a", "_b"),
    )

    # Per-sequence agreement
    merged["agree"] = merged["classification_a"] == merged["classification_b"]
    agreement_rate = merged["agree"].mean()

    # Disagreements
    disagree = merged[~merged["agree"]].copy()

    # Build comparison report
    comparison = {
        "label_a": label_a,
        "label_b": label_b,
        "n_sequences": len(merged),
        "agreement_rate": float(agreement_rate),
        "n_disagreements": len(disagree),
        "metrics_comparison": {},
    }

    # Compare metrics per dataset
    for dataset in set(list(a_metrics.get("datasets", {})) + list(b_metrics.get("datasets", {}))):
        a_ds = a_metrics.get("datasets", {}).get(dataset, {})
        b_ds = b_metrics.get("datasets", {}).get(dataset, {})
        comparison["metrics_comparison"][dataset] = {
            label_a: {k: a_ds.get(k) for k in ["accuracy", "f1", "precision", "recall", "auc"]},
            label_b: {k: b_ds.get(k) for k in ["accuracy", "f1", "precision", "recall", "auc"]},
        }

    # Overall comparison
    a_overall = a_metrics.get("overall", {})
    b_overall = b_metrics.get("overall", {})
    comparison["overall"] = {
        label_a: {k: a_overall.get(k) for k in ["accuracy", "f1", "precision", "recall", "auc"]},
        label_b: {k: b_overall.get(k) for k in ["accuracy", "f1", "precision", "recall", "auc"]},
    }

    # Disagreement analysis
    if len(disagree) > 0:
        viral_cats = {"phage", "rna_virus"}
        disagree["true_viral"] = disagree["true_category"].isin(viral_cats)
        disagree["a_correct"] = (
            (disagree["viral_score_a"] >= 0.5) == disagree["true_viral"]
        )
        disagree["b_correct"] = (
            (disagree["viral_score_b"] >= 0.5) == disagree["true_viral"]
        )
        comparison["disagreement_analysis"] = {
            "a_correct_b_wrong": int(disagree["a_correct"].sum() - (disagree["a_correct"] & disagree["b_correct"]).sum()),
            "b_correct_a_wrong": int(disagree["b_correct"].sum() - (disagree["a_correct"] & disagree["b_correct"]).sum()),
            "both_wrong": int((~disagree["a_correct"] & ~disagree["b_correct"]).sum()),
            "by_category": dict(Counter(disagree["true_category"])),
        }

    # Score correlation
    comparison["score_correlation"] = float(
        np.corrcoef(merged["viral_score_a"], merged["viral_score_b"])[0, 1]
    )

    # Write outputs
    with open(output / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    disagree.to_csv(output / "disagreements.tsv", sep="\t", index=False)

    # Write side-by-side summary table
    _write_comparison_table(comparison, merged, output, label_a, label_b)

    # Print summary
    logger.info("=" * 60)
    logger.info(f"TIER COMPARISON: {label_a} vs {label_b}")
    logger.info("=" * 60)
    logger.info(f"Agreement: {agreement_rate:.1%} ({len(merged) - len(disagree)}/{len(merged)})")
    logger.info(f"Score correlation: {comparison['score_correlation']:.4f}")
    logger.info(f"Disagreements: {len(disagree)}")

    for ds, info in comparison["metrics_comparison"].items():
        logger.info(f"\n{ds}:")
        for label, m in info.items():
            f1 = m.get("f1", "N/A")
            auc = m.get("auc", "N/A")
            f1_str = f"{f1:.3f}" if isinstance(f1, float) else str(f1)
            auc_str = f"{auc:.3f}" if isinstance(auc, float) else str(auc)
            logger.info(f"  {label}: F1={f1_str}, AUC={auc_str}")


def _write_comparison_table(
    comparison: dict, merged: pd.DataFrame, output: Path,
    label_a: str, label_b: str,
) -> None:
    """Write a markdown comparison table."""
    lines = [
        f"# ViroSense Tier Comparison: {label_a} vs {label_b}",
        "",
        f"**Sequences**: {comparison['n_sequences']}",
        f"**Agreement**: {comparison['agreement_rate']:.1%}",
        f"**Score correlation**: {comparison['score_correlation']:.4f}",
        "",
        "## Per-Dataset Metrics",
        "",
        f"| Dataset | Metric | {label_a} | {label_b} | Delta |",
        "|---------|--------|-----------|-----------|-------|",
    ]

    for ds, info in comparison["metrics_comparison"].items():
        a_m = info.get(label_a, {})
        b_m = info.get(label_b, {})
        for metric in ["accuracy", "f1", "precision", "recall", "auc"]:
            av = a_m.get(metric)
            bv = b_m.get(metric)
            if av is not None and bv is not None:
                delta = av - bv
                sign = "+" if delta >= 0 else ""
                lines.append(f"| {ds} | {metric} | {av:.3f} | {bv:.3f} | {sign}{delta:.3f} |")

    lines.extend(["", "## Score Distribution by Category", ""])
    lines.append(f"| Category | N | {label_a} mean | {label_b} mean | Diff |")
    lines.append("|----------|---|------------|------------|------|")
    for cat in sorted(merged["true_category"].unique()):
        cat_df = merged[merged["true_category"] == cat]
        a_mean = cat_df["viral_score_a"].mean()
        b_mean = cat_df["viral_score_b"].mean()
        diff = a_mean - b_mean
        sign = "+" if diff >= 0 else ""
        lines.append(f"| {cat} | {len(cat_df)} | {a_mean:.3f} | {b_mean:.3f} | {sign}{diff:.3f} |")

    (output / "comparison_table.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ViroSense unified benchmark for 40B vs 7B head-to-head",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- manifest ---
    p_manifest = subparsers.add_parser("manifest", help="Generate stratified benchmark samples")
    p_manifest.add_argument("--gyp-phage", help="GYP phage FASTA")
    p_manifest.add_argument("--gyp-chromosome", help="GYP chromosome FASTA")
    p_manifest.add_argument("--gyp-plasmid", help="GYP plasmid FASTA")
    p_manifest.add_argument("--rna-fasta", help="RNA virus database FASTA")
    p_manifest.add_argument("--cellular-fasta", help="Cellular reference FASTA")
    p_manifest.add_argument("--cellular-labels", help="Cellular reference labels TSV")
    p_manifest.add_argument("--n-phage", type=int, default=0, help="Phage sample size (0=all)")
    p_manifest.add_argument("--n-chromosome", type=int, default=2000, help="Chromosome sample size")
    p_manifest.add_argument("--n-plasmid", type=int, default=0, help="Plasmid sample size (0=all)")
    p_manifest.add_argument("--n-rna-virus", type=int, default=1000, help="RNA virus sample size")
    p_manifest.add_argument("--n-cellular", type=int, default=1000, help="Cellular control sample size")
    p_manifest.add_argument("--seed", type=int, default=42, help="Random seed")
    p_manifest.add_argument("--output", required=True, help="Output directory")
    p_manifest.set_defaults(func=cmd_manifest)

    # --- run ---
    p_run = subparsers.add_parser("run", help="Run benchmark on one tier")
    p_run.add_argument("--manifest", required=True, help="Manifest directory from 'manifest' command")
    p_run.add_argument("--classifier", required=True, help="Path to classifier .joblib")
    p_run.add_argument("--backend", default="nim", help="Backend name (nim, mlx, local)")
    p_run.add_argument("--model", default="evo2_7b", help="Evo2 model name")
    p_run.add_argument("--layer", default="blocks.28.mlp.l3", help="Embedding layer")
    p_run.add_argument("--nim-url", help="Self-hosted NIM URL")
    p_run.add_argument("--cache-dir", help="Embedding cache directory")
    p_run.add_argument("--output", required=True, help="Output directory")
    p_run.set_defaults(func=cmd_run)

    # --- compare ---
    p_compare = subparsers.add_parser("compare", help="Compare results from two tiers")
    p_compare.add_argument("--tier-a", required=True, help="Tier A results directory")
    p_compare.add_argument("--tier-b", required=True, help="Tier B results directory")
    p_compare.add_argument("--label-a", help="Label for tier A")
    p_compare.add_argument("--label-b", help="Label for tier B")
    p_compare.add_argument("--output", required=True, help="Output directory")
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
