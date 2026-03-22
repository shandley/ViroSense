#!/usr/bin/env python3
"""
Benchmark exon-intron detection: Evo2 inversion vs Augustus.

Runs Augustus on the same 36 genes and compares per-position exon
detection accuracy against our offset-3 cosine inversion method.

Usage:
    uv run python scripts/benchmark_exon_intron.py
"""

import json
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np

DATA_DIR = Path("results/experiments/exon_intron")
OUT_DIR = Path("results/experiments/exon_intron/benchmark")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Species to Augustus model mapping
SPECIES_MODEL = {
    "human": "human",
    "mouse": "human",  # human model works for mouse
    "zebrafish": "zebrafish",
    "chicken": "chicken",
    "xenopus": "xenopus",
    "drosophila": "fly",
    "celegans": "caenorhabditis",
    "arabidopsis": "arabidopsis",
    "rice": "rice",
    "maize": "maize",
    "yeast": "saccharomyces",  # S. cerevisiae specific
    "neurospora": "neurospora_crassa",
    "toxoplasma": "toxoplasma",
}


def run_augustus(fasta_path: Path, species: str) -> list[dict]:
    """Run Augustus on a FASTA file and return predicted CDS regions."""
    try:
        result = subprocess.run(
            ["conda", "run", "-n", "genefinders", "augustus",
             f"--species={species}", "--gff3=on", str(fasta_path)],
            capture_output=True, text=True, timeout=60
        )

        if result.returncode != 0:
            return []

        # Parse GFF3 output for CDS features
        cds_regions = []
        for line in result.stdout.split("\n"):
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 9 and parts[2] == "CDS":
                start = int(parts[3]) - 1  # GFF is 1-based
                end = int(parts[4])
                cds_regions.append({"start": start, "end": end})

        return cds_regions

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"    Augustus failed: {e}")
        return []


def compute_accuracy(predicted_positions: np.ndarray, truth: np.ndarray):
    """Compute precision, recall, F1, accuracy."""
    tp = float(((predicted_positions == 1) & (truth == 1)).sum())
    fp = float(((predicted_positions == 1) & (truth == 0)).sum())
    fn = float(((predicted_positions == 0) & (truth == 1)).sum())

    accuracy = float((predicted_positions == truth).sum()) / len(truth)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def main():
    # Load annotations (ground truth)
    ann = {}
    for ann_file in [DATA_DIR / "annotations_all.json", DATA_DIR / "annotations_fixed.json"]:
        if ann_file.exists():
            with open(ann_file) as f:
                ann.update(json.load(f))

    # Load Evo2 metrics
    metrics_dir = DATA_DIR / "metrics"
    seq_dir = DATA_DIR / "sequences"

    results = []
    augustus_failures = []

    # Get all genes with both metrics and sequences
    gene_files = sorted(metrics_dir.glob("*_perpos.json"))

    print(f"Benchmarking {len(gene_files)} genes: Evo2 inversion vs Augustus")
    print("=" * 90)
    print(f"{'Gene':<28s} {'Species':<14s} {'Evo2 F1':<9s} {'Aug F1':<9s} {'Evo2 Rec':<9s} {'Aug Rec':<9s} {'Evo2 Pre':<9s} {'Aug Pre':<9s}")
    print("-" * 90)

    for mp in gene_files:
        gene_name = mp.stem.replace("_perpos", "")
        fasta_path = seq_dir / f"{gene_name}.fasta"

        if not fasta_path.exists():
            continue

        gene_ann = ann.get(gene_name, {})
        regions = gene_ann.get("cds", []) if gene_ann.get("cds") else gene_ann.get("exons", [])

        # Load sequence
        with open(fasta_path) as f:
            seq = "".join(l.strip() for l in f if not l.startswith(">"))
        seq_len = len(seq)

        if not regions:
            continue

        # Ground truth
        truth = np.zeros(seq_len)
        for r in regions:
            s = max(0, r["start"])
            e = min(r["end"], seq_len)
            truth[s:e] = 1

        coding_fraction = truth.mean()
        if coding_fraction < 0.01 or coding_fraction > 0.99:
            continue

        # ── Evo2 inversion prediction ──
        with open(mp) as f:
            data = json.load(f)
        cos1 = np.array(data["cos1"])
        cos3 = np.array(data["cos3"])

        kernel = np.ones(100) / 100
        cos1_s = np.convolve(cos1, kernel, mode="same")
        cos3_s = np.convolve(cos3, kernel, mode="same")
        inversion = cos3_s - cos1_s

        evo2_pred = (inversion > 0).astype(int)
        evo2_metrics = compute_accuracy(evo2_pred, truth)

        # ── Augustus prediction ──
        # Determine species model
        species_key = None
        for key in SPECIES_MODEL:
            if gene_name.startswith(key):
                species_key = key
                break

        aug_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

        if species_key and species_key in SPECIES_MODEL:
            model = SPECIES_MODEL[species_key]
            aug_cds = run_augustus(fasta_path, model)

            if aug_cds:
                aug_pred = np.zeros(seq_len)
                for r in aug_cds:
                    s = max(0, r["start"])
                    e = min(r["end"], seq_len)
                    aug_pred[s:e] = 1
                aug_metrics = compute_accuracy(aug_pred.astype(int), truth)
            else:
                augustus_failures.append(gene_name)
        else:
            augustus_failures.append(gene_name)

        species_short = species_key[:12] if species_key else "unknown"
        print(f"{gene_name:<28s} {species_short:<14s} "
              f"{evo2_metrics['f1']:<9.3f} {aug_metrics['f1']:<9.3f} "
              f"{evo2_metrics['recall']:<9.3f} {aug_metrics['recall']:<9.3f} "
              f"{evo2_metrics['precision']:<9.3f} {aug_metrics['precision']:<9.3f}")

        results.append({
            "gene": gene_name,
            "species": species_key or "unknown",
            "seq_len": seq_len,
            "coding_fraction": coding_fraction,
            "evo2": evo2_metrics,
            "augustus": aug_metrics,
        })

    # Summary
    evo2_f1s = [r["evo2"]["f1"] for r in results]
    aug_f1s = [r["augustus"]["f1"] for r in results if r["augustus"]["f1"] > 0]
    evo2_recalls = [r["evo2"]["recall"] for r in results]
    aug_recalls = [r["augustus"]["recall"] for r in results if r["augustus"]["recall"] > 0]

    print(f"\n{'=' * 90}")
    print(f"SUMMARY")
    print(f"{'=' * 90}")
    print(f"Genes benchmarked: {len(results)}")
    print(f"Augustus failures: {len(augustus_failures)} ({augustus_failures})")
    print(f"Augustus successful: {len(aug_f1s)}")
    print(f"")
    print(f"{'Metric':<25s} {'Evo2 (unsupervised)':<25s} {'Augustus (trained)':<25s}")
    print(f"{'Mean F1':<25s} {np.mean(evo2_f1s):<25.3f} {np.mean(aug_f1s) if aug_f1s else 0:<25.3f}")
    print(f"{'Mean Recall':<25s} {np.mean(evo2_recalls):<25.3f} {np.mean(aug_recalls) if aug_recalls else 0:<25.3f}")
    print(f"{'Mean Precision':<25s} {np.mean([r['evo2']['precision'] for r in results]):<25.3f} {np.mean([r['augustus']['precision'] for r in results if r['augustus']['precision'] > 0]) if aug_f1s else 0:<25.3f}")

    # Evo2 wins/ties/losses
    if aug_f1s:
        both = [(r["evo2"]["f1"], r["augustus"]["f1"]) for r in results if r["augustus"]["f1"] > 0]
        evo2_wins = sum(1 for e, a in both if e > a)
        aug_wins = sum(1 for e, a in both if a > e)
        ties = sum(1 for e, a in both if abs(e - a) < 0.01)
        print(f"\nHead-to-head (F1): Evo2 wins {evo2_wins}, Augustus wins {aug_wins}, ties {ties}")

    # Save
    with open(OUT_DIR / "benchmark_results.json", "w") as f:
        json.dump({"results": results, "augustus_failures": augustus_failures}, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/benchmark_results.json")


if __name__ == "__main__":
    main()
