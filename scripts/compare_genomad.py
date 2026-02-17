#!/usr/bin/env python3
"""Head-to-head comparison of ViroSense vs geNomad on benchmark datasets.

Produces summary metrics for both tools on:
1. Simulated benchmark (100 viral + 100 bacterial RefSeq fragments) - perfect ground truth
2. Gut virome (SRR5747446, 99 contigs) - CheckV-confirmed ground truth
"""

import csv
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "data" / "test"


def load_virosense_results(path: Path) -> dict[str, dict]:
    """Load ViroSense detection_results.tsv."""
    results = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            results[row["contig_id"]] = {
                "viral_score": float(row["viral_score"]),
                "classification": row["classification"],
            }
    return results


def load_genomad_virus_summary(path: Path) -> dict[str, dict]:
    """Load geNomad virus_summary.tsv (only contains viral predictions)."""
    results = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            results[row["seq_name"]] = {
                "virus_score": float(row["virus_score"]),
                "n_hallmarks": int(row["n_hallmarks"]),
                "taxonomy": row["taxonomy"],
            }
    return results


def load_simulated_labels(path: Path) -> dict[str, str]:
    """Load ground truth labels for simulated dataset."""
    labels = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            labels[row["sequence_id"]] = row["class"]
    return labels


def load_checkv_quality(path: Path, sample: str = "SRR5747446") -> dict[str, str]:
    """Load CheckV quality for a specific sample. Returns contig -> quality."""
    qualities = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sample"] == sample:
                qualities[row["contig_id"]] = row["checkv_quality"]
    return qualities


def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    """Compute classification metrics from confusion matrix."""
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def compare_simulated():
    """Compare on simulated benchmark with perfect ground truth."""
    print("=" * 70)
    print("SIMULATED BENCHMARK (100 viral + 100 bacterial RefSeq fragments)")
    print("=" * 70)

    labels = load_simulated_labels(BASE / "simulated_mixed_200_labels.tsv")
    vs = load_virosense_results(BASE / "simulated_results" / "detection_results.tsv")
    gn = load_genomad_virus_summary(
        BASE / "genomad_simulated" / "simulated_mixed_200_summary" / "simulated_mixed_200_virus_summary.tsv"
    )

    all_ids = set(labels.keys())

    # ViroSense metrics
    vs_tp = vs_fp = vs_tn = vs_fn = 0
    for sid, truth in labels.items():
        pred = vs.get(sid, {}).get("classification", "cellular")
        if truth == "viral" and pred == "viral":
            vs_tp += 1
        elif truth == "cellular" and pred == "viral":
            vs_fp += 1
        elif truth == "cellular" and pred == "cellular":
            vs_tn += 1
        elif truth == "viral" and pred == "cellular":
            vs_fn += 1

    # geNomad metrics (presence in virus_summary = viral prediction)
    gn_tp = gn_fp = gn_tn = gn_fn = 0
    for sid, truth in labels.items():
        pred_viral = sid in gn
        if truth == "viral" and pred_viral:
            gn_tp += 1
        elif truth == "cellular" and pred_viral:
            gn_fp += 1
        elif truth == "cellular" and not pred_viral:
            gn_tn += 1
        elif truth == "viral" and not pred_viral:
            gn_fn += 1

    vs_metrics = compute_metrics(vs_tp, vs_fp, vs_tn, vs_fn)
    gn_metrics = compute_metrics(gn_tp, gn_fp, gn_tn, gn_fn)

    print(f"\n{'Metric':<15} {'ViroSense':>12} {'geNomad':>12}")
    print("-" * 40)
    for key in ["accuracy", "precision", "recall", "f1"]:
        print(f"{key:<15} {vs_metrics[key]:>11.1%} {gn_metrics[key]:>11.1%}")
    print(f"{'TP':<15} {vs_metrics['tp']:>12} {gn_metrics['tp']:>12}")
    print(f"{'FP':<15} {vs_metrics['fp']:>12} {gn_metrics['fp']:>12}")
    print(f"{'TN':<15} {vs_metrics['tn']:>12} {gn_metrics['tn']:>12}")
    print(f"{'FN':<15} {vs_metrics['fn']:>12} {gn_metrics['fn']:>12}")

    # Disagreements
    print("\nDisagreements (ViroSense vs geNomad):")
    vs_only_viral = []
    gn_only_viral = []
    both_wrong = []
    for sid, truth in labels.items():
        vs_pred = vs.get(sid, {}).get("classification", "cellular") == "viral"
        gn_pred = sid in gn
        if vs_pred and not gn_pred:
            vs_only_viral.append((sid, truth, vs.get(sid, {}).get("viral_score", 0)))
        elif gn_pred and not vs_pred:
            gn_only_viral.append((sid, truth, gn[sid]["virus_score"]))
        elif vs_pred and gn_pred and truth == "cellular":
            both_wrong.append(sid)

    print(f"  ViroSense-only viral calls: {len(vs_only_viral)}")
    for sid, truth, score in sorted(vs_only_viral, key=lambda x: -x[2])[:5]:
        print(f"    {sid} (truth={truth}, vs_score={score:.3f})")
    print(f"  geNomad-only viral calls: {len(gn_only_viral)}")
    for sid, truth, score in sorted(gn_only_viral, key=lambda x: -x[2])[:5]:
        print(f"    {sid} (truth={truth}, gn_score={score:.3f})")
    if both_wrong:
        print(f"  Both flagged as viral (truth=cellular): {both_wrong}")

    return {"virosense": vs_metrics, "genomad": gn_metrics}


def compare_gut_virome():
    """Compare on gut virome with CheckV ground truth."""
    print("\n" + "=" * 70)
    print("GUT VIROME (SRR5747446, 99 contigs, CheckV ground truth)")
    print("=" * 70)

    checkv = load_checkv_quality(BASE / "set5_gut_virome_dataset" / "checkv_quality_summary.csv")
    vs = load_virosense_results(BASE / "gut_virome_results" / "detection_results.tsv")
    gn = load_genomad_virus_summary(
        BASE / "genomad_gut_virome" / "gut_virome_test_summary" / "gut_virome_test_virus_summary.tsv"
    )

    # Only evaluate contigs that were in our test input (present in ViroSense results)
    all_contigs = set(vs.keys())

    # CheckV-confirmed viral = any quality other than "Not-determined", restricted to test contigs
    confirmed_viral = {cid for cid, q in checkv.items() if q != "Not-determined" and cid in all_contigs}

    print(f"\nDataset: {len(all_contigs)} contigs, {len(confirmed_viral)} CheckV-confirmed viral")

    # Sensitivity on CheckV-confirmed viral contigs
    vs_detected = {cid for cid, v in vs.items() if v["classification"] == "viral"}
    gn_detected = set(gn.keys())

    vs_sens = len(vs_detected & confirmed_viral)
    gn_sens = len(gn_detected & confirmed_viral)

    print(f"\nSensitivity on CheckV-confirmed viral ({len(confirmed_viral)} contigs):")
    print(f"  ViroSense: {vs_sens}/{len(confirmed_viral)} ({vs_sens/len(confirmed_viral):.1%})")
    print(f"  geNomad:   {gn_sens}/{len(confirmed_viral)} ({gn_sens/len(confirmed_viral):.1%})")

    # Overall predictions
    print(f"\nTotal viral predictions:")
    print(f"  ViroSense: {len(vs_detected)}/{len(all_contigs)}")
    print(f"  geNomad:   {len(gn_detected)}/{len(all_contigs)}")

    # Overlap
    both_viral = vs_detected & gn_detected
    vs_only = vs_detected - gn_detected
    gn_only = gn_detected - vs_detected
    neither = all_contigs - vs_detected - gn_detected

    print(f"\nAgreement:")
    print(f"  Both viral:    {len(both_viral)}")
    print(f"  ViroSense only: {len(vs_only)}")
    print(f"  geNomad only:  {len(gn_only)}")
    print(f"  Neither:       {len(neither)}")

    # CheckV breakdown for ViroSense-only calls
    vs_only_confirmed = vs_only & confirmed_viral
    vs_only_unknown = vs_only - confirmed_viral
    print(f"\n  ViroSense-only calls: {len(vs_only_confirmed)} CheckV-confirmed + {len(vs_only_unknown)} not-determined")

    # Missed by ViroSense
    vs_missed = confirmed_viral - vs_detected
    if vs_missed:
        print(f"\n  ViroSense missed ({len(vs_missed)} CheckV-confirmed):")
        for cid in vs_missed:
            score = vs.get(cid, {}).get("viral_score", "N/A")
            quality = checkv.get(cid, "N/A")
            print(f"    {cid} (score={score}, checkv={quality})")

    # Missed by geNomad
    gn_missed = confirmed_viral - gn_detected
    print(f"\n  geNomad missed ({len(gn_missed)} CheckV-confirmed):")
    for cid in sorted(gn_missed):
        quality = checkv.get(cid, "N/A")
        vs_score = vs.get(cid, {}).get("viral_score", "N/A")
        print(f"    {cid} (checkv={quality}, vs_score={vs_score})")

    return {
        "checkv_confirmed": len(confirmed_viral),
        "virosense_sensitivity": vs_sens / len(confirmed_viral),
        "genomad_sensitivity": gn_sens / len(confirmed_viral),
        "virosense_viral": len(vs_detected),
        "genomad_viral": len(gn_detected),
        "both_viral": len(both_viral),
        "virosense_only": len(vs_only),
        "genomad_only": len(gn_only),
    }


def main():
    sim_results = compare_simulated()
    virome_results = compare_gut_virome()

    # Write summary JSON
    output_dir = BASE / "comparison_results"
    output_dir.mkdir(exist_ok=True)

    summary = {
        "tools": {
            "virosense": "v0.1.0 (Evo2 NIM backend, default threshold 0.5)",
            "genomad": "v1.11.2 (database v1.9, default threshold 0.7)",
        },
        "simulated_benchmark": sim_results,
        "gut_virome": virome_results,
    }

    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n\nSummary written to {output_dir / 'comparison_summary.json'}")


if __name__ == "__main__":
    main()
