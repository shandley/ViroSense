#!/usr/bin/env python3
"""Post-hoc prophage noise detection and filtering.

Uses a trained ViroSense classifier to identify cellular training fragments
that are likely prophage-contaminated (i.e., viral DNA mislabeled as cellular).
Outputs a cleaned training set with suspect fragments removed.

This is a bootstrapping approach: the classifier learned enough signal to
spot its own label noise. Fragments from the cellular class that score
highly viral are almost certainly integrated prophage regions.

Usage:
    python scripts/filter_prophage_noise.py \\
        --model data/reference/model/classifier.joblib \\
        --cache-dir data/reference/cache \\
        --labels data/reference/labels.tsv \\
        --fasta data/reference/sequences.fasta \\
        --output data/reference/cleaned/ \\
        --threshold 0.8

After filtering, retrain with cached embeddings (fast):
    virosense build-reference \\
        -i data/reference/cleaned/sequences.fasta \\
        --labels data/reference/cleaned/labels.tsv \\
        -o data/reference/cleaned/model/ \\
        --cache-dir data/reference/cache \\
        --install
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Filter likely prophage contamination from cellular training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to trained classifier (classifier.joblib)",
    )
    parser.add_argument(
        "--cache-dir", required=True,
        help="Embedding cache directory (contains NPZ files)",
    )
    parser.add_argument(
        "--labels", required=True,
        help="Original labels TSV (sequence_id, label)",
    )
    parser.add_argument(
        "--fasta", required=True,
        help="Original sequences FASTA",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for cleaned data",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.8,
        help="Viral score threshold for flagging cellular fragments (default: 0.8)",
    )
    parser.add_argument(
        "--layer", default="blocks.28.mlp.l3",
        help="Evo2 layer used for embeddings (default: blocks.28.mlp.l3)",
    )
    parser.add_argument(
        "--model-name", default="evo2_7b",
        help="Evo2 model name (default: evo2_7b)",
    )
    args = parser.parse_args()

    from virosense.io.fasta import read_fasta
    from virosense.models.detector import ViralClassifier

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load classifier
    print("Loading classifier...")
    classifier = ViralClassifier.load(args.model)
    class_names = classifier.metadata.get("class_names", ["cellular", "viral"])
    print(f"  Classes: {class_names}")

    # 2. Load labels
    print("Loading labels...")
    labels_df = pd.read_csv(args.labels, sep="\t")
    cellular_ids = set(labels_df[labels_df["label"] == 0]["sequence_id"])
    viral_ids = set(labels_df[labels_df["label"] == 1]["sequence_id"])
    print(f"  {len(viral_ids)} viral, {len(cellular_ids)} cellular")

    # 3. Load cached embeddings
    print("Loading cached embeddings...")
    cache_dir = Path(args.cache_dir)
    layer_safe = args.layer.replace(".", "_")
    cache_path = cache_dir / f"{args.model_name}_{layer_safe}_embeddings.npz"

    if not cache_path.exists():
        print(f"ERROR: Cache file not found: {cache_path}")
        sys.exit(1)

    data = np.load(cache_path, allow_pickle=True)
    cached_ids = list(data["sequence_ids"])
    cached_embeddings = data["embeddings"]
    print(f"  Loaded {len(cached_ids)} cached embeddings ({cached_embeddings.shape[1]}d)")

    # Build ID -> index lookup
    id_to_idx = {sid: i for i, sid in enumerate(cached_ids)}

    # 4. Score all cellular fragments
    print(f"\nScoring {len(cellular_ids)} cellular fragments...")
    cellular_list = [sid for sid in cellular_ids if sid in id_to_idx]
    if len(cellular_list) < len(cellular_ids):
        print(f"  Warning: {len(cellular_ids) - len(cellular_list)} cellular IDs not in cache")

    cellular_indices = [id_to_idx[sid] for sid in cellular_list]
    cellular_embeddings = cached_embeddings[cellular_indices]

    # Get viral probabilities
    probas = classifier.predict_proba(cellular_embeddings)

    # Find the viral class index
    viral_class_idx = 1  # default: class 1 = viral
    if "viral" in class_names:
        viral_class_idx = class_names.index("viral")

    viral_scores = probas[:, viral_class_idx]

    # 5. Identify suspect fragments
    suspect_mask = viral_scores >= args.threshold
    suspect_ids = set(np.array(cellular_list)[suspect_mask])
    n_suspect = len(suspect_ids)

    print(f"\n=== Prophage Noise Detection Results ===")
    print(f"  Threshold: {args.threshold}")
    print(f"  Cellular fragments scored: {len(cellular_list)}")
    print(f"  Suspect prophage fragments: {n_suspect} ({100 * n_suspect / len(cellular_list):.1f}%)")

    # Score distribution for cellular fragments
    print(f"\n  Viral score distribution (cellular class):")
    for cutoff in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        count = (viral_scores >= cutoff).sum()
        print(f"    >= {cutoff:.2f}: {count} fragments ({100 * count / len(cellular_list):.1f}%)")

    # 6. Write suspect fragment report
    report = []
    for i, sid in enumerate(cellular_list):
        if suspect_mask[i]:
            report.append({
                "sequence_id": sid,
                "viral_score": round(float(viral_scores[i]), 4),
            })
    report.sort(key=lambda x: x["viral_score"], reverse=True)

    report_path = output_dir / "suspect_prophage_fragments.tsv"
    report_df = pd.DataFrame(report)
    report_df.to_csv(report_path, sep="\t", index=False)
    print(f"\n  Suspect fragments written to {report_path}")

    if n_suspect > 0:
        print(f"\n  Top 10 most suspicious:")
        for r in report[:10]:
            print(f"    {r['sequence_id']}: viral_score={r['viral_score']}")

    # 7. Write cleaned FASTA and labels
    print(f"\nWriting cleaned dataset (removing {n_suspect} suspect fragments)...")
    sequences = read_fasta(args.fasta)

    clean_fasta_path = output_dir / "sequences.fasta"
    clean_labels_path = output_dir / "labels.tsv"

    n_written = 0
    with open(clean_fasta_path, "w") as fasta_f, open(clean_labels_path, "w") as labels_f:
        labels_f.write("sequence_id\tlabel\n")

        for _, row in labels_df.iterrows():
            sid = row["sequence_id"]
            label = row["label"]

            # Skip suspect prophage fragments
            if sid in suspect_ids:
                continue

            seq = sequences.get(sid)
            if seq is None:
                continue

            fasta_f.write(f">{sid}\n{seq}\n")
            labels_f.write(f"{sid}\t{label}\n")
            n_written += 1

    n_viral_out = len(viral_ids)
    n_cellular_out = len(cellular_ids) - n_suspect
    print(f"  Written: {n_written} fragments ({n_viral_out} viral + {n_cellular_out} cellular)")
    print(f"  Removed: {n_suspect} suspect prophage fragments")

    # 8. Write summary
    summary = {
        "threshold": args.threshold,
        "cellular_scored": len(cellular_list),
        "suspect_prophage_count": n_suspect,
        "suspect_prophage_fraction": round(n_suspect / len(cellular_list), 4) if cellular_list else 0,
        "fragments_before": len(labels_df),
        "fragments_after": n_written,
        "viral_after": n_viral_out,
        "cellular_after": n_cellular_out,
        "score_distribution": {
            f">=0.{d}": int((viral_scores >= d / 10).sum())
            for d in range(5, 10)
        },
    }
    summary_path = output_dir / "filtering_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary written to {summary_path}")

    print(f"\n=== Next Steps ===")
    print(f"  Retrain with cleaned data (uses cached embeddings, ~1 min):")
    print(f"  virosense build-reference \\")
    print(f"    -i {clean_fasta_path} \\")
    print(f"    --labels {clean_labels_path} \\")
    print(f"    -o {output_dir / 'model/'} \\")
    print(f"    --cache-dir {args.cache_dir} \\")
    print(f"    --install")


if __name__ == "__main__":
    main()
