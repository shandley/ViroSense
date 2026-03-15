#!/usr/bin/env python3
"""Train a binary (viral/cellular) classifier from cached 7B embeddings.

Uses the 3-class embedding cache (chromosome/plasmid/viral) and collapses
chromosome + plasmid → cellular (0), viral → viral (1).

Usage:
    uv run python scripts/train_binary_from_cache.py \
        --cache /scratch/sahlab/shandley/virosense/embedding_cache_3class/evo2_7b_blocks_28_mlp_l3_embeddings.npz \
        --labels /scratch/sahlab/shandley/virosense/virosense_repo/data/reference/3class/labels.tsv \
        --output /scratch/sahlab/shandley/virosense/model_binary_7b/ \
        --install
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from virosense.models.training import train_classifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train binary classifier from cached 3-class embeddings"
    )
    parser.add_argument(
        "--cache", required=True, help="Path to cached NPZ embeddings"
    )
    parser.add_argument(
        "--labels", required=True, help="Path to 3-class labels TSV"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for model"
    )
    parser.add_argument(
        "--install", action="store_true",
        help="Install as default reference classifier (~/.virosense/models/)"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--normalize-l2", action="store_true",
        help="L2-normalize embeddings before classification (fixes length-dependent recall)"
    )
    args = parser.parse_args()

    # Load cached embeddings
    logger.info(f"Loading embeddings from {args.cache}")
    data = np.load(args.cache)
    seq_ids = data["sequence_ids"]
    embeddings = data["embeddings"]
    logger.info(f"Loaded {len(seq_ids)} sequences, {embeddings.shape[1]}-D embeddings")

    # Load 3-class labels
    logger.info(f"Loading labels from {args.labels}")
    labels_df = pd.read_csv(args.labels, sep="\t")
    label_map = dict(zip(labels_df["sequence_id"], labels_df["label"]))

    # Match embeddings to labels and collapse to binary
    matched_ids = []
    matched_embeddings = []
    matched_labels = []

    for i, seq_id in enumerate(seq_ids):
        if seq_id in label_map:
            label_3class = label_map[seq_id]
            # Collapse: chromosome/plasmid → 0 (cellular), viral → 1
            if label_3class in ("chromosome", "plasmid"):
                binary_label = 0
            elif label_3class == "viral":
                binary_label = 1
            else:
                logger.warning(f"Unknown label '{label_3class}' for {seq_id}, skipping")
                continue
            matched_ids.append(seq_id)
            matched_embeddings.append(embeddings[i])
            matched_labels.append(binary_label)

    X = np.array(matched_embeddings)
    y = np.array(matched_labels)

    n_cellular = (y == 0).sum()
    n_viral = (y == 1).sum()
    logger.info(
        f"Matched {len(y)} sequences: {n_cellular} cellular "
        f"(chr+plasmid), {n_viral} viral"
    )

    # Train binary classifier with calibration
    metrics = train_classifier(
        embeddings=X,
        labels=y,
        output_dir=Path(args.output),
        epochs=args.epochs,
        lr=args.lr,
        val_split=0.2,
        task="viral_vs_cellular_7b",
        class_names=["cellular", "viral"],
        layer="blocks.28.mlp.l3",
        model="evo2_7b",
        normalize_l2=args.normalize_l2,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("BINARY CLASSIFIER (7B) RESULTS")
    print("=" * 60)
    print(f"Training samples:    {metrics['n_train']}")
    print(f"Calibration samples: {metrics['n_cal']}")
    print(f"Test samples:        {metrics['n_test']}")
    print(f"")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"AUC:       {metrics.get('auc', 'N/A')}")
    if "brier_score" in metrics:
        print(f"")
        print(f"Brier (uncal): {metrics.get('brier_score_uncalibrated', 'N/A'):.4f}")
        print(f"Brier (cal):   {metrics['brier_score']:.4f}")
        print(f"ECE (uncal):   {metrics.get('ece_uncalibrated', 'N/A'):.4f}")
        print(f"ECE (cal):     {metrics['ece']:.4f}")
    print("=" * 60)

    # Install as default reference classifier
    if args.install:
        install_dir = Path.home() / ".virosense" / "models"
        install_dir.mkdir(parents=True, exist_ok=True)
        src = Path(args.output) / "classifier.joblib"
        dst = install_dir / "reference_classifier.joblib"
        import shutil
        shutil.copy2(src, dst)
        logger.info(f"Installed as reference classifier: {dst}")


if __name__ == "__main__":
    main()
