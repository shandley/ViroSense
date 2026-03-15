#!/bin/bash
# Retrain 7B reference classifier on HTCF using existing embedding cache.
#
# No GPU needed — loads cached embeddings and trains sklearn MLP on CPU.
#
# Usage:
#   bash htcf/retrain_7b_classifier.sh
#
# Prerequisites:
#   - Embedding cache at virosense_repo/data/reference/cache/evo2_7b_blocks_28_mlp_l3_embeddings.npz
#   - 3-class reference data at virosense_repo/data/reference/3class/

set -euo pipefail

VIROSENSE_HOME="/scratch/sahlab/shandley/virosense/ViroSense"
VIROSENSE_DATA="/scratch/sahlab/shandley/virosense/virosense_repo/data"
export PATH="$HOME/.local/bin:$PATH"

OUTPUT_DIR="/scratch/sahlab/shandley/virosense/model_7b"
mkdir -p "$OUTPUT_DIR"

echo "=== Retraining 7B Reference Classifier ==="
echo "Data: $VIROSENSE_DATA/reference/3class/"
echo "Cache: $VIROSENSE_DATA/reference/cache/"
echo "Output: $OUTPUT_DIR"
echo ""

cd "$VIROSENSE_HOME"

# Use build-reference with the 2-class reference data (prophage-filtered).
# The cache has all 6,210 original sequences embedded at 7B (4096-D).
# The cleaned dataset has 6,158 (6,210 minus prophage suspects).
# --nim-url points to localhost (no server) but all embeddings are cached
# so no API calls will be made.
#
# Note: 3-class (chromosome/plasmid/viral) requires embedding the 3,053
# plasmid sequences, which needs a running NIM server. Use 2-class for now.

# Create string labels from numeric (0=cellular, 1=viral)
awk -F'\t' 'NR==1{print $1"\t"$2} NR>1{label=($2=="1"?"viral":"cellular"); print $1"\t"label}' \
    "$VIROSENSE_DATA/reference/cleaned/labels.tsv" > "$OUTPUT_DIR/labels_string.tsv"

NVIDIA_API_KEY=dummy uv run virosense build-reference \
    -i "$VIROSENSE_DATA/reference/cleaned/sequences.fasta" \
    --labels "$OUTPUT_DIR/labels_string.tsv" \
    -o "$OUTPUT_DIR" \
    --backend nim \
    --model evo2_7b \
    --nim-url http://localhost:8000 \
    --layer blocks.28.mlp.l3 \
    --epochs 200 \
    --lr 1e-3 \
    --val-split 0.2 \
    --cache-dir "$VIROSENSE_DATA/reference/cache"

echo ""
echo "=== Retraining complete ==="
echo "Classifier: $OUTPUT_DIR/classifier.joblib"
echo "Metrics: $OUTPUT_DIR/metrics.json"
cat "$OUTPUT_DIR/metrics.json"
