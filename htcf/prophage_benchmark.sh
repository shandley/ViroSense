#!/usr/bin/env bash
# Run ViroSense prophage detection on Philympics benchmark genomes.
# Requires: NIM server running on HTCF (start with htcf/start_nim_server.sbatch)
#
# Usage:
#   # 1. Start NIM server first
#   export NGC_API_KEY='nvapi-...'
#   sbatch htcf/start_nim_server.sbatch
#   # Wait for NIM to be ready (~3-5 min)
#
#   # 2. Run this script
#   bash htcf/prophage_benchmark.sh http://localhost:8000

set -uo pipefail

NIM_URL="${1:?Usage: bash htcf/prophage_benchmark.sh <NIM_URL>}"
VIROSENSE_DIR="/scratch/sahlab/shandley/virosense/ViroSense"
BENCHMARK_DIR="$VIROSENSE_DIR/data/benchmarks/philympics"
OUTPUT_DIR="$VIROSENSE_DIR/results/benchmark/prophage"
CACHE_DIR="$OUTPUT_DIR/cache"

# Use the 7B classifier (matching 7B NIM embeddings)
CLASSIFIER="$VIROSENSE_DIR/results/classifiers/7b_16kb/classifier.joblib"

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"
cd "$VIROSENSE_DIR"

echo "============================================================"
echo "ViroSense Prophage Benchmark (Philympics)"
echo "Date: $(date)"
echo "NIM URL: $NIM_URL"
echo "============================================================"

# Check NIM is ready
echo "Checking NIM server..."
for i in $(seq 1 30); do
    if curl -s "$NIM_URL/v1/health/ready" | grep -q "true"; then
        echo "NIM server is ready"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: NIM server not ready after 5 min"
        exit 1
    fi
    sleep 10
done

# Process each genome
GENOMES=$(ls "$BENCHMARK_DIR"/*.fasta 2>/dev/null)
if [ -z "$GENOMES" ]; then
    echo "ERROR: No FASTA files found in $BENCHMARK_DIR"
    exit 1
fi

for FASTA in $GENOMES; do
    GENOME=$(basename "$FASTA" .fasta)
    GENOME_OUT="$OUTPUT_DIR/$GENOME"

    # Skip if already done
    if [ -f "$GENOME_OUT/prophage_summary.json" ]; then
        echo "Skipping $GENOME (already done)"
        continue
    fi

    echo ""
    echo "=== Processing $GENOME ==="

    GT_FILE="$BENCHMARK_DIR/${GENOME}_ground_truth.json"
    if [ -f "$GT_FILE" ]; then
        echo "Ground truth: $(python3 -c "import json; d=json.load(open('$GT_FILE')); print(f'{d[\"phage_genes\"]} phage genes, {len(d[\"prophage_regions\"])} regions')")"
    fi

    START_TIME=$SECONDS

    uv run virosense prophage \
        -i "$FASTA" \
        -o "$GENOME_OUT" \
        --backend nim \
        --nim-url "$NIM_URL" \
        --model evo2_7b \
        --layer blocks.28.mlp.l3 \
        --scan-mode adaptive \
        --threshold 0.5 \
        --cache-dir "$CACHE_DIR" \
        --classifier-model "$CLASSIFIER" \
        2>&1 | tee "$GENOME_OUT/run.log"

    ELAPSED=$((SECONDS - START_TIME))
    echo "Completed $GENOME in ${ELAPSED}s"

    # Quick evaluation against ground truth
    if [ -f "$GT_FILE" ] && [ -f "$GENOME_OUT/prophage_summary.json" ]; then
        python3 -c "
import json

gt = json.load(open('$GT_FILE'))
pred = json.load(open('$GENOME_OUT/prophage_summary.json'))

gt_regions = gt['prophage_regions']
pred_regions = pred.get('regions', [])

print(f'  Ground truth: {len(gt_regions)} regions')
print(f'  Predicted:    {len(pred_regions)} regions')

# Match predictions to ground truth (overlap-based)
hits = 0
for gt_r in gt_regions:
    for p_r in pred_regions:
        # Check overlap
        overlap_start = max(gt_r['start'], p_r['start'])
        overlap_end = min(gt_r['end'], p_r['end'])
        if overlap_end > overlap_start:
            overlap = overlap_end - overlap_start
            gt_len = gt_r['end'] - gt_r['start']
            if overlap / gt_len > 0.1:  # >10% overlap = hit
                hits += 1
                print(f'  HIT: GT {gt_r[\"start\"]}-{gt_r[\"end\"]} ({gt_len}bp) ↔ Pred {p_r[\"start\"]}-{p_r[\"end\"]} ({p_r[\"length\"]}bp)')
                break

print(f'  Recall: {hits}/{len(gt_regions)} ({hits/len(gt_regions):.0%})' if gt_regions else '  No ground truth regions')
print(f'  False positives: {len(pred_regions) - hits}')
"
    fi
done

echo ""
echo "============================================================"
echo "Benchmark complete!"
echo "Results in: $OUTPUT_DIR/"
echo "============================================================"
