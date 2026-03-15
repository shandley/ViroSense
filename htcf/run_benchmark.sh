#!/bin/bash
# Run ViroSense 7B benchmark on HTCF
#
# Orchestrates: start NIM server -> wait for ready -> run benchmark -> cleanup
#
# Usage:
#   # Generate manifest first (locally or on HTCF):
#   uv run python scripts/benchmark_unified.py manifest \
#       --gyp-phage data/benchmarks/gauge_your_phage/phage_fragment_set.fasta \
#       --gyp-chromosome data/benchmarks/gauge_your_phage/host_chr_pvog_fragments.fasta \
#       --gyp-plasmid data/benchmarks/gauge_your_phage/host_plasmid_pvog_fragments.fasta \
#       --rna-fasta data/reference/rna_viruses/RNA_virus_database.fasta \
#       --cellular-fasta data/reference/3class/sequences.fasta \
#       --cellular-labels data/reference/3class/labels.tsv \
#       --output results/benchmark/manifest/
#
#   # Then run benchmark:
#   bash htcf/run_benchmark.sh
#
# Environment:
#   NGC_API_KEY     - required for first NIM run (model weight download)
#   NIM_JOB_ID      - optional: reuse an existing NIM server job

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIROSENSE_HOME="/scratch/sahlab/shandley/virosense/ViroSense"
VIROSENSE_DATA="/scratch/sahlab/shandley/virosense/virosense_repo/data"
export PATH="$HOME/.local/bin:$PATH"

# Benchmark configuration
MANIFEST_DIR="${MANIFEST_DIR:-$VIROSENSE_HOME/results/benchmark/manifest}"
CLASSIFIER="${CLASSIFIER:-/scratch/sahlab/shandley/virosense/model_7b/classifier.joblib}"
CACHE_DIR="${CACHE_DIR:-$VIROSENSE_HOME/results/benchmark/cache_7b}"
OUTPUT_DIR="${OUTPUT_DIR:-$VIROSENSE_HOME/results/benchmark/7b}"

echo "=== ViroSense 7B Benchmark ==="
echo "Manifest: $MANIFEST_DIR"
echo "Classifier: $CLASSIFIER"
echo "Cache: $CACHE_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# --- Step 0: Generate manifest if it doesn't exist ---
if [ ! -f "$MANIFEST_DIR/manifest.tsv" ]; then
    echo "=== Generating benchmark manifest ==="
    cd "$VIROSENSE_HOME"
    uv run python scripts/benchmark_unified.py manifest \
        --gyp-phage "$VIROSENSE_DATA/benchmarks/gauge_your_phage/phage_fragment_set.fasta" \
        --gyp-chromosome "$VIROSENSE_DATA/benchmarks/gauge_your_phage/host_chr_pvog_fragments.fasta" \
        --gyp-plasmid "$VIROSENSE_DATA/benchmarks/gauge_your_phage/host_plasmid_pvog_fragments.fasta" \
        --rna-fasta "$VIROSENSE_DATA/reference/rna_viruses/RNA_virus_database.fasta" \
        --cellular-fasta "$VIROSENSE_DATA/reference/3class/sequences.fasta" \
        --cellular-labels "$VIROSENSE_DATA/reference/3class/labels.tsv" \
        --output "$MANIFEST_DIR"
    echo ""
fi

# --- Step 1: Start or reuse NIM server ---
get_nim_host() {
    local jobid="$1"
    local logfile="nim-server-${jobid}.log"
    local waited=0
    while [ ! -f "$logfile" ] && [ $waited -lt 120 ]; do
        sleep 5
        waited=$((waited + 5))
    done
    [ -f "$logfile" ] && grep -m1 'NIM_SERVER_HOST=' "$logfile" 2>/dev/null | cut -d= -f2
}

wait_for_nim() {
    local host="$1"
    local max_wait="${2:-600}"
    local waited=0
    echo "Waiting for NIM server at http://${host}:8000 ..."
    while [ $waited -lt $max_wait ]; do
        if curl -sf "http://${host}:8000/v1/health/ready" >/dev/null 2>&1; then
            echo "NIM server is ready! (waited ${waited}s)"
            return 0
        fi
        sleep 10
        waited=$((waited + 10))
        [ $((waited % 60)) -eq 0 ] && echo "  Still waiting... (${waited}s / ${max_wait}s)"
    done
    echo "ERROR: NIM server did not become ready within ${max_wait}s"
    return 1
}

cleanup() {
    if [ -n "${NIM_JOB_STARTED:-}" ] && [ -n "${NIM_JOB_ID:-}" ]; then
        echo ""
        echo "Cancelling NIM server job $NIM_JOB_ID ..."
        scancel "$NIM_JOB_ID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [ -n "${NIM_JOB_ID:-}" ]; then
    echo "=== Reusing existing NIM server job: $NIM_JOB_ID ==="
    NIM_HOST=$(get_nim_host "$NIM_JOB_ID")
else
    echo "=== Starting NIM Evo2 server ==="
    NIM_JOB_ID=$(sbatch --parsable "$SCRIPT_DIR/start_nim_server.sbatch")
    NIM_JOB_STARTED=1
    echo "Submitted NIM server job: $NIM_JOB_ID"
    echo "Log: nim-server-${NIM_JOB_ID}.log"
    NIM_HOST=""
    for i in $(seq 1 60); do
        NIM_HOST=$(get_nim_host "$NIM_JOB_ID")
        [ -n "$NIM_HOST" ] && break
        sleep 5
    done
fi

if [ -z "${NIM_HOST:-}" ]; then
    echo "ERROR: Could not determine NIM server hostname"
    exit 1
fi

NIM_URL="http://${NIM_HOST}:8000"
echo "NIM URL: $NIM_URL"

if ! wait_for_nim "$NIM_HOST" 600; then
    echo "NIM server failed to start. Check log: cat nim-server-${NIM_JOB_ID}.log"
    exit 1
fi

# --- Step 2: Run benchmark ---
echo ""
echo "=== Running 7B Benchmark ==="
cd "$VIROSENSE_HOME"
uv run python scripts/benchmark_unified.py run \
    --manifest "$MANIFEST_DIR" \
    --classifier "$CLASSIFIER" \
    --backend nim \
    --model evo2_7b \
    --nim-url "$NIM_URL" \
    --layer blocks.28.mlp.l3 \
    --cache-dir "$CACHE_DIR" \
    --output "$OUTPUT_DIR"

echo ""
echo "=== Benchmark complete ==="
echo "Results: $OUTPUT_DIR"
echo "NIM server job $NIM_JOB_ID will be cancelled on exit."
