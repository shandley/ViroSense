#!/bin/bash
# ViroSense HTCF Pipeline
#
# Orchestrates: start NIM server -> wait for ready -> run ViroSense -> cleanup
#
# Usage:
#   bash htcf/virosense_pipeline.sh detect -i contigs.fasta -o results/
#   bash htcf/virosense_pipeline.sh prophage -i chromosome.fasta -o results/
#   bash htcf/virosense_pipeline.sh build-reference -i seqs.fasta --labels labels.tsv -o model/
#
# Environment:
#   NGC_API_KEY     - required for first run (model weight download)
#   NIM_JOB_ID      - optional: reuse an existing NIM server job

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIROSENSE_HOME="/scratch/sahlab/shandley/virosense/ViroSense"
export PATH="$HOME/.local/bin:$PATH"

# --- Parse arguments ---
if [ $# -lt 1 ]; then
    echo "Usage: $0 <virosense-command> [args...]"
    echo ""
    echo "Examples:"
    echo "  $0 detect -i contigs.fasta -o results/"
    echo "  $0 prophage -i chromosome.fasta -o results/"
    echo "  $0 build-reference -i seqs.fasta --labels labels.tsv -o model/"
    exit 1
fi

VIROSENSE_CMD="$1"
shift
VIROSENSE_ARGS="$@"

# --- Functions ---
get_nim_host() {
    local jobid="$1"
    local logfile="nim-server-${jobid}.log"

    # Wait for log file to appear
    local waited=0
    while [ ! -f "$logfile" ] && [ $waited -lt 120 ]; do
        sleep 5
        waited=$((waited + 5))
    done

    if [ ! -f "$logfile" ]; then
        echo ""
        return 1
    fi

    grep -m1 'NIM_SERVER_HOST=' "$logfile" 2>/dev/null | cut -d= -f2
}

wait_for_nim() {
    local host="$1"
    local port="${2:-8000}"
    local max_wait="${3:-600}"  # 10 minutes default
    local waited=0

    echo "Waiting for NIM server at http://${host}:${port} ..."
    while [ $waited -lt $max_wait ]; do
        if curl -sf "http://${host}:${port}/v1/health/ready" >/dev/null 2>&1; then
            echo "NIM server is ready! (waited ${waited}s)"
            return 0
        fi
        sleep 10
        waited=$((waited + 10))
        if [ $((waited % 60)) -eq 0 ]; then
            echo "  Still waiting... (${waited}s / ${max_wait}s)"
        fi
    done

    echo "ERROR: NIM server did not become ready within ${max_wait}s"
    return 1
}

cleanup() {
    if [ -n "${NIM_JOB_STARTED:-}" ] && [ -n "${NIM_JOB_ID:-}" ]; then
        echo ""
        echo "Cancelling NIM server job $NIM_JOB_ID ..."
        scancel "$NIM_JOB_ID" 2>/dev/null || true
        echo "NIM server job cancelled."
    fi
}
trap cleanup EXIT

# --- Step 1: Start or reuse NIM server ---
if [ -n "${NIM_JOB_ID:-}" ]; then
    echo "=== Reusing existing NIM server job: $NIM_JOB_ID ==="
    NIM_HOST=$(get_nim_host "$NIM_JOB_ID")
else
    echo "=== Starting NIM Evo2 server ==="
    NIM_JOB_ID=$(sbatch --parsable "$SCRIPT_DIR/start_nim_server.sbatch")
    NIM_JOB_STARTED=1
    echo "Submitted NIM server job: $NIM_JOB_ID"
    echo "Log: nim-server-${NIM_JOB_ID}.log"
    echo ""

    # Wait for job to start and get hostname
    echo "Waiting for SLURM job to start..."
    NIM_HOST=""
    for i in $(seq 1 60); do
        NIM_HOST=$(get_nim_host "$NIM_JOB_ID")
        if [ -n "$NIM_HOST" ]; then
            break
        fi
        sleep 5
    done
fi

if [ -z "${NIM_HOST:-}" ]; then
    echo "ERROR: Could not determine NIM server hostname"
    echo "Check job status: squeue -j $NIM_JOB_ID"
    echo "Check log: cat nim-server-${NIM_JOB_ID}.log"
    exit 1
fi

NIM_URL="http://${NIM_HOST}:8000"
echo "NIM server host: $NIM_HOST"
echo "NIM URL: $NIM_URL"

# --- Step 2: Wait for NIM to be ready ---
# NIM needs time to load the model into GPU memory (2-5 minutes typical)
if ! wait_for_nim "$NIM_HOST" 8000 600; then
    echo "NIM server failed to start. Check log:"
    echo "  cat nim-server-${NIM_JOB_ID}.log"
    exit 1
fi

# --- Step 3: Run ViroSense ---
echo ""
echo "=== Running ViroSense ==="
echo "Command: virosense $VIROSENSE_CMD $VIROSENSE_ARGS --nim-url $NIM_URL"
echo ""

cd "$VIROSENSE_HOME"
uv run virosense "$VIROSENSE_CMD" $VIROSENSE_ARGS --backend nim --nim-url "$NIM_URL"

echo ""
echo "=== ViroSense complete ==="
echo "NIM server job $NIM_JOB_ID will be cancelled on exit."
