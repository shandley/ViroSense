#!/usr/bin/env bash
#SBATCH --job-name=virosense_tool_benchmark
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/sahlab/shandley/benchmark_tools/benchmark_%j.log
#
# Run all viral detection tools on the GYP + RNA virus benchmark dataset.
# Produces standardized output for comparison.
#
# Prerequisites:
#   1. Run htcf/benchmark_tools_setup.sh first
#   2. Copy benchmark data to HTCF:
#      scp results/benchmark/manifest/sequences.fasta \
#          shandley@login.htcf.wustl.edu:/scratch/sahlab/shandley/benchmark_tools/input/
#      scp results/benchmark/manifest/manifest.tsv \
#          shandley@login.htcf.wustl.edu:/scratch/sahlab/shandley/benchmark_tools/input/
#
# Usage:
#   sbatch htcf/benchmark_tools_run.sbatch

set -euo pipefail

BENCH_DIR="/scratch/sahlab/shandley/benchmark_tools"
INPUT_DIR="$BENCH_DIR/input"
OUTPUT_DIR="$BENCH_DIR/results"
FASTA="$INPUT_DIR/sequences.fasta"
CPUS=${SLURM_CPUS_PER_TASK:-8}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Add system deps to PATH
export PATH="$BENCH_DIR/bin:$PATH"

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "ViroSense Head-to-Head Benchmark"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $CPUS"
echo "Input: $FASTA"
echo "============================================================"

# Count input sequences
N_SEQS=$(grep -c "^>" "$FASTA")
echo "Input sequences: $N_SEQS"
echo ""

# --------------------------------------------------------------------------
# 1. geNomad
# --------------------------------------------------------------------------
GENOMAD="$BENCH_DIR/genomad_env/bin/genomad"
GENOMAD_DB="$BENCH_DIR/genomad_db"
GENOMAD_OUT="$OUTPUT_DIR/genomad"

if [ -x "$GENOMAD" ] && [ -d "$GENOMAD_DB" ]; then
    echo "=== [1/4] Running geNomad ==="
    rm -rf "$GENOMAD_OUT"
    mkdir -p "$GENOMAD_OUT"

    START_TIME=$SECONDS
    "$GENOMAD" end-to-end \
        "$FASTA" \
        "$GENOMAD_OUT" \
        "$GENOMAD_DB" \
        --threads "$CPUS" \
        --cleanup \
        2>&1 | tee "$GENOMAD_OUT/genomad.log" || {
        echo "geNomad end-to-end failed, retrying without provirus finding..."
        rm -rf "$GENOMAD_OUT"
        mkdir -p "$GENOMAD_OUT"
        "$GENOMAD" end-to-end \
            "$FASTA" \
            "$GENOMAD_OUT" \
            "$GENOMAD_DB" \
            --threads "$CPUS" \
            --cleanup \
            --disable-find-proviruses \
            2>&1 | tee "$GENOMAD_OUT/genomad.log"
    }
    GENOMAD_TIME=$((SECONDS - START_TIME))

    echo "geNomad completed in ${GENOMAD_TIME}s"
    echo "$GENOMAD_TIME" > "$GENOMAD_OUT/runtime_seconds.txt"

    # Record version
    "$GENOMAD" --version > "$GENOMAD_OUT/version.txt" 2>&1 || true
else
    echo "=== [1/4] geNomad: SKIPPED (not installed) ==="
    GENOMAD_TIME=0
fi

# --------------------------------------------------------------------------
# 2. DeepVirFinder
# --------------------------------------------------------------------------
DVF_PYTHON="$BENCH_DIR/dvf_env/bin/python"
DVF_SCRIPT="$BENCH_DIR/DeepVirFinder/dvf.py"
DVF_OUT="$OUTPUT_DIR/deepvirfinder"

if [ -x "$DVF_PYTHON" ] && [ -f "$DVF_SCRIPT" ]; then
    echo ""
    echo "=== [2/4] Running DeepVirFinder ==="
    rm -rf "$DVF_OUT"
    mkdir -p "$DVF_OUT"

    START_TIME=$SECONDS
    "$DVF_PYTHON" "$DVF_SCRIPT" \
        -i "$FASTA" \
        -o "$DVF_OUT" \
        -c "$CPUS" \
        2>&1 | tee "$DVF_OUT/dvf.log"
    DVF_TIME=$((SECONDS - START_TIME))

    echo "DeepVirFinder completed in ${DVF_TIME}s"
    echo "$DVF_TIME" > "$DVF_OUT/runtime_seconds.txt"
else
    echo "=== [2/4] DeepVirFinder: SKIPPED (not installed) ==="
    DVF_TIME=0
fi

# --------------------------------------------------------------------------
# 3. VirSorter2
# --------------------------------------------------------------------------
VS2="$BENCH_DIR/vs2_env/bin/virsorter"
VS2_DB="$BENCH_DIR/vs2_db"
VS2_OUT="$OUTPUT_DIR/virsorter2"

if [ -x "$VS2" ] && [ -d "$VS2_DB" ]; then
    echo ""
    echo "=== [3/4] Running VirSorter2 ==="
    rm -rf "$VS2_OUT"
    mkdir -p "$VS2_OUT"

    START_TIME=$SECONDS
    PATH="$BENCH_DIR/bin:$PATH" "$VS2" run \
        -i "$FASTA" \
        -w "$VS2_OUT" \
        --db-dir "$VS2_DB" \
        -j "$CPUS" \
        --include-groups "dsDNAphage,NCLDV,RNA,ssDNA,lavidaviridae" \
        --min-length 500 \
        --keep-original-seq \
        all \
        2>&1 | tee "$VS2_OUT/vs2.log"
    VS2_TIME=$((SECONDS - START_TIME))

    echo "VirSorter2 completed in ${VS2_TIME}s"
    echo "$VS2_TIME" > "$VS2_OUT/runtime_seconds.txt"
else
    echo "=== [3/4] VirSorter2: SKIPPED (not installed) ==="
    VS2_TIME=0
fi

# --------------------------------------------------------------------------
# 4. VIBRANT
# --------------------------------------------------------------------------
VIBRANT="$BENCH_DIR/vibrant_env/bin/VIBRANT"
VIBRANT_OUT="$OUTPUT_DIR/vibrant"

VIBRANT_SCRIPT="$BENCH_DIR/VIBRANT/VIBRANT_run.py"
VIBRANT_PYTHON="$BENCH_DIR/vibrant_env/bin/python"

if [ -f "$VIBRANT_SCRIPT" ] && [ -x "$VIBRANT_PYTHON" ]; then
    echo ""
    echo "=== [4/4] Running VIBRANT ==="
    rm -rf "$VIBRANT_OUT"
    mkdir -p "$VIBRANT_OUT"

    START_TIME=$SECONDS
    # VIBRANT needs prodigal and hmmsearch on PATH
    PATH="$BENCH_DIR/bin:$PATH" "$VIBRANT_PYTHON" "$VIBRANT_SCRIPT" \
        -i "$FASTA" \
        -folder "$VIBRANT_OUT" \
        -t "$CPUS" \
        -l 500 \
        2>&1 | tee "$VIBRANT_OUT/vibrant.log" || {
        echo "VIBRANT failed."
    }
    VIBRANT_TIME=$((SECONDS - START_TIME))

    echo "VIBRANT completed in ${VIBRANT_TIME}s"
    echo "$VIBRANT_TIME" > "$VIBRANT_OUT/runtime_seconds.txt"
else
    echo "=== [4/4] VIBRANT: SKIPPED (not installed) ==="
    VIBRANT_TIME=0
fi

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Benchmark complete!"
echo "============================================================"
echo "  geNomad:        ${GENOMAD_TIME}s"
echo "  DeepVirFinder:  ${DVF_TIME}s"
echo "  VirSorter2:     ${VS2_TIME}s"
echo "  VIBRANT:        ${VIBRANT_TIME}s"
echo ""
echo "Results in: $OUTPUT_DIR/"
echo "Run scripts/compare_tools.py to analyze results."

# Write timing summary
cat > "$OUTPUT_DIR/timing_summary.json" << ENDJSON
{
    "timestamp": "$TIMESTAMP",
    "node": "$(hostname)",
    "cpus": $CPUS,
    "n_sequences": $N_SEQS,
    "runtime_seconds": {
        "genomad": $GENOMAD_TIME,
        "deepvirfinder": $DVF_TIME,
        "virsorter2": $VS2_TIME,
        "vibrant": $VIBRANT_TIME
    }
}
ENDJSON
