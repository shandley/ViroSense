#!/bin/bash
#SBATCH --job-name=virosense_comprehensive
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n099
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/sahlab/shandley/virosense/logs/comprehensive_%j.out
#SBATCH --error=/scratch/sahlab/shandley/virosense/logs/comprehensive_%j.err

# Comprehensive Evo2 embedding extraction for 489 sequences
# Extracts per-position embeddings from blocks.10 (optimal layer)
# Computes periodicity metrics + mean-pooled vectors on-the-fly
# No large .npy files saved — only metrics JSON + mean-pooled .npy (~32KB each)

set -euo pipefail

# ── Environment ──
export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=/scratch/sahlab/shandley/.cache/uv
export NGC_API_KEY="${NGC_API_KEY:?Set NGC_API_KEY before submitting}"

VIROSENSE_DIR=/scratch/sahlab/shandley/virosense/ViroSense
NIM_SIF=/scratch/sahlab/shandley/containers/evo2_nim.sif
NIM_CACHE=/scratch/sahlab/shandley/containers/nim_cache
LOG_DIR=/scratch/sahlab/shandley/virosense/logs

mkdir -p "$LOG_DIR"
cd "$VIROSENSE_DIR"

echo "=== Starting NIM server ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Time: $(date)"

# ── Start NIM Evo2 server ──
apptainer exec \
    --nv \
    --writable-tmpfs \
    --no-home \
    --env HOME=/opt/nim/.cache \
    --env "PATH=/opt/nim/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --env "NGC_API_KEY=$NGC_API_KEY" \
    --env "NIM_CACHE_PATH=/nim_cache" \
    --env "NIM_MODEL_PROFILE=d14c09c133087abaeeb2149f2b9c19503225b922eb4fe4519d49b6f1209e6cd0" \
    --env "NIM_EVO2_FORWARD_SEQUENCE_LENGTH_LIMIT=16000" \
    --bind "$NIM_CACHE:/nim_cache" \
    "$NIM_SIF" \
    bash /opt/nim/start_server.sh &

NIM_PID=$!
echo "NIM PID: $NIM_PID"

# ── Wait for NIM to be ready ──
echo "Waiting for NIM server..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8000/v1/health/ready 2>/dev/null | grep -q "ready"; then
        echo "NIM ready after ${i}0 seconds"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "NIM failed to start after 20 minutes"
        kill $NIM_PID 2>/dev/null
        exit 1
    fi
    sleep 10
done

# ── Run extraction ──
echo ""
echo "=== Starting extraction ==="
echo "Time: $(date)"

# Use the codon_periodicity_panel.py extract command
# It reads FASTA files from results/codon_periodicity_panel/fasta/
# and extracts per-position embeddings via the local NIM server
# Computing metrics on-the-fly, saving only metrics JSON + mean-pooled npy

uv run python scripts/comprehensive_validation.py extract \
    --input results/codon_periodicity_panel/fasta \
    --output results/codon_periodicity_panel \
    --nim-url http://localhost:8000 \
    --panel panel.json 2>&1

EXTRACT_EXIT=$?

echo ""
echo "=== Extraction complete ==="
echo "Exit code: $EXTRACT_EXIT"
echo "Time: $(date)"

# ── Count results ──
N_METRICS=$(ls results/codon_periodicity_panel/embeddings/*_metrics.json 2>/dev/null | wc -l)
N_MEAN=$(ls results/codon_periodicity_panel/embeddings/*_mean.npy 2>/dev/null | wc -l)
echo "Metrics files: $N_METRICS"
echo "Mean-pooled files: $N_MEAN"

# ── Run analysis ──
echo ""
echo "=== Starting analysis ==="
uv run python scripts/comprehensive_validation.py analyze \
    --input results/codon_periodicity_panel \
    --output results/codon_periodicity_panel/analysis \
    --panel panel.json 2>&1

echo ""
echo "=== All done ==="
echo "Time: $(date)"

# ── Cleanup ──
kill $NIM_PID 2>/dev/null
wait $NIM_PID 2>/dev/null
