#!/bin/bash
#SBATCH --job-name=genome_scan
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n099
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/sahlab/shandley/virosense/logs/genome_scan_%j.out
#SBATCH --error=/scratch/sahlab/shandley/virosense/logs/genome_scan_%j.err

set -euo pipefail

export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=/scratch/sahlab/shandley/.cache/uv
export NGC_API_KEY="${NGC_API_KEY:?Set NGC_API_KEY}"

VIROSENSE_DIR=/scratch/sahlab/shandley/virosense/ViroSense
NIM_SIF=/scratch/sahlab/shandley/containers/evo2_nim.sif
NIM_CACHE=/scratch/sahlab/shandley/containers/nim_cache

cd "$VIROSENSE_DIR"
mkdir -p results/genome_scan /scratch/sahlab/shandley/virosense/logs

echo "=== Starting NIM ==="
apptainer exec --nv --writable-tmpfs --no-home \
    --env HOME=/opt/nim/.cache \
    --env "PATH=/opt/nim/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --env "NGC_API_KEY=$NGC_API_KEY" \
    --env "NIM_CACHE_PATH=/nim_cache" \
    --env "NIM_MODEL_PROFILE=d14c09c133087abaeeb2149f2b9c19503225b922eb4fe4519d49b6f1209e6cd0" \
    --env "NIM_EVO2_FORWARD_SEQUENCE_LENGTH_LIMIT=16000" \
    --bind "$NIM_CACHE:/nim_cache" \
    "$NIM_SIF" bash /opt/nim/start_server.sh &

NIM_PID=$!

echo "Waiting for NIM..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8000/v1/health/ready 2>/dev/null | grep -q "ready"; then
        echo "NIM ready after ${i}0s"
        break
    fi
    [ $i -eq 120 ] && { echo "NIM failed"; kill $NIM_PID; exit 1; }
    sleep 10
done

echo "=== Running genome scan ==="
PYTHONUNBUFFERED=1 uv run python scripts/genome_scan.py --nim-url http://localhost:8000 2>&1

echo "=== Done ==="
ls -lh results/genome_scan/
kill $NIM_PID 2>/dev/null
