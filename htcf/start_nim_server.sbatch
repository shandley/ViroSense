#!/bin/bash
# Start NIM Evo2 server on an HTCF GPU node
#
# Usage: sbatch htcf/start_nim_server.sbatch
#
# The server listens on port 8000. Connect from login node or another job:
#   virosense detect -i contigs.fasta -o results/ --nim-url http://<GPU_NODE>:8000
#
# Check the log for the hostname:
#   grep NIM_SERVER_HOST nim-server-*.log

#SBATCH --job-name=nim-evo2
#SBATCH -p gpu
#SBATCH --gpus=L40S:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=nim-server-%j.log

set -euo pipefail

SCRATCH="/scratch/sahlab/shandley"
CONTAINER="$SCRATCH/containers/evo2_nim.sif"
NIM_CACHE="$SCRATCH/containers/nim_cache"

# Verify container exists
if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: NIM container not found at $CONTAINER"
    echo "Run setup.sh first: sbatch htcf/setup.sh"
    exit 1
fi

mkdir -p "$NIM_CACHE"

# Log connection info for pipeline script to parse
echo "NIM_SERVER_HOST=$(hostname)"
echo "NIM_SERVER_PORT=8000"
echo "NIM_SERVER_START=$(date -Iseconds)"
echo "NIM_SERVER_JOBID=$SLURM_JOB_ID"
echo "NIM_SERVER_GPU=$(nvidia-smi -L | head -1)"
echo ""

# Start NIM server with GPU passthrough
# --nv: NVIDIA GPU support (passes through /dev/nvidia* and driver libs)
# --bind: mount NIM model cache to persist downloaded models between runs
# Note: Using 'exec' instead of 'run' because the SIF was built with a
# /root symlink workaround that skips Docker ENV import. We set PATH
# explicitly and call /opt/nim/start_server.sh directly.
# NGC API key is needed on first run to download model weights (~14 GB for 7B)
# After first run, weights are cached in NIM_CACHE
# NIM_MODEL_PROFILE selects 7B model (requires FP8: L40S/H100/Ada, not A100/V100)
if [ -z "${NGC_API_KEY:-}" ]; then
    echo "WARNING: NGC_API_KEY not set. If model weights aren't cached, this will fail."
    echo "Set it with: export NGC_API_KEY='nvapi-...'"
fi

exec apptainer exec --nv \
    --no-home \
    --writable-tmpfs \
    --bind "$NIM_CACHE:/opt/nim/.cache" \
    --env "PATH=/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/sbin:/bin" \
    --env "NGC_API_KEY=${NGC_API_KEY:-}" \
    --env "HOME=/opt/nim/.cache" \
    --env "NIM_CACHE_PATH=/opt/nim/.cache" \
    --env "NIM_MODEL_PROFILE=d14c09c133087abaeeb2149f2b9c19503225b922eb4fe4519d49b6f1209e6cd0" \
    --env "NIM_EVO2_FORWARD_SEQUENCE_LENGTH_LIMIT=16000" \
    "$CONTAINER" \
    bash /opt/nim/start_server.sh
