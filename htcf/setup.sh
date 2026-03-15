#!/bin/bash
# ViroSense HTCF Setup Script
# One-time setup: install uv, ViroSense, and pull NIM container
#
# Run this as a SLURM job on a GPU node (Apptainer only available there):
#   export NVIDIA_API_KEY='nvapi-...'
#   sbatch htcf/setup.sh
#
# Or run interactively:
#   srun -p gpu --gpus=1 --time=02:00:00 --mem=16G --pty bash
#   bash htcf/setup.sh

#SBATCH --job-name=virosense-setup
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=virosense-setup-%j.log

set -euo pipefail

# Use scratch for large files (home is ceph with 20 GiB quota)
SCRATCH="/scratch/sahlab/shandley"
VIROSENSE_HOME="$SCRATCH/virosense"
CONTAINER_DIR="$SCRATCH/containers"
NIM_CACHE="$SCRATCH/containers/nim_cache"
APPTAINER_CACHE="/tmp/apptainer_cache_${USER}"
APPTAINER_TMP="/tmp/apptainer_tmp_${USER}"

echo "=== ViroSense HTCF Setup ==="
echo "Date: $(date -Iseconds)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi -L 2>/dev/null | head -1)"
echo ""

# --- Step 1: Install uv ---
echo "=== Step 1: Installing uv ==="
if [ -f "$HOME/.local/bin/uv" ]; then
    echo "uv already installed: $($HOME/.local/bin/uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "uv installed: $($HOME/.local/bin/uv --version)"
fi
export PATH="$HOME/.local/bin:$PATH"

# --- Step 2: Clone and install ViroSense ---
echo ""
echo "=== Step 2: Setting up ViroSense ==="
mkdir -p "$VIROSENSE_HOME"

if [ ! -d "$VIROSENSE_HOME/ViroSense" ]; then
    echo "Cloning ViroSense..."
    cd "$VIROSENSE_HOME"
    git clone https://github.com/scotthandley/ViroSense.git 2>/dev/null || \
        echo "Clone failed - copy manually: scp -r /path/to/ViroSense shandley@login.htcf.wustl.edu:$VIROSENSE_HOME/"
fi

if [ -d "$VIROSENSE_HOME/ViroSense" ]; then
    cd "$VIROSENSE_HOME/ViroSense"
    echo "Installing ViroSense with NIM backend..."
    # Use /tmp for uv cache to avoid home quota issues
    UV_CACHE_DIR="/tmp/uv_cache_${USER}" UV_LINK_MODE=copy uv sync --extra nim --extra dev
    echo "ViroSense installed: $(uv run virosense --version 2>/dev/null || echo 'checking...')"
    uv run virosense --help | head -5
else
    echo "WARNING: ViroSense directory not found at $VIROSENSE_HOME/ViroSense"
fi

# --- Step 3: Pull NIM Evo2 container ---
echo ""
echo "=== Step 3: Pulling NIM Evo2 container ==="
mkdir -p "$CONTAINER_DIR" "$NIM_CACHE" "$APPTAINER_CACHE" "$APPTAINER_TMP"

export APPTAINER_CACHEDIR="$APPTAINER_CACHE"
export APPTAINER_TMPDIR="$APPTAINER_TMP"

if [ -f "$CONTAINER_DIR/evo2_nim.sif" ]; then
    echo "NIM container already exists: $CONTAINER_DIR/evo2_nim.sif"
    echo "Size: $(du -h "$CONTAINER_DIR/evo2_nim.sif" | cut -f1)"
else
    # NGC authentication - uses NVIDIA API key
    if [ -z "${NVIDIA_API_KEY:-}" ]; then
        echo "ERROR: NVIDIA_API_KEY not set. Export it before running:"
        echo "  export NVIDIA_API_KEY='nvapi-...'"
        exit 1
    fi

    export APPTAINER_DOCKER_USERNAME='$oauthtoken'
    export APPTAINER_DOCKER_PASSWORD="$NVIDIA_API_KEY"

    echo "Pulling nvcr.io/nim/arc/evo2:2 ..."
    echo "NOTE: The NIM Docker image has /root as a symlink, which causes"
    echo "Apptainer build to fail. Using --no-cleanup workaround."
    echo ""

    # Step 3a: Pull image (will fail at 'mkdir /root' but keeps rootfs)
    apptainer build --no-cleanup --force \
        --tmpdir "$APPTAINER_TMP" \
        /tmp/nim_attempt.sif \
        docker://nvcr.io/nim/arc/evo2:2 2>&1 || true

    # Step 3b: Fix /root symlink in extracted rootfs
    ROOTFS=$(find "$APPTAINER_TMP" -maxdepth 3 -name rootfs -type d | head -1)
    if [ -z "$ROOTFS" ] || [ ! -d "$ROOTFS" ]; then
        echo "ERROR: Could not find extracted rootfs in $APPTAINER_TMP"
        exit 1
    fi
    echo "Rootfs: $ROOTFS"

    if [ -L "$ROOTFS/root" ]; then
        echo "Removing /root symlink and creating directory..."
        rm -f "$ROOTFS/root"
        mkdir -p "$ROOTFS/root"
    elif [ ! -d "$ROOTFS/root" ]; then
        mkdir -p "$ROOTFS/root"
    fi

    # Step 3c: Build SIF from fixed rootfs
    echo "Building SIF from fixed rootfs..."
    apptainer build --force \
        --tmpdir "$APPTAINER_TMP" \
        /tmp/evo2_nim.sif \
        "$ROOTFS" 2>&1 || {
            echo "ERROR: SIF build failed"
            exit 1
        }

    # Step 3d: Move to scratch (home quota too small for 11 GB SIF)
    cp /tmp/evo2_nim.sif "$CONTAINER_DIR/evo2_nim.sif"
    rm -f /tmp/evo2_nim.sif /tmp/nim_attempt.sif

    # Cleanup temp build dirs
    rm -rf "$APPTAINER_TMP"/build-temp-* "$APPTAINER_TMP"/bundle-temp-*

    echo "Container built: $(du -h "$CONTAINER_DIR/evo2_nim.sif" | cut -f1)"
fi

# --- Step 4: Verify ---
echo ""
echo "=== Step 4: Verification ==="
echo "Apptainer: $(apptainer --version)"
echo "Container: $(ls -lh "$CONTAINER_DIR/evo2_nim.sif" 2>/dev/null || echo 'NOT FOUND')"
echo "uv: $(uv --version)"
if [ -d "$VIROSENSE_HOME/ViroSense" ]; then
    cd "$VIROSENSE_HOME/ViroSense"
    echo "ViroSense: $(uv run python -c 'import virosense; print("OK")' 2>&1)"
fi
echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Export NGC_API_KEY: export NGC_API_KEY='nvapi-...'"
echo "  2. Submit NIM server: cd $VIROSENSE_HOME/ViroSense && sbatch htcf/start_nim_server.sbatch"
echo "  3. Or use pipeline: bash htcf/virosense_pipeline.sh detect -i contigs.fasta -o results/"
