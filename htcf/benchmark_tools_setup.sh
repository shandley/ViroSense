#!/usr/bin/env bash
# One-time setup: install viral detection tools on HTCF for head-to-head benchmark
#
# Run on a compute node (needs internet + memory):
#   srun --mem=16G --cpus-per-task=4 -p general --time=02:00:00 \
#       bash /scratch/sahlab/shandley/benchmark_tools/benchmark_tools_setup.sh
#
# Tools installed:
#   1. geNomad       — Camargo et al., Nature Biotechnology 2024
#   2. DeepVirFinder — Ren et al., Quantitative Biology 2020
#   3. VirSorter2    — Guo et al., Microbiome 2021
#   4. VIBRANT       — Kieft et al., Microbiome 2020
#
# System dependencies (downloaded as static binaries):
#   mmseqs2, hmmer, prodigal, aragorn
#
# All tools go to /scratch/sahlab/shandley/benchmark_tools/

set -euo pipefail

BENCH_DIR="/scratch/sahlab/shandley/benchmark_tools"
UV="$HOME/.local/bin/uv"
BIN_DIR="$BENCH_DIR/bin"
PLATFORM="linux-x86_64"

mkdir -p "$BENCH_DIR" "$BIN_DIR"
export PATH="$BIN_DIR:$PATH"
cd "$BENCH_DIR"

echo "============================================================"
echo "Setting up benchmark tools in $BENCH_DIR"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "============================================================"

# ==========================================================================
# Step 0: Download system dependencies as static binaries
# ==========================================================================
echo ""
echo "=== [0] System dependencies ==="

# --- MMseqs2 (needed by geNomad) ---
if [ ! -x "$BIN_DIR/mmseqs" ]; then
    echo "Downloading MMseqs2..."
    cd "$BENCH_DIR"
    wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
    tar xzf mmseqs-linux-avx2.tar.gz
    cp mmseqs/bin/mmseqs "$BIN_DIR/"
    rm -rf mmseqs mmseqs-linux-avx2.tar.gz
    echo "  mmseqs: $($BIN_DIR/mmseqs version 2>&1 || echo 'installed')"
else
    echo "  mmseqs already installed"
fi

# --- HMMER (needed by VirSorter2, VIBRANT) ---
if [ ! -x "$BIN_DIR/hmmsearch" ]; then
    echo "Downloading HMMER..."
    cd "$BENCH_DIR"
    wget -q http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz
    tar xzf hmmer-3.4.tar.gz
    cd hmmer-3.4
    ./configure --prefix="$BENCH_DIR/hmmer_install" --quiet
    make -j4 --quiet
    make install --quiet
    cp "$BENCH_DIR/hmmer_install/bin/"hmm* "$BIN_DIR/"
    cd "$BENCH_DIR"
    rm -rf hmmer-3.4 hmmer-3.4.tar.gz hmmer_install
    echo "  hmmer: $($BIN_DIR/hmmsearch -h | head -2 | tail -1)"
else
    echo "  hmmer already installed"
fi

# --- Prodigal (needed by VirSorter2, VIBRANT) ---
if [ ! -x "$BIN_DIR/prodigal" ]; then
    echo "Downloading Prodigal..."
    wget -q -O "$BIN_DIR/prodigal" \
        https://github.com/hyattpd/Prodigal/releases/download/v2.6.3/prodigal.linux
    chmod +x "$BIN_DIR/prodigal"
    echo "  prodigal: $($BIN_DIR/prodigal -v 2>&1 | head -1)"
else
    echo "  prodigal already installed"
fi

# --- ARAGORN (needed by geNomad) ---
if [ ! -x "$BIN_DIR/aragorn" ]; then
    echo "Downloading ARAGORN..."
    cd "$BENCH_DIR"
    wget -q http://www.ansikte.se/ARAGORN/Downloads/aragorn1.2.41.c
    gcc -O3 -o "$BIN_DIR/aragorn" aragorn1.2.41.c -lm
    rm -f aragorn1.2.41.c
    echo "  aragorn: $($BIN_DIR/aragorn -h 2>&1 | head -1 || echo 'installed')"
else
    echo "  aragorn already installed"
fi

echo ""
echo "System deps in $BIN_DIR:"
ls -la "$BIN_DIR/"

# ==========================================================================
# 1. geNomad
# ==========================================================================
echo ""
echo "=== [1/4] geNomad ==="
if [ ! -d "$BENCH_DIR/genomad_env" ]; then
    echo "Creating geNomad venv..."
    $UV venv "$BENCH_DIR/genomad_env" --python 3.10
    $UV pip install --python "$BENCH_DIR/genomad_env/bin/python" genomad
else
    echo "geNomad venv already exists, skipping install"
fi

# Download database
if [ ! -d "$BENCH_DIR/genomad_db" ]; then
    echo "Downloading geNomad database (~3.8 GB)..."
    "$BENCH_DIR/genomad_env/bin/genomad" download-database "$BENCH_DIR"
else
    echo "geNomad database already exists, skipping download"
fi

echo "geNomad version:"
"$BENCH_DIR/genomad_env/bin/genomad" --version 2>&1 || true

# ==========================================================================
# 2. DeepVirFinder
# ==========================================================================
echo ""
echo "=== [2/4] DeepVirFinder ==="
if [ ! -d "$BENCH_DIR/DeepVirFinder" ]; then
    echo "Cloning DeepVirFinder..."
    git clone https://github.com/jessieren/DeepVirFinder.git "$BENCH_DIR/DeepVirFinder"
else
    echo "DeepVirFinder repo already exists"
fi

if [ ! -d "$BENCH_DIR/dvf_env" ]; then
    echo "Creating DeepVirFinder venv..."
    $UV venv "$BENCH_DIR/dvf_env" --python 3.10
    # DeepVirFinder uses TensorFlow 1.x/Keras 2.x patterns
    # Install compatible versions
    $UV pip install --python "$BENCH_DIR/dvf_env/bin/python" \
        "numpy<2" scipy scikit-learn biopython h5py "keras<3" "tensorflow-cpu<2.16"
else
    echo "DeepVirFinder venv already exists, skipping install"
fi

echo "DeepVirFinder: testing import..."
"$BENCH_DIR/dvf_env/bin/python" -c "import tensorflow; print(f'TF {tensorflow.__version__}')" 2>&1 || true

# ==========================================================================
# 3. VirSorter2
# ==========================================================================
echo ""
echo "=== [3/4] VirSorter2 ==="
if [ ! -d "$BENCH_DIR/vs2_env" ]; then
    echo "Creating VirSorter2 venv..."
    $UV venv "$BENCH_DIR/vs2_env" --python 3.10
    # VirSorter2 is a snakemake pipeline; needs hmmer + prodigal on PATH
    $UV pip install --python "$BENCH_DIR/vs2_env/bin/python" \
        virsorter scikit-learn pandas click "ruamel.yaml<0.18" snakemake pulp
else
    echo "VirSorter2 venv already exists, skipping install"
fi

# Database setup
if [ ! -d "$BENCH_DIR/vs2_db" ]; then
    echo "Setting up VirSorter2 database..."
    PATH="$BIN_DIR:$PATH" "$BENCH_DIR/vs2_env/bin/virsorter" setup \
        -d "$BENCH_DIR/vs2_db" -j 4 2>&1 || {
        echo "WARNING: VirSorter2 database setup failed."
        echo "VirSorter2 will be skipped in the benchmark."
    }
else
    echo "VirSorter2 database already exists"
fi

# ==========================================================================
# 4. VIBRANT
# ==========================================================================
echo ""
echo "=== [4/4] VIBRANT ==="
if [ ! -d "$BENCH_DIR/VIBRANT" ]; then
    echo "Cloning VIBRANT..."
    git clone https://github.com/AnantharamanLab/VIBRANT.git "$BENCH_DIR/VIBRANT"
else
    echo "VIBRANT repo already exists"
fi

if [ ! -d "$BENCH_DIR/vibrant_env" ]; then
    echo "Creating VIBRANT venv..."
    $UV venv "$BENCH_DIR/vibrant_env" --python 3.10
    $UV pip install --python "$BENCH_DIR/vibrant_env/bin/python" \
        biopython pandas numpy scikit-learn matplotlib
else
    echo "VIBRANT venv already exists, skipping install"
fi

# VIBRANT database download
if [ ! -d "$BENCH_DIR/VIBRANT/databases/VIBRANT_setup_done" ] && [ ! -f "$BENCH_DIR/VIBRANT/databases/.setup_done" ]; then
    echo "Setting up VIBRANT databases (~12 GB)..."
    cd "$BENCH_DIR/VIBRANT"
    "$BENCH_DIR/vibrant_env/bin/python" VIBRANT_setup.py 2>&1 || {
        # Try the download script
        bash scripts/VIBRANT_setup.sh 2>&1 || {
            echo "WARNING: VIBRANT database setup failed."
            echo "VIBRANT will be skipped in the benchmark."
        }
    }
    touch "$BENCH_DIR/VIBRANT/databases/.setup_done" 2>/dev/null || true
    cd "$BENCH_DIR"
else
    echo "VIBRANT databases already set up"
fi

# ==========================================================================
# Summary
# ==========================================================================
echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "Tool locations:"
echo "  System deps:    $BIN_DIR/{mmseqs,hmmsearch,prodigal,aragorn}"
echo "  geNomad:        $BENCH_DIR/genomad_env/bin/genomad"
echo "  geNomad DB:     $BENCH_DIR/genomad_db/"
echo "  DeepVirFinder:  $BENCH_DIR/dvf_env/bin/python $BENCH_DIR/DeepVirFinder/dvf.py"
echo "  VirSorter2:     $BENCH_DIR/vs2_env/bin/virsorter"
echo "  VirSorter2 DB:  $BENCH_DIR/vs2_db/"
echo "  VIBRANT:        $BENCH_DIR/vibrant_env/bin/python $BENCH_DIR/VIBRANT/VIBRANT_run.py"
echo ""
echo "Space used:"
du -sh "$BENCH_DIR"/* 2>/dev/null | sort -h || true
echo ""
echo "Next: sbatch /scratch/sahlab/shandley/benchmark_tools/benchmark_tools_run.sbatch"
