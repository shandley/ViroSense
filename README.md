# ViroSense

Multi-modal viral detection and characterization using DNA and protein structural analysis.

ViroSense combines DNA-level analysis (Evo2 foundation model) with protein-level analysis (ProstT5/vHold) for viral sequence detection, annotation, and classification from metagenomic data.

## Overview

Traditional viral detection tools rely on sequence homology, missing the vast "viral dark matter" -- divergent viruses with no detectable similarity to known references. ViroSense uses learned representations from the Evo2 DNA foundation model to capture genomic patterns independent of coding potential, enabling detection and characterization of novel viral sequences.

### Key features

- **DNA-level detection**: Classify metagenomic contigs as viral or cellular using Evo2 embeddings -- no gene calling required
- **Multi-modal analysis**: Fuse DNA (Evo2) and protein structural (ProstT5/vHold) embeddings for comprehensive characterization
- **Backend abstraction**: Run anywhere via NVIDIA NIM cloud API; no local GPU required
- **Lightweight classifiers**: scikit-learn MLP heads on frozen embeddings -- no PyTorch in the core install
- **Incremental processing**: Embedding cache with checkpointing supports resume after interruption

## Installation

Requires Python 3.10+.

```bash
# Install with NIM backend support (recommended)
pip install -e ".[nim]"

# Install with development tools
pip install -e ".[dev]"

# Install with optional vHold protein integration
pip install -e ".[vhold]"

# Install with local GPU support (requires CUDA H100/Ada+)
pip install -e ".[gpu]"
```

Or using [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra nim --extra dev
```

## Quick start

### 1. Set up API access

ViroSense uses the NVIDIA NIM API for Evo2 inference by default. Get an API key from [build.nvidia.com](https://build.nvidia.com/settings/api-keys):

```bash
export NVIDIA_API_KEY="nvapi-..."
```

### 2. Detect viral sequences

```bash
virosense detect \
    -i contigs.fasta \
    -o results/ \
    --backend nim
```

This produces a scored TSV with per-contig viral/cellular classifications and confidence scores.

### 3. Build a custom reference model

Train a classifier from labeled viral and cellular sequences:

```bash
virosense build-reference \
    -i labeled_sequences.fasta \
    --labels labels.tsv \
    -o model/ \
    --install
```

The labels file is tab-separated with `sequence_id` and `label` columns (0 = cellular, 1 = viral). The `--install` flag copies the trained model to the default location (`~/.virosense/models/`).

## Commands

### detect

Classify metagenomic contigs as viral or cellular.

```bash
virosense detect -i contigs.fasta -o results/ [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--backend` | nim | Inference backend (nim, local, modal) |
| `--model` | evo2_7b | Evo2 model |
| `--threshold` | 0.5 | Viral classification threshold |
| `--min-length` | 500 | Minimum contig length (bp) |
| `--layer` | blocks.28.mlp.l3 | Evo2 layer for embeddings |
| `--cache-dir` | None | Directory to cache embeddings |

Output: `detect_results.tsv` with columns `contig_id`, `contig_length`, `viral_score`, `classification`.

### context

Annotate ORFs with genomic context from Evo2 embeddings, optionally merged with vHold protein annotations.

```bash
virosense context \
    -i viral_contigs.fasta \
    --orfs orfs.gff3 \
    -o results/ \
    --vhold-output vhold_annotations.tsv
```

Accepts ORFs in GFF3, prodigal protein FASTA, or plain protein FASTA format. Extracts Evo2 embeddings from genomic windows centered on each ORF.

### cluster

Organize unclassified viral sequences into putative families using multi-modal embeddings.

```bash
virosense cluster \
    -i unknown_viral.fasta \
    -o clusters/ \
    --mode multi \
    --vhold-embeddings prostt5_embeddings.npz
```

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | multi | Embedding modality (dna, protein, multi) |
| `--algorithm` | hdbscan | Clustering method (hdbscan, leiden, kmeans) |
| `--min-cluster-size` | 5 | Minimum cluster size (HDBSCAN) |
| `--n-clusters` | auto | Number of clusters (KMeans only) |

### classify

Train a custom classifier on Evo2 embeddings for any label scheme (viral family, host range, etc.).

```bash
# Train
virosense classify \
    -i sequences.fasta \
    --labels labels.tsv \
    -o model/ \
    --task family

# Predict
virosense classify \
    -i new_sequences.fasta \
    --labels dummy.tsv \
    -o predictions/ \
    --predict new_sequences.fasta \
    --classifier-model model/classifier.joblib
```

### build-reference

Build the default viral/cellular reference classifier from labeled training data.

```bash
virosense build-reference \
    -i labeled.fasta \
    --labels labels.tsv \
    -o model/ \
    --install \
    --cache-dir cache/
```

The `--cache-dir` option enables incremental checkpointing: embeddings are saved every 50 sequences and reused on restart.

## Preparing reference data

The `scripts/prepare_reference_data.py` script downloads and fragments RefSeq genomes to build balanced training data:

```bash
python scripts/prepare_reference_data.py \
    --output data/reference/ \
    --n-viral 250 \
    --n-cellular 250 \
    --n-crass 50
```

This produces a FASTA file and labels TSV with balanced viral/cellular fragments at multiple length scales (500, 1000, 2000, 3000, 5000 bp).

### Prophage noise filtering

After training an initial model, suspected prophage contamination in the cellular class can be identified and removed:

```bash
python scripts/filter_prophage_noise.py \
    --model data/reference/model/classifier.joblib \
    --cache-dir data/reference/cache \
    --labels data/reference/labels.tsv \
    --fasta data/reference/sequences.fasta \
    --output data/reference/cleaned/ \
    --threshold 0.8
```

### Model validation

Validate a trained model with cross-validation and confidence analysis:

```bash
python scripts/validate_model.py \
    --model data/reference/model/classifier.joblib \
    --cache-dir data/reference/cache \
    --labels data/reference/labels.tsv \
    --output data/reference/validation/
```

## Architecture

```
virosense/
  backends/
    base.py          # Evo2Backend ABC + factory
    nim.py           # NVIDIA NIM API client (default)
    local.py         # Local GPU backend (stub)
    modal.py         # Modal.com backend (stub)
  features/
    evo2_embeddings.py   # Embedding extraction + NPZ cache
    prostt5_bridge.py    # Optional vHold integration
  models/
    detector.py      # Viral classifier (sklearn MLP)
    training.py      # Training loop + evaluation
  clustering/
    multimodal.py    # Embedding fusion + clustering algorithms
  io/
    fasta.py         # FASTA I/O
    orfs.py          # ORF parser (GFF3/prodigal/FASTA)
    results.py       # TSV/JSON output
  subcommands/
    detect.py        # Viral detection pipeline
    context.py       # ORF context annotation
    cluster.py       # Sequence clustering
    classify.py      # Custom classifier training/prediction
    build_reference.py   # Reference model builder
  cli.py             # Click CLI entry point
```

### Backend system

Evo2 requires NVIDIA H100/Ada GPUs for local inference. ViroSense abstracts this behind a backend interface:

- **NIM** (default): NVIDIA cloud API. Works on any machine with an API key. Processes sequences up to 16,000 bp.
- **Local**: Direct Evo2 Python package. Requires CUDA GPU with FP8 support.
- **Modal**: Modal.com serverless GPU. Planned for future.

The NIM backend translates between native Evo2 layer names (`blocks.[n].*`) and the NIM API format (`decoder.layers.[n].*`) automatically.

### Embedding caching

All embeddings are cached as NPZ files keyed by model + layer. The cache supports:

- Incremental checkpointing (saves every 50 sequences)
- Resume after interruption (skips already-cached sequences)
- Reuse across commands (detect, classify, and build-reference share the same cache)

## Environment variables

| Variable | Purpose |
|----------|---------|
| `NVIDIA_API_KEY` | NVIDIA NIM API authentication |
| `VIROSENSE_DATA_DIR` | Override default data directory (~/.virosense) |
| `VIROSENSE_CACHE_DIR` | Override default embedding cache location |

## Development

```bash
# Install in dev mode
uv sync --extra dev --extra nim

# Run tests (119 tests)
uv run pytest tests/ -v

# Run linter
uv run ruff check src/ tests/

# Run single test file
uv run pytest tests/test_classify.py -v
```

## Biosecurity

Evo2 deliberately excludes eukaryotic viral sequences from its training data for biosecurity reasons. ViroSense uses Evo2 only for discriminative tasks (classification, clustering, embedding extraction) -- never for sequence generation. This is a safe, defensive application focused on detection and characterization of existing sequences.

## License

MIT
