# ViroSense - Claude Code Context

Last updated: 2026-02-13

## Project Overview

**ViroSense** is a multi-modal viral detection and characterization tool that combines DNA-level analysis (Evo2 foundation model) with protein-level analysis (ProstT5/vHold) for viral sequence detection, annotation, and classification.

### Relationship to vHold
- vHold: Viral protein annotation via structural homology (ProstT5 + Foldseek)
- ViroSense: DNA + protein multi-modal analysis using Evo2 + vHold integration
- vHold is an optional dependency (`[vhold]` extra), integrated via file-based data exchange

### Key Constraint
Evo2 requires NVIDIA GPU (H100/Ada+ with CUDA 12.1+ and FP8). Developer machine is Apple M4. Solution: backend abstraction with NVIDIA NIM API as default.

## Architecture

### Four Modules

1. **detect** — Classify metagenomic contigs as viral vs cellular using Evo2 DNA embeddings
2. **context** — Enhance ORF annotation with genomic context (Evo2 windows + vHold merge)
3. **cluster** — Organize viral dark matter using fused DNA + protein embeddings
4. **classify** — Train discriminative classifiers on frozen Evo2 embeddings

### Backend Abstraction

```
Evo2Backend (ABC)
├── NIMBackend   — NVIDIA NIM API (default, works anywhere, needs NVIDIA_API_KEY)
├── LocalBackend — Direct evo2 package (needs CUDA H100/Ada GPU)
└── ModalBackend — Modal.com serverless GPU (stub)
```

All backends implement: `extract_embeddings()`, `is_available()`, `max_context_length()`

### Dependencies
- **Core**: click, biopython, numpy, pandas, loguru, scikit-learn (no torch!)
- **[gpu]**: torch, evo2
- **[vhold]**: torch, transformers, sentencepiece
- **[nim]**: httpx
- **[dev]**: pytest, pytest-cov, ruff

## Commands

```bash
virosense detect -i contigs.fasta -o results/ --backend nim
virosense context -i viral.fasta --orfs orfs.gff3 -o results/
virosense cluster -i unknown.fasta -o clusters/ --mode multi
virosense classify -i seqs.fasta --labels labels.tsv -o model/
virosense build-reference -i labeled.fasta --labels labels.tsv -o model/ --install
```

## Implementation Status

### Phase 1: Scaffold — COMPLETE
- Project structure, pyproject.toml, CLI with 5 commands (detect, context, cluster, classify, build-reference)
- Utils: constants, logging, config, device
- I/O: FASTA reader, ORF parser, TSV/JSON result writers

### Phase 2: Backend Layer — COMPLETE
- NIM API client (`backends/nim.py`) with httpx
- Layer name translation: `blocks.[n].*` (native) ↔ NIM format in `utils/constants.py`
- Per-sequence HTTP calls with base64 NPZ decoding and mean-pooling
- Handles dual NIM response: 200 JSON (short seqs) and 302 S3 redirect (long seqs)
- Retry logic with exponential backoff for 429/503 and `httpx.TransportError` (5 attempts)
- Sequence validation (16,000 bp max), N-base sanitization
- Local/Modal backends remain as stubs (NIM is default and production-ready)
- 28 mocked HTTP tests

### Phase 3: Embedding Infrastructure — COMPLETE
- `features/evo2_embeddings.py` with incremental checkpointing (saves every 50 sequences)
- Resume-capable: loads partial cache on restart, skips completed sequences
- NPZ cache format, batch-based extraction
- Embedding dimension: 8192 (Evo2 40B via NIM)

### Phase 4: detect module — COMPLETE
- `ViralClassifier` sklearn MLP wrapper in `models/detector.py` (train/predict/save/load via joblib)
- `classify_contigs()` for binary viral/cellular classification with threshold
- `train_classifier()` in `models/training.py` (train/val split, metrics)
- `run_detect` pipeline: FASTA → filter → embeddings → classify → TSV
- `build-reference` subcommand for training from labeled RefSeq data
- Cold-start approach: ship reference model + custom training support

### Phase 5: vHold bridge — COMPLETE
- `features/prostt5_bridge.py` loads vHold NPZ embeddings and TSV annotations
- Optional integration via file-based data exchange
- Used by context (annotation merge) and cluster (protein embedding fusion)

### Phase 6: cluster module — COMPLETE
- `clustering/multimodal.py` with HDBSCAN, Leiden, and KMeans algorithms
- Embedding fusion: DNA-only, protein-only, or multi-modal concatenation
- Auto-k estimation via elbow method (KMeans)
- Centroid distances, representative selection, silhouette scoring
- `run_cluster` pipeline: FASTA → Evo2 embeddings → optional vHold fusion → cluster → results
- 21 tests

### Phase 7: context module — COMPLETE
- `io/orfs.py` ORF parser: GFF3, prodigal protein FASTA, plain protein FASTA with auto-detection
- `run_context` pipeline: contigs + ORFs → genomic windows (±window/2 from midpoint) → Evo2 embeddings → optional vHold merge → annotated TSV
- 14 tests

### Phase 8: classify module — COMPLETE
- `run_classify` with training mode (FASTA + labels → classifier) and prediction mode (model + sequences → predictions TSV)
- Supports multi-class classification with string label encoding
- 6 tests

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `NVIDIA_API_KEY` | NIM API access |
| `VIROSENSE_DATA_DIR` | Override ~/.virosense |
| `VIROSENSE_CACHE_DIR` | Embedding cache location |

## Key Source Files

| File | Purpose |
|------|---------|
| `src/virosense/cli.py` | Click CLI with 5 subcommands |
| `src/virosense/backends/base.py` | Evo2Backend ABC + factory |
| `src/virosense/backends/nim.py` | NIM API client (production default) |
| `src/virosense/features/evo2_embeddings.py` | Embedding extraction + incremental NPZ cache |
| `src/virosense/features/prostt5_bridge.py` | Optional vHold integration |
| `src/virosense/clustering/multimodal.py` | Multi-modal fusion + HDBSCAN/Leiden/KMeans |
| `src/virosense/models/detector.py` | Viral classifier head (sklearn MLP) |
| `src/virosense/models/training.py` | Training loop + evaluation metrics |
| `src/virosense/io/fasta.py` | DNA FASTA I/O |
| `src/virosense/io/orfs.py` | ORF parser (GFF3/prodigal/FASTA) |
| `src/virosense/io/results.py` | TSV/JSON result writers |
| `src/virosense/subcommands/detect.py` | Viral detection pipeline |
| `src/virosense/subcommands/context.py` | ORF context annotation pipeline |
| `src/virosense/subcommands/cluster.py` | Sequence clustering pipeline |
| `src/virosense/subcommands/classify.py` | Classifier training/prediction pipeline |
| `src/virosense/subcommands/build_reference.py` | Reference model builder |

## Development

```bash
# Install in dev mode
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Run CLI
uv run virosense --help
```

## Biosecurity Note

Evo2 deliberately excludes eukaryotic viral sequences from training for biosecurity. ViroSense uses Evo2 only for discriminative tasks (classification, clustering) — not sequence generation. This is a safe, defensive application focused on detection and characterization of existing sequences.
