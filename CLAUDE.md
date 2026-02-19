# ViroSense - Claude Code Context

Last updated: 2026-02-19

## Project Overview

**ViroSense** is a multi-modal viral detection and characterization tool that combines DNA-level analysis (Evo2 foundation model) with protein-level analysis (ProstT5/vHold) for viral sequence detection, annotation, and classification.

### Relationship to vHold
- vHold: Viral protein annotation via structural homology (ProstT5 + Foldseek)
- ViroSense: DNA + protein multi-modal analysis using Evo2 + vHold integration
- vHold is an optional dependency (`[vhold]` extra), integrated via file-based data exchange

### Key Constraint
Evo2 requires NVIDIA GPU (H100/Ada+ with CUDA 12.1+ and FP8). Developer machine is Apple M4. Solution: backend abstraction with NVIDIA NIM API as default.

## Architecture

### Five Modules

1. **detect** — Classify metagenomic contigs as viral vs cellular using Evo2 DNA embeddings
2. **context** — Enhance ORF annotation with genomic context (Evo2 windows + vHold merge)
3. **cluster** — Organize viral dark matter using fused DNA + protein embeddings
4. **classify** — Train discriminative classifiers on frozen Evo2 embeddings
5. **prophage** — Sliding-window scan of bacterial chromosomes for integrated prophage regions

### Backend Abstraction

```
Evo2Backend (ABC)
├── NIMBackend   — NVIDIA NIM API (default, async concurrent, needs NVIDIA_API_KEY)
│                  Supports cloud API and self-hosted via --nim-url
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
virosense prophage -i chromosome.fasta -o results/ --window-size 5000 --step-size 2000
virosense build-reference -i labeled.fasta --labels labels.tsv -o model/ --install
```

## Implementation Status

### Phase 1: Scaffold — COMPLETE
- Project structure, pyproject.toml, CLI with 5 commands (detect, context, cluster, classify, build-reference)
- Utils: constants, logging, config, device
- I/O: FASTA reader, ORF parser, TSV/JSON result writers

### Phase 2: Backend Layer — COMPLETE
- NIM API client (`backends/nim.py`) with async concurrent httpx
- Async concurrency: `asyncio.gather()` + `asyncio.Semaphore(10)` for 5-10x throughput
- Self-hosted NIM support: `--nim-url` flag on all commands (unlimited concurrency)
- Layer name translation: `blocks.[n].*` (native) ↔ NIM format in `utils/constants.py`
- Per-sequence HTTP calls with base64 NPZ decoding and mean-pooling
- Handles dual NIM response: 200 JSON (short seqs) and 302 S3 redirect (long seqs)
- Retry logic with exponential backoff for 429/503 and `httpx.TransportError` (5 attempts)
- Sequence validation (16,000 bp max), N-base sanitization
- Local/Modal backends remain as stubs (NIM is default and production-ready)
- 43 mocked async HTTP tests (including constructor, concurrency, retry logic)

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
- PCA dimensionality reduction as default preprocessing (auto ~90% variance, `--pca-dims`)
- Embedding fusion: DNA-only, protein-only, or multi-modal concatenation
- Auto-k estimation via elbow method (KMeans)
- Centroid distances, representative selection, silhouette scoring
- `run_cluster` pipeline: FASTA → Evo2 embeddings → optional vHold fusion → PCA → cluster → results
- Validated on real gut virome: 9 biologically coherent clusters separating Caudoviricetes from novel viruses
- 27 tests

### Phase 7: context module — COMPLETE
- `io/orfs.py` ORF parser: GFF3, prodigal protein FASTA, plain protein FASTA with auto-detection
- `run_context` pipeline: contigs + ORFs → genomic windows (±window/2 from midpoint) → Evo2 embeddings → optional vHold merge → annotated TSV
- 14 tests

### Phase 8: classify module — COMPLETE
- `run_classify` with training mode (FASTA + labels → classifier) and prediction mode (model + sequences → predictions TSV)
- Supports multi-class classification with string label encoding
- 6 tests

### Phase 9: prophage detection — COMPLETE
- `models/prophage.py` with `generate_windows()`, `score_windows()`, `merge_prophage_regions()`
- Sliding window approach: configurable window_size, step_size, merge_gap, min_region_length
- Auto-clamps window_size to backend `max_context_length()` (16,000 bp for NIM)
- Outputs: `prophage_windows.tsv`, `prophage_regions.tsv`, `prophage_regions.bed`, `prophage_summary.json`
- BED output for genome browser visualization (IGV, UCSC)
- Reuses `_load_classifier` from detect module, `extract_embeddings` with caching
- Validated on Salmonella enterica (2 prophage regions: 11 kb + 19 kb) and Lelliottia amnigena (1 region: 41 kb)
- Sharp boundary detection: scores transition 0.99→0.00 within a single window step
- 21 tests, 159 total passing

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `NVIDIA_API_KEY` | NIM API access |
| `VIROSENSE_DATA_DIR` | Override ~/.virosense |
| `VIROSENSE_CACHE_DIR` | Embedding cache location |

## Key Source Files

| File | Purpose |
|------|---------|
| `src/virosense/cli.py` | Click CLI with 6 subcommands |
| `src/virosense/backends/base.py` | Evo2Backend ABC + factory |
| `src/virosense/backends/nim.py` | NIM API client (production default) |
| `src/virosense/features/evo2_embeddings.py` | Embedding extraction + incremental NPZ cache |
| `src/virosense/features/prostt5_bridge.py` | Optional vHold integration |
| `src/virosense/clustering/multimodal.py` | Multi-modal fusion + HDBSCAN/Leiden/KMeans |
| `src/virosense/models/detector.py` | Viral classifier head (sklearn MLP) |
| `src/virosense/models/training.py` | Training loop + evaluation metrics |
| `src/virosense/io/fasta.py` | DNA FASTA I/O |
| `src/virosense/io/orfs.py` | ORF parser (GFF3/prodigal/FASTA) |
| `src/virosense/io/results.py` | TSV/JSON/BED result writers |
| `src/virosense/models/prophage.py` | Prophage window generation, scoring, region merging |
| `src/virosense/subcommands/detect.py` | Viral detection pipeline |
| `src/virosense/subcommands/prophage.py` | Prophage detection pipeline |
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

## Benchmarks

### Reference Classifier
- Trained on 6,158 RefSeq fragments (3,105 viral + 3,053 cellular), prophage-filtered
- Training accuracy: 99.0%, CV accuracy: 97.8%, CV AUC: 0.998

### Head-to-head vs geNomad v1.11.2

| Dataset | Metric | ViroSense | geNomad |
|---------|--------|-----------|---------|
| Simulated (200 RefSeq) | Accuracy | 96.0% | 84.0% |
| | Precision | 95.1% | 96.0% |
| | Recall | 97.0% | 71.0% |
| | F1 | 96.0% | 81.6% |
| Gut virome (CheckV truth) | Sensitivity | 98.6% | 47.9% |

ViroSense's composition-based approach detects viral sequences that gene-dependent tools miss, especially short fragments and novel viruses lacking recognizable marker genes.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/prepare_reference_data.py` | Download + fragment RefSeq genomes for training |
| `scripts/validate_model.py` | Cross-validation, hard examples, misclassification analysis |
| `scripts/filter_prophage_noise.py` | Post-hoc prophage contamination detection |
| `scripts/compare_genomad.py` | Head-to-head ViroSense vs geNomad comparison |
| `scripts/analyze_clusters.py` | PCA + HDBSCAN clustering with taxonomy cross-reference |

## Performance / Speed

### Current State
NIM cloud API: ~27s per sequence (Evo2 40B inference), now with async concurrent requests.
The 40 RPM rate limit allows up to ~13 concurrent requests (since each takes 27s).

### Phase A — NIM Async Concurrency — COMPLETE
- `NIMBackend` uses `httpx.AsyncClient` + `asyncio.gather()` + `asyncio.Semaphore(10)`
- 10 concurrent requests (conservative; 40 RPM limit allows ~18)
- Self-hosted NIM: unlimited concurrency via `--nim-url` flag on all commands
- Expected: 5-10x speedup on cloud, no rate-limit overhead on self-hosted
- Removed serial `time.sleep()` between requests

### Speed Optimization Plan (remaining)

**Phase B — Modal backend implementation:**
- Serverless A100/H100 via Modal.com Python SDK
- True batch inference (4-8 sequences per GPU invocation)
- Per-second billing: ~$0.87 for a full 5Mb chromosome scan
- Expected: 20 min for a full chromosome vs 20 hours on NIM serial

**Phase C — Algorithmic optimizations:**
- Adaptive prophage scanning: coarse pass (15kb windows) → fine pass on hits only
- K-mer pre-filtering for detect: skip obviously bacterial contigs
- Larger windows (10-15 kb) where resolution permits

### Self-hosted NIM
Docker: `nvcr.io/nim/arc/evo2:2` — same HTTP API as cloud, no rate limits.
- 7B model: single H100 (80GB), ~45 nt/s generation, ~1-3s forward pass
- 40B model: 2x H100 or 1x H200, ~26 nt/s
- Point existing NIMBackend at `localhost:8000` via `--nim-url`

### Benchmark Numbers (NVIDIA official)
| Model | GPU | Throughput |
|-------|-----|------------|
| 40B | 2x H100 80GB | 26 nt/sec |
| 40B | 1x H200 141GB | 33 nt/sec |
| 7B | 1x H100 80GB | 45 nt/sec |
| 7B | 1x H200 141GB | 52 nt/sec |

## Future Work / Notes

- **UHGV** (Unified Human Gut Virome Catalog): 873K virus genomes / 168K vOTUs. Potential benchmarking resource.
- **Multi-modal detection**: Fuse DNA + protein embeddings for improved precision
- **Prophage benchmark**: Validate against PHASTER/PhiSpy on curated prophage datasets
- **Calibration**: Platt scaling for well-calibrated confidence scores
- **Methods paper**: Novel DNA foundation model approach to viral detection

## Biosecurity Note

Evo2 deliberately excludes eukaryotic viral sequences from training for biosecurity. ViroSense uses Evo2 only for discriminative tasks (classification, clustering) — not sequence generation. This is a safe, defensive application focused on detection and characterization of existing sequences.
