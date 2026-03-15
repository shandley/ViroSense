# ViroSense - Claude Code Context

Last updated: 2026-03-15

## Project Overview

**ViroSense** is an end-to-end viral detection and annotation tool. It uses the Evo2 DNA foundation model for sequence-level viral detection, then integrates structural annotation via ColabFold + BFVD + Foldseek + FoldMason for protein-level functional characterization.

### Architecture Vision
```
Raw contigs → ViroSense detect (Evo2) → viral contigs
  → Pyrodigal-gv gene calling → protein ORFs
  → AlphaFold DB lookup / ColabFold → protein structures (PDB)
  → Foldseek search against BFVD (351K viral structures) → structural neighbors
  → FoldMason structural MSA → guide tree + LDDT
  → Functional classification → categories from hit descriptions
  → Export → anvi'o, DRAM-v, vConTACT2/3, GFF3
```

### Relationship to Other Tools
- **pHold**: Structural annotation via ProstT5 + Foldseek (ViroSense replaces ProstT5 with ColabFold/AlphaFold DB)
- **Phynteny**: Synteny-based functional prediction via LSTM/transformer (complementary signal)
- **vHold** (sunset): Portable utility modules (gene calling, Foldseek/FoldMason wrappers, export formats) are being migrated to ViroSense's `annotate/` directory. ProstT5-dependent modules are NOT migrating.
- **Foldseek/FoldMason/BFVD**: Steinegger lab tools used as-is for structural search and alignment
- **ColabFold**: Structure prediction for novel proteins; AlphaFold DB lookup for known proteins

### `annotate` Module (in progress — 109 tests passing)
The `annotate` module extends ViroSense from detection ("is this viral?") to functional annotation ("what do the proteins do?"). Standalone modules in `annotate/`:

| Module | File | Tests | Status |
|--------|------|:-----:|:------:|
| Structure acquisition | `annotate/structure.py` | 27 | COMPLETE |
| Foldseek PDB search | `annotate/foldseek.py` | 27 | COMPLETE |
| FoldMason alignment | `annotate/foldmason.py` | 13 | COMPLETE |
| Functional classification | `annotate/categories.py` | 42 | COMPLETE |
| Gene calling (Pyrodigal-gv) | `annotate/genecall.py` | — | Pending |
| Metagenomic export | `annotate/export.py` | — | Pending |
| CLI wiring | — | — | Pending |

See `annotate/README.md` for full integration plan.

### Key Constraint
Evo2 requires NVIDIA GPU with FP8 support (L40S/H100/Ada+, NOT A100/V100). Developer machine is Apple M4. Solution: backend abstraction with NVIDIA NIM API as default, self-hosted NIM on HPC clusters, plus MLX backend for local Apple Silicon inference.

### Two Model Tiers
| | Evo2 40B (Cloud NIM) | Evo2 7B (Self-hosted / MLX) |
|---|---|---|
| Embedding dim | 8,192 | 4,096 |
| Speed | ~27s/seq (rate-limited) | ~3.3s/seq (unlimited) |
| GPU requirement | 2x H100 80GB | 1x L40S 48GB (or Apple Silicon) |
| Best for | Publications, high-stakes | Screening, exploration, local dev |

Classifiers are NOT interchangeable between tiers — each requires its own trained reference model. Cloud NIM always serves 40B; the NIM backend auto-corrects `--model` to `evo2_40b` for cloud usage. Benchmark results are stored in `data/reference/model/metrics.json` and script output directories.

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
├── MLXBackend   — Apple Silicon local inference (Evo2 7B via MLX, no CUDA needed)
│                  Needs: `python scripts/download_evo2_weights.py` (~13.2 GB)
└── LocalBackend — Direct evo2 package (needs CUDA H100/Ada GPU)
```

All backends implement: `extract_embeddings()`, `is_available()`, `max_context_length()`

### Dependencies
- **Core**: click, biopython, numpy, pandas, loguru, scikit-learn (no torch!)
- **[gpu]**: torch, evo2
- **[vhold]**: torch, transformers, sentencepiece
- **[nim]**: httpx
- **[mlx]**: mlx, safetensors, huggingface-hub
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
- Self-hosted NIM support: `--nim-url` flag on all commands
  - Auto-detects self-hosted mode: uses `/biology/arc/evo2/forward` endpoint, `decoder.layers.N` layer names
  - Handles different tensor shape `(seq_len, 1, hidden_dim)` vs cloud `(1, seq_len, hidden_dim)`
  - Max context: 10,000bp (vs 16,000bp cloud), concurrency: ~2 (vs 3 cloud)
- Layer name translation: `blocks.[n].*` (native) ↔ NIM format in `utils/constants.py`
- Per-sequence HTTP calls with base64 NPZ decoding and mean-pooling
- Handles dual NIM response: 200 JSON (short seqs) and 302 S3 redirect (long seqs)
- Retry logic with exponential backoff for 429/503 and `httpx.TransportError` (5 attempts)
- Sequence validation (16,000 bp max), N-base sanitization
- Local backend remains as stub (NIM is default and production-ready)
- 43 mocked async HTTP tests (including constructor, concurrency, retry logic)

### Phase 3: Embedding Infrastructure — COMPLETE
- `features/evo2_embeddings.py` with incremental checkpointing (saves every 50 sequences)
- Resume-capable: loads partial cache on restart, skips completed sequences
- NPZ cache format, batch-based extraction
- Embedding dimension: 8,192 (Evo2 40B via cloud NIM) or 4,096 (Evo2 7B via self-hosted NIM / MLX)

### Phase 4: detect module — COMPLETE
- `ViralClassifier` sklearn MLP wrapper in `models/detector.py` (train/predict/save/load via joblib)
- `classify_contigs()` for binary viral/cellular classification with threshold
- `train_classifier()` in `models/training.py` with 3-way split (train/calibrate/test)
- Platt scaling (sigmoid calibration) via `CalibratedClassifierCV` + `FrozenEstimator`
- Reports Brier score and ECE (Expected Calibration Error) before and after calibration
- Graceful fallback: skips calibration when holdout < 40 samples
- `run_detect` pipeline: FASTA → filter → embeddings → classify → TSV
- `build-reference` subcommand for training from labeled RefSeq data
- Cold-start approach: ship reference model + custom training support
- **L2-normalization**: Optional `--normalize-l2` flag on `build-reference` — eliminates length-dependent magnitude effects that cause RNA virus detection failure at longer lengths (34% → 99% recall at 10-16kb). Stored in classifier metadata; loaded models auto-apply correct preprocessing. See `docs/rna_virus_length_analysis.md`.

### Phase 5: vHold bridge — LEGACY (replaced by annotate module)
- `features/prostt5_bridge.py` loads vHold NPZ embeddings and TSV annotations
- **Superseded by `annotate` module**: ColabFold/AlphaFold DB replaces ProstT5 dependency
- Retained for backward compatibility with existing cluster/context workflows

### Phase 6: cluster module — COMPLETE
- `clustering/multimodal.py` with HDBSCAN, Leiden, and KMeans algorithms
- PCA dimensionality reduction as default preprocessing (auto ~90% variance, `--pca-dims`)
- Embedding fusion: DNA-only, protein-only, or multi-modal concatenation
- Auto-k estimation via elbow method (KMeans)
- Centroid distances, representative selection, silhouette scoring
- `run_cluster` pipeline: FASTA → Evo2 embeddings → optional vHold fusion → PCA → cluster → results
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

### Phase 10: adaptive prophage scanning — COMPLETE
- Two-pass coarse→fine scanning (`--scan-mode adaptive`, default)
- Coarse pass: 15kb windows, 10kb step → identifies candidate regions (score >= 0.3)
- Fine pass: 5kb windows, 2kb step, restricted to candidate regions + 20kb margin
- ~5x reduction in API calls for typical bacterial chromosomes
- `identify_candidate_regions()`: interval expansion, overlap merging, chromosome-aware
- `generate_windows_for_regions()`: fine-resolution tiling within candidate intervals
- Auto-bypass: inputs < 100 fine windows skip coarse pass (backward compatible)
- Summary JSON includes adaptive scan statistics (n_coarse_windows, n_candidate_regions, etc.)
- CLI: `--scan-mode`, `--coarse-window-size`, `--coarse-step-size`, `--coarse-threshold`, `--margin`
- 35 prophage tests

### Phase 11: MLX backend (Apple Silicon) — IN PROGRESS
- Full Evo2 7B forward pass reimplemented in Apple MLX (`backends/mlx_model.py`)
- StripedHyena 2 architecture: 32 blocks (HCS/HCM/HCL Hyena + MHA attention)
- HCS: short FIR filter (k=7), HCM: medium FIR filter (k=128, FFT), HCL: IIR filter (poles/residues, FFT)
- Byte-level DNA tokenizer (A=65, C=67, G=71, T=84)
- Lazy model loading, mean-pooled embedding extraction from any named layer
- `MLXBackend` implements `Evo2Backend` ABC, registered in factory as `--backend mlx`
- Weight download script: `python scripts/download_evo2_weights.py` (13.2 GB from HuggingFace)
- 46 tests (all synthetic weights, no download required), 223 total passing
- **Pending**: weight download, numerical validation against NIM, reference classifier retraining
- Produces 4096-D embeddings (Evo2 7B) — different representations from NIM (Evo2 40B), requires separate reference classifier

### Phase 12: HTCF deployment (self-hosted NIM) — COMPLETE
- Self-hosted NIM Evo2 7B on HTCF (Washington University HPC cluster) via Apptainer
- Container: `nvcr.io/nim/arc/evo2:2` converted to SIF with `/root` symlink workaround
- GPU: L40S 48GB on n099 (only HTCF node with FP8 support, compute capability 8.9)
- A100s (n095-n098) cannot run NIM Evo2 — FP8 requires cc 8.9+
- Apptainer workarounds: `--writable-tmpfs` (read-only SIF), `--no-home` (20 GiB quota), explicit PATH/HOME env vars
- NIM model profile `d14c09c133...` selects 7B variant (40B needs 3+ GPUs)
- Scripts: `htcf/setup.sh` (one-time), `htcf/start_nim_server.sbatch` (SLURM job), `htcf/virosense_pipeline.sh` (orchestrator)
- 3-class embedding cache and trained 7B classifiers on scratch

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `NVIDIA_API_KEY` | NIM API access (cloud and self-hosted first-run) |
| `NGC_API_KEY` | NGC authentication for NIM model weight download (same as NVIDIA_API_KEY) |
| `VIROSENSE_DATA_DIR` | Override ~/.virosense |
| `VIROSENSE_CACHE_DIR` | Embedding cache location |

## Key Source Files

| File | Purpose |
|------|---------|
| `src/virosense/cli.py` | Click CLI with 6 subcommands |
| `src/virosense/backends/base.py` | Evo2Backend ABC + factory |
| `src/virosense/backends/nim.py` | NIM API client (production default) |
| `src/virosense/backends/mlx_backend.py` | MLX backend for Apple Silicon |
| `src/virosense/backends/mlx_model.py` | Evo2 7B model in MLX (StripedHyena 2) |
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
| `src/virosense/annotate/structure.py` | AlphaFold DB lookup + ColabFold prediction |
| `src/virosense/annotate/foldseek.py` | Foldseek structural search against BFVD |
| `src/virosense/annotate/foldmason.py` | FoldMason structural MSA |
| `src/virosense/annotate/categories.py` | Functional classification (11 categories) |

## Development

```bash
# Install in dev mode (NIM backend)
uv sync --extra dev --extra nim

# Install for Apple Silicon (MLX backend)
uv sync --extra dev --extra mlx
python scripts/download_evo2_weights.py  # ~13.2 GB

# Run tests
uv run pytest tests/ -v

# Run CLI
uv run virosense --help

# Self-hosted NIM (HTCF or any GPU server with L40S/H100/Ada)
virosense detect -i contigs.fasta -o results/ --backend nim --nim-url http://<host>:8000

# Train 7B classifier from cached embeddings
uv run python scripts/train_binary_from_cache.py \
    --cache embeddings.npz --labels labels.tsv --output model/ --install
```

## Available Datasets (not in git)

| Dataset | Location | Description |
|---------|----------|-------------|
| RNA Virus Database | `data/reference/rna_viruses/RNA_virus_database.fasta` | 385,732 RNA virus sequences (NCBI + RVMT + terrestrial) |
| Gauge Your Phage - Phage | `data/benchmarks/gauge_your_phage/phage_fragment_set.fasta` | 6,664 phage fragments (1-15 kbp) |
| Gauge Your Phage - Chromosome | `data/benchmarks/gauge_your_phage/host_chr_pvog_fragments.fasta` | 104,003 chromosome fragments |
| Gauge Your Phage - Plasmid | `data/benchmarks/gauge_your_phage/host_plasmid_pvog_fragments.fasta` | 2,754 plasmid fragments |

Benchmark results are stored in script output directories and `data/reference/model/metrics.json`, not in this file.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/prepare_reference_data.py` | Download + fragment RefSeq genomes for training |
| `scripts/prepare_reference_data_3class.py` | Prepare 3-class (chromosome/plasmid/viral) reference data |
| `scripts/prepare_reference_data_4class.py` | Prepare 4-class reference data |
| `scripts/train_binary_from_cache.py` | Train binary classifier from cached 7B embeddings |
| `scripts/validate_model.py` | Cross-validation, hard examples, misclassification analysis |
| `scripts/filter_prophage_noise.py` | Post-hoc prophage contamination detection |
| `scripts/compare_genomad.py` | Head-to-head ViroSense vs geNomad comparison |
| `scripts/analyze_clusters.py` | PCA + HDBSCAN clustering with taxonomy cross-reference |
| `scripts/download_evo2_weights.py` | Download Evo2 7B weights from HuggingFace for MLX backend |
| `scripts/validate_rna_viruses.py` | Zero-shot RNA virus validation (classifier generalization) |
| `scripts/benchmark_gauge_your_phage.py` | Gauge Your Phage community benchmark (phage/chr/plasmid) |
| `scripts/benchmark_unified.py` | Unified benchmark runner (GYP + RNA virus, 7B/40B, manifest-based) |
| `scripts/analyze_rna_length.py` | RNA virus length-dependent detection diagnostics (PCA, cosine sim, norms) |
| `htcf/setup.sh` | One-time HTCF setup: uv, ViroSense, NIM container pull |
| `htcf/start_nim_server.sbatch` | SLURM job: NIM Evo2 7B server on L40S GPU |
| `htcf/virosense_pipeline.sh` | Orchestrator: start NIM → wait → run ViroSense → cleanup |

## Performance / Speed

### Current State
NIM cloud API: ~27s per sequence (Evo2 40B inference), now with async concurrent requests.
The 40 RPM rate limit allows up to ~13 concurrent requests (since each takes 27s).

### Phase A — NIM Async Concurrency — COMPLETE
- `NIMBackend` uses `httpx.AsyncClient` + `asyncio.gather()` + `asyncio.Semaphore(3)`
- Empirically validated: NVIDIA cloud NIM serializes requests per user; concurrency=3 is optimal
- Self-hosted NIM: unlimited concurrency via `--nim-url` flag on all commands
- Timeout: 300s (prevents queue timeouts for concurrent requests)

### Phase C1 — Adaptive Prophage Scanning — COMPLETE
- Two-pass coarse→fine scanning reduces API calls ~5x for typical chromosomes
- Coarse: 15kb windows, 10kb step → ~500 windows for 5Mb chromosome
- Fine: 5kb/2kb only on candidate regions (score >= 0.3, ±20kb margin)
- Auto-bypass for small inputs (< 100 fine windows)
- CLI: `--scan-mode adaptive|full` (default: adaptive)

### Phase B — MLX Backend (Apple Silicon) — PARTIAL
- Replaces the original Modal backend plan with local Apple Silicon inference
- Full Evo2 7B (StripedHyena 2) forward pass in MLX, no CUDA needed
- Embedding extraction only (no generation, no KV cache)
- Estimated: ~10-20s per 5kb sequence on M4 Max, no rate limits
- **Pending**: weight download (13.2 GB), numerical validation, reference classifier retraining
- Uses 7B model directly (4096-D embeddings) vs NIM's 40B model

### Speed Optimization Plan (remaining)

**Phase C2 — K-mer pre-filtering:**
- Tetranucleotide frequency pre-filter for detect: skip obviously bacterial contigs
- Expected: 50-80% of contigs pre-classified by k-mer alone

### Self-hosted NIM
Docker: `nvcr.io/nim/arc/evo2:2` — API differs from cloud.
- 7B model: single L40S (48GB), ~3.7s per 1000bp forward pass
- Point existing NIMBackend at `localhost:8000` via `--nim-url`
- **API differences from cloud**: endpoint is `/biology/arc/evo2/forward` (not `/evo2-40b/forward`), layer names use `decoder.layers.N` (not `blocks.N.mlp.l3`), tensor shape is `(seq_len, 1, 4096)` (not `(1, seq_len, 4096)`), max context is 10,000bp (not 16,000bp)
- NIM backend auto-handles these differences when `--nim-url` is set
- Concurrency: L40S 7B handles ~2 concurrent requests (returns "Too Busy" beyond that)
- 54 NIM backend tests including self-hosted mode

### HTCF Deployment
Self-hosted NIM running on Washington University HTCF cluster via Apptainer.

**Infrastructure:**
- Login: `ssh shandley@login.htcf.wustl.edu`
- GPU: L40S 48GB on n099 (only FP8-capable node; A100/V100 nodes cannot run NIM Evo2)
- Filesystem: scratch (`/scratch/sahlab/shandley`, 2 TiB) for everything large; home is only 20 GiB

**HTCF Paths:**
| Path | Contents |
|------|----------|
| `/scratch/sahlab/shandley/virosense/ViroSense/` | ViroSense installation |
| `/scratch/sahlab/shandley/containers/evo2_nim.sif` | NIM Apptainer container |
| `/scratch/sahlab/shandley/containers/nim_cache/` | NIM model weight cache (~13 GB) |
| `/scratch/sahlab/shandley/virosense/embedding_cache_3class/` | 9,159 x 4,096 cached embeddings (145 MB) |
| `/scratch/sahlab/shandley/virosense/model_binary_7b/` | Binary classifier (with plasmid) |
| `/scratch/sahlab/shandley/virosense/model_binary_7b_noplasmid/` | Binary classifier (no plasmid) |
| `results/benchmark/cache_7b/` (on ViroSense) | 13,417 x 4,096 full GYP+RNA benchmark embeddings |
| `results/benchmark/7b_16kb/` (on ViroSense) | 7B benchmark results + RNA analysis |
| `results/classifiers/7b_16kb/` (on ViroSense) | 7B binary classifier for benchmark |

**Apptainer workarounds (documented in `htcf/` scripts):**
1. `/root` symlink in NIM Docker image breaks Apptainer build → `--no-cleanup`, fix rootfs, rebuild SIF
2. Read-only SIF filesystem → `--writable-tmpfs`
3. No Docker ENTRYPOINT in SIF → `apptainer exec` with explicit `bash /opt/nim/start_server.sh`
4. Missing Docker ENV → explicit `--env PATH=...`
5. Home mount quota → `--no-home` + `--env HOME=/opt/nim/.cache`
6. NGC auth for weights → export `NGC_API_KEY` before `sbatch`

**Usage:**
```bash
# One-time setup (GPU node required for Apptainer)
export NVIDIA_API_KEY='nvapi-...'
sbatch htcf/setup.sh

# Run pipeline (starts NIM, waits, runs ViroSense, cleans up)
export NGC_API_KEY='nvapi-...'
bash htcf/virosense_pipeline.sh detect -i contigs.fasta -o results/
```

## Future Work / Notes

### Annotate module (next major feature — partially built)
- **Structure acquisition** (COMPLETE): `annotate/structure.py` — AlphaFold DB lookup + ColabFold prediction → PDB files. Two-pass: instant download for known UniProt proteins, ColabFold for novel proteins.
- **Foldseek PDB search** (COMPLETE): `annotate/foldseek.py` — PDB → `foldseek createdb` → search BFVD. Includes prob/LDDT/TM-score output (not available with 3Di-only input).
- **FoldMason alignment** (COMPLETE): `annotate/foldmason.py` — Structural MSA with full 3D coordinate refinement and LDDT scoring.
- **Functional classification** (COMPLETE): `annotate/categories.py` — 11 categories via keyword/Pfam/GO/SUPERFAMILY matching.
- **Gene calling** (PENDING): Port Pyrodigal-gv from vHold (drop miniprot).
- **Metagenomic export** (PENDING): Port anvi'o, DRAM-v, vConTACT2/3, GFF3 writers from vHold.
- **CLI wiring** (PENDING): Wire `virosense annotate` and `virosense run` subcommands.
- **Design doc**: `annotate/README.md`

### Benchmark Status (2026-03-15)
- **7B GYP benchmark**: COMPLETE (13,417 sequences via HTCF NIM). Results: 93.0% accuracy, 95.75% phage sensitivity, 98.2% chromosome specificity, 82.5% plasmid specificity, 63.1% RNA virus recall (without L2-norm). With L2-normalization retrain: 92.5% RNA recall. See `docs/rna_virus_length_analysis.md`.
- **40B GYP benchmark**: IN PROGRESS (~83% complete, running via cloud NIM). 11,183/13,417 classified.
- **L2-normalization fix**: IMPLEMENTED. Eliminates length-dependent RNA virus failure. Retrain with `--normalize-l2` needed for production classifiers.

### Existing planned work
- **Retrain official classifiers with L2-norm**: Both 7B and 40B classifiers need retraining with `--normalize-l2` for production use
- **Complete 40B GYP benchmark**: ~17% remaining, then repeat L2-norm analysis for 40B comparison
- **Improve 7B classifier**: Larger/more diverse training set, hyperparameter tuning, ensemble methods
- **RNA virus / metatranscriptomics support**: L2-normalization dramatically improves RNA virus detection. Consider including RNA virus cDNA in training data for further gains (especially short fragments <1kb).
- **Multi-class viral classification**: Train a multi-class classifier for virus type identification (lytic phage, temperate phage, euk RNA virus, euk DNA virus, cellular)
- **MLX backend optimization**: Replace Python conv loops with vectorized MLX ops or mx.conv1d
- **MLX numerical validation**: Compare MLX (7B) vs NIM (7B self-hosted) embeddings on same sequences
- **UHGV** (Unified Human Gut Virome Catalog): 873K virus genomes / 168K vOTUs. Potential benchmarking resource.
- **Prophage benchmark**: Validate against PHASTER/PhiSpy on curated prophage datasets
- **Methods paper**: Novel DNA foundation model approach to viral detection
- **RNA foundation models**: AIDO.RNA, RiNALMo, LucaOne as alternative backends for RNA-specific tasks
- **Biosurveillance research**: Perplexity forensics, distilled models, anomaly detection. See `docs/biosurveillance_research_plan.md`.

## Biosecurity Note

Evo2 deliberately excludes eukaryotic viral sequences from training for biosecurity. ViroSense uses Evo2 only for discriminative tasks (classification, clustering) — not sequence generation. This is a safe, defensive application focused on detection and characterization of existing sequences. The RNA virus embedding experiment confirms that while Evo2 has high perplexity (poor generation quality) for eukaryotic viruses, the intermediate hidden state representations still capture compositional signatures useful for detection — a discriminative use that does not compromise the biosecurity exclusion.
