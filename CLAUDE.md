# ViroSense - Claude Code Context

Last updated: 2026-02-27

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
Evo2 requires NVIDIA GPU (H100/Ada+ with CUDA 12.1+ and FP8). Developer machine is Apple M4. Solution: backend abstraction with NVIDIA NIM API as default, plus MLX backend for local Apple Silicon inference.

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
- Embedding dimension: 8192 (Evo2 40B via NIM)

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

### Comparison vs Published Tools on Gauge Your Phage Benchmark

GYP RefSeq Artificial Contigs dataset (Ho et al., Microbiome 2023): 6,664 phage + 104,003 chromosome + 2,754 plasmid fragments, 1-15 kbp. ViroSense results from stratified 1,200-sequence sample of same dataset.

| Tool | Precision | Recall | F1 | Notes |
|------|-----------|--------|-----|-------|
| **ViroSense** | **0.92** | **0.99** | **0.96** | Evo2 embeddings, composition-based |
| VirSorter2 | 0.92 | 0.93 | 0.93 | Gene + composition hybrid |
| VIBRANT | 0.97 | 0.89 | 0.93 | Gene-based (requires >=4 ORFs) |
| PPR-Meta | 0.88 | 0.96 | 0.92 | Deep learning |
| DeepVirFinder | 0.85 | 0.89 | 0.87 | k-mer composition |
| Kraken2 | ~1.00 | ~0.85 | 0.87 | k-mer taxonomy |
| MetaPhinder | 0.85 | 0.83 | 0.84 | BLAST homology |
| VirFinder | 0.82 | 0.85 | 0.83 | k-mer composition |
| VirSorter | 0.80 | 0.82 | 0.81 | Gene-based |
| Seeker | 0.48 | 0.41 | 0.45 | LSTM |

**ViroSense achieves the highest F1 (0.96) and recall (0.99) on this benchmark.** Precision matches VirSorter2 (0.92); the gap vs VIBRANT's precision (0.97) is due to plasmid false positives. geNomad and PhaMer/PhaBOX were not tested in the original GYP study.

**Key differentiators:**
- **Highest recall**: 0.994 — misses only 3/500 phages (an archaeal pleomorphic virus, a host-ameliorated temperate phage, and a borderline short fragment)
- **Length-independent**: >99% sensitivity from 1kb to 15kb (composition-based tools like DeepVirFinder also show this; gene-based tools like VIBRANT/VirSorter2 degrade on short fragments)
- **Zero-shot RNA virus generalization**: No other tool on this list can detect eukaryotic RNA viruses without retraining
- **Precision limited by plasmid FPs**: 41/200 plasmids called viral — the same biological ambiguity that affects all composition-based tools

**GYP Mock Community caveat:** Published tools showed 40.6% mean F1 drop on real metagenomic data vs RefSeq benchmark (best: Kraken2 at F1=0.86, most tools F1<0.50). ViroSense has not been tested on the GYP mock community.

### RNA Virus Embedding Experiment (Evo2 40B via NIM)

Tested whether Evo2 embeddings discriminate eukaryotic RNA viruses (as cDNA) from cellular DNA, despite eukaryotic viruses being deliberately excluded from Evo2 training (biosecurity).

**Sequences tested** (11 total, ~5kb fragments):
- Eukaryotic RNA viruses (cDNA): SARS-CoV-2, HIV-1, Influenza A, HCV
- DNA bacteriophages (positive control): T4, Lambda, P22
- Bacteria (negative control): E. coli, B. subtilis
- Eukaryotic DNA viruses (also excluded): HSV-1, Vaccinia

**Cosine similarity results:**

| Comparison | Mean Cosine Sim |
|---|---|
| RNA viruses (within-group) | 0.818 |
| RNA viruses ↔ Euk DNA viruses | 0.722 |
| Phages ↔ Bacteria | 0.643 |
| RNA viruses ↔ Phages | 0.379 |
| RNA viruses ↔ Bacteria | **0.229** |

**PCA analysis** (PC1=59.5%, PC2=23.0%, total 83%):
- **PC1 is a viral↔cellular axis**: all viruses (RNA, DNA phage, eukaryotic DNA) on one side, bacteria on the other
- **PC2 separates by composition**: AT-rich viruses (T4, Vaccinia, SARS-CoV-2) vs GC-rich (HCV, HSV-1)
- Lambda/P22 are intermediate — compositionally ameliorated toward their bacterial hosts

**Key findings:**
1. Evo2 embeddings carry strong discriminative signal for eukaryotic RNA viruses despite training exclusion
2. T4 phage clusters with eukaryotic viruses (AT-rich, modified bases), not with temperate phages — Evo2 captures base modification signatures
3. Temperate phages (Lambda, P22) are the hardest to distinguish from bacteria (cos sim 0.87 with E. coli)
4. HCV↔HSV-1 similarity (0.919) transcends Baltimore class — driven by shared GC-rich composition
5. A classifier trained only on phages + bacteria should generalize to eukaryotic viruses — they occupy the same compositional region of embedding space

**Implications:**
- ViroSense detection may work on metatranscriptomic (RNA virus cDNA) data without retraining
- Multi-class classification (lytic phage, temperate phage, euk RNA virus, euk DNA virus, cellular) is feasible
- The hardest detection boundary is temperate phage vs host, not RNA virus vs cellular

### Reference Classifier on Eukaryotic Viruses (zero-shot generalization)

Tested the existing reference classifier (trained only on prokaryotic phages + bacteria) on the RNA virus experiment embeddings. **11/11 correct (100%):**

| Sequence | Category | P(viral) | Correct |
|---|---|---|---|
| SARS-CoV-2 ORF1a | Euk RNA virus (cDNA) | 1.0000 | YES |
| HIV-1 gag-pol | Euk RNA virus (cDNA) | 0.9557 | YES |
| Influenza A PB2 | Euk RNA virus (cDNA) | 0.9997 | YES |
| HCV NS3-NS5 | Euk RNA virus (cDNA) | 1.0000 | YES |
| T4 gene23 | DNA phage | 0.9999 | YES |
| Lambda CI-N | DNA phage | 0.9965 | YES |
| P22 tailspike | DNA phage | 0.9995 | YES |
| E. coli rpoB | Bacteria | 0.0004 | YES |
| B. subtilis sporulation | Bacteria | 0.0000 | YES |
| HSV-1 UL30 | Euk DNA virus | 0.9821 | YES |
| Vaccinia F13L | Euk DNA virus | 0.9621 | YES |

**The classifier generalizes to eukaryotic viruses (RNA and DNA) with zero retraining.** Metatranscriptomic data (RNA virus cDNA) can be fed directly to `virosense detect`.

### RNA Virus Zero-Shot Validation (200 RNA viruses + 200 cellular)

Tested reference classifier (trained only on phages + bacteria) on eukaryotic RNA viruses (cDNA from NCBI RefSeq). All sequences sent through NIM API for Evo2 40B embedding extraction.

| Metric | Value |
|--------|-------|
| Overall accuracy | 93.8% |
| RNA virus sensitivity | 89.0% |
| Cellular specificity | 98.5% |
| F1 score | 0.934 |
| AUC | 0.994 |
| Precision (viral) | 98.3% |
| False negatives | 22/200 |
| False positives | 3/200 |

**Per-length-bin RNA virus sensitivity:**

| Length bin | Sensitivity | Mean score | N |
|-----------|-------------|------------|---|
| 500-1000 bp | 77.5% | 0.726 | 40 |
| 1000-3000 bp | 80.0% | 0.828 | 40 |
| 3000-5000 bp | 92.5% | 0.926 | 40 |
| 5000-10000 bp | 97.5% | 0.952 | 40 |
| 10000-16000 bp | 97.5% | 0.957 | 40 |

**Key findings:**
- Sensitivity scales strongly with sequence length (77.5% at <1kb to 97.5% at >5kb)
- AUC of 0.994 indicates near-perfect separability — most errors are threshold-dependent, not embedding failures
- False negatives are concentrated in short (<3kb) fragments: 17/22 FN are <3kb (mean 2,224 bp vs 5,592 bp for true positives)
- Only 3 false positives out of 200 cellular sequences (all 500bp fragments)
- Zero-shot generalization works: no RNA virus training data needed

**Error analysis — false negative categories:**
1. **Retroviruses with cellular oncogenes** (2 sequences, scores 0.00-0.02): Harvey murine sarcoma virus (v-Has) and avian myelocytomatosis virus (v-Myc) carry captured host genes — compositionally chimeric. The classifier correctly detects that these sequences contain host-derived DNA.
2. **Individual segments of segmented RNA viruses** (~15 sequences, scores 0.04-0.49): Single segments of reoviruses, bunyaviruses, orbiviruses, and emaraviruses at 500-1800 bp. Insufficient compositional signal in short individual segments.
3. **Borderline cases** (7 sequences, scores 0.40-0.50): Would be rescued by lowering threshold to 0.40 with zero additional false positives.

**Threshold analysis:** Lowering threshold from 0.50 to 0.40 improves sensitivity from 89.0% to 92.5% with no change in specificity (98.5%). The score distribution is bimodal — cellular sequences cluster below 0.10 (99.5%) while most RNA viruses score above 0.30 (92.5%).

### Gauge Your Phage Benchmark (500 phage + 500 chromosome + 200 plasmid)

Community-standard benchmark for phage detection tools. Pre-fragmented 1-15 kbp contigs, stratified sample of 1,200 sequences across 4 length bins.

**Binary classification (phage vs non-viral):**

| Metric | Value |
|--------|-------|
| Overall accuracy | 96.2% |
| Phage sensitivity | 99.4% (497/500) |
| Chromosome specificity | 99.6% (498/500) |
| Precision | 92.0% |
| F1 score | 0.956 |
| AUC | 0.997 |

**Per-length-bin phage sensitivity:**

| Length | Sensitivity | Mean score |
|--------|-------------|------------|
| 1-3kb | 99.2% | 0.994 |
| 3-5kb | 99.2% | 0.991 |
| 5-10kb | 100.0% | 0.999 |
| 10-15kb | 99.2% | 0.990 |

**Phage detection is near-perfect across all length bins.** Unlike RNA viruses, phages are within the training domain — the classifier was trained on phage fragments. Sensitivity is uniformly >99% from 1kb to 15kb.

**Plasmid false positive analysis:**
- 41/200 plasmids (20.5%) classified as viral — the primary source of false positives
- Plasmid mean viral score: 0.203 (most score low, but some score very high)
- Many high-scoring plasmids carry phage-derived genes (conjugation machinery, anti-restriction)
- Chromosome FP rate is only 0.4% — plasmids are the hard boundary, not chromosomes

**Only 3 phage false negatives:**
1. Haloarcula hispanica pleomorphic virus 4 (score 0.015) — archaeal pleomorphic virus, highly unusual morphology
2. Leptospira biflexa temperate phage LE1 (score 0.258) — host-ameliorated temperate phage
3. Mycobacterium phage Pippy fragment (score 0.459) — borderline, short 2.7kb fragment

**Threshold sweep (F1 optimization):**
- F1 peaks at threshold 0.9 (F1=0.970, sens=98.8%, spec=96.4%)
- Raising threshold from 0.5 to 0.8 eliminates 12 plasmid FPs while losing zero phages
- At threshold 0.5: best sensitivity (99.4%), at threshold 0.9: best F1 (0.970)

### Validation Datasets (downloaded, not in git)

| Dataset | Location | Size | Sequences |
|---------|----------|------|-----------|
| RNA Virus Database | `data/reference/rna_viruses/RNA_virus_database.fasta` | 1.1 GB | 385,732 (6,621 NCBI + 378K RVMT + 858 terrestrial) |
| Gauge Your Phage - Phage | `data/benchmarks/gauge_your_phage/phage_fragment_set.fasta` | 54 MB | 6,664 phage fragments (1-15 kbp) |
| Gauge Your Phage - Chromosome | `data/benchmarks/gauge_your_phage/host_chr_pvog_fragments.fasta` | 844 MB | 104,003 chromosome fragments |
| Gauge Your Phage - Plasmid | `data/benchmarks/gauge_your_phage/host_plasmid_pvog_fragments.fasta` | 22 MB | 2,754 plasmid fragments |

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/prepare_reference_data.py` | Download + fragment RefSeq genomes for training |
| `scripts/validate_model.py` | Cross-validation, hard examples, misclassification analysis |
| `scripts/filter_prophage_noise.py` | Post-hoc prophage contamination detection |
| `scripts/compare_genomad.py` | Head-to-head ViroSense vs geNomad comparison |
| `scripts/analyze_clusters.py` | PCA + HDBSCAN clustering with taxonomy cross-reference |
| `scripts/download_evo2_weights.py` | Download Evo2 7B weights from HuggingFace for MLX backend |
| `scripts/validate_rna_viruses.py` | Zero-shot RNA virus validation (classifier generalization) |
| `scripts/benchmark_gauge_your_phage.py` | Gauge Your Phage community benchmark (phage/chr/plasmid) |

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

### Benchmark Numbers
**Measured (self-hosted NIM 7B on L40S 48GB, HTCF):**

| Sequence Length | Wall Time | Server Time |
|----------------|-----------|-------------|
| 100bp | 0.4s | 0.4s |
| 500bp | 1.7s | 1.6s |
| 1000bp | 3.8s | 3.7s |
| 2000bp | 6.7s | 6.4s |
| 5000bp | 16.6s | 15.8s |
| 8000bp | 27.6s | 25.9s |
| 10000bp | 33.2s | 31.6s |

**NVIDIA official throughput (generation, not embedding):**

| Model | GPU | Throughput |
|-------|-----|------------|
| 40B | 2x H100 80GB | 26 nt/sec |
| 40B | 1x H200 141GB | 33 nt/sec |
| 7B | 1x H100 80GB | 45 nt/sec |
| 7B | 1x H200 141GB | 52 nt/sec |

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

### Existing planned work
- **RNA virus / metatranscriptomics support**: Validated — classifier achieves 89% sensitivity / 99.4% AUC on 200 RNA virus RefSeq sequences with zero retraining. Short fragments (<3kb) are the main challenge (77-80% sensitivity). Consider: (1) including RNA virus cDNA in training data to boost short-fragment sensitivity, (2) adding metatranscriptomics to supported input types in documentation.
- **Multi-class viral classification**: Embeddings naturally separate lytic phage, temperate phage, eukaryotic RNA virus, eukaryotic DNA virus, and cellular. Train a multi-class classifier for virus type identification.
- **MLX backend optimization**: Current 62s/5kb sequence. Replace Python conv loops with vectorized MLX ops or mx.conv1d for ~3-5x speedup.
- **MLX numerical validation**: Compare MLX (7B) vs NIM (40B) embeddings on same sequences. Retrain reference classifier on 7B embeddings.
- **UHGV** (Unified Human Gut Virome Catalog): 873K virus genomes / 168K vOTUs. Potential benchmarking resource.
- **Prophage benchmark**: Validate against PHASTER/PhiSpy on curated prophage datasets
- **Methods paper**: Novel DNA foundation model approach to viral detection. RNA virus generalization is a strong differentiating result. Annotate module adds end-to-end story.
- **RNA foundation models**: AIDO.RNA (1.6B, 2048-D), RiNALMo (650M, 1280-D) as alternative backends for RNA-specific tasks. LucaOne (1.8B, unified DNA/RNA/protein) as potential single-model solution.

## Biosecurity Note

Evo2 deliberately excludes eukaryotic viral sequences from training for biosecurity. ViroSense uses Evo2 only for discriminative tasks (classification, clustering) — not sequence generation. This is a safe, defensive application focused on detection and characterization of existing sequences. The RNA virus embedding experiment confirms that while Evo2 has high perplexity (poor generation quality) for eukaryotic viruses, the intermediate hidden state representations still capture compositional signatures useful for detection — a discriminative use that does not compromise the biosecurity exclusion.
