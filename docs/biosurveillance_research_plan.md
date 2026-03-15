# Foundation Model Biosurveillance Research Plan

Last updated: 2026-03-15

## Vision

Develop foundation model-based tools for pathogen detection and engineered sequence attribution, extending ViroSense's Evo2 embedding infrastructure beyond viral detection into pan-pathogen biosurveillance.

## Motivation

- DARPA Bio-Attribution Challenge highlights need for petabyte-scale pathogen detection + engineering attribution
- DNA foundation models (Evo2) capture deep sequence composition that k-mer databases cannot
- ViroSense benchmark demonstrates 95.8% phage sensitivity (7B) / 99.7% (40B) — proof that Evo2 embeddings work for detection
- L2-normalization fix (2026-03-15) eliminates length-dependent RNA virus detection failure: 34% → 99% recall at 10-16kb
- No existing tools use foundation model perplexity for engineered sequence forensics
- WashU RIS infrastructure (H100s, Docker-native, petabyte storage) makes large-scale work feasible

## Infrastructure

### WashU RIS (Primary — needs onboarding)
- **GPUs**: H100 (SXM), A100 80GB (SXM), 100+ datacenter GPUs total
- **Storage**: 16 PB Ceph + 2 PB BeeGFS NVMe scratch + 3 PB ZFS
- **Containers**: Docker-native (no Apptainer workarounds)
- **Interconnect**: HDR InfiniBand
- **Key advantage**: Self-hosted NIM Evo2 40B on H100s — no rate limits, no cloud dependency

### WashU HTCF (Current)
- **GPUs**: 1x L40S 48GB (n099) — FP8 capable, runs NIM Evo2 7B
- **Limitations**: 8hr SLURM wall time, Apptainer only, small scratch quota
- **Role**: Continue 7B work, transition large-scale jobs to RIS

### NVIDIA Resources
- Cloud NIM API (Evo2 40B, rate-limited to 40 RPM)
- Academic Grant Program — apply for DGX Cloud credits
- BioNeMo collaboration potential (showcase Evo2 biosurveillance application)

## Research Directions

### 1. Perplexity Forensics for Engineered Sequence Detection

**Paper**: "Detecting Engineered DNA Using Foundation Model Perplexity Profiles"

**Core idea**: Evo2 assigns per-nucleotide probabilities. Natural genomes have smooth, organism-characteristic perplexity profiles. Engineered sequences show discontinuities:
- Codon-optimized genes → unusually low perplexity (too "perfect" for host)
- Synthetic promoters → high perplexity spikes (alien regulatory elements)
- Chimeric constructs → sharp transitions at assembly junctions
- BioBrick scars, Gibson junctions → stereotyped perplexity signatures

**Existing evidence**: ViroSense RNA virus experiment showed Evo2 captures compositional signatures even for sequences outside its training distribution. The L2-normalization analysis (March 2026) proved that Evo2 embedding *direction* encodes stable viral signatures independent of sequence length — the information is there, the classifier just needs to use it correctly.

**Data**:
- Positive (engineered): Addgene (~100K deposited plasmids), iGEM Registry, JBEI-ICE
- Negative (natural): NCBI RefSeq complete genomes
- Validation: Known engineered organisms with documented modifications

**Compute requirements**:
- Evo2 forward pass with per-token log-likelihoods (not just embeddings)
- Options: Self-hosted NIM on RIS H100s, MLX backend (local, 7B), or direct Evo2 package on A100s
- Sliding window perplexity computation along genomes

**Novelty**: High. Nobody has applied DNA foundation model perplexity to forensic detection of engineering.

**Priority**: Phase 1 — lowest hanging fruit, most novel, uses existing infrastructure.

### 2. Distilled Foundation Model for Read-Level Metagenomics

**Paper**: "Distilling DNA Foundation Models for Petabyte-Scale Pathogen Detection"

**Core idea**: Evo2 is too slow for read-level classification (~3-27s per sequence). Train a lightweight student model (small CNN or MLP on k-mer frequencies) that approximates Evo2's embedding space at 1000x speed.

**Approach**:
- Teacher: Evo2 40B embeddings (8192-D) for reference genome fragments
- Student: 1D CNN or MLP on tetranucleotide/hexanucleotide frequencies → 256-D embedding
- Loss: MSE or contrastive (align student embeddings to teacher)
- Classification: Nearest centroid in distilled embedding space → NCBI TaxID

**Training data**:
- NCBI pathogen reference genomes fragmented to read-length (150bp, 250bp, 1kb, 5kb)
- ~50K genomes × ~100 fragments = 5M embedding pairs
- Current benchmark caches (40B: 16,867; 7B: 13,417) as initial seed

**Compute requirements**:
- Embedding generation: Self-hosted 40B on RIS H100s (weeks for 5M sequences at ~3-5s/seq unlimited)
- Student training: A100 nodes, standard PyTorch distributed training
- Benchmarking: Compare against Kraken2, CLARK, MetaPhlAn4 on CAMI, mockrobiota

**Novelty**: High. Knowledge distillation from DNA foundation models is unexplored for metagenomics.

**Priority**: Phase 2 — requires embedding corpus from Phase 0/1.

### 3. Embedding-Space Anomaly Detection for Biosurveillance

**Paper**: "Anomaly Detection in DNA Foundation Model Embedding Space for Biosurveillance"

**Core idea**: Embed a large reference panel of known pathogens + environmental background. New metagenomic samples scored by distance to known clusters. Anomalies = novel pathogens, engineering, or unusual co-occurrence.

**Approach**:
- Reference panel: All NCBI pathogen genomes + representative environmental genomes
- Embedding: Evo2 40B (or distilled student from Direction 2)
- Anomaly methods: Isolation forest, local outlier factor, autoencoder on embedding space
- Metadata integration: Geospatial + temporal signals (organism range maps, seasonal patterns)

**Validation data**:
- Known outbreak datasets (NCBI BioProject)
- Synthetic spike-ins (engineered sequences mixed into environmental backgrounds)
- UHGV (873K gut virus genomes) as environmental reference

**Novelty**: Moderate-high. Embedding-based anomaly detection exists but not with DNA foundation models or at biosurveillance scale.

**Priority**: Phase 3 — builds on embedding corpus and potentially distilled model.

### 4. Complementary Approaches (Lower Priority)

**Codon usage forensics**: Compare observed vs expected codon frequencies per organism. Computationally trivial, pairs with perplexity analysis. Could be a section in the perplexity paper rather than standalone.

**Compression-based taxonomy**: Normalized Compression Distance as ultra-fast sequence similarity. Scales to petabytes trivially. Good for a speed-focused comparison in the distillation paper.

**Graph topology anomaly detection**: Assembly graph structure analysis for detecting chimeric/engineered constructs. More speculative, potentially a later direction.

## Implementation Roadmap

### Phase 0: Infrastructure Setup (weeks 1-2)
- [ ] Onboard to WashU RIS
- [ ] Request H100 GPU allocation
- [ ] Test NIM Evo2 Docker container on RIS H100s (should work natively)
- [ ] Migrate embedding caches and reference data to RIS storage
- [ ] Validate self-hosted 40B NIM (multi-GPU, no rate limits)

### Phase 1: Perplexity Forensics Pilot (months 1-2)
- [ ] Implement perplexity computation (per-token log-likelihoods from Evo2)
  - Option A: Add to MLX backend (forward pass already implemented)
  - Option B: Self-hosted NIM on RIS H100s (if logits accessible)
  - Option C: Direct Evo2 package on A100s (standard PyTorch inference)
- [ ] Download Addgene plasmid dataset + matched natural genomes
- [ ] Compute perplexity profiles: sliding windows across engineered vs natural
- [ ] Identify signature patterns (junction detection, codon optimization detection)
- [ ] Write preprint

### Phase 2: Large-Scale Embedding Corpus (months 2-4)
- [ ] Fragment all NCBI pathogen reference genomes (multiple length scales)
- [ ] Generate 40B embeddings via self-hosted NIM on RIS (target: 5M sequences)
- [ ] Build reference embedding database with taxonomy labels
- [ ] Train + evaluate distilled student model on A100 nodes
- [ ] Benchmark against Kraken2/CLARK/MetaPhlAn4 on standard datasets (CAMI)
- [ ] Write distillation paper

### Phase 3: Biosurveillance Framework (months 4-8)
- [ ] Build anomaly detection pipeline on embedding space
- [ ] Integrate geospatial/temporal metadata scoring
- [ ] Validate on known outbreak datasets
- [ ] Combine perplexity + distilled classifier + anomaly detection into unified system
- [ ] System paper

### Phase 4: External Engagement (ongoing)
- [ ] Apply for NVIDIA Academic Grant / BioNeMo collaboration
- [ ] Monitor DARPA BTO solicitations for relevant BAAs
- [ ] Consider next Bio-Attribution Challenge cycle with mature tools
- [ ] Present at RECOMB, ISMB, or ASM

## Key Datasets

| Dataset | Size | Purpose |
|---------|------|---------|
| Addgene plasmids | ~100K sequences | Engineered sequence ground truth |
| iGEM Registry | ~20K parts | Standardized synthetic biology parts |
| NCBI RefSeq complete genomes | ~50K genomes | Natural reference panel |
| UHGV | 873K viral genomes | Environmental viral reference |
| BFVD | 351K viral structures | Structural annotation (existing ViroSense) |
| CAMI benchmarks | Standard | Metagenomics classifier evaluation |
| Gauge Your Phage | 113K fragments | Phage/chr/plasmid benchmark (in progress) |

## NVIDIA Pitch

> "We're developing the first foundation model-based biosurveillance toolkit using Evo2 on NVIDIA NIM. Our published viral detection benchmark shows 99.7% phage sensitivity on the community Gauge Your Phage benchmark. We're extending to: (1) perplexity-based detection of engineered sequences — a novel forensic application, (2) knowledge distillation for petabyte-scale pathogen screening, and (3) embedding-space anomaly detection for biosurveillance. We request [X] H100-hours for large-scale embedding generation and NIM Enterprise access for rate-unlimited 40B inference."

## Connection to ViroSense

ViroSense remains the viral detection tool. This research plan extends the underlying technology (Evo2 embeddings) into new applications:

- **Perplexity forensics**: New capability, could become a ViroSense subcommand (`virosense forensics`)
- **Distilled model**: Could replace NIM backend for speed-critical workflows
- **Anomaly detection**: Could integrate with ViroSense detect for flagging unusual sequences
- **Pan-pathogen taxonomy**: Separate tool, but shares embedding infrastructure

The ViroSense codebase (backend abstraction, caching, CLI framework) provides a strong foundation to build on.
