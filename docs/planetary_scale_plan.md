# Planetary-Scale Viral Discovery with ViroSense × Logan

Last updated: 2026-03-17

## Vision

Apply ViroSense's foundation model embedding framework to [Logan](https://github.com/IndexThePlanet/Logan) — assembled contigs from the entire NCBI Sequence Read Archive (27.3 million samples, 50 petabases) — to conduct the largest viral discovery survey ever attempted.

## Logan Dataset

| Metric | Value |
|--------|-------|
| Raw data | 50 petabases |
| Samples assembled | 27.3 million |
| Compressed contigs | 385 TB |
| Metagenomes | 4,791,129 samples (43.4 TB) |
| Metatranscriptomes | 129,974 samples (2.6 TB) |
| Transcriptomes | 4,929,361 samples (80.9 TB) |
| Hosting | AWS S3 (public, Registry of Open Data) |

Logan already did the hard part — assembling all public sequencing data into contigs. The sequences are ready for analysis.

## The Scale Problem

| Approach | Sequences (est.) | Time (1 GPU) | Time (100 GPUs) |
|----------|-----------------|-------------|-----------------|
| Evo2 40B on all contigs | ~100 billion | 158,000 years | 1,580 years |
| Evo2 7B on all contigs | ~100 billion | 10,000 years | 100 years |
| **Distilled model on all contigs** | ~100 billion | 3 years | **1 day (1000 CPUs)** |
| Distilled + Evo2 on flagged 1% | 1 billion | 100 years | **1 year (100 GPUs)** |

Full Evo2 on all of Logan is impossible. Knowledge distillation is the critical enabler.

## Strategy Overview

```
Logan (385 TB contigs)
    │
    ▼
Strategy 4: Target metatranscriptomes first (2.6 TB, 129K samples)
    │
    ▼
Distilled model: k-mer CNN screening (~hours on cloud cluster)
    ├── 95% cellular → discard
    └── 5% interesting → Evo2 characterization
    │
    ▼
Evo2 7B on ~50M flagged contigs (weeks on GPU cluster)
    ├── Viral detection + typing
    ├── RNA dark matter identification (codon periodicity)
    ├── Clustering → novel lineage discovery
    └── Characterization → DNA passports
    │
    ▼
Result: Planetary-scale viral census
```

## Four Strategies

### Strategy 1: Distilled Model on All Logan

**Goal**: Screen every contig in Logan for viral signatures.

1. Train distilled model on 5M Evo2 embeddings (one-time, ~8 months on 8 H100s)
2. Run distilled model on all Logan contigs (hours on AWS Batch with 1000+ CPUs)
3. Flag 1-5% as viral/interesting
4. Full Evo2 characterization on flagged sequences (weeks-months on GPU cluster)

**Result**: Complete viral census of all public sequencing data.

**Cost estimate**: ~$5K-50K in cloud compute (distillation training + screening + Evo2 on flagged).

### Strategy 2: Logan as Reference Database

**Goal**: Use Logan to build the ultimate reference panel for `virosense characterize`.

- Pre-compute embeddings for all viral contigs in Logan (~5% = ~5 billion contigs)
- Any new sequence → compare against the planetary viral catalog
- "How novel is this virus compared to everything ever sequenced?"

With distilled model: pre-computation takes ~1 hour on 1000 CPUs.
Result: The definitive novelty scoring reference.

### Strategy 3: Sample-Level Viral Fingerprints

**Goal**: Create a "viral fingerprint" for each of 4.8M metagenomes.

- Embed only the top 100 contigs per sample (longest, most likely complete genomes)
- 4.8M × 100 = 480M embeddings
- With distilled: 8 minutes on 1000 CPUs
- With Evo2 7B: ~6 months on 100 GPUs

**Applications**:
- Global viral diversity survey by environment type
- Cross-environment comparisons (gut vs ocean vs soil viral communities)
- Anomaly detection (which samples have novel viruses?)
- Temporal surveillance (viral emergence across years)
- Geographic mapping of viral diversity

### Strategy 4: Targeted Metatranscriptome Mining (RECOMMENDED FIRST)

**Goal**: Survey all 129,974 metatranscriptomes for novel RNA elements.

This is the most immediately actionable because:
1. Metatranscriptomes are where RNA viruses live
2. Our RNA dark matter detector (97.5% accuracy) is uniquely suited
3. The subset is small enough: 2.6 TB compressed, ~1-2 billion contigs
4. Codon periodicity provides a signal no other tool can access

**Concrete plan**:
1. Download 129K metatranscriptome contigs from Logan S3 (2.6 TB)
2. For each sample, extract longest 10-50 contigs
3. Screen with distilled model (or k-mer pre-filter): ~hours
4. Flag RNA-origin candidates using periodicity features
5. Full Evo2 characterization on top 100K candidates: ~days on GPU cluster
6. HDBSCAN clustering → discover novel RNA virus lineages
7. Characterize → DNA passports for every novel element

**Expected discoveries**:
- Novel RNA virus families not in any database
- Obelisk-like elements across diverse environments
- RNA dark matter across all metatranscriptomes ever published
- Geographic/temporal patterns of RNA viral diversity

**Paper potential**: "Surveying 129,974 metatranscriptomes for novel RNA elements using foundation model embeddings" — this is a Nature paper.

## Critical Path

### Prerequisites (in order)

1. **Knowledge distillation** — Train the student model. This is the single bottleneck for planetary scale. Everything else scales with CPUs.

2. **AWS integration** — Logan lives on S3. Need Nextflow/Snakemake pipeline that processes S3 data in-place using AWS Batch.

3. **WashU RIS onboarding** — GPU access for Evo2 full characterization of flagged sequences.

4. **Pre-computed reference embeddings** — Embed all known viral genomes (RefSeq viral + UHGV + IMG/VR) for novelty scoring.

### Timeline

| Phase | Task | Duration | Compute |
|-------|------|----------|---------|
| 0 | Distillation training data (5M Evo2 embeddings) | 2-4 months | 8 H100s |
| 1 | Train distilled model | 1-2 weeks | 4 A100s |
| 2 | Validate distilled on GYP benchmark | 1 week | CPU |
| 3 | Screen Logan metatranscriptomes | 1 day | 1000 CPUs (AWS) |
| 4 | Evo2 characterization of candidates | 2-4 weeks | 8-16 H100s |
| 5 | Clustering + analysis + paper | 2-3 months | CPU |

Total: ~6-9 months from distillation start to paper submission.

## Why ViroSense Is Uniquely Positioned

No other tool can do this because:

1. **Foundation model embeddings** capture biological signals that k-mer methods miss — we've proven this with RNA virus zero-shot detection and codon periodicity
2. **The "embed once" architecture** means we compute once and analyze many ways
3. **Codon periodicity** provides a signal for RNA dark matter that requires no reference database
4. **The characterize framework** produces rich profiles, not just binary labels — enabling novel element discovery beyond what classification can find
5. **Knowledge distillation** bridges the gap between foundation model quality and planetary scale

## Connection to Other Plans

- **Biosurveillance plan**: Logan mining is the data source for anomaly detection (Direction 3)
- **Distillation plan**: Logan is the motivation for training the distilled model (Direction 2)
- **RNA dark matter**: Logan metatranscriptomes are the validation dataset
- **Obelisk case study**: Logan is where Obelisks were originally found — we could rediscover them and find their relatives

## Hardware Requirements

### By Phase

| Phase | Hardware | Duration | Cloud cost estimate |
|-------|----------|----------|-------------------|
| 0. Distillation training data | 8× L40S or H100 | 2-4 weeks | $9-18K |
| 1. Train distilled model | 4× A100 | 1-2 weeks | $2-4K |
| 2. Screen Logan metatranscriptomes | 1000 CPU cores (AWS Batch) | 1 day | $0.5-2K |
| 3. Evo2 characterization of candidates | 32-100× L40S | 1-3 weeks | $6-28K |
| 4. Clustering + analysis | 1× high-memory node | Days | Minimal |
| **Total** | | **2-3 months** | **$18-52K** |

### Phase 0: Distillation Training Data (one-time investment)

Generate 5M Evo2 embeddings as teacher signal for the student model.

| Setup | Time | Cost |
|-------|------|------|
| 8× L40S (7B) | **24 days** | ~$9K |
| 8× H100 (7B) | **12 days** | ~$18K |
| 24× H100 (40B, 8 instances) | 36 days | ~$140K (overkill) |

**Recommended**: 8× L40S for 24 days on WashU RIS. Standard HPC allocation.

### Phase 1: Train Distilled Model

Standard deep learning training: 1-4× A100 for 1-2 weeks. The student model is tiny (~10M parameters vs Evo2's 7B). Any lab with GPU access can do this.

### Phase 2: Screen Logan (the cheap phase)

The distilled model runs on **CPU only**. No GPUs needed.

| Setup | Time | Cost |
|-------|------|------|
| 100 CPU cores (1 HPC node) | 11 days | Free (HPC allocation) |
| 1000 CPU cores (AWS Batch) | **1 day** | $500-1000 |
| 10,000 CPU cores (AWS auto-scale) | **2 hours** | $500-2000 |

AWS Batch processes Logan data directly from S3 — no data transfer. A single $1000 job screens every public metatranscriptome ever generated.

### Phase 3: Evo2 Characterization (the GPU phase)

Full characterization on the 1-5% flagged by screening.

| Volume | Setup | Time | Cost |
|--------|-------|------|------|
| 10M contigs | 8× L40S | 48 days | $5.5K |
| 10M contigs | 32× L40S | **12 days** | $5.5K |
| 50M contigs | 100× L40S | **19 days** | $28K |

### Phase 4: Analysis

All cached embeddings. HDBSCAN on 50M sequences needs a high-memory node (512GB RAM, hours). Everything else runs on a laptop.

### What WashU Has

| Resource | Available | Sufficient |
|----------|-----------|-----------|
| WashU RIS H100s | Yes (needs onboarding) | Yes — 8-32 GPU jobs |
| WashU RIS storage | 16 PB Ceph + 2 PB NVMe | Yes — embedding caches fit easily |
| WashU RIS CPU cluster | Yes | Yes — 1000+ cores |
| HTCF L40S | 1× on n099 | No — too slow for planetary scale |

### What We'd Need to Acquire

| Resource | Purpose | How to get it |
|----------|---------|--------------|
| WashU RIS access | Multi-GPU H100, storage | Onboarding application (priority #1) |
| AWS credits | Process Logan on S3 in-place | AWS Open Science program or NVIDIA Academic Grant |
| NVIDIA Academic Grant | DGX Cloud access | Application to NVIDIA (accelerates Phase 0 and 3) |

### Cost Context

- A single NovaSeq sequencing run: $5-15K
- This entire project: $18-52K (2-5 sequencing runs)
- Result: survey of ALL public metatranscriptomes for novel viruses
- That is extraordinary value per dollar

## Sources

- [Logan: Planetary-Scale Genome Assembly Surveys Life's Diversity](https://www.biorxiv.org/content/10.1101/2024.07.30.605881v1)
- [Logan GitHub](https://github.com/IndexThePlanet/Logan)
- [Logan on AWS Registry of Open Data](https://registry.opendata.aws/pasteur-logan/)
