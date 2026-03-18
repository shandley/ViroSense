# Unified Research Roadmap: From ViroSense to Planetary-Scale Viral Discovery

Last updated: 2026-03-18

## The Insight

The biosurveillance research directions (perplexity forensics, knowledge distillation, anomaly detection) are not just interesting science — they are **engineering optimizations** that make planetary-scale analysis feasible. Each research paper produces a tool that reduces the cost and time of the Logan project by an order of magnitude.

## The Pipeline

```
Logan (385 TB, 100B contigs)
    │
    ▼ Paper 1: NCD pre-filter
Compression filter ($50, hours)
    │  90% eliminated — obviously chromosomal
    ▼
~10B candidates
    │
    ▼ Paper 2: K-mer periodicity proxy
K-mer periodicity screening ($100, hours)
    │  Flag RNA-origin, viral-composition candidates
    ▼
~2B viral/interesting candidates
    │
    ▼ Paper 3: Knowledge distillation
Distilled CNN classifier ($500, 1 day)
    │  Classify into virus/plasmid/chromosome/novel
    ▼
~100M flagged sequences
    │
    ▼ Paper 4: Evo2 characterization
Full Evo2 DNA passports ($3K, 1-2 weeks)
    │  Identity, origin, structure, novelty
    ▼
~5M novel/interesting sequences
    │
    ▼ Paper 5: Anomaly detection + clustering
Novel lineage discovery ($50, hours)
    │  HDBSCAN, FAISS nearest-neighbor, anomaly scoring
    ▼
Publication: "Planetary-scale viral census"
```

Each arrow is a 10× reduction in volume. Five papers, each producing a tool, each making the next step feasible.

## Paper 1: NCD Pre-Filter

**"Compression Distance as a Pre-Filter for Foundation Model Inference on Metagenomic Data"**

### Scientific question
Can Normalized Compression Distance (NCD) predict which sequences will benefit from expensive foundation model analysis?

### Method
- Compute NCD between each contig and a small panel of viral reference sequences
- NCD uses gzip compression — runs at disk I/O speed, no model needed
- Sequences with high NCD to all viral references are obviously non-viral → skip

### Expected result
- 90% of contigs eliminated before any ML inference
- <1% false negative rate (viral sequences incorrectly filtered)
- Microseconds per sequence — effectively free at any scale

### Connection to Logan
Reduces the input to the distilled model from 100B to ~10B sequences. Saves ~$900 in Phase 2 screening and proportionally reduces downstream costs.

### Effort
- Implementation: 1-2 weeks
- Validation on GYP benchmark: 1 week
- Paper: 1 month
- **Can start immediately — no GPU needed**

## Paper 2: K-mer Periodicity Proxy

**"K-mer Frequency Signatures Approximate Foundation Model Codon Periodicity for RNA Virus Detection"**

### Scientific question
Can the codon periodicity signal we discovered in Evo2 per-position embeddings (lag-3 = 0.875 for RNA viruses) be approximated from hexanucleotide frequencies alone?

### Method
- Compute hexanucleotide (6-mer) frequencies for each sequence
- 6-mers capture 2 complete codons — should encode periodicity
- Train a small MLP: 4096 hexanuc frequencies → periodicity score
- Compare against Evo2 ground truth (our 203-sequence dark matter dataset)

### Expected result
- RNA virus identification at ~85-90% accuracy from k-mers alone (vs 97.5% with Evo2)
- Microseconds per sequence
- Good enough for screening, with Evo2 reserved for confirmation

### Connection to Logan
Flags RNA-origin candidates in the NCD-filtered pool without any GPU inference. Focuses the distilled model on the most promising sequences.

### Effort
- Implementation: 1 week (k-mer computation is trivial)
- Validation: 1 week (use existing 203-sequence dataset)
- Paper: Could be a section in the distillation paper or standalone
- **Can start immediately — uses existing data**

## Paper 3: Knowledge Distillation

**"Distilling DNA Foundation Models for Petabyte-Scale Pathogen Detection"**

### Scientific question
Can a lightweight student model approximate Evo2's embedding space well enough for viral detection and contig typing?

### Method
- Teacher: Evo2 7B embeddings for 5M reference genome fragments
- Student: 1D CNN on hexanucleotide frequencies → 256-D embedding
- Loss: contrastive (align student to teacher in embedding space)
- Evaluation: compare student vs teacher on GYP benchmark, RNA dark matter, prophage

### Expected result
- ~85-90% of Evo2 classification accuracy
- 1000× faster (0.001s vs 3.3s per sequence)
- Runs on CPU — no GPU needed for inference

### Connection to Logan
This is the **critical enabler**. Without distillation, Logan screening requires nation-state GPU budgets. With it, a single cloud job screens all metatranscriptomes in a day.

### Effort
- Phase 0: Generate 5M teacher embeddings (8× L40S, 24 days, ~$9K)
- Phase 1: Train student (4× A100, 1-2 weeks, ~$3K)
- Phase 2: Validate on benchmarks (CPU, 1 week)
- Paper: 2-3 months total
- **Requires WashU RIS onboarding first**

## Paper 4: Evo2 Characterization at Scale

**"DNA Foundation Model Embeddings as a Universal Sequence Characterization Framework"**

This is the **ViroSense methods paper** — the one we've been building toward.

### Content
- Viral detection (99.7% phage, 93.0% RNA virus vs geNomad)
- Contig typing (94.5% 3-class accuracy)
- Unsupervised clustering (ARI=0.903)
- Codon periodicity discovery (the genetic code in embeddings)
- RNA dark matter detection (97.5% accuracy)
- Gene boundary detection (91.7% coding accuracy)
- Prophage detection (37bp boundary accuracy)
- DNA passport characterization framework
- Per-position analysis reveals gene structure without gene calling

### Connection to Logan
Establishes ViroSense as the characterization tool applied to Logan candidates in Paper 5. The methods and validation are already done.

### Effort
- Generate figures: 1-2 weeks
- Write manuscript: 2-4 weeks
- **Can start now — all data exists**

## Paper 5: Planetary-Scale Viral Census

**"Surveying All Public Metatranscriptomes for Novel RNA Elements Using Foundation Model Embeddings"**

### Scientific question
What novel RNA viruses and RNA elements exist across all 129,974 public metatranscriptomes?

### Method
- Screen Logan metatranscriptomes with NCD → k-mer periodicity → distilled model pipeline
- Full Evo2 characterization on flagged candidates
- HDBSCAN clustering → identify novel lineages
- DNA passports for every novel element
- Cross-environment and temporal analysis

### Expected discoveries
- Novel RNA virus families with no database representation
- Obelisk-related elements across diverse environments
- Geographic and temporal patterns of RNA viral diversity
- The first complete census of RNA dark matter

### Connection to everything
This paper uses every tool from Papers 1-4. It's the capstone.

### Effort
- Requires Papers 1-3 completed first
- Compute: ~$3-5K with optimizations
- Analysis: 2-3 months
- Paper: 2-3 months after analysis

## Unified Timeline

```
Month 1-2:  Paper 4 (ViroSense methods — write + submit)
            Paper 1 (NCD filter — implement + validate)
            Paper 2 (K-mer periodicity — implement + validate)
            WashU RIS onboarding

Month 2-4:  Paper 3 Phase 0 (generate 5M teacher embeddings on RIS)
            Paper 1+2 submission

Month 4-5:  Paper 3 Phase 1 (train + validate distilled model)
            Paper 3 submission

Month 5-7:  Paper 5 (Logan metatranscriptome screening)
            Analysis + novel lineage discovery

Month 7-9:  Paper 5 writing + submission
```

Five papers in 9 months. Each one builds on the previous. The engineering work and the science are the same.

## Cost Summary (with optimizations)

| Paper | Compute cost | GPU time | Key output |
|-------|-------------|----------|-----------|
| 1. NCD filter | ~$0 | 0 | 10× volume reduction |
| 2. K-mer periodicity | ~$0 | 0 | RNA-origin screening |
| 3. Distillation | ~$12K | 5 weeks | 1000× speedup tool |
| 4. ViroSense methods | ~$0 (data exists) | 0 | Methods paper |
| 5. Logan census | ~$3-5K | 1-2 weeks | Nature paper |
| **Total** | **~$15-17K** | **7 weeks GPU** | **5 papers** |

Compare: $15-17K and 9 months for 5 papers including a planetary-scale viral census. The cost of 1-2 sequencing runs for the most comprehensive viral discovery project ever attempted.

## What We Have Already

| Asset | Status | Papers it feeds |
|-------|--------|----------------|
| ViroSense codebase (9 commands, 283 tests) | ✅ Complete | All |
| GYP benchmark (13,417 seqs, geNomad comparison) | ✅ Complete | Paper 4 |
| Codon periodicity discovery (40+ sequences) | ✅ Complete | Papers 2, 4 |
| RNA dark matter detection (203 seqs, 97.5%) | ✅ Complete | Papers 2, 4, 5 |
| Prophage pilot (E. coli K12, 37bp accuracy) | ✅ Complete | Paper 4 |
| Forensics pilot (40 gene pairs, 75%) | ✅ Complete | Paper 4 (supplementary) |
| Characterize framework | ✅ Complete | Papers 4, 5 |
| 40B embedding cache (18,631 sequences) | ✅ Complete | Paper 3 (seed data) |
| HTCF NIM deployment | ✅ Complete | Paper 3 |
| Philympics prophage data (5 genomes) | ✅ Running | Paper 4 |
