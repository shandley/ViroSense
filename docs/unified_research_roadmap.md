# Unified Research Roadmap: From ViroSense to Planetary-Scale Viral Discovery

Last updated: 2026-03-18

## The Insight

The biosurveillance research directions (perplexity forensics, knowledge distillation, anomaly detection) are not just interesting science — they are **engineering optimizations** that make planetary-scale analysis feasible. Each research paper produces a tool that reduces the cost and time of the Logan project by an order of magnitude.

## The Pipeline

```
Logan (385 TB, 100B contigs)
    │
    ▼ Paper 1: K-mer pre-filter (VALIDATED: 91.4% accuracy, 458 seq/s)
Trinucleotide classifier ($500, ~2.5 days on 1000 CPUs)
    │  ~90% eliminated as non-viral
    ▼
~10B viral/ambiguous candidates
    │
    ▼ Paper 2: Knowledge distillation
Distilled CNN classifier ($500, ~1 day)
    │  Refine from 91% to ~95%, add contig typing
    ▼
~500M high-confidence viral + novel
    │
    ▼ Paper 3: Evo2 characterization (ViroSense methods)
Full Evo2 DNA passports ($3K, 1-2 weeks)
    │  Identity, origin, structure, novelty
    ▼
~5-50M characterized sequences
    │
    ▼ Paper 4: Anomaly detection + clustering
Novel lineage discovery ($50, hours)
    │  HDBSCAN, FAISS nearest-neighbor, anomaly scoring
    ▼
Publication: "Planetary-scale viral census"
```

Each arrow is a 10-20× reduction in volume. Four papers (NCD dropped after negative feasibility test), each producing a tool, each making the next step feasible.

## ~~Paper 1: NCD Pre-Filter~~ — TESTED, DOES NOT WORK

**Feasibility test (2026-03-18)**: NCD cannot separate viral from non-viral sequences.

- AUC: **0.475** (worse than random)
- All categories have NCD ~0.95 to viral references (Cohen's d = 0.01)
- NCD measures compression redundancy, not biological category
- Works for within-species comparison, NOT cross-category screening

**Verdict**: Dropped from the pipeline. Replaced by k-mer classifier (Paper 2).

## Paper 1 (revised): K-mer Classifier as Pre-Filter

**"Trinucleotide Frequency Classifiers as Fast Pre-Filters for Foundation Model Metagenomic Analysis"**

### Feasibility test — VALIDATED (2026-03-18)

Tested on 5,000 GYP benchmark sequences with trinucleotide + dinucleotide features:

| Task | K-mer accuracy | K-mer AUC | Evo2 accuracy | Speed |
|------|---------------|-----------|---------------|-------|
| **Viral vs non-viral** | **91.4%** | **0.966** | 95.4% | **1,527× faster** |
| **RNA virus vs all** | **95.0%** | **0.985** | 97.5% | 1,527× |
| **5-class typing** | **74.2%** | — | 94.5% | 1,527× |

- 458 sequences/second on CPU (vs 0.3 seq/s for Evo2 7B)
- Top features: trinucleotide frequencies (CTA, AGA, ACT) and dinucleotide (GC, AC)
- 91.4% viral detection is sufficient for first-pass screening
- Captures codon-level composition that approximates Evo2's periodicity signal

### Scientific question
What fraction of Evo2's discriminative power comes from k-mer composition vs higher-order sequence features? Can simple k-mer classifiers replace foundation models for initial screening?

### Method
- Random forest on trinucleotide (64) + dinucleotide (16) + GC + frame entropy features
- Compare against Evo2 on same benchmark at multiple tasks
- Characterize the "accuracy gap" — what do the 4-8% of sequences that k-mers miss have in common?
- Test as a pre-filter: k-mer screens, only ambiguous cases sent to Evo2

### Connection to Logan
Replaces both the NCD filter AND the distilled model for initial screening. At 458 seq/s:
- 100B Logan contigs on 1000 CPUs: **~2.5 days (~$500)**
- 129K metatranscriptome samples: **~hours**
- Cost is essentially zero compared to GPU inference

### Effort
- Implementation: **DONE** (feasibility test is the implementation)
- Full validation on complete 13K benchmark: 1 day
- Paper: standalone or section in ViroSense methods paper
- **Immediately actionable**

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

## Unified Timeline (revised)

```
Month 1:    Paper 3 (ViroSense methods — write + submit, data exists)
            Paper 1 (K-mer classifier — full validation on 13K benchmark)
            WashU RIS onboarding

Month 2-3:  Paper 2 Phase 0 (generate 5M teacher embeddings on RIS)
            Paper 1 submission (or fold into Paper 3 as supplementary)

Month 3-4:  Paper 2 Phase 1 (train + validate distilled model)
            Paper 2 submission

Month 4-6:  Paper 4 (Logan metatranscriptome screening)
            K-mer pre-filter → distilled refinement → Evo2 characterization
            Novel lineage discovery

Month 6-8:  Paper 4 writing + submission
```

Four papers in 8 months. NCD dropped after negative feasibility test.
K-mer classifier validated and ready for immediate use.

## Cost Summary (updated after feasibility tests)

| Paper | Compute cost | GPU time | Key output | Status |
|-------|-------------|----------|-----------|--------|
| ~~NCD filter~~ | — | — | DOES NOT WORK | ❌ Dropped |
| 1. K-mer classifier | ~$500 | 0 | 91.4% pre-filter, 1527× speedup | ✅ Validated |
| 2. Distillation | ~$12K | 5 weeks | 95%+ accuracy, 1000× speedup | Planned |
| 3. ViroSense methods | ~$0 (data exists) | 0 | Methods paper | Ready to write |
| 4. Logan census | ~$3-5K | 1-2 weeks | Nature paper | After 1-2 |
| **Total** | **~$16-18K** | **7 weeks GPU** | **4 papers** |

The k-mer pre-filter alone reduces Logan screening to **~$500 on CPUs**. Combined with distillation for refinement, the total planetary-scale project is ~$16-18K — still under the cost of 2 sequencing runs.

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
| Philympics prophage data (5 genomes) | ✅ Running | Paper 3 |
| K-mer classifier feasibility (5K seqs) | ✅ Validated | Paper 1 |
| NCD pre-filter feasibility | ❌ Does not work | Dropped |
| Forensics pilot (codon optimization) | ✅ Complete | Paper 3 (supplementary) |
