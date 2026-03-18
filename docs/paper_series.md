# ViroSense Paper Series

Last updated: 2026-03-18

## Paper 1: Methods Paper (NOW)

**"Frozen DNA foundation model embeddings reveal the genetic code and enable universal sequence characterization"**

Target: Nature Methods

### What's included
- Codon periodicity discovery (genetic code in embeddings)
- RNA dark matter detection (97.5%, database-free)
- Universal characterization framework (DNA passports)
- K-mer baseline comparison (93% detection, honest framing)
- Viral detection benchmark (geNomad head-to-head)
- Multi-task: contig typing, clustering, phylogenomics
- Prophage amelioration gradient (as characterization example)
- Cryptic prophage detection (CP4-6, Qin not in curation)
- e14 invisibility (honest limitation)
- Two-tier --fast pipeline
- Forensics pilot (supplementary)

### Status: All data complete. Figures + writing remain.

---

## Paper 2: RNA Dark Matter at Scale

**"Database-free identification of novel RNA elements across all public metatranscriptomes"**

Target: Nature / Nature Biotechnology

### What it needs from Paper 1
- Validated periodicity features (cos3, lag3)
- Characterize framework for profiling novel elements
- K-mer pre-filter for initial screening

### What's new
- Knowledge distillation (train student model for speed)
- Application to Logan metatranscriptomes (129K samples)
- Novel RNA virus lineage discovery
- Obelisk-related element census
- Geographic/temporal RNA viral diversity patterns

### Depends on
- Paper 1 published (establishes the methods)
- Knowledge distillation completed (~3 months)
- WashU RIS onboarding (GPU access)

### Timeline: ~6-9 months after Paper 1

---

## Paper 3: Prophage Evolution from Embeddings

**"Compositional age of integrated viral elements revealed by DNA foundation model embeddings"**

Target: Genome Research / PNAS

### What it needs from Paper 1
- Per-position norm analysis infrastructure
- Viral score as amelioration proxy
- Codon periodicity for host-similarity assessment

### What's new
- Full amelioration index (max score, score variance, boundary sharpness, periodicity host-similarity)
- Multi-genome survey (E. coli, B. subtilis, S. aureus, S. Typhi + more)
- Correlation with literature-derived prophage ages
- Cryptic prophage census across well-studied bacterial genomes
- Active vs degraded vs domesticated classification
- Evolutionary dynamics: rate of compositional convergence

### Depends on
- Paper 1 published
- HTCF/RIS GPU access for full-genome scans
- Literature review of known prophage ages

### Timeline: ~4-6 months after Paper 1

---

## Paper 4: Perplexity Forensics

**"Detecting engineered DNA using foundation model embedding signatures"**

Target: Nature Biotechnology / Nucleic Acids Research

### What it needs from Paper 1
- Per-position lag-1 autocorrelation (codon optimization signal)
- norm_cv (compositional uniformity)
- Characterize framework

### What's new
- Validation on real engineered sequences (NCBI synthetic constructs, Addgene)
- Chimera junction detection (sliding-window lag-1)
- Systematic characterization of engineering signatures
- Comparison against existing forensic tools
- DARPA Bio-Attribution alignment

### Depends on
- Paper 1 published
- Addgene API access or NCBI synthetic construct download
- Possibly MLX backend for true perplexity (logits, not just embeddings)

### Timeline: ~4-6 months after Paper 1

---

## Paper 5: Planetary-Scale Viral Census

**"A foundation model survey of viral diversity across the Sequence Read Archive"**

Target: Nature

### What it needs from Papers 1-2
- Knowledge distillation (Paper 2)
- K-mer pre-filter (Paper 1)
- Characterize framework (Paper 1)
- RNA dark matter periodicity features (Paper 1)

### What's new
- Application to all of Logan (385 TB, 27.3M samples)
- Complete viral census of all public sequencing data
- Novel viral lineage discovery at planetary scale
- Cross-environment viral diversity analysis
- Temporal surveillance (emergence patterns across years)

### Depends on
- Papers 1 and 2 published
- AWS integration (Logan on S3)
- Significant compute allocation (~$15-20K)

### Timeline: ~12-18 months after Paper 1

---

## Paper 6: Alignment-Free Phylogenomics (optional)

**"Embedding-space phylogenetics from DNA foundation models"**

Target: Systematic Biology / MBE

### What it needs from Paper 1
- Validated embedding distance ↔ taxonomic distance correlation (r=0.504)

### What's new
- Full validation against known viral phylogenies (ICTV)
- Comparison with alignment-based trees (VICTOR, ViPTree)
- Cross-kingdom distance comparisons
- Resolution limits (at what divergence does it break?)

### Timeline: could be written in parallel with Paper 1 if desired

---

## Dependencies

```
Paper 1 (Methods) ─────────────────────────────────────┐
    │                                                    │
    ├── Paper 2 (RNA Dark Matter + Distillation)         │
    │       │                                            │
    │       └── Paper 5 (Planetary Census)               │
    │                                                    │
    ├── Paper 3 (Prophage Evolution)                     │
    │                                                    │
    ├── Paper 4 (Forensics)                              │
    │                                                    │
    └── Paper 6 (Phylogenomics) ── can parallel Paper 1  │
```

Paper 1 is the foundation. Everything else builds on it.

---

## What Goes Where — Decision Matrix

| Finding | Paper 1 (Methods) | Paper 2+ (Follow-up) |
|---------|-------------------|---------------------|
| Codon periodicity | **Main result** | — |
| RNA dark matter (203 seqs) | **Main result** | Expanded in Paper 2 |
| DNA passport framework | **Main result** | Used in all papers |
| K-mer baseline (93%) | **Main result** | Pre-filter in Paper 2 |
| Viral detection (geNomad) | Supporting result | — |
| 3-class typing | Supporting result | — |
| HDBSCAN clustering | Supporting result | — |
| Phylogenomic signal (r=0.504) | Supporting result | Paper 6 if expanded |
| Prophage amelioration (3 states) | **Fig 3 panel** | Paper 3 if expanded |
| CP4-6/Qin novel detection | 1-2 sentences | Paper 3 |
| e14 invisibility | Honest limitation | Paper 3 |
| Two-tier --fast pipeline | Methods section | Paper 2 |
| Forensics pilot (75%) | **Supplementary** | Paper 4 |
| Logan strategy | Discussion/future | Paper 5 |
| Knowledge distillation plan | Discussion/future | Paper 2 |
