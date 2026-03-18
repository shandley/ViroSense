# Universal DNA Characterization Framework — Applications

Last updated: 2026-03-18

## Core Principle

ViroSense is a **universal DNA characterization framework** built on Evo2 foundation model embeddings. The framework follows a single pattern:

1. **Embed sequences** (one-time cost, cached)
2. **Apply task-specific analysis** on cached embeddings (instant)
3. **The embedding captures more than any single analysis uses**

The framework is the product. Each application is a configuration — a different lens on the same embedding.

## What the Embedding Encodes

From our experimental validation (March 2026):

| Signal | Evidence | Method |
|--------|----------|--------|
| Viral vs cellular composition | 95.4% accuracy, 0.994 AUC | Mean-pooled → MLP |
| Contig type (virus/plasmid/chromosome) | 94.5% cross-validated | Mean-pooled → 3-class MLP |
| Biological category (5 types) | ARI=0.903, RNA virus cluster 99% pure | HDBSCAN clustering |
| RNA vs DNA origin | 100% interpretation accuracy | Cosine similarity ratio |
| Coding vs non-coding regions | 91.7% accuracy, 1.72× norm ratio | Per-position norms |
| Triplet genetic code | Lag-3 autocorr 0.635, dominant FFT frequency | Per-position autocorrelation |
| Codon usage bias | RNA virus periodicity 0.822 vs phage 0.624 | Per-position cos3 |
| Gene boundaries | 73.2% recall, median 6bp precision | Norm derivative peaks |
| Engineering signatures | 75% codon optimization detection | Per-position lag-1 |
| Sequence novelty | Anomaly percentile against reference panel | Nearest-neighbor distance |

## Application Areas

### Tier 1: Validated (data exists, tested)

**Viral detection and characterization**
- Binary detection: 99.7% phage sensitivity, 93.0% RNA virus recall
- 3-class contig typing: virus/plasmid/chromosome
- RNA dark matter identification: 97.5% from periodicity alone
- DNA passport profiling: identity, origin, structure, novelty
- Prophage detection: 37bp boundary accuracy on E. coli K12
- Two-tier pipeline: k-mer screening (93%) → Evo2 characterization
- *Status: publication-ready*

### Tier 2: Directly Testable (same architecture, new training data)

**AMR element detection** *(highest clinical impact)*
- Same sliding-window architecture as prophage detection
- AMR cassettes are mobile genetic elements with distinctive composition
- Mobile element score from `characterize` already flags these
- Training data: CARD database (known resistance genes) + chromosomal context
- Pilot: scan known AMR-carrying plasmids, check if resistance islands produce viral-like scores
- *Effort: 2-4 weeks pilot, existing infrastructure*

**Horizontal gene transfer detection**
- Per-position norm transitions at HGT boundaries (same signal as prophage)
- Mobile element scoring for transferred segments
- Recent HGT = higher norm contrast (foreign composition); ancient HGT = ameliorated (host-like)
- Pilot: the Philympics prophage data includes HGT regions beyond prophages
- *Effort: 1-2 weeks analysis of existing prophage results*

**Alignment-free phylogenomics**
- Embedding cosine distance as evolutionary distance proxy
- Test with GYP phage benchmark: do embedding distances correlate with known viral taxonomy?
- Works on sequences too divergent for alignment
- Pilot: compute pairwise distances for 6,663 phage fragments, compare to ICTV taxonomy
- *Effort: 1 week, existing cached embeddings*

**Assembly quality assessment**
- Per-position compositional transitions detect chimeric misassemblies
- Anomaly scoring flags contigs with unexpected composition
- No reference database needed
- Pilot: create synthetic chimeric contigs, test if per-position analysis detects junctions
- *Effort: 1-2 weeks*

### Tier 3: Feasible (new data collection needed)

**Pathogen screening** *(high public health impact)*
- Two-tier pipeline (k-mer → Evo2) on clinical metagenomes
- Anomaly score as "pathogen alert" — unknown sequences not matching normal microbiome
- Reference panel: healthy microbiome embeddings from HMP
- Validation: known infection samples with documented pathogens
- *Effort: 1-2 months, needs clinical metagenome data*

**Source tracking / forensic identification**
- Sample-level characterize profiles as "microbiome fingerprints"
- Compare against reference environment database
- Applications: contamination detection, pollution tracking, forensic identification
- *Effort: 2-3 months, needs diverse environmental samples*

**Strain-level typing**
- Embedding distance between isolates as rapid typing method
- Faster than MLST, captures more information than serotyping
- Validation: known outbreak clusters with WGS data
- *Effort: 1-2 months, needs clinical isolate collections*

**Engineered sequence detection** *(biosurveillance)*
- Perplexity forensics: lag-1, norm_cv detect codon optimization (75% PoC accuracy)
- Chimera detection via per-position analysis (needs development)
- Reference panel: Addgene plasmids + NCBI synthetic constructs
- *Effort: 2-3 months, needs real engineered sequence validation*

### Tier 4: Speculative (unexplored)

- **Plant pathogen detection** from agricultural metagenomes
- **eDNA monitoring** for invasive species
- **Soil/water health scoring** from environmental microbiome fingerprints
- **Biocontainment verification** for synthetic biology
- **Genome evolution studies** tracking compositional changes over evolutionary time
- **Endosymbiont detection** from host genome scans

## Recommended Pilot Order

| Priority | Application | Why first | Effort | Data needed |
|----------|------------|-----------|--------|-------------|
| 1 | **Alignment-free phylogenomics** | Uses existing embeddings, testable in a week | 1 week | None (GYP cached) |
| 2 | **HGT detection** | Falls out of prophage benchmark analysis | 1-2 weeks | Prophage results (running) |
| 3 | **AMR element detection** | Highest clinical demand, same architecture | 2-4 weeks | CARD database |
| 4 | **Assembly QC** | Unique capability, no competitor | 1-2 weeks | Synthetic chimeras |
| 5 | **Pathogen screening** | Public health impact | 1-2 months | Clinical metagenomes |

## The "ViroSense" Name Question

The tool started as viral detection ("Viro" + "Sense") but has become a general DNA characterization framework. The viral detection is one application — and not even the strongest one compared to k-mer baselines (93% vs 95.4%).

The unique capabilities — per-position gene structure, codon periodicity, RNA dark matter, DNA passports, anomaly detection — are general-purpose, not virus-specific.

Options:
- Keep "ViroSense" — brand recognition, viral detection is the flagship application
- Rename to something general — "EvoSense", "GenomeSense", "DNAPassport"
- Keep the name but expand the tagline: "ViroSense: DNA characterization powered by foundation models"

For the paper, the title should emphasize the general framework, not just viral detection.
