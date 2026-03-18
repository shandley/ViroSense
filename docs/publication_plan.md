# ViroSense Publication Plan

**Working title**: "Frozen DNA foundation model embeddings reveal the genetic code and enable universal sequence characterization"

**Target journal**: Nature Methods

Last updated: 2026-03-18

---

## Central Thesis (Revised)

Frozen Evo2 embeddings are a **universal DNA characterization framework**, not merely a classifier. The key contributions are things that only foundation model embeddings can do — and that k-mer baselines and marker-gene tools cannot:

1. **Per-position embeddings encode the genetic code** — 3bp codon periodicity emerges from unsupervised next-nucleotide prediction
2. **Zero-shot generalization** — 93% RNA virus recall without any RNA virus training data
3. **Database-free RNA dark matter detection** — 97.5% accuracy from periodicity features alone
4. **Compositional characterization** — DNA passports that reveal identity, origin, structure, and novelty
5. **Prophage amelioration gradients** — viral scores measure evolutionary age of integrated elements

Binary viral detection (where we compete with geNomad and k-mer baselines) is a **demonstration**, not the main contribution. K-mer baselines achieve 93% for detection — the 2.3% gap is honest and well-characterized.

---

## What's Changed Since the Original Plan

| Original framing | Revised framing |
|-----------------|-----------------|
| "Viral detection tool that beats geNomad" | "Universal characterization framework with viral detection as one application" |
| Detection accuracy is the main result | **Codon periodicity discovery** and **characterization** are the main results |
| Per-position analysis is a PoC | Per-position analysis is a **primary finding** — the genetic code in embeddings |
| Prophage detection as a feature | Prophage **amelioration gradient** as a characterization example |
| Speed is a weakness | **Two-tier pipeline** (k-mer screening → Evo2 characterization) addresses speed |
| Single-purpose tool | **"Embed once, analyze many ways"** — detection, typing, clustering, gene structure, phylogenomics, characterization all from one embedding |

---

## Key Findings (ranked by novelty, not by order discovered)

### 1. The Genetic Code in Embeddings (most novel)
- 3bp codon periodicity is the **dominant FFT frequency** in Evo2 per-position representations
- Lag-3 autocorrelation: 0.635 across 40 diverse sequences, universal across all categories
- Offset-3 cosine inversion: 94.7% coding detection accuracy, present in 100% of sequences
- Evo2 learned the triplet code from unsupervised next-nucleotide prediction
- **No prior work has shown this.** This is a fundamental insight about DNA foundation models.

### 2. RNA Dark Matter Detection (high novelty + high impact)
- 97.5% accuracy, 0.990 AUC distinguishing eukaryotic RNA viruses from all other categories
- From per-position periodicity features alone — **no database, no homology, no gene calling**
- cos3 (offset-3 cosine) is the dominant feature (Cohen's d = 2.83)
- Zero Firmicutes phage false positives (0/25)
- 203 sequences validated across 4 categories
- **First database-free RNA virus identifier from DNA composition**

### 3. Universal Characterization Framework (architectural contribution)
- `virosense characterize` produces multi-dimensional "DNA passports"
- Identity (nearest category + anomaly score), Origin (viral/RNA/mobile signatures), Structure (coding, periodicity), Novelty (percentile against reference)
- 100% RNA origin interpretation accuracy on 25 diverse test sequences
- Anomaly scoring flags novel elements (Obelisk-like discoveries)
- **One Evo2 forward pass → comprehensive biological profile**

### 4. K-mer Baseline Comparison (honest, strengthens the paper)
- Trinucleotide frequencies achieve **93% viral detection** at **1527× the speed** of Evo2
- Gap analysis: 6.3% of sequences need Evo2 (host-adapted phages, low-GC false positives, short fragments)
- The gap is **biologically meaningful** — Evo2 captures contextual composition beyond k-mer bags
- Two-tier pipeline: k-mers screen everything, Evo2 characterizes the borderline 15%
- **This is the honest comparison most papers don't include**

### 5. Viral Detection Benchmark (validation, not main contribution)
- 40B: 99.7% phage sensitivity, 93.0% RNA virus recall, 95.4% overall accuracy
- vs geNomad: ViroSense wins on short fragments (99.7% vs 51.2% at 1-3kb), geNomad wins on plasmid specificity (99.3% vs 81.5%) and speed (500×)
- Bootstrap CIs: all key comparisons non-overlapping
- 13,417 sequences, complete benchmark

### 6. Additional Validated Capabilities
- **3-class contig typing**: 94.5% CV accuracy (virus/plasmid/chromosome)
- **Unsupervised clustering**: HDBSCAN ARI=0.903, RNA viruses separate from dsDNA phages (99% pure cluster)
- **Alignment-free phylogenomics**: Spearman r=0.504 between embedding distance and taxonomic distance
- **Prophage amelioration**: Coarse-pass viral scores reveal evolutionary age gradient (DLP12 active → e14 invisible)
- **Gene boundary detection**: 91.7% coding accuracy, 73.2% boundary recall, 1.72× norm ratio across 41 sequences

### 7. Preprocessing Insights
- L2-normalization: essential for 7B (63%→93% RNA recall), counterproductive for 40B
- Training data composition: adding plasmids improves RNA virus recall 4.7%
- Model scale: preprocessing must match model tier
- **Transferable findings for anyone building classifiers on foundation model embeddings**

---

## Manuscript Outline (Revised)

### Title
"Frozen DNA foundation model embeddings reveal the genetic code and enable universal sequence characterization"

### Prior Art Context (see docs/prior_art.md for full analysis)
- 3bp periodicity in coding DNA: known since 1982 (Fickett), NOT novel
- Foundation models learn gene structure: shown qualitatively (Goodfire blog 2025, NOT peer-reviewed; Nucleotide Transformer, Nature Methods 2024)
- Codon-level foundation models: cdsFM/CodonFM (2024) trains ON codons, learns genetic code — but given the reading frame by tokenization
- **Our novelty**: quantitative per-position periodicity in nucleotide-resolution model (lag-3 autocorr, offset-3 cosine inversion, coding detection, RNA dark matter). The model discovers codon structure without being told about it. Applications (94.7% coding detection, 97.5% RNA dark matter) are entirely new.

### Abstract
DNA foundation models learn rich representations of nucleotide sequences through unsupervised training, but how to extract and apply these representations for biological analysis remains unclear. We show that frozen per-position embeddings from Evo2 encode the triplet genetic code as their dominant structural feature — a 3-nucleotide periodicity that emerges without any supervised signal. This periodicity enables 94.7% coding region detection without gene calling and 97.5% database-free RNA virus identification from codon periodicity features alone. We develop ViroSense, a universal DNA characterization framework that produces multi-dimensional biological profiles ("DNA passports") from a single embedding extraction, supporting viral detection (99.7% phage sensitivity), contig typing, unsupervised clustering, gene structure analysis, and compositional characterization. We show that simple trinucleotide frequency classifiers achieve 93% of Evo2's accuracy for binary viral detection, with the foundation model's unique value concentrated in per-position analysis, zero-shot generalization to unseen sequence types, and compositional anomaly detection for novel element discovery.

### Results

**1. Evo2 embeddings encode the genetic code** (Figure 1)
- Per-position norm signal: coding 1.72× intergenic (41 sequences, 5 categories)
- Codon periodicity: lag-3 autocorrelation 0.635, dominant FFT at 3bp
- Offset-3 cosine inversion: universal binary coding signature (94.7% accuracy)
- RNA viruses have strongest periodicity (0.822) — compositional signature of RNA-origin

**2. Database-free RNA dark matter detection** (Figure 2)
- 97.5% accuracy, 0.990 AUC from periodicity features (203 sequences)
- Zero Firmicutes false positives
- cos3 is the dominant feature (Cohen's d = 2.83)
- Implications for novel RNA element discovery (Obelisks, novel viral lineages)

**3. Universal characterization framework** (Figure 3)
- DNA passport: identity + origin + structure + novelty from one embedding
- Anomaly scoring for novel element detection
- Characterize vs classify: multi-dimensional profiles vs binary labels
- Application examples: prophage amelioration gradient, phylogenomic signal

**4. Viral detection and the k-mer baseline** (Figure 4)
- ViroSense 40B: 99.7% phage, 93.0% RNA virus, 95.4% accuracy
- geNomad head-to-head: complementary strengths (short fragments vs plasmid specificity)
- K-mer baseline: 93% accuracy at 1527× speed — honest comparison
- Gap analysis: 6.3% of sequences need Evo2 (host-adapted phages, short fragments)
- Two-tier pipeline: k-mer screening → Evo2 characterization

**5. Multi-task from one embedding** (Figure 5)
- 3-class contig typing (94.5%)
- Unsupervised clustering (ARI=0.903)
- Alignment-free phylogenomics (r=0.504)
- Prophage amelioration gradient (3 states visible from coarse scores)
- "Embed once, analyze many ways"

### Discussion
- The genetic code discovery: what it means for understanding DNA foundation models
- Characterization vs classification: why multi-dimensional profiles matter
- Honest speed comparison: Evo2 is 1500× slower but the "embed once" architecture amortizes the cost
- K-mer baselines are strong for detection — foundation models add unique per-position and zero-shot capabilities
- Implications: alignment-free phylogenomics, RNA dark matter, prophage evolution, forensic detection
- Limitations: speed, GPU requirement, e14-like fully ameliorated elements invisible
- Future: knowledge distillation for planetary-scale screening (Logan)

### Methods
- Evo2 40B/7B embedding extraction (NIM API, per-position)
- Mean-pooling, L2-normalization, and preprocessing
- MLP classifier (sklearn, 512→128, Platt calibration)
- K-mer feature computation (trinucleotide + dinucleotide frequencies)
- Benchmark datasets (GYP 13,417 seqs, RNA Virus Database 1,000 seqs)
- Bootstrap CIs (10,000 resamples)
- geNomad v1.11.2 comparison
- HDBSCAN clustering with PCA preprocessing
- Per-position periodicity analysis (autocorrelation, FFT, cosine similarity)
- DNA passport characterization (cosine to category centroids, anomaly scoring)

---

## Figure Set (5 main + supplementary)

### Figure 1: The Genetic Code in Embeddings
- A: Embedding norm trajectory along a phage genome (coding vs intergenic)
- B: Autocorrelation function — peak at lag-3 (codon periodicity)
- C: FFT spectrum — dominant frequency at exactly 3.0 bp
- D: Offset-3 cosine inversion: coding > intergenic at 3bp offset
- E: Cross-category validation (norm ratio 1.72× across 41 sequences)

### Figure 2: RNA Dark Matter Detection
- A: Periodicity features by category (cos3, lag3 — RNA virus highest)
- B: Classification performance (97.5% accuracy, 0.990 AUC)
- C: Feature importance (cos3 dominates at 50.9%)
- D: Obelisk case study — how characterization would flag novel RNA elements

### Figure 3: Universal Characterization (DNA Passports)
- A: Architecture diagram (embed → multi-task analysis)
- B: Example DNA passports for phage, RNA virus, chromosome, plasmid
- C: Prophage amelioration gradient (DLP12 active → e14 invisible)
- D: Anomaly scoring: RNA viruses at 99.8th percentile

### Figure 4: Viral Detection and K-mer Baseline
- A: ViroSense vs geNomad: sensitivity by fragment length (1-15kb)
- B: Bootstrap CIs for key metrics
- C: K-mer baseline: 93% accuracy, 1527× faster
- D: Gap analysis: where Evo2 adds value (host-adapted phages, short fragments)
- E: Two-tier pipeline diagram

### Figure 5: Multi-Task from One Embedding
- A: HDBSCAN clustering (UMAP, colored by category, ARI=0.903)
- B: 3-class contig typing confusion matrix
- C: Alignment-free phylogenomics (distance hierarchy by taxonomic level)
- D: Speed comparison table (embed once → instant analysis)

### Supplementary
- S1: L2-normalization analysis (7B vs 40B)
- S2: Training data composition effects (plasmid inclusion improves RNA recall)
- S3: Plasmid FP embedding space analysis
- S4: Detailed benchmark tables with bootstrap CIs
- S5: Per-position analysis on all 41 sequences (norm ratio, periodicity by category)
- S6: Full E. coli K12 prophage scores (all 9 known cryptic prophages)

---

## What's Ready vs What's Needed

| Component | Status | Action needed |
|-----------|--------|--------------|
| All benchmark data | ✅ Complete | None |
| K-mer baseline comparison | ✅ Complete | None |
| RNA dark matter (203 seqs) | ✅ Complete | None |
| Codon periodicity (40 seqs) | ✅ Complete | None |
| Gene boundaries (41 seqs) | ✅ Complete | None |
| Phylogenomics pilot | ✅ Complete | None |
| Prophage amelioration | ✅ Complete (coarse) | Full scan nice-to-have |
| Forensics pilot | ✅ Complete | Supplementary |
| Characterize framework | ✅ Complete | None |
| Two-tier --fast pipeline | ✅ Complete | None |
| **Figures** | ❌ Not started | **START HERE** |
| **Manuscript text** | ❌ Not started | After figures |
| DVF/VIBRANT/VS2 comparison | ❌ Not done | Nice-to-have, not required |
| Real metagenome validation | ❌ Not done | Would strengthen but not essential |

---

## Timeline

| Task | Effort | Priority |
|------|--------|----------|
| **Generate figures** | 3-5 days | **NOW** |
| **Write manuscript** | 2-3 weeks | After figures |
| Real metagenome validation | 3-5 days | If time permits |
| Additional tool comparisons | 2-3 days | Nice-to-have |
| Revisions | Ongoing | After submission |
