# ViroSense Publication Plan

**Working title**: "DNA foundation model embeddings as a general-purpose representation for metagenomic sequence classification"

**Target journal**: Nature Methods / Genome Research

Last updated: 2026-03-16. This is a living document — iterate as new data and findings emerge.

---

## Central Thesis

Frozen embeddings from the Evo2 DNA foundation model serve as a **general-purpose representation** for metagenomic analysis. A single forward pass through Evo2 produces per-position embeddings that support viral detection, contig typing, unsupervised clustering, and gene boundary detection — all through lightweight downstream analyses on the same cached representation. This "embed once, analyze many ways" paradigm contrasts with traditional pipelines where each task requires a separate tool with its own models and databases.

---

## Key Findings (data in hand)

### 1. Viral Detection (binary)
- ViroSense 40B: 99.7% phage sensitivity, 91.1% RNA virus recall (zero-shot)
- Outperforms geNomad on short fragments: **99.7% vs 51.2% at 1-3kb**
- Bootstrap CIs: all key comparisons non-overlapping
- **Status**: COMPLETE, publication-ready

### 2. Contig Typing (3-class: virus/plasmid/chromosome)
- Cross-validated: 94.5% ± 0.7% accuracy, 91.5% ± 2.9% plasmid detection
- Held-out evaluation: 94.8% plasmid detection, 99.2% plasmid specificity
- Matches geNomad plasmid specificity (99.3%) while adding detection capability
- **Status**: COMPLETE, publication-ready

### 3. L2-Normalization / Preprocessing Insights
- 7B model: embedding magnitude decays with length for RNA viruses → L2-norm fixes (34% → 99% at 10-16kb)
- 40B model: magnitude is informative → L2-norm hurts
- Training data composition: adding plasmids to cellular class improves RNA virus recall by 4.7%
- **Status**: COMPLETE, publication-ready

### 4. Unsupervised Clustering
- HDBSCAN on PCA-reduced 40B embeddings: ARI=0.903 (best config), NMI=0.782
- **Naturally separates eukaryotic RNA viruses from dsDNA phages** (cluster of 277, 99% pure) — fully unsupervised, no labels used
- GYP phage category is 100% dsDNA (936 genomes); RNA virus category is 99.8% eukaryotic RNA viruses + 2 RNA phages (Leviviridae)
- Evo2, trained only on DNA, still distinguishes cDNA-converted RNA viral genomes from DNA viral genomes — the model captures fundamental compositional differences between DNA-origin and RNA-origin sequences
- Best config: min_cluster_size=200, min_samples=5 → 4 clusters, 47% noise (typical for HDBSCAN on biological data)
- K-Means weaker (ARI=0.2) — embedding manifold is non-linear
- **Status**: VALIDATED on full GYP benchmark (13,055 sequences with 40B embeddings), needs figures

**Important terminology note**: Use "dsDNA phages" (not "phages") and "eukaryotic RNA viruses" (not "RNA viruses") throughout the paper to avoid conflating with RNA phages (Leviviricetes, Cystoviridae). The GYP benchmark is specifically dsDNA phages.

### 5. Per-Position Embeddings: Gene Structure & Codon Periodicity
- **Gene boundary detection**: Coding regions have 1.72 ± 0.23× higher embedding norms than intergenic — consistent across 41 diverse sequences (phage, chromosome, plasmid, RNA virus, cellular). 91.7% coding/non-coding accuracy, 73.2% boundary recall.
- **Codon periodicity** (KEY FINDING): Lag-3 autocorrelation of embedding norms is 0.55-0.59 in coding regions, near zero at lag-1/lag-2. The 3bp period is the **dominant FFT frequency**. Confirmed on both phage and E. coli chromosome.
- **Offset-3 cosine inversion**: At 1bp offset, coding similarity < intergenic. At 3bp offset, it INVERTS (coding > intergenic). This is a binary signature of protein-coding DNA requiring no ORF analysis, codon tables, or training.
- **Interpretation**: Evo2 implicitly learns the triplet genetic code from next-nucleotide prediction. The reading frame is the strongest structural feature in its per-position representations.
- **RNA virus dark matter detection**: Codon periodicity can classify RNA virus vs dsDNA phage at 91.3% accuracy (5-fold CV, n=23). cos3_coding has Cohen's d = 2.83 — a massive effect size. Could be the first database-free RNA virus identifier.
- **Status**: VALIDATED (periodicity on 40 sequences, RNA virus classification PoC on 23 sequences). Needs expansion to 100+ RNA virus sequences for robust validation.

### 6. Head-to-Head with geNomad
- geNomad completed on 13,417 sequences (22 min, 8 CPUs)
- ViroSense wins: short fragments, overall sensitivity, AUC
- geNomad wins: speed (500×), plasmid specificity, gene-level annotation, provirus detection
- **Status**: COMPLETE for geNomad. DeepVirFinder/VIBRANT pending.

---

## TODO List

### Priority 1: Complete for Submission

- [x] **Complete 40B RNA virus gap fill** — DONE (2026-03-17). All 13,417 sequences. RNA virus 10-16kb: 98.5%.
- [x] **Expand gene boundary PoC to 20+ sequences** — DONE. 41 sequences across 5 categories.
- [x] **Codon periodicity validation** — DONE. 40 sequences. lag-3 autocorr 0.635, offset-3 inversion 100% universal, 94.7% coding accuracy.
- [x] **Bootstrap CIs with complete 40B** — DONE. `results/benchmark/comparison/bootstrap_ci_complete.json`.
- [x] **Implement `virosense embed` command** — DONE. Separates expensive embedding step from analysis.
- [x] **Implement `virosense scan` command** — DONE. Per-position analysis via CLI.
- [x] **Fix `virosense classify`** — DONE. Custom task names, warning suppression.
- [x] **CLI-test `virosense cluster`** — DONE. Works end-to-end on real data.
- [x] **RNA dark matter validation** — COMPLETE (203/220 sequences). **97.5% accuracy**, 0.990 AUC. Zero Firmicutes false positives. cos3 is the dominant feature (50.9% importance).
- [x] **Perplexity forensics pilot** — COMPLETE. 20 natural + 20 codon-optimized gene pairs. 75% accuracy from embedding features alone. Lag-1 autocorrelation distinguishes natural from optimized. Norm CV is the best feature (d = -0.92).

- [ ] **Generate publication figures** (matplotlib/seaborn)
  - Fig 1: Architecture concept diagram ("embed once, analyze many ways")
  - Fig 2: Detection benchmark (ROC, length curves, bootstrap CIs)
  - Fig 3: Embedding space (PCA, L2-norm, training data effects)
  - Fig 4: 3-class contig typing (confusion matrix, vs geNomad)
  - Fig 5: Per-position embeddings — codon periodicity FFT, offset-3 inversion, norm trajectory
  - Fig 6: RNA virus dark matter (periodicity classification, pending batch completion)
  - Fig S1: HDBSCAN clustering (UMAP visualization)
  - Fig S2: Speed benchmarking
  - Fig S3: Plasmid FP analysis (embedding space, training data composition effects)

- [ ] **Run UMAP** for publication-quality clustering visualization

### Priority 2: Strengthen for Reviewers

- [ ] **DeepVirFinder benchmark** — needs working DVF installation (Theano/TF compat issue)
  - Consider using the newer `deepvirfinder` pip package if available
  - Or download pre-computed predictions if published on GYP data

- [ ] **VIBRANT benchmark** — needs database download on HTCF
  - Database is ~12 GB, requires figshare download
  - Could submit separate SLURM job for database setup + benchmark

- [ ] **VirSorter2 benchmark** — needs conda environment
  - Most practical: install micromamba on HTCF scratch, create VS2 env
  - Or use published VS2 results on GYP if available

- [ ] **Real metagenome validation**
  - Minimum: 1 sample with CheckV ground truth (e.g., SRR5747446 gut virome)
  - Better: 3-5 diverse samples (gut, ocean, soil)
  - Run ViroSense + geNomad on same assemblies, compare with CheckV

- [ ] **Independent validation dataset**
  - UHGV (873K virus genomes) or IMG/VR for an independent test set
  - Shows generalization beyond GYP benchmark

- [ ] **Retrain 7B classifier with L2-norm** on HTCF (formal production classifier)
  - Also retrain 7B 3-class classifier
  - Compare 7B+L2 vs 40B in head-to-head

### Priority 3: Nice-to-Have

- [ ] **Extract 3class plasmid training embeddings via 40B NIM**
  - 3,053 sequences × ~50s = ~42 hrs
  - Enables clean 3-class training without GYP leakage
  - Would strengthen the 3-class results

- [ ] **Ensemble: ViroSense + geNomad**
  - Show that union of both tools outperforms either alone
  - ViroSense catches short-fragment viruses geNomad misses; geNomad catches plasmid-like FPs
  - Quantify the unique contribution of each tool

- [ ] **Prophage benchmark**
  - Need curated prophage dataset (PHASTER, PhiSpy, or manual curation)
  - Validate the adaptive scanning approach on real prophages
  - Could be supplementary or a separate paper

- [ ] **Per-position gene calling model**
  - Train a 1D-CNN/BiLSTM on per-position embeddings for gene prediction
  - Benchmark against Prodigal on diverse genomes
  - "Evo2 as a gene caller" — could be a separate short paper

- [ ] **Functional region annotation from per-position**
  - Label gene types (capsid, tail, integrase, etc.) in known phages
  - Train per-position classifier for functional categories
  - Would replace the annotate module's protein-based pipeline

- [ ] **RNA virus training data for 3-class**
  - Sample from RNA Virus Database (385K sequences)
  - Extract 40B embeddings for ~3,000 RNA virus fragments
  - Retrain 3-class with RNA viruses in viral class → fix 3-class RNA recall

---

## Validation Status Matrix

| Module | Unit tests | Real data benchmark | Publication-ready? |
|--------|-----------|--------------------|--------------------|
| **embed** (NEW) | 0 | ✅ CLI tested on real data with cached embeddings | **YES** |
| **detect** (binary) | 47 | ✅ 13,417 seq + geNomad head-to-head + bootstrap CIs | **YES** |
| **classify** (3-class) | 12 | ✅ 5-fold CV + held-out GYP + CLI training/prediction tested | **YES** |
| **cluster** | 27 | ✅ HDBSCAN ARI=0.903 on 13K sequences; CLI tested on real data | **YES** (needs figures) |
| **scan** (NEW) | 0 | ✅ 42 sequences via CLI; norm ratio, periodicity, boundaries all validated | **YES** |
| **per-position analysis** | — | ✅ 40-seq codon periodicity (lag-3=0.635, 94.7% coding), RNA dark matter PoC (91.3%) | **YES** |
| **prophage** | 36 | ❌ No real-data validation | Needs prophage benchmark |
| **context** | 14 | Deprecated — superseded by scan | Not in this paper |
| **annotate** | 109 | ❌ No real-data validation | Separate future paper |

---

## Manuscript Outline (Draft)

### Title
"DNA foundation model embeddings as a general-purpose representation for metagenomic sequence classification"

### Abstract
- DNA foundation model embeddings serve as a universal sequence characterization framework
- Trinucleotide frequencies achieve 93% of Evo2's accuracy for binary viral detection — the foundation model's unique value is in qualitatively different analyses: per-position gene structure, zero-shot generalization, compositional characterization, and anomaly detection
- Per-position embeddings reveal the triplet genetic code (3bp codon periodicity) and enable 94.7% coding region detection without gene calling
- RNA dark matter detection at 97.5% accuracy from periodicity features alone — database-free
- Same embeddings support detection, contig typing, clustering, gene structure, prophage boundaries, and comprehensive biological profiling ("DNA passports")

### Introduction
- Viral metagenomics depends on detecting viral sequences in mixed microbial communities
- Current tools (geNomad, VIBRANT, VirSorter2) use marker genes → fail on short/fragmentary data
- DNA foundation models (Evo2) learn sequence representations from next-nucleotide prediction
- We show these frozen representations are sufficient for multiple metagenomic tasks

### Results
1. **ViroSense architecture**: embed once, classify many ways
2. **Viral detection benchmark**: head-to-head with geNomad on GYP + RNA virus
3. **Short-fragment advantage**: mechanistic explanation (composition vs gene content)
4. **Embedding preprocessing**: L2-normalization, training data composition effects
5. **3-class contig typing**: virus/plasmid/chromosome from same embeddings (94.5% CV accuracy)
6. **Unsupervised structure**: HDBSCAN recovers biological categories (ARI=0.903), separates euk. RNA viruses from dsDNA phages (99% pure)
7. **Per-position codon periodicity**: Evo2 implicitly learns the triplet genetic code — the dominant FFT frequency in coding regions is exactly 3bp, with offset-3 cosine inversion as a universal binary coding signature (94.7% coding accuracy, 100% of sequences)
8. **RNA virus dark matter**: Eukaryotic RNA viruses have distinctively strong codon periodicity (lag-3 = 0.822 vs 0.624 for dsDNA phages). Enables database-free RNA virus identification — 91.3% accuracy (PoC on 23 sequences, expanded 220-sequence validation running)
9. **CLI architecture**: "Embed once, analyze many ways" — 8 commands (embed, detect, classify, cluster, scan, prophage, build-reference, context), 6 validated on real data

### Discussion
- Foundation models as general-purpose extractors vs task-specific tools
- Complementarity with geNomad (different signals: composition vs gene content)
- Practical deployment: two-tier architecture (7B screening, 40B publication)
- Speed trade-off: 500× slower embedding extraction, but instant reclassification
- Implications for other foundation model applications in genomics

### Methods
- Evo2 embedding extraction (NIM API, mean pooling, per-position)
- MLP classifier architecture (sklearn, 512→128, Platt calibration)
- Benchmark datasets (GYP, RNA Virus Database, 3-class reference)
- Bootstrap confidence intervals (10,000 resamples)
- geNomad comparison (v1.11.2, default parameters)

---

## Timeline Estimate

| Task | Effort | Blocked by |
|------|--------|-----------|
| 40B gap fill | Running now | NIM API |
| Gene boundary expansion (20 seqs) | 1-2 hours | NIM API calls |
| Generate figures | 1-2 days | Gap fill, gene boundary |
| DVF/VIBRANT/VS2 benchmarks | 1 day | Tool installation |
| Real metagenome validation | 2-3 days | Data + geNomad runs |
| Write manuscript | 1-2 weeks | Figures |
| Revisions | Ongoing | Reviewer feedback |

---

## Files Reference

| Document | Contents |
|----------|----------|
| `docs/rna_virus_length_analysis.md` | All benchmark findings (detect, 3-class, plasmid analysis, L2-norm) |
| `docs/poc_gene_boundaries.md` | Per-position embedding gene boundary PoC |
| `docs/speed_benchmarking.md` | Speed comparison across tools |
| `docs/biosurveillance_research_plan.md` | Future directions (perplexity forensics, etc.) |
| `results/benchmark/comparison/` | Comparison results JSON + LaTeX table |
| `results/benchmark/comparison/bootstrap_ci.json` | Bootstrap confidence intervals |
| `results/poc_gene_boundaries*/` | Per-position embedding analysis data |
| `scripts/compare_tools.py` | Tool comparison analysis |
| `scripts/bootstrap_ci.py` | Bootstrap CI computation |
| `scripts/poc_gene_boundaries.py` | Gene boundary detection PoC |
