# Paper 1: Gene Structure from DNA Foundation Model Embeddings

**Title**: "Per-position DNA foundation model embeddings encode gene structure from codons to splice sites across all domains of life"

**Target**: Nature Methods (Nature as stretch)

**Last updated**: 2026-03-21

---

## Thesis

A DNA foundation model trained on next-nucleotide prediction learns gene structure at every scale — from triplet codon periodicity to eukaryotic exon-intron boundaries — detectable as a simple geometric property (offset-3 cosine inversion) of per-position embeddings. This is universal across all domains of life, independent of GC content and genetic code variant, and requires no training, no database, and no reference genome.

---

## Key Numbers

| Finding | Value | N |
|---------|-------|---|
| Codon periodicity (coding detection) | 98.5% sensitivity, 100% >500bp | 459 sequences, 55 phyla |
| Comb filter (mod-3 vs non-mod-3) | 0.45 vs 0.29 | 50 sequences |
| Exon-intron detection recall | 98.0% ± 3.6% | 36 genes, 13 species, 9 kingdoms |
| Exon-intron detection F1 | 0.703 ± 0.132 | 36 genes |
| Non-coding specificity | 76.5% (excl. tRNA/intergenic) | 20 controls |
| GC range tested | 9.8% – 78.8% | 459 sequences |
| GC correlation | r = 0.26 (weak) | 459 sequences |
| Stop codon clustering | 1.55× vs sense codons | 64 codons |
| Amino acid identity | NOT encoded (silhouette -0.40) | 64 codons |
| Protein identity clustering | NOT encoded (NN 13-20%) | 287 sequences, 3 configs |
| Optimal smoothing window | 100bp (F1 sweet spot 75-150bp) | 6 genes, 8 windows |

---

## Figures

### Figure 1: The Triplet Genetic Code in Embedding Space
*Discovery and universality*

| Panel | Content | Data source |
|-------|---------|-------------|
| A | Schematic: what offset-3 cosine measures | Diagram |
| B | Multi-offset comb filter (N=36 coding, N=14 non-coding, ± SE) | `multi_offset_expanded.json` |
| C | E. coli lac operon trajectory (cos3 vs cos1 along 6kb) | `fig1_data/` (needs re-extraction for final) |
| D | Cross-domain box plot (N=459, 55 phyla) | `codon_periodicity_panel/embeddings/*_metrics.json` |
| E | GC independence scatter (9.8-78.8%) | Same as D |

**Status**: v5 generated. Needs final polish.

### Figure 2: Gene Structure Detection Across All Life
*Exon-intron boundaries from embedding geometry*

| Panel | Content | Data source |
|-------|---------|-------------|
| A | Human HBB: 3 exons perfectly resolved | `exon_intron/metrics/human_HBB_perpos.json` |
| B | Cross-kingdom montage: TP53, Arabidopsis AGAMOUS, Drosophila eve, C. elegans lin-12 | `exon_intron/metrics/` |
| C | Quantification: recall by kingdom (36 genes, 98% mean) | `exon_intron/quantification_all.json` |
| D | Smoothing optimization: precision-recall tradeoff | `exon_intron/smoothing_optimization.json` |

**Status**: Data complete. Figure NOT YET GENERATED.

### Figure 3: Evo2 vs Augustus Gene Finder Benchmark
*Unsupervised embedding geometry matches trained gene prediction*

| Panel | Content | Data source |
|-------|---------|-------------|
| A | F1 scatter: Augustus vs Evo2 (35 paired genes) | `exon_intron/benchmark/benchmark_results.json` |
| B | Recall comparison: Evo2 matches Augustus (bar chart, 35 genes sorted) | Same |
| C | Precision gap: where Augustus wins (grouped bars + annotation) | Same |
| D | Evo2 advantage: no species model needed (Xenopus + top recall advantages) | Same |

**Status**: GENERATED (`scripts/generate_paper1_fig3_augustus.py` → `results/paper1/figures/fig3.png`)

### Figure 4: Non-Model Organism Gene Discovery
*Unsupervised gene detection validated by independent reannotation*

| Panel | Content | Data source |
|-------|---------|-------------|
| A | Per-gene detection rate across 6 organisms, 3 supergroups | `nonmodel_genome/multi_organism_results.json` |
| B | MCC vs coding density — robust across genomes | Same |
| C | Novel gene detection: 31/33 (94%) of reannotated genes found | `nonmodel_genome/zt_reannotation/` |
| D | Example region: novel genes validated in Z. tritici | Same |

**Status**: GENERATED (`scripts/generate_paper1_fig4_discovery.py` → `results/paper1/figures/fig4.png`)

### Supplementary Figures

| Figure | Content | Status |
|--------|---------|--------|
| S1 | Layer profiling (blocks 0-31) | Data in `nim_api_layer_investigation.md` |
| S2 | Comprehensive validation table (489 seqs by domain/phylum/GC/length) | Data in `comprehensive_validation_results.md` |
| S3 | Non-coding specificity by category | Data computed |
| S4 | E. coli K12 full genome circular map | `genome_scan/ecoli_k12_circular_v2.png` — DONE |
| S5 | All 36 exon-intron gene profiles | `exon_intron/figures/` — DONE |
| S6 | Smoothing window optimization details | `smoothing_optimization.json` |
| S7 | Stop codon clustering (1.55×) + AA identity NOT encoded | `codon_table_embeddings/` |
| S8 | Protein identity clustering NEGATIVE (NN 13-20%) | `functional_clustering_comparison.json` |
| S9 | Non-model organism individual scan plots (6 organisms) | `nonmodel_genome/*_scan.png` |
| S10 | Coding detection context: k-mer comparison, length dependence, non-coding specificity | Old Fig 4 content |
| S11 | HyenaDNA comparison: same inversion in different architecture (5/5 correct) | `hyenadna_comparison/` |

---

## Results Outline

**1. Per-position embeddings encode the triplet genetic code** (Fig 1)
- Offset-3 cosine inversion in coding DNA, inverting in non-coding
- 3-periodic comb filter at offsets 1-15
- Universal: 452/459 (98.5%) across 55 phyla, GC 9.8-78.8%
- 100% above 500bp; 7 failures all in short sequences

**2. The inversion detects eukaryotic exon-intron boundaries** (Fig 2)
- 98% recall across 36 genes, 13 species, 9 kingdoms
- No splice site model, no RNA-seq, no reference genome
- Works on human, Drosophila, C. elegans, Arabidopsis, zebrafish, chicken, Xenopus, yeast, Neurospora, Toxoplasma, rice, maize
- Fills Arc Institute Issue #72

**3. Unsupervised Evo2 matches trained Augustus gene finder** (Fig 3)
- Augustus wins F1 in all 35 paired genes (mean 0.966 vs 0.703)
- But Evo2 matches recall (98% vs 96%) — finds the same exons
- Precision gap (56% vs 98%) due to boundary blurring from smoothing
- Evo2 advantage: works where Augustus has no species model (Xenopus)
- No training, no database, no reference genome required
- Negative results (stop codons, AA identity, protein clustering) → S7-S9

**4. Non-model organism gene discovery** (Fig 4)
- 6 non-model organisms, 5 eukaryotic kingdoms, no Augustus models
- Per-gene detection 95-99% in 5/6 organisms (76% oyster)
- Z. tritici reannotation: 91% of novel genes detected, 24% FPs genuinely coding
- RNA-seq validation: FP transcription enrichment 1.17-2.32× in ALL 5 organisms (p < 10^-56)
- Boundary resolution: linear probe doubles F1 to 0.88, boundary distance 116bp (17bp for best genes)
- K-mer context and limitations → S10

---

## Discussion Points

- The model learned gene structure from DNA prediction alone — codon triplets, stop codons, splice sites
- This delineates DNA-learnable features (syntax) from protein-learnable features (semantics)
- Database-free gene annotation for the 99.9% of species without references
- Exon-intron detection without RNA-seq — relevant for unculturable organisms, ancient DNA, non-model species
- Should be testable in other DNA models (Nucleotide Transformer, DNABERT-2, Caduceus)
- Limitations: 100bp boundary resolution, tRNA false positives, short sequence sensitivity

---

## NOT in Paper 1 (→ Paper 2: ViroSense)

- RNA dark matter detection (95.2%)
- Viral detection benchmark (13,417 sequences, geNomad comparison)
- Prophage amelioration gradient
- ViroSense tool and CLI
- Two-tier k-mer → Evo2 pipeline
- Contig typing (virus/plasmid/chromosome)
- HTCF deployment infrastructure
- L2-normalization for 7B RNA virus detection
- Clustering (ARI=0.903) on viral data
- Phylogenomics on phage data
