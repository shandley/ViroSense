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

### Figure 3: What the Model Learned — and What It Didn't
*Boundaries of DNA-level learning*

| Panel | Content | Data source |
|-------|---------|-------------|
| A | Stop codon clustering (1.55×) | `codon_table_embeddings/` |
| B | Amino acid identity NOT encoded | `codon_table_embeddings/` |
| C | Protein identity clustering NEGATIVE | `functional_clustering_comparison.json` |
| D | Summary: syntax vs semantics of the genetic code | Diagram |

**Status**: Data complete. Figure NOT YET GENERATED.

### Figure 4: Coding Detection in Context
*Comparison to existing approaches*

| Panel | Content | Data source |
|-------|---------|-------------|
| A | Coding detection accuracy by kingdom | `codon_periodicity_panel/` |
| B | K-mer baseline: 93% at 1,527× speed | docs/ |
| C | Capability comparison: what k-mers can't do (exon-intron, per-position) | Table/diagram |
| D | Length dependence: 100% >500bp, 85% <300bp | `comprehensive_validation_results.md` |

**Status**: Data complete. Figure NOT YET GENERATED.

### Supplementary Figures

| Figure | Content | Status |
|--------|---------|--------|
| S1 | Layer profiling (blocks 0-31) | Data in `nim_api_layer_investigation.md` |
| S2 | Comprehensive validation table (489 seqs by domain/phylum/GC/length) | Data in `comprehensive_validation_results.md` |
| S3 | Non-coding specificity by category | Data computed |
| S4 | E. coli K12 full genome circular map | `genome_scan/ecoli_k12_circular_v2.png` — DONE |
| S5 | All 36 exon-intron gene profiles | `exon_intron/figures/` — DONE |
| S6 | Smoothing window optimization details | `smoothing_optimization.json` |

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

**3. The model learned DNA syntax, not protein semantics** (Fig 3)
- Stop codons cluster (1.55×) — gene boundaries are learnable from DNA
- Amino acid identity NOT encoded — requires protein-level selection
- Protein identity clustering NEGATIVE — UMAP was misleading
- Offset-1 > offset-2 is a sequential property, not wobble-specific

**4. Practical context and limitations** (Fig 4)
- K-mer baselines achieve 93% for binary coding detection
- Foundation models add per-position resolution (exon-intron, gene boundaries)
- Length dependence: reliable above 500bp
- Non-coding specificity: tRNA shows inversion (codon-anticodon structure)

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
