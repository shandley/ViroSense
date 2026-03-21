# ViroSense Paper Assets Index

**Paper**: "Per-position DNA foundation model embeddings encode the triplet genetic code across all domains of life"
**Target**: Nature Methods
**Last updated**: 2026-03-21

---

## Main Figures

### Figure 1: Per-Position Embeddings Encode the Triplet Genetic Code
- **File**: `figures/fig1_v5.png` / `.pdf` — **CURRENT**
- **Caption**: `figures/fig1_caption.md`
- **Panels**: A (schematic) + B (comb filter, N=36/14) + C (lac operon trajectory) + D (box plot, N=459) + E (GC scatter)
- **Data sources**:
  - Panel B: `codon_periodicity_panel/multi_offset_expanded.json` (50 sequences)
  - Panel C: `figures/fig1_data/ecoli_lacz_genes.json` (gene annotations); embedding deleted, regenerate via NIM
  - Panel D-E: `codon_periodicity_panel/embeddings/*_metrics.json` (489 sequences, 7B HTCF)

### Figure 2: Applications of Per-Position Periodicity — NOT YET GENERATED
- RNA dark matter (97.5%, 0.990 AUC)
- Coding detection (94.7%)
- Prophage amelioration gradient
- **Data**: `poc_rna_dark_matter/batch_results.csv`, `poc_gene_boundaries_expanded/`

### Figure 3: Multi-Task DNA Analysis — NOT YET GENERATED
- Viral detection (99.7% phage), contig typing (94.5%)
- HDBSCAN clustering (ARI=0.903)
- Phylogenomics (r=0.504)
- Functional clustering negative result
- **Data**: `benchmark/40b_complete/`, `benchmark/comparison/`

### Figure 4: Foundation Models vs K-mer Baselines — NOT YET GENERATED
- K-mer: 93% at 1,527x speed
- Gap analysis, capability matrix, two-tier pipeline
- **Data**: in docs/

---

## Supplementary Figures

### S1: Layer Profiling
- Not yet generated; data in `docs/nim_api_layer_investigation.md`

### S2: Comprehensive Validation Data
- Not yet generated; data in `codon_periodicity_panel/embeddings/*_metrics.json` (489 sequences)
- Full breakdown by domain/phylum/family/GC/length in `docs/comprehensive_validation_results.md`

### S3: Non-Coding Specificity by Category
- Not yet generated; data analyzed in `docs/comprehensive_validation_results.md`

### S4: Functional Clustering Negative Result
- Not yet generated; data in `codon_periodicity_panel/functional_clustering_comparison.json`

### S5: L2-Normalization (7B vs 40B)
- Data in `benchmark/40b_l2norm/`, `docs/rna_virus_length_analysis.md`

### S6: ViroSense vs geNomad Head-to-Head
- Data in `benchmark/comparison/bootstrap_ci_complete.json`

### S7: E. coli K12 Full Genome Inversion Profile
- **File**: `genome_scan/ecoli_k12_circular_v2.png` / `.pdf` — **CURRENT**
- **Data**: `genome_scan/window_metrics.npz` (100 MB, 4.6M positions), `genome_scan/ecoli_k12_genes.json`
- 386 windows × 16kb, Evo2 7B decoder.layers.10, HTCF

### S8: Codon Table in Embedding Space
- Not yet generated as figure; data in `codon_table_embeddings/codon_embeddings.json`
- Stop codons cluster (1.55x); amino acid identity not encoded

### S9: Prophage Amelioration Scores
- Data in `docs/prophage_reassessment.md`

---

## Superseded (do not use)

| File | Reason |
|------|--------|
| `figures/fig1_genetic_code.png` | Superseded by v5 |
| `figures/fig1_genetic_code_v2.png` | Superseded by v5 |
| `figures/fig1_v3.png` | Superseded by v5 |
| `figures/fig1_v4.png` | Superseded by v5 |
| `figures/fig1a_norm_trajectory.png` | Exploratory draft |
| `figures/fig1_options_exploration.png` | Layout exploration |
| `figures/fig_training_summary.png` | Legacy |
| `figures/benchmark_summary.md` | Outdated |
| `genome_scan/ecoli_k12_circular.png` | Superseded by v2 |
| `genome_scan/ecoli_k12_genome_scan.png` | Linear plot — replaced by circular |
| `universal_validation_v2/figures/*` | From 72-seq pilot, superseded by 489-seq comprehensive |

---

## Captions

All captions should be in `figures/fig1_caption.md` or equivalent files per figure.
Currently only Figure 1 has a caption file. Remaining captions to be written with manuscript.

---

## Key Numbers for Text

| Metric | Value | Source |
|--------|-------|--------|
| Coding sensitivity | 452/459 (98.5%) | comprehensive_validation_results.md |
| Sensitivity >500bp | 100% (316/316) | comprehensive_validation_results.md |
| Non-coding specificity (excl tRNA/intergenic) | 76.5% (15/20) | comprehensive_validation_results.md |
| Phyla tested | 55 | comprehensive_validation_results.md |
| Gene families tested | 10 × ~29 orthologs | comprehensive_validation_results.md |
| GC range | 9.8% - 78.8% | comprehensive_validation_results.md |
| GC correlation | r = 0.26 | comprehensive_validation_results.md |
| Comb filter: mod-3 mean | 0.45 | multi_offset_expanded.json |
| Comb filter: non-mod-3 mean | 0.29 | multi_offset_expanded.json |
| Stop codon clustering | 1.55× ratio | codon_table_embeddings/ |
| Functional clustering (40B) | Silhouette -0.064, NN 19.9% | functional_clustering_comparison.json |
| RNA dark matter | 97.5%, AUC 0.990 | poc_rna_dark_matter/ |
| Coding detection | 94.7% | poc_gene_boundaries_expanded/ |
| Viral detection (40B) | 95.4% accuracy, 99.7% phage | benchmark/40b_complete/ |
| K-mer baseline | 93% at 1,527× speed | docs/ |
| HDBSCAN ARI | 0.903 | docs/cluster_validation.md |
| Phylogenomics | r = 0.504 | docs/ |
| Same-genome NN (taxonomy pilot) | 13.5% top-1 (135× random) | findings_log.md Finding 14 |
