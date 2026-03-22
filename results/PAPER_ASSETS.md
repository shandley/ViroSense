# Paper Assets Index

**Last updated**: 2026-03-21

Two papers planned. Paper 1 is the priority.

---

## Paper 1: Gene Structure from DNA Foundation Model Embeddings
**Plan**: `docs/paper1_plan.md`

### Main Figures

| Figure | Status | Key file |
|--------|--------|----------|
| Fig 1: Codon periodicity + universality | v5 DONE, needs final polish | `figures/fig1_v5.png` |
| Fig 2: Exon-intron detection | NOT GENERATED | `exon_intron/quantification_all.json` |
| Fig 3: Syntax vs semantics | NOT GENERATED | `codon_table_embeddings/`, `functional_clustering_comparison.json` |
| Fig 4: Coding detection in context | NOT GENERATED | `codon_periodicity_panel/` |

### Supplementary

| Figure | Status | Key file |
|--------|--------|----------|
| S4: E. coli genome circular | DONE | `genome_scan/ecoli_k12_circular_v2.png` |
| S5: All 36 exon-intron profiles | DONE | `exon_intron/figures/*.png` |
| S1-S3, S6 | NOT GENERATED | Data exists |

### Key Data Files

| File | Description |
|------|-------------|
| `codon_periodicity_panel/embeddings/*_metrics.json` (489) | Periodicity metrics (7B) |
| `codon_periodicity_panel/multi_offset_expanded.json` | Multi-offset comb filter (50 seqs, 40B) |
| `codon_table_embeddings/codon_embeddings.json` | 64 codon-repeat embeddings (40B) |
| `codon_periodicity_panel/functional_clustering_comparison.json` | Negative result (40B+7B) |
| `exon_intron/quantification_all.json` | 36 genes quantified |
| `exon_intron/annotations_all.json` | Gene annotations |
| `exon_intron/smoothing_optimization.json` | Window optimization |
| `exon_intron/metrics/*.json` (36) | Per-position cosine for each gene |
| `exon_intron/figures/*.png` (36) | Exon-intron profiles |
| `genome_scan/window_metrics.npz` (100MB) | Full E. coli genome scan |
| `figures/fig1_data/` | Lac operon data for Fig 1 Panel C |

---

## Paper 2: ViroSense
**Plan**: `docs/paper2_plan.md`

### Key Data Files

| File | Description |
|------|-------------|
| `benchmark/40b_complete/` | 13,417-seq benchmark |
| `benchmark/comparison/` | ViroSense vs geNomad + bootstrap CIs |
| `rna_dark_matter_v2/` | Recomputed RNA dark matter (95.2%) |
| `classifiers/40b/` | Production classifier |

---

## Superseded Files (ignore)

| File | Reason |
|------|--------|
| `figures/fig1_v3.png`, `fig1_v4.png` | Superseded by v5 |
| `figures/fig1_genetic_code*.png` | Superseded by v5 |
| `figures/fig1a_norm_trajectory.png` | Exploratory |
| `figures/fig1_options_exploration.png` | Exploratory |
| `figures/fig_training_summary.png` | Legacy |
| `figures/fig2_applications.png` | Virology-focused, → Paper 2 |
| `figures/fig3_multitask.png` | Virology-focused, → Paper 2 |
| `genome_scan/ecoli_k12_circular.png` | Superseded by v2 |
| `genome_scan/ecoli_k12_genome_scan.png` | Linear plot, replaced by circular |
| `universal_validation_v2/figures/*` | From 72-seq pilot, superseded |
| `docs/archive_publication_plan_v4.md` | Historical, superseded by paper1_plan.md |
