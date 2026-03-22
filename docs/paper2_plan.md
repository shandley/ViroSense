# Paper 2: ViroSense — Viral Detection and Characterization

**Title**: "ViroSense: DNA foundation model embeddings for viral detection, characterization, and RNA dark matter discovery"

**Target**: Nature Methods or Genome Biology

**Depends on**: Paper 1 (establishes the per-position periodicity signal)

**Last updated**: 2026-03-21

---

## Thesis

Building on the per-position periodicity signal discovered in Paper 1, ViroSense applies frozen Evo2 embeddings to viral metagenomics — achieving 99.7% phage sensitivity, database-free RNA dark matter detection (95.2%), and prophage evolutionary characterization from DNA composition alone.

---

## Key Results (all data complete)

| Finding | Value | Source |
|---------|-------|--------|
| Viral detection (40B) | 95.4% accuracy, 99.7% phage, 93% RNA virus | `benchmark/40b_complete/` |
| ViroSense vs geNomad | Complementary: ViroSense wins short frags, geNomad wins speed | `benchmark/comparison/` |
| RNA dark matter | 95.2% accuracy, 0.982 AUC (periodicity features) | `rna_dark_matter_v2/` |
| Prophage amelioration | DLP12 active → CP4-6 mosaic → e14 invisible | `docs/prophage_reassessment.md` |
| K-mer baseline | 93% at 1,527× speed | docs/ |
| Two-tier pipeline | K-mer screening → Evo2 characterization | Implemented in CLI |
| 3-class contig typing | 94.5% (virus/plasmid/chromosome) | `benchmark/40b_3class/` |
| L2-normalization | Essential for 7B (63%→93% RNA recall) | `docs/rna_virus_length_analysis.md` |
| HDBSCAN clustering | ARI=0.903, RNA virus cluster 99% pure | `docs/cluster_validation.md` |
| Phylogenomics | r=0.504 (embedding distance vs taxonomy) | `docs/applications_framework.md` |
| Taxonomy pilot | 4.34× within-genome fragment ratio | `docs/findings_log.md` Finding 14 |

---

## Figures (to be designed)

### Figure 1: ViroSense Detection Performance
- Benchmark on 13,417 sequences
- Head-to-head vs geNomad with bootstrap CIs
- Performance by fragment length

### Figure 2: RNA Dark Matter Detection
- Periodicity features distinguish RNA viruses
- ROC curve (0.982 AUC)
- Database-free, no homology needed

### Figure 3: Characterization Beyond Detection
- Prophage amelioration gradient
- DNA passport framework
- Contig typing

### Figure 4: Multi-Task from One Embedding
- Clustering (ARI=0.903)
- Phylogenomics (r=0.504)
- K-mer comparison and two-tier pipeline

---

## Status

All data exists. No new experiments needed. Paper can be written after Paper 1 is submitted. ViroSense tool is functional and deployed on HTCF.
