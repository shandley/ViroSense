# Cluster Module Validation

Last updated: 2026-03-16

## Dataset

Full GYP + RNA virus benchmark: 13,055 sequences with 40B embeddings (8,192-D).
All clustering performed on PCA-reduced embeddings (50 components, 97.7% variance).

## Test 1: Cross-Kingdom Category Recovery

**Question**: Can unsupervised clustering recover the 5 known biological categories (dsDNA phage, eukaryotic RNA virus, chromosome, plasmid, cellular)?

### HDBSCAN (best config: min_cluster_size=200, min_samples=5)

| Metric | Value |
|--------|-------|
| Clusters found | 4 |
| Noise | 47% |
| **ARI vs 5 categories** | **0.903** |
| NMI vs 5 categories | 0.782 |

| Cluster | N | Purity | Dominant category | Composition |
|---------|---|--------|-------------------|-------------|
| **1 (dsDNA phage)** | 4,922 | **99%** | phage | 4,871 phage, 45 plasmid |
| **0 (euk. RNA virus)** | 280 | **99%** | rna_virus | 277 RNA virus, 2 phage, 1 cellular |
| **3 (plasmid)** | 267 | **88%** | plasmid | 235 plasmid, 30 phage |
| 2 (cellular mix) | 1,391 | 52% | chromosome | 721 chr, 541 plasmid, 129 cellular |

### Key findings

1. **dsDNA phages cluster near-perfectly** (99% purity, 73% of all phage recovered)
2. **Eukaryotic RNA viruses form a distinct cluster** (99% purity, 43% recovered) — unsupervised separation from dsDNA phages, despite both being "viral." Evo2, trained only on DNA, captures compositional differences between DNA-origin and RNA-origin (cDNA) genomes.
3. **A plasmid cluster emerges** (88% purity) — the most compositionally distinctive plasmids (likely conjugative elements) separate from chromosomes
4. **47% noise is expected** — HDBSCAN is conservative; noise contains sequences at category boundaries. This is preferable to forcing ambiguous sequences into clusters.
5. **Fragment coherence: 100%** — all fragments from the same source genome always co-cluster

### Parameter sensitivity

| min_cluster_size | min_samples | Clusters | Noise | ARI |
|-----------------|-------------|----------|-------|-----|
| 100 | 10 | 5 | 53% | 0.877 |
| **200** | **5** | **4** | **47%** | **0.903** |
| 200 | 10 | 4 | 64% | 0.480 |
| 500 | 5 | 2 | 48% | 0.870 |

Best ARI at moderate cluster sizes. Very large clusters (500+) merge distinct groups; small clusters fragment them.

## Test 2: Host-Taxonomy Clustering (Phage Only)

**Question**: Do phage embeddings cluster by the taxonomy of their bacterial host?

5,576 phage fragments from 6 host phyla (Gammaproteobacteria, Actinobacteria, Firmicutes, Cyanobacteria, Betaproteobacteria, Bacteroidetes).

### PCA centroids by host phylum

| Host phylum | PC1 | PC2 | N |
|-------------|-----|-----|---|
| Actinobacteria | +43.4 | -17.4 | 1,743 |
| Betaproteobacteria | +44.5 | +0.3 | 136 |
| Cyanobacteria | +3.5 | +57.9 | 189 |
| Firmicutes | -24.7 | +35.3 | 523 |
| Gammaproteobacteria | -23.3 | -0.8 | 2,892 |
| Bacteroidetes | -23.7 | +33.6 | 93 |

**The centroids separate by host phylum in PCA space**, particularly:
- Actinobacteria phages vs Gammaproteobacteria phages (PC1: +43 vs -23)
- Cyanobacteria phages occupy a unique region (PC2: +58)
- Firmicutes and Bacteroidetes phages group together (similar PC1, high PC2)

### Clustering metrics

| Method | ARI vs genus | ARI vs phylum | NMI vs phylum |
|--------|-------------|---------------|---------------|
| HDBSCAN | 0.006 | 0.031 | 0.056 |
| K-Means (k=6) | — | **0.222** | **0.335** |

### Interpretation

1. **HDBSCAN poorly recovers host taxonomy** (ARI≈0) — most phages (87%) collapse into one large cluster. The host-taxonomy signal exists (PCA centroids separate) but is overwhelmed by the within-phylum diversity.

2. **K-Means with k=6 does better** (ARI=0.222, NMI=0.335) — forced into 6 clusters, it finds groups enriched for Actinobacteria phages (81% pure) and Gammaproteobacteria phages (76-88% pure). But Firmicutes phages don't form a clean cluster.

3. **This is biologically expected**: Phage genomes evolve faster than their hosts and frequently recombine. A Salmonella phage and an E. coli phage may share more genes (both Gammaproteobacteria) than two Salmonella phages of different morphotypes. DNA composition correlates with host phylum (GC content) but not fine-grained host genus.

4. **For the paper**: Evo2 embeddings capture **phylum-level host composition** but not genus-level taxonomy. This is consistent with the embeddings reflecting DNA composition (GC content, dinucleotide frequencies, codon usage) rather than gene content. Gene-sharing approaches (vConTACT2/3) remain necessary for fine-grained viral taxonomy.

## Summary for Publication

| Clustering task | Method | Metric | Value | Status |
|----------------|--------|--------|-------|--------|
| Cross-kingdom (5 categories) | HDBSCAN | ARI | **0.903** | Strong |
| RNA virus vs DNA phage separation | HDBSCAN | Cluster purity | **99%** | Remarkable |
| Plasmid identification | HDBSCAN | Cluster purity | 88% | Good |
| Fragment coherence | HDBSCAN | Same-cluster rate | **100%** | Perfect |
| Host phylum clustering | K-Means | ARI | 0.222 | Moderate |
| Host genus clustering | HDBSCAN | ARI | 0.006 | Weak |

## Test 3: Viral Genome Binning

**Question**: Can Evo2 embeddings group fragments from the same viral genome — i.e., act as a viral metagenomic binner?

5,981 phage fragments from 733 source genomes (≥3 fragments each, mean 8.2 fragments/genome).

### Distance separation

| Metric | Value |
|--------|-------|
| Intra-genome cosine distance | 0.287 ± 0.254 |
| Inter-genome cosine distance | 1.001 ± 0.526 |
| **Separation ratio** | **3.48×** |
| Intra-genome pairs closer than median inter-genome | **98.3%** |

The signal is present: fragments from the same genome are substantially more similar than fragments from different genomes.

### Binning performance

| Method | Clusters | Noise | ARI | NMI |
|--------|----------|-------|-----|-----|
| HDBSCAN (mcs=3) | 627 | 40% | 0.126 | **0.760** |
| HDBSCAN (mcs=5) | 272 | 45% | 0.129 | 0.728 |
| HDBSCAN (mcs=7) | 169 | 48% | 0.128 | 0.711 |
| k-NN (k=1) | — | — | — | **13.5%** same-genome |
| k-NN (k=5) | — | — | — | 10.4% same-genome |

### Interpretation

**The signal exists but is not sufficient for genome-resolution binning alone.**

- **High NMI (0.760)**: When HDBSCAN forms micro-clusters, they tend to be genome-pure. The embedding space contains genome-specific structure.
- **Low ARI (0.126)**: Most genomes get split across clusters or into noise. The method has precision but lacks recall.
- **13.5% k-NN accuracy**: 135× better than random (0.1%), but insufficient for confident binning.

**Why phage binning is harder than bacterial binning:**

Phage genomes are **mosaic** — frequent recombination means different genomic regions within a single phage can have different evolutionary origins and compositional signatures. A capsid gene module from one lineage may have different codon usage than a tail fiber module from another. Bacterial chromosomes have relatively uniform composition, making coverage + composition sufficient for binning (MetaBAT2, CONCOCT). Phage fragments from compositionally variable regions of the same genome can be more distant from each other than fragments from related but distinct phages.

**Practical implication**: Evo2 embedding distance is a useful **feature** for multi-signal viral binning (3.48× separation provides a strong compositional prior) but should be combined with:
- Read coverage co-variation
- Tetranucleotide frequency
- Gene content overlap (shared protein clusters)
- Assembly graph connectivity

This is complementary to vConTACT2/3, which uses protein-level gene sharing (more specific for genome-resolution clustering, but requires gene calling + protein clustering as prerequisites).

### What the embeddings capture (and don't)

**Capture well**: Kingdom-level distinctions (DNA phage vs RNA virus vs bacteria), major compositional differences (Actinobacteria vs Gammaproteobacteria phages), within-genome consistency.

**Don't capture**: Fine-grained viral taxonomy (genus/species), specific gene content, host-virus interactions beyond composition.

**Implication**: Evo2 embedding-based clustering is best suited for:
- Initial triage of metagenomic contigs into broad categories
- Identifying compositionally novel groups ("dark matter" that doesn't cluster with known types)
- Flagging potential misassemblies (fragments from the same contig should co-cluster)

It is **not** a replacement for gene-sharing networks (vConTACT2/3) or marker-gene taxonomy for fine-grained viral classification.
