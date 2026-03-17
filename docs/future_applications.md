# Future Applications of DNA Foundation Model Embeddings

Last updated: 2026-03-17

## Core Insight

Evo2 learns fundamental biological structure from unsupervised next-nucleotide prediction:

1. **The genetic code**: Per-position embeddings show 3bp codon periodicity as the dominant signal (lag-3 autocorr 0.635, universal across all 40 tested sequences). Offset-3 cosine inversion detects coding regions at 94.7% accuracy.
2. **Gene structure**: Coding regions have 1.72× higher embedding norms than intergenic (41 sequences across 5 categories).
3. **Compositional signatures**: RNA viruses have distinctively strong periodicity (0.822 vs 0.624 for phages) enabling database-free identification.
4. **Cross-domain generalization**: 93.0% RNA virus recall without RNA virus training data; HDBSCAN separates euk. RNA viruses from dsDNA phages unsupervised (ARI=0.903).

This makes the embeddings a **universal DNA feature extractor** at two levels:
- **Mean-pooled**: sequence-level classification, clustering, detection
- **Per-position**: gene structure, codon analysis, boundary detection, compositional segmentation

Every ViroSense module maps to a general pattern.

## Direct Generalizations

### Prophage detector -> Any genomic island detector

The sliding window + binary classifier is a "detect foreign DNA in a chromosome" framework. With different training labels, the same architecture detects:

- **Pathogenicity islands** — virulence gene clusters acquired by HGT
- **Resistance islands** — AMR cassettes in clinical isolates
- **Metabolic islands** — xenobiotic degradation clusters
- **ICEs** (integrative conjugative elements) — self-transmissible elements

The adaptive coarse-to-fine scanning is ideal here — bacterial chromosomes are mostly "self," with sparse islands. Directly publishable as a general HGT detection tool.

### Viral detect -> Plasmid detection

Plasmid vs. chromosomal classification is the same binary problem. Plasmids have distinct replication origins, mobilization genes, and compositional signatures. The existing `detect` module with different reference training data becomes a plasmid detector. Tools like PlasFlow and MOB-suite do this with k-mer frequencies — Evo2 embeddings likely capture much richer features.

### Clustering module -> Metagenome binning

Assigning metagenomic contigs to taxonomic bins is currently done with tetranucleotide frequencies + coverage (MetaBAT2, CONCOCT, VAMB). The clustering module already does embedding-based binning with HDBSCAN/Leiden. Evo2 embeddings encode far more information than 4-mer frequencies. A direct comparison against VAMB (which uses a VAE on k-mers) would be interesting.

## Novel Applications

### Alignment-free phylogenomics

If Evo2 embeddings capture evolutionary relationships (which the clustering results suggest — Caudoviricetes grouped separately from novel viruses), then:

- **Rapid taxonomic placement** without BLAST or alignment
- **Embedding-space phylogenies** — cosine distance as a proxy for evolutionary distance
- **Cross-kingdom comparisons** — compare sequences that are too divergent for alignment

Fundamentally new approach to phylogenetics. Key question: does embedding distance correlate with evolutionary distance, and at what divergence levels does it break down? Testable with known phylogenies.

### Dark matter characterization

Metagenomes are 40-90% "dark matter" — sequences matching nothing in databases. Evo2 embeddings provide a feature space for characterizing sequences **without any reference database**. Can ask "what is this sequence most similar to compositionally?" even when BLAST returns nothing.

### Ancient DNA / Contamination detection

Ancient DNA has systematic compositional differences from modern DNA (C->T deamination, fragmentation patterns). A classifier trained on ancient vs. modern sequences could detect contamination in paleogenomic datasets — a persistent problem in the field.

## What Makes This Framework Unique

Every existing tool for these problems relies on **hand-crafted features** (k-mer frequencies, GC content, codon adaptation index) or **database homology** (BLAST, HMMs, marker genes). This approach:

1. **No feature engineering** — the foundation model learns features
2. **No reference database** — works on novel/divergent sequences
3. **Modular** — same embedding extraction, different downstream head
4. **Runs locally** — the MLX backend democratizes access

## Already Realized (March 2026)

| Proposed Application | Status | Result |
|---------------------|--------|--------|
| **Plasmid detection** | ✅ DONE | 3-class classifier: 91.5% plasmid detection, 99.2% specificity |
| **Dark matter characterization** | ✅ DONE | RNA virus dark matter: 91.3% classification from periodicity alone |
| **Prophage → genomic island** | ✅ PILOT | E. coli K12 prophage: 37bp start accuracy, clean signal |
| **Metagenome binning** | ✅ TESTED | HDBSCAN ARI=0.903; fragment coherence 100%; but genome-resolution limited |

## Recommended Next Extensions

**1. Perplexity forensics** — Per-position embedding norms are a proxy for sequence information density. Compare natural vs codon-optimized genes. Infrastructure exists (`virosense scan`). See `docs/biosurveillance_research_plan.md`.

**2. HGT/genomic island detection** — Direct reuse of the prophage architecture. Per-position norm transitions at foreign DNA boundaries provide single-nucleotide resolution. Enormous demand in clinical microbiology, no foundation-model tool doing this.

**3. Alignment-free phylogenomics** — Embedding distance as evolutionary distance proxy. Testable now with known phylogenies from our benchmark data.
