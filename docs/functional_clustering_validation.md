# Functional Clustering Validation Experiments

Last updated: 2026-03-19

## The Claim

Mean-pooled Evo2 40B embeddings cluster protein-coding DNA sequences by **protein identity** across all domains of life, with function dominating over taxonomy and nucleotide composition.

## Current Evidence (Preliminary — Quantified 2026-03-19)

**UMAP (qualitative)**: Actins, GAPDHs, EFs visually cluster in Embedding Atlas. No taxonomic or GC structure.

**Quantitative (N=72, 9 gene families with N≥2)**:

| Metric | Evo2 (PCA-50) | K-mers (3-mer) | Evo2/K-mer ratio |
|--------|---------------|----------------|-----------------|
| Silhouette (gene) | **0.003** | -0.238 | Better |
| NN gene accuracy (multi-gene) | **23.5%** | 5.9% | **4.0×** |
| NN gene accuracy (all) | 11.1% | 2.8% | 4.0× |

**Within-family / between-family cosine distances**:
- Actin (N=10): 1.11× (Evo2), 1.19× (k-mer)
- RuBisCO large (N=3): **2.90×** (Evo2)
- GAPDH-like (N=2): **1.77×** (Evo2)
- Most families: 1.05-1.3× (modest but consistent)

**Assessment**: Signal is **real but weak** at N=72. Evo2 consistently outperforms k-mers (4× NN accuracy) but absolute performance is low. UMAP visualization amplified a modest signal into apparent tight clusters. Needs expanded gene families (Exp 2) to determine if this is a publishable finding or a supplementary observation.

## What Reviewers Will Ask

1. "Is this just codon usage bias?" — k-mers capture codon usage. If trinucleotide frequencies also cluster by gene, the embedding isn't doing anything special.
2. "N=72 is small." — need more sequences per gene family to quantify clustering robustly.
3. "UMAP is qualitative." — need quantitative metrics (silhouette, nearest-neighbor accuracy).
4. "Is this driven by sequence length?" — genes of similar length might cluster for trivial reasons.
5. "Is this specific to Evo2, or any DNA model?" — ideally test on Nucleotide Transformer or DNABERT-2.

## Validation Experiments

### Experiment 1: K-mer Baseline (CRITICAL — no API calls needed)

**Question**: Do trinucleotide frequencies also cluster by gene identity?

**Method**:
- Compute 64-D trinucleotide frequency vectors for all 72 sequences (existing FASTA files)
- Run UMAP, color by gene
- Compute silhouette scores for gene-based clustering
- Compare to Evo2 embedding silhouette scores

**Expected outcome**: If k-mers cluster by gene → the functional signal is partially compositional (but Evo2 may still be better). If k-mers do NOT cluster → Evo2 captures something beyond composition.

**Effort**: 30 minutes, no API calls, runs locally.

### Experiment 2: Expanded Gene Families (needs API calls — run on HTCF)

**Question**: With more examples per gene family, does the clustering hold statistically?

**Method**:
- Select 5 gene families with many orthologs in RefSeq: actin, GAPDH, EF-Tu/EF-1a, HSP70, cytochrome b
- Download 20-30 orthologs per family from diverse species (NCBI Orthologs database)
- Total: ~100-150 new CDS sequences
- Extract mean-pooled Evo2 embeddings (blocks.10)
- Compute: silhouette score, nearest-neighbor gene identity accuracy (leave-one-out), ARI

**Expected outcome**: Silhouette > 0.3 for gene-based clusters would be strong. Nearest-neighbor accuracy > 80% would be very strong.

**Effort**: 1-2 days including NCBI download + HTCF extraction.

### Experiment 3: Quantitative Metrics on Current 72 Sequences

**Question**: How well do the current 72 sequences cluster by gene?

**Method**:
- Using existing mean-pooled embeddings (parquet file, 72 sequences)
- Compute pairwise cosine distance matrix
- For each sequence: is its nearest neighbor the same gene? → nearest-neighbor accuracy
- Silhouette score using gene labels (only for genes with N≥2)
- Compare: raw 8192-D vs PCA-reduced vs UMAP-2D

**Expected outcome**: With only 72 sequences and many singleton genes, metrics will be noisy. But for actin (N=10), GAPDH (N=5), EF (N=3), we can compute within-family vs between-family distances.

**Effort**: 1 hour, no API calls, runs locally.

### Experiment 4: Protein LM Comparison

**Question**: How does Evo2 DNA-only functional clustering compare to ESM-2 protein embedding clustering?

**Method**:
- Translate our 72 CDS sequences to protein
- Compute ESM-2 embeddings for the 72 proteins
- Compare UMAP structure, silhouette scores, NN accuracy
- If Evo2 DNA ≈ ESM-2 protein → Evo2 has learned protein-level information from DNA
- If Evo2 DNA < ESM-2 protein → expected, but the gap matters

**Effort**: Requires ESM-2 (can run locally on M4, or use ESM API). 2-3 hours.

### Experiment 5: Per-Position UMAP of a Single Gene Region

**Question**: Do coding and intergenic positions separate in per-position embedding space?

**Method**:
- Re-extract per-position embeddings for lacZ region (6kb, blocks.10) — on HTCF
- Subsample to every 3rd position (2000 points)
- UMAP colored by coding/intergenic/gene identity
- This directly visualizes the per-position signal

**Effort**: 1 extraction (~30s on NIM) + analysis. Must be on HTCF for disk space.

### Experiment 6: Shuffled Sequence Controls

**Question**: Does shuffling the nucleotide order destroy the functional clustering?

**Method**:
- For each of the 72 sequences, create a dinucleotide-preserving shuffle (maintains composition)
- Extract mean-pooled embeddings for shuffled sequences
- Compare: real vs shuffled silhouette scores and NN accuracy
- If shuffling destroys clustering → signal is context-dependent, not just compositional

**Effort**: 72 new API calls (~12 min on NIM). Runs locally.

## Execution Priority

| Priority | Experiment | Why first | Effort | API calls |
|----------|-----------|-----------|--------|-----------|
| **1** | Exp 3: Quantify current 72 | Immediate metrics, no new data | 1 hour | None |
| **2** | Exp 1: K-mer comparison | Critical control, no API calls | 30 min | None |
| **3** | Exp 6: Shuffled controls | Tests composition vs context | 12 min | 72 calls |
| **4** | Exp 2: Expanded gene families | Statistical power | 1-2 days | ~150 calls (HTCF) |
| **5** | Exp 4: ESM-2 comparison | Protein LM benchmark | 2-3 hours | ESM API |
| **6** | Exp 5: Per-position UMAP | Visualizes the other signal | 30 min | 1 call (HTCF) |

## Minimum Viable Validation for the Paper

Experiments 1-3 are the minimum needed to support the claim. Together they answer:
- Is the clustering real? (Exp 3: quantitative metrics)
- Is it beyond k-mers? (Exp 1: trinucleotide comparison)
- Is it context-dependent? (Exp 6: shuffled controls)

Experiments 4-6 strengthen the paper but aren't strictly required.

## What Would Kill the Claim

- If k-mers cluster actins equally well → the signal is just codon usage, not learned context
- If shuffled sequences cluster identically → the signal is compositional, not structural
- If nearest-neighbor accuracy is < 50% → the clustering is noise in UMAP visualization

## What Would Make It a Nature Paper

- Exp 1 shows k-mers DON'T cluster by gene (or much worse) → model captures something beyond composition
- Exp 2 shows NN accuracy > 80% on expanded families → robust, quantified
- Exp 4 shows Evo2 DNA ≈ ESM-2 protein for functional clustering → the model bridges the central dogma
- Exp 6 shows shuffling destroys clustering → context-dependent, not compositional
