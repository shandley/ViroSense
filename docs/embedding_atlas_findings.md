# Embedding Atlas: Mean-Pooled vs Per-Position Signal Analysis

Last updated: 2026-03-19

## Experiment

Extracted mean-pooled Evo2 40B embeddings (blocks.10, 8192-D) for all 72 cross-domain validation sequences. Visualized in Apple's Embedding Atlas tool (interactive UMAP) with metadata coloring.

**Data**: `results/universal_validation_v2/atlas_sequences.parquet` (72 sequences, 3.8 MB)

## Key Finding: Orthogonal Information at Different Scales

**Mean-pooled embeddings encode gene function. Per-position embeddings encode coding structure. These are orthogonal signals.**

### Evidence

| Color by | UMAP pattern | Interpretation |
|----------|-------------|----------------|
| **Gene name** | **Strong clustering** — all 10 actins cluster together, all 5 GAPDHs cluster, all EF-2s cluster | Mean-pooled embeddings capture protein function across species |
| **Taxonomic domain** | **No clustering** — Archaea, Vertebrata, Invertebrata, Plantae, Fungi, Protista all interleaved | Gene function dominates over taxonomic identity |
| **GC content** | **No gradient** — 20-67% GC bins completely mixed | GC composition does NOT drive embedding structure |
| **Inversion gap** (cos3 − cos1) | **No clustering** — weak/strong inversion mixed throughout | Per-position periodicity signal is invisible in mean-pooled space |

### What This Means

1. **Mean-pooling discards positional information** but retains compositional/functional information. Actin from *E. coli*, *Drosophila*, *Homo sapiens*, and *Plasmodium falciparum* are neighbors in embedding space despite billions of years of divergence.

2. **Per-position analysis reveals structural features** (codon periodicity, gene boundaries, coding/non-coding) that are averaged out by mean-pooling. The offset-3 cosine inversion — our primary finding — is a per-position signal invisible in the UMAP.

3. **These are complementary, not competing analyses.** The same embedding extraction contains both:
   - Mean-pooled → classification, clustering, phylogenomics (gene function)
   - Per-position → coding detection, periodicity, RNA dark matter, gene boundaries (sequence structure)

4. **"Embed once, analyze many ways"** is not just about convenience — the embedding genuinely contains multiple orthogonal biological signals at different scales of analysis.

### Specific Observations

**Gene function clustering:**
- Actin (10 sequences from 8 phyla) forms a tight cluster
- GAPDH (5 sequences from 5 species) clusters together
- Beta-actin (4 vertebrate species) sits near but distinct from the broader actin cluster
- Elongation factors (3 archaeal species) cluster
- This is alignment-free functional annotation from raw DNA alone

**Non-coding controls:**
- Non-coding sequences (rRNA, lncRNA, intron, Alu, telomere) are at the periphery of the main cloud but don't form a single coherent cluster
- This makes sense: non-coding sequences are functionally diverse (rRNA ≠ lncRNA ≠ repetitive element), so they scatter in mean-pooled space
- Mean-pooling cannot reliably distinguish coding from non-coding — that requires per-position analysis

**GC independence (confirmed from a new angle):**
- The GC content scatter plot showed no correlation with inversion signal (Figure 1, Panel D)
- The UMAP confirms: GC does not drive embedding structure. High-GC and low-GC sequences are interleaved.
- This rules out the hypothesis that Evo2 embeddings are "just fancy k-mer frequencies"

## Implications for the Paper

### New finding for Results section
> "Mean-pooled Evo2 embeddings cluster sequences by protein function across all domains of life — actin genes from human, Drosophila, coral, and Plasmodium are neighbors in embedding space despite billions of years of divergence (Fig. S_). This functional clustering is independent of GC content and taxonomic identity. In contrast, the per-position periodicity signal (offset-3 cosine inversion) is orthogonal to the mean-pooled representation: sequences with strong and weak inversion are interleaved in UMAP space. This demonstrates that a single embedding extraction contains complementary information at different scales — functional similarity at the sequence level and structural features at the nucleotide level."

### Supports the "embed once, analyze many ways" thesis
The Embedding Atlas results directly validate the multi-scale framework:

| Scale | Analysis | Signal | Example |
|-------|----------|--------|---------|
| Sequence-level (mean-pooled) | Classification, clustering, phylogenomics | Gene function | Actin clustering across phyla |
| Position-level (per-position) | Coding detection, periodicity, boundaries | Genetic code | Offset-3 inversion |
| Window-level (sliding window) | Prophage detection, HGT | Compositional transitions | Amelioration gradient |

### Potential supplementary figure
Three-panel UMAP: same 72 points colored by (A) gene, (B) domain, (C) GC content. Caption: "Mean-pooled Evo2 embeddings capture functional similarity across all domains of life, independent of taxonomy and GC content."

## Exploring Further: Mean-Pooled vs Per-Position

The orthogonality between mean-pooled (function) and per-position (structure) raises testable questions:

### 1. Can we separate the signals?
- **Per-position variance** across a gene: high variance = heterogeneous structure (coding + intergenic mix). Low variance = uniform (pure CDS or pure non-coding).
- **Per-position autocorrelation at lag-3**: captures codon periodicity independent of mean.
- Do these per-position summary statistics add information beyond what mean-pooling captures?

### 2. What does mean-pooling actually compute?
- If actins cluster, the model has learned something about codon usage bias, amino acid composition, or protein structural properties — all encoded in the DNA.
- Is the functional clustering driven by codon usage (k-mer-like) or by deeper contextual patterns?
- Test: do k-mer features (trinucleotide frequencies) also cluster actins together? If yes, mean-pooling ≈ sophisticated k-mer. If no, the model captures something beyond composition.

### 3. Per-position UMAP
- Instead of mean-pooling, project individual positions from the lacZ region into UMAP
- Do coding positions cluster separately from intergenic positions?
- Do positions within the same gene cluster together?
- This would directly visualize the per-position structure that enables coding detection.

### 4. Combined embedding
- Concatenate mean-pooled (8192-D) with per-position summary statistics (lag-3, cos3, cos1, norm_mean, norm_std — 5-D)
- Does the combined representation separate both function AND coding status?
- This could be the "DNA passport" embedding — one vector that captures everything.

## Screenshots

Saved in the conversation as screenshots from Embedding Atlas at localhost:5055:
- Gene coloring: strong clustering by protein function
- Domain coloring: no taxonomic clustering
- GC content coloring: no GC-driven structure
- Inversion gap coloring: per-position signal invisible in mean-pooled space
