# Proof-of-Concept: Gene Boundary Detection from Evo2 Per-Position Embeddings

Last updated: 2026-03-16

## Summary

Evo2's per-position hidden-state representations implicitly encode gene structure. Without any gene-calling training, a simple norm-based analysis on a single phage fragment achieves **87.9% coding/non-coding accuracy** and **82.4% gene boundary recall** with median precision of 6-15 bp.

## Background

The current ViroSense pipeline extracts per-position embeddings from Evo2 (shape: `seq_len × hidden_dim`) and immediately mean-pools them to a single vector (`hidden_dim`). This discards all spatial information. We asked: does the per-position data encode biologically meaningful features like gene boundaries?

## Methods

- **Sequence**: Erwinia phage Hena1 fragment (5,299 bp, positions 11913-17212)
- **Model**: Evo2 40B via NIM cloud API, layer `blocks.20.mlp.l3`
- **Per-position embeddings**: 5,299 × 8,192 (43.4M values)
- **Ground truth**: Pyrodigal-gv gene calls (10 genes, all minus strand)
- **No training**: All analyses use raw Evo2 representations, no fine-tuning

## Key Findings

### 1. Embedding Norm Distinguishes Coding from Intergenic Regions

| Region type | Mean embedding norm | Adjacent cosine similarity |
|-------------|--------------------|-----------------------------|
| **Coding** | **236.9** | 0.238 |
| **Intergenic** | **166.4** | 0.421 |
| **Gene boundaries (±20bp)** | — | 0.346 |
| **Ratio (coding/intergenic)** | **1.42×** | **0.57×** |

Interpretation:
- **Coding regions** produce higher-norm embeddings — the model recognizes higher information density from codon structure and amino acid constraints
- **Intergenic regions** have more self-similar adjacent embeddings (higher cosine) — repetitive/low-complexity DNA produces more uniform representations
- **Gene boundaries** show intermediate cosine similarity — transition zones

### 2. Coding/Non-Coding Prediction

Simple threshold on smoothed embedding norm (window=30bp):
- **Accuracy: 87.9%**
- Sensitivity (coding detected): 88.5%
- Specificity (intergenic detected): 75.3%

This uses zero training — just a threshold derived from the coding/intergenic norm means on this single sequence.

### 3. Gene Boundary Detection

Peaks in the derivative of the smoothed norm signal:
- **14/17 actual boundaries detected** (82.4% recall)
- 22 peaks detected, 14 true positives (63.6% precision)
- Median distance to true boundary: **6 bp**

Selected matches:
| True boundary | Detected peak | Distance |
|--------------|---------------|----------|
| 469 | 469 | **0 bp** (exact) |
| 2258 | 2259 | **1 bp** |
| 1941 | 1935 | 6 bp |
| 170 | 164 | 6 bp |
| 2803 | 2818 | 15 bp |
| 689 | 707 | 18 bp |
| 676 | 707 | 31 bp |

3 missed boundaries were at positions 0, 3191, and 3688 — sequence edges and densely packed gene junctions.

## Implications

### For the ViroSense Paper

1. **Evo2 learns gene structure implicitly** from next-nucleotide prediction. This is a significant finding about DNA foundation models — they capture not just compositional patterns but structural features of genomes.

2. **Mean pooling discards rich spatial information.** The current classification pipeline collapses 43M values to 8K. Per-position analysis could enable:
   - Gene calling without Prodigal
   - Provirus boundary detection at nucleotide resolution
   - Functional region annotation from DNA alone
   - Chimeric contig detection

3. **No additional model training needed.** These signals come directly from the pre-trained Evo2 representations. The foundation model approach means new capabilities emerge from analysis of existing representations.

### For Future Work

- **Train a per-position classifier**: With labeled gene boundaries from thousands of genomes, a lightweight 1D-CNN on the per-position embeddings could rival Prodigal for gene calling.
- **Provirus boundary detection**: The norm transition signal (coding → intergenic at phage/host junctions) could replace the current sliding-window approach.
- **Functional annotation**: Different gene types (capsid, tail fiber, integrase) may produce characteristic local embedding patterns.
- **The "one model" thesis**: Evo2 embeddings simultaneously support classification (mean-pooled), gene calling (per-position norm), and potentially functional annotation — all from a single forward pass.

## Cross-Sequence Validation

Tested on 3 sequences of different types and organisms:

| Sequence | Length | Genes | Coding norm | Intergenic norm | Ratio | Coding cosine | Intergenic cosine |
|----------|--------|-------|------------|----------------|-------|--------------|------------------|
| Erwinia phage Hena1 | 5,299 bp | 10 | 236.9 | 166.4 | **1.42×** | 0.229 | 0.418 |
| Flavobacterium phage pippi8-1 | 7,296 bp | 9 | 202.3 | 140.2 | **1.44×** | 0.258 | 0.506 |
| E. coli Stbl4 chromosome | 4,959 bp | 6 | 238.1 | 169.6 | **1.40×** | 0.351 | 0.453 |

Key findings:
1. **Coding/intergenic norm ratio is remarkably consistent** (1.40–1.44×) across all sequences, organisms, and sequence types (phage vs chromosome)
2. **Works on bacterial chromosome** — not phage-specific, a general property of Evo2 representations
3. **Cosine similarity pattern is universal**: intergenic > boundary > coding interior
4. **Absolute norms differ** between organisms (Flavobacterium phage ~200 vs E. coli ~238) — likely reflects GC content and organism-specific composition captured by Evo2

### Boundary Detection on Erwinia Phage (best result)

Using norm derivative peak detection on the initial Erwinia phage fragment:
- **14/17 boundaries detected** (82.4% recall, 63.6% precision)
- Median distance to true boundary: **6 bp**
- One exact match (0 bp), several within 1-6 bp

The PC1-based approach failed on the other two sequences (PCA numerical overflow with raw 8192-D embeddings). The **norm-based approach is more robust** and doesn't require dimensionality reduction.

## Codon Periodicity — Evo2 Learns the Genetic Code

### Finding

Evo2 per-position embeddings exhibit **strong 3-nucleotide periodicity** within coding regions. Positions 3bp apart have ~0.55 autocorrelation, while adjacent positions (1-2bp) have near-zero or negative autocorrelation. This periodicity is absent in intergenic regions.

### Evidence (40 sequences, 5 categories)

**Lag-3 autocorrelation by category** (largest coding region per sequence):

| Category | Lag-3 autocorr | Lag-1 autocorr | N |
|----------|---------------|---------------|---|
| **RNA virus (cDNA)** | **0.822 ± 0.111** | 0.337 ± 0.292 | 8 |
| Plasmid | 0.649 ± 0.063 | 0.347 ± 0.162 | 5 |
| dsDNA phage | 0.624 ± 0.140 | 0.103 ± 0.164 | 15 |
| Cellular | 0.566 ± 0.147 | 0.135 ± 0.181 | 5 |
| Chromosome | 0.486 ± 0.079 | 0.174 ± 0.183 | 7 |
| **Overall** | **0.635 ± 0.157** | **0.197 ± 0.217** | **40** |

**3bp is the dominant FFT frequency** in 82% of sequences (33/40).

**RNA viruses show the strongest periodicity** (0.822) — Evo2, trained only on DNA, captures codon structure in cDNA-converted RNA genomes even better than in native DNA genomes. This likely reflects stronger codon usage bias in RNA viruses.

### Offset-3 Cosine Inversion

Cosine similarity between positions at different offsets reveals a **coding/non-coding signature**:

| Offset | Phage coding | Phage intergenic | E. coli coding | E. coli intergenic |
|--------|-------------|-----------------|---------------|-------------------|
| 1 bp | 0.238 | **0.421** | 0.353 | **0.454** |
| **3 bp** | **0.540** | 0.502 | **0.466** | 0.389 |
| 6 bp | **0.511** | 0.437 | **0.462** | 0.367 |

- At 1bp offset: coding < intergenic (within-codon diversity)
- At 3bp offset: coding > intergenic (**inverts** — same codon position)
- At 6bp offset: coding > intergenic (two codons apart, same pattern)

This **inversion at offset 3** is a binary signature of protein-coding DNA that requires no ORF analysis, no codon tables, and no training — it emerges purely from Evo2's learned representations.

**100% of sequences** (40/40) exhibit this inversion. It is universal across all tested sequence types.

### Coding Region Detection via Offset-3 Inversion

Using the inversion signal (local smoothed offset-3 cosine > offset-1 cosine) as a coding region detector achieves **94.7% ± 4.9% accuracy** across 40 sequences — **outperforming the norm-threshold method** (91.7% ± 5.2%):

| Category | Inversion accuracy | Norm-threshold accuracy | N |
|----------|-------------------|----------------------|---|
| Chromosome | **96.5% ± 2.5%** | 93.7% ± 3.1% | 7 |
| dsDNA phage | **96.6% ± 2.2%** | 91.2% ± 4.0% | 15 |
| Cellular | **95.7% ± 4.2%** | 94.4% ± 2.2% | 5 |
| RNA virus | **94.8% ± 2.7%** | 91.8% ± 6.3% | 8 |
| Plasmid | 85.3% ± 6.8% | 86.9% ± 8.7% | 5 |
| **Overall** | **94.7% ± 4.9%** | 91.7% ± 5.2% | 40 |

The codon periodicity signal is a **more discriminative coding detector** than the embedding norm alone, and captures a fundamentally different biological property (reading frame structure vs information density).

### Interpretation

Evo2 was trained to predict the next nucleotide in DNA sequences. To do this well in coding regions, it must implicitly learn:
1. **Codon structure** — the triplet reading frame constrains which nucleotides can follow
2. **Amino acid preferences** — codon usage bias and protein constraints
3. **Frame consistency** — positions in the same reading frame are more predictable from each other

This is the **genetic code emerging from unsupervised sequence modeling**. The model was never told about codons, yet its representations encode codon-level periodicity as the dominant signal.

### Systematic Follow-up (Experiments 1-4)

**Experiment 1: Reading Frame Detection** — Can we determine which of 3 frames is active?
- Result: **35.3% accuracy (essentially random, 33.3% baseline)**
- Neither same-frame cosine similarity nor FFT phase reliably encodes the specific frame
- The periodicity tells us "this is coding" but not "which frame" — the absolute frame position depends on the gene start, which is a local property not captured in the per-position embedding alone
- **Conclusion**: The codon signal detects coding regions, not reading frames

**Experiment 2: Combined Norm + Periodicity Detector**

| Method | Accuracy | Sensitivity | Specificity |
|--------|---------|-------------|-------------|
| Norm threshold only | 91.6% ± 5.2% | 91.7% ± 5.0% | 91.9% ± 8.8% |
| **Inversion only** | **94.7% ± 4.9%** | **98.8% ± 1.6%** | 64.7% ± 22.4% |
| Both agree (AND) | 91.6% ± 5.4% | 91.4% ± 5.4% | **93.5% ± 8.8%** |
| Either (OR) | 94.7% ± 4.8% | 99.1% ± 1.1% | 63.1% ± 22.0% |
| Weighted average | 90.7% ± 7.4% | 90.5% ± 7.7% | 94.6% ± 9.7% |

- Inversion alone has highest accuracy (94.7%) with near-perfect sensitivity (98.8%)
- Norm has balanced sensitivity/specificity — better for avoiding false coding calls
- **"Both agree" gives highest specificity (93.5%)** — use when precision matters
- The signals are complementary: periodicity detects coding; norms detect boundaries

**Experiment 3: RNA Virus Codon Periodicity**

| Category | Lag-3 autocorr | Codon diversity | GC% |
|----------|---------------|----------------|-----|
| **RNA virus** | **0.822 ± 0.111** | **0.953 ± 0.021** | 43.5% |
| Plasmid | 0.649 ± 0.063 | 0.941 ± 0.030 | 56.7% |
| dsDNA phage | 0.624 ± 0.140 | 0.857 ± 0.065 | 42.1% |
| Cellular | 0.566 ± 0.147 | 0.816 ± 0.146 | 37.0% |
| Chromosome | 0.486 ± 0.079 | 0.834 ± 0.105 | 58.7% |

- Codon diversity correlates positively with periodicity strength (Spearman r=0.477, p=0.001)
- **RNA viruses use the most diverse codons (61/64) AND have the strongest periodicity**
- This is counterintuitive: more diverse codons = more information per codon position = Evo2 must encode the structure more strongly
- GC content does NOT correlate with periodicity (r=-0.083, p=0.60)

**Experiment 4: Connection to Viral Detection**

The per-position periodicity strength does not linearly predict mean embedding norm (r=0.179, p=0.28). The relationship is more nuanced: per-position biological structure (codons, gene boundaries, composition) contributes to the mean embedding's character through complex non-linear interactions across the 8,192 embedding dimensions. The periodicity is one component of what makes RNA virus embeddings distinctive, not the sole driver.

### Potential applications

- **Coding region detection without ORF analysis**: offset-3 cosine inversion achieves 94.7% accuracy
- **High-sensitivity coding detection**: inversion signal has 98.8% sensitivity — misses almost nothing
- **High-specificity gene calling**: combine norm AND inversion for 93.5% specificity
- **Non-standard genetic code detection**: organisms with reassigned codons should still show periodicity
- **Frameshift detection**: a programmed frameshift would cause a phase shift in the 3bp periodicity (untested — needs known frameshift examples)

## RNA Virus Dark Matter Detection via Codon Periodicity

### Concept

Eukaryotic RNA viruses have the **strongest codon periodicity** (lag-3 autocorr 0.822 ± 0.111) — significantly higher than dsDNA phages (0.624 ± 0.140), chromosomes (0.486 ± 0.079), or any other category. This means codon periodicity could be a **database-free RNA virus identifier** for viral dark matter that has no homology to known sequences.

### Proof-of-Concept Classification

**RNA virus vs dsDNA phage** (5-fold CV, n=23):
- Accuracy: **91.3%**
- RNA virus recall: **100%** (8/8 detected)
- dsDNA phage specificity: 87% (2 FPs — Lactobacillus and Staphylococcus phages)

**Best feature**: cos3_coding (offset-3 cosine in coding regions)
- Cohen's d = **2.83** between RNA virus and all other categories (huge effect size)
- RNA virus: 0.625 ± 0.027 vs other: 0.490 ± 0.061

**Simple threshold** (lag-3 ≥ 0.75): 88% recall, 88% specificity, 64% PPV for RNA virus

### False Positives

The 2-3 false positive phages are from **Firmicutes hosts** (Lactobacillus, Staphylococcus) — low-GC organisms with potentially stronger codon usage bias. The cos3_coding feature partially resolves this (RNA viruses cluster at 0.59-0.66, while FP phages are 0.54-0.59).

### Why This Matters for Dark Matter

Current RNA virus detection requires RdRp homology search or protein domain databases. For truly novel RNA viruses with no detectable homology, these methods fail. Codon periodicity requires only:
1. One Evo2 forward pass (per-position embeddings)
2. Simple autocorrelation computation
3. **No database, no gene calling, no homology search**

This could be the **first approach to identify likely RNA viral sequences from DNA/cDNA composition alone**, without any reference database — addressing a fundamental gap in viral dark matter detection.

### Definitive Validation (203 sequences, 2026-03-17)

Expanded to 100 RNA viruses + 72 dsDNA phages + 20 plasmids + 18 chromosomes with full periodicity features.

| Metric | RNA virus vs ALL | RNA virus vs dsDNA phage |
|--------|-----------------|-------------------------|
| **Accuracy** | **97.5%** | **94.5%** |
| **AUC** | **0.990** | **0.987** |
| **RNA virus recall** | 95% | 96% |
| **Specificity** | 100% | 93% |
| **Firmicutes phage FP** | **0/25 (0%)** | — |

Best discriminating features (Random Forest importance):
- **cos3** (offset-3 cosine): 50.9% — RNA virus 0.638 vs phage 0.528
- **lag3** (autocorrelation): 18.0% — RNA virus 0.875 vs phage 0.766
- **cos1** (adjacent cosine): 15.9% — RNA virus 0.331 vs phage 0.267

The Firmicutes false positive concern is **resolved** — 0/25 Firmicutes-host phages misclassified as RNA virus.

### Remaining Caveats

- Validated on benchmark fragments, not real metatranscriptomic assemblies
- Need to test on dsRNA viruses, retroviruses, and negative-sense ssRNA specifically
- Potential confounders: highly AT-biased organisms (Plasmodium, some archaea), organellar DNA
- Should validate on completely independent dataset (not from the same RNA virus database)

## Novel Gene Detection

### Can Evo2 find genes that Pyrodigal misses?

Across 41 sequences, Evo2 flags ~1.9% additional potentially coding regions (1,349 bp beyond Pyrodigal's 70,042 bp). These are predominantly small ORFs (30-76 aa) in intergenic regions that fall below standard gene-calling thresholds.

### Best candidate: 76 aa ORF on Rhodococcus plasmid

On plasmid NZ_CP070610.1 (Rhodococcus koreensis, 12,373 bp), Evo2 identifies 3 high-norm intergenic regions (618 bp total) that both Pyrodigal-gv and standard Pyrodigal miss:

| Region | Length | Best ORF | Norm vs intergenic | Context |
|--------|--------|----------|-------------------|---------|
| 3121-3411 | 290 bp | **76 aa** (+2, ATG) | 1.54× | Between two called genes |
| 3576-3754 | 178 bp | 46 aa (-1, GTG) | 1.57× | After short gene, 524bp gap |
| 7028-7178 | 150 bp | 32 aa (+2, ATG) | 1.47× | After gene, 163bp gap |

### AlphaFold3 structural validation

All three candidate ORFs were submitted to AlphaFold3 Server for structure prediction:

| ORF | Length | pTM score | Interpretation |
|-----|--------|-----------|---------------|
| 76 aa | 76 aa | **0.23** | No confident fold — disordered |
| 46 aa | 46 aa | **0.40** | Low confidence — marginally structured |
| 32 aa | 32 aa | **0.34** | No confident fold — disordered |

None produce a confident fold (pTM < 0.5 for all three). The 46 aa candidate is closest (pTM 0.40) but still below the threshold for a reliable structure prediction.

The 76 aa candidate has a pTM of 0.23 (very low confidence), indicating it does not adopt a stable protein fold. This does not necessarily mean the region is non-functional — it could be:
- An intrinsically disordered protein (IDPs are common in phage/plasmid genomes)
- A non-coding RNA with secondary structure
- A regulatory element
- A degraded pseudogene

### Interpretation

**Evo2 per-position norms reliably distinguish coding from non-coding regions** (91.7% accuracy, 1.72× norm ratio), but high-norm intergenic regions are not always novel protein-coding genes. The norm signal reflects **information density** in the DNA sequence, which encompasses coding regions, structured RNAs, and regulatory elements. Validation with structural prediction or experimental evidence is needed to confirm specific candidates.

For the paper, this is best framed as: "Per-position embeddings identify candidate functional regions that complement gene callers, particularly for sub-threshold features. Structural validation using AlphaFold3 can distinguish likely protein-coding ORFs from other high-information-density regions."

## Files

| File | Contents |
|------|----------|
| `results/poc_gene_boundaries/per_position_embeddings.npy` | 5299 × 8192 per-position embeddings |
| `results/poc_gene_boundaries/genes.json` | Pyrodigal-gv gene calls (ground truth) |
| `results/poc_gene_boundaries/embedding_landscape.npz` | Derived signals (cosine, norms, PCA, masks) |
| `results/poc_gene_boundaries/analysis_results.json` | Quantitative results |
| `scripts/poc_gene_boundaries.py` | Analysis script |
