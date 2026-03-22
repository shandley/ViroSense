# ViroSense Findings Log

Running document of all findings from Evo2 embedding analysis.
Each finding includes: what we tested, what we found, significance, and status for the paper.

Last updated: 2026-03-20

---

## Finding 1: Offset-3 Cosine Inversion (PRIMARY)

**What**: Per-position Evo2 embeddings show cos(offset-3) > cos(offset-1) in coding DNA, inverting in non-coding DNA.

**Evidence**:
- 452/459 coding sequences (98.5%), 100% above 500 bp
- 55 phyla, all 3 domains, GC 9.8–78.8%
- 10 gene families × ~29 orthologs each
- 7 failures all in short sequences (<300 bp for 5 of 7)
- Non-coding specificity: 76.5% for rRNA/lncRNA/intron/repeats. tRNA shows inversion (codon-related). Bacterial intergenic unreliable.

**Significance**: Novel signal. No prior work identified this. Enables 94.7% coding detection without gene calling.

**Status**: CONFIRMED at N=459. Publication-ready.

## Finding 2: Multi-Offset Triplet Periodicity

**What**: Cosine similarity at offsets 1-15 shows a 3-periodic "comb filter" pattern in coding DNA.

**Evidence**:
- N=36 coding, N=14 non-coding (40B blocks.10)
- Mod-3 offsets: 0.41–0.49; non-mod-3: 0.29–0.31 (~0.17 elevation)
- Non-coding: smooth monotonic decay, no periodicity
- Pattern persists through offset 15 (5 codon lengths)

**Significance**: Definitive proof the signal is specifically triplet structure, not some other periodicity. Strongest single visualization of "the model learned the genetic code."

**Status**: CONFIRMED. In Figure 1 Panel B.

## Finding 3: GC Independence

**What**: The offset-3 inversion is independent of nucleotide composition.

**Evidence**:
- Works from 9.8% GC (Toxoplasma apicoplast) to 78.8% GC (Chlamydomonas ChR2)
- Pearson r = 0.26 (weak positive correlation with GC, but signal present across full range)
- 459 sequences spanning the full GC spectrum

**Significance**: Rules out the hypothesis that the inversion is a compositional artifact.

**Status**: CONFIRMED. Figure 1 Panel E.

## Finding 4: Genetic Code Variant Independence

**What**: The inversion works with non-standard genetic codes.

**Evidence**:
- Human mitochondrial CO1 (UGA = Trp instead of Stop): gap = +0.298
- Yeast mitochondrial COX2: gap = +0.281
- Drosophila mitochondrial ND5: gap = +0.299
- Chloroplast genes (standard code but plastid context): 5/6 positive

**Significance**: The model detects triplet STRUCTURE, not specific codon ASSIGNMENTS.

**Status**: CONFIRMED. Mentioned in text + Figure 1E edge cases.

## Finding 5: Length Dependence

**What**: Short sequences have noisier inversion signal.

**Evidence**:
- >500 bp: 100% (316/316)
- 300-500 bp: 98.5% (65/66)
- <300 bp: 85.3% (29/34)

**Significance**: Practical limitation — need ~500 bp for reliable detection. Biologically reasonable (need several codons for periodicity).

**Status**: CHARACTERIZED. In results text.

## Finding 6: tRNA Shows Inversion (Expected)

**What**: tRNA sequences show the offset-3 inversion despite being non-coding.

**Evidence**: 5/5 tRNA sequences show inversion (gaps +0.01 to +0.39)

**Significance**: tRNAs have inherent codon-anticodon structure — the anticodon loop directly encodes triplet information. This is biologically correct, not a false positive.

**Status**: CHARACTERIZED. Discussed as expected exception.

## Finding 7: Bacterial Intergenic Often Contains Coding

**What**: "Intergenic" regions in compact prokaryotic genomes frequently show inversion.

**Evidence**:
- E. coli lacY-lacA "intergenic": was 95% CDS (lacY + lacZ overlap)
- 4/5 bacterial/yeast intergenic controls show inversion
- Bacteria are ~87% coding — true >1kb intergenic regions are rare

**Significance**: The model catches mislabeled sequences. Compact genomes are unreliable negative controls.

**Status**: CHARACTERIZED. Supplementary.

## Finding 8: RNA Dark Matter Detection

**What**: Periodicity features alone distinguish RNA viruses from all other categories.

**Evidence (recomputed 2026-03-21, 7B decoder.layers.10)**:
- **95.2% accuracy, 0.982 AUC** (207 sequences, 4 categories, 5-fold CV)
- inversion_gap is top feature (46.6% importance, Cohen's d = 2.48)
- cos1 is 2nd (27.3%, d = 2.10) — RNA viruses have elevated adjacent-nucleotide similarity
- inversion_gap alone: 92.3% accuracy, 0.959 AUC
- Database-free, no homology, no gene calling

**Original claim (40B blocks.28.mlp.l3, March 2026)**: 97.5% / 0.990 AUC — from a layer that is no longer accessible via the current NIM API. The recomputed numbers at blocks.10 are lower but still strong.

**Significance**: Novel capability — database-free RNA virus identification from periodicity features alone.

**Status**: RECONFIRMED with updated numbers. Figure 2.

## Finding 9: Prophage Amelioration Gradient

**What**: Viral embedding scores serve as a proxy for prophage evolutionary age.

**Evidence**:
- E. coli K12: DLP12/rac/Qin (active, score 0.957) → CP4-6 (mosaic, 1/5 windows) → e14 (invisible, 0.120)
- Score variance across windows = mosaicism indicator

**Significance**: Novel method for measuring prophage age from composition alone.

**Status**: CONFIRMED (coarse pass). In Figure 2.

## Finding 10: Functional Clustering is NEGATIVE

**What**: Mean-pooled Evo2 embeddings do NOT cluster by protein identity.

**Evidence**:
- 40B blocks.10: Silhouette -0.064, NN accuracy 19.9%
- 40B blocks.28: Silhouette -0.140, NN accuracy 12.9%
- 7B layer.10: Silhouette -0.165, NN accuracy 20.2%
- N=287, 10 gene families × ~29 orthologs
- Embedding Atlas UMAP was a visualization artifact

**Significance**: Important negative result. Mean-pooled supports classification but not functional annotation via embedding similarity.

**Status**: CONFIRMED NEGATIVE. Figure 3 Panel D / Supplementary.

## Finding 11: Layer Dependence

**What**: The periodicity signal is strongest at intermediate layers.

**Evidence**:
- blocks.0: no signal (lag3 = -0.05)
- blocks.5-15: strong signal (gap +0.17 to +0.23)
- blocks.20: weaker (gap +0.05)
- blocks.25-31: residual stream explodes (norms ~1e16), MLP near-zero

**Significance**: Practical guidance for which layer to use. Also reveals model internals — periodicity is learned in middle layers.

**Status**: CHARACTERIZED. Supplementary S1.

## Finding 12: NIM API Change (Technical)

**What**: NVIDIA NIM API changed format between early March and mid-March 2026.

**Evidence**:
- Old: `layer` parameter, `embedding` response field, float32
- New: `output_layers` list, `data` response field (NPZ), float64
- Self-hosted 7B returns (seq_len, 1, 4096); cloud 40B returns (1, seq_len, 8192)

**Significance**: Technical note for reproducibility.

**Status**: Documented in `docs/nim_api_layer_investigation.md`.

## Finding 13: K-mer Baselines are Strong for Detection

**What**: Trinucleotide frequencies achieve 93% viral detection at 1,527× the speed.

**Evidence**:
- Random Forest on 84 features (64 trinuc + 16 dinuc + GC + frame entropy)
- 93% accuracy vs Evo2 95.4% (2.4% gap)
- 1,527× faster

**Significance**: Honest comparison. Foundation models add per-position and zero-shot, not better binary detection.

**Status**: CONFIRMED. Figure 4.

---

## Datasets Analyzed

| Dataset | N | What we tested | Key finding |
|---------|---|---------------|-------------|
| Original metagenomic panel | 40 | Periodicity discovery | Offset-3 inversion, 94.7% coding detection |
| Cross-domain pilot | 72 | Universality at blocks.10 (40B) | 98.5% (64/65), all domains |
| Comprehensive panel (Component A) | 172 | Universality at layer.10 (7B) | 98.5% (138/142 coding), 55 phyla |
| Comprehensive panel (Component B) | 287 | Functional clustering (7B + 40B) | Negative: no protein-identity clustering |
| Comprehensive panel (Component C) | 30 | Negative controls | 29/30 coding inversion |
| Multi-offset analysis | 50 | Offset specificity 1-15 | 3-periodic comb filter |
| E. coli lac operon | 1 (6kb) | Genomic trajectory | Inversion tracks gene boundaries |
| Benchmark (GYP + RNA virus) | 13,417 | Viral detection | 95.4% accuracy, 99.7% phage |
| RNA dark matter | 203 | Periodicity-based classification | 97.5% RNA virus detection |
| E. coli K12 prophage | 1 (partial) | Prophage amelioration | DLP12→CP4-6→e14 gradient |

---

## Datasets NOT YET Analyzed (Opportunities)

### High Priority — Would Strengthen the Paper

| Dataset | What it would test | Expected finding | Effort |
|---------|-------------------|-----------------|--------|
| **Complete bacterial genome scan** (E. coli K12 full 4.6Mb) | Per-position profile of a full genome | All genes detected, operons visible, regulatory elements? | 1 NIM call (6 min at 16kb windows) |
| **Eukaryotic gene with introns** (human BRCA1 genomic, ~80kb) | Exon-intron boundary detection | Inversion should flip at splice sites | Multiple NIM calls, need to window |
| **Overlapping reading frames** (phage genomes with +1/+2 frame overlaps) | Does the model detect multiple frames? | Unknown — could be novel | A few NIM calls |
| **rRNA operon** (E. coli rrn, 5.5kb) | Per-position profile of structured non-coding | rRNA should lack inversion; tRNA within operon may show it | 1 NIM call |

### Medium Priority — Interesting but Not Essential for Paper 1

| Dataset | What it would test | Expected finding | Effort |
|---------|-------------------|-----------------|--------|
| **CRISPR array** | Repeat vs spacer structure | Repeats uniform, spacers from different organisms may differ | A few NIM calls |
| **Horizontal gene transfer** (genomic islands) | Compositional foreignness in per-position | Recently acquired DNA should show embedding transitions | Need annotated HGT regions |
| **Plasmid vs chromosome** | Compositional signatures | Resistance cassettes may have different periodicity | Already have data from ViroSense |
| **Metagenomic contigs** (real stool sample) | Real-world application | Identify coding regions, viral sequences in metagenome | Need metagenome assembly |
| **Transposable elements** (IS elements, Tn) | Mobile element per-position structure | Inverted repeats, coding regions within TEs | A few NIM calls |
| **Codon-optimized vs wild-type** (same protein, different organisms) | Engineering detection | Optimized may have altered periodicity pattern | Already have eGFP data |

### Lower Priority — Future Papers

| Dataset | What it would test | Paper |
|---------|-------------------|-------|
| Logan assembled contigs (385 TB) | Planetary-scale screening | Paper 5 |
| Full metatranscriptome | RNA dark matter at scale | Paper 2 |
| Multi-genome prophage survey | Amelioration across species | Paper 3 |
| Addgene synthetic constructs | Forensic detection | Paper 4 |

---

## Finding 14: Taxonomic Signal in Mean-Pooled Embeddings (Pilot)

**What**: Mean-pooled Evo2 40B embeddings carry strong compositional signal for same-genome fragment retrieval and broad category classification.

**Evidence**:
- Same-genome top-1 NN: 13.5% (135× random baseline of 0.1%) on 6,526 phage fragments from 797 genomes
- Within-genome cosine distance: 0.063 vs between-genome: 0.275 (4.34× ratio)
- 5-class KNN (k=5, cosine): 85.3% accuracy (phage/plasmid/chromosome/RNA virus/cellular)
- Zero new API calls — used cached 40B benchmark embeddings

**Significance**: Embeddings capture compositional similarity (same genome, same category) but not functional identity (same protein). This is consistent with the functional clustering negative result. For viral taxonomy, family/genus-level classification should be achievable from embeddings alone.

**Status**: PILOT. Not in Paper 1. Could be Paper 6 (alignment-free taxonomy).

---

## Finding 15: Intra-Codon Structure — Offset-1 > Offset-2 (100% Consistent)

**What**: Within coding sequences, cosine similarity at offset-1 (adjacent codon positions, e.g., pos 1→2) consistently exceeds offset-2 (skip-one positions, e.g., pos 1→3).

**Evidence**:
- 36/36 coding sequences (100%) show offset-1 > offset-2
- Mean difference: +0.012 (small but perfectly consistent)
- The asymmetry disappears at offset-4 vs offset-5 (only 17%), confirming it's specific to within-codon transitions
- The two non-mod-3 residue classes (offsets 1,4,7,10,13 vs 2,5,8,11,14) are indistinguishable at the aggregate level (0.2937 vs 0.2938)

**Significance**: Suggests the model captures the functional coupling between the first two codon positions (which together largely determine the amino acid) vs the wobble position (3rd, more free to vary). The model has learned not just that codons are triplets, but something about the INTERNAL structure of codons.

**Caveats**: The effect is small (+0.012). Needs per-position analysis with known reading frames to verify it's truly the wobble effect vs some other adjacent-nucleotide bias.

**Status**: PRELIMINARY. Worth a sentence in the paper, not a main claim.

## Finding 16: Genetic Code Table in Embedding Space

**What**: Do mean-pooled embeddings of codon-repeat sequences (300bp of one codon repeated) cluster by amino acid?

**Evidence**:
- 64 codon-repeat sequences (61 sense + 3 stop) extracted at blocks.10 (40B)
- Synonymous codons are 1.32× closer than non-synonymous (within=0.099, between=0.131)
- Silhouette by amino acid: -0.401 (no clean clustering)
- **Stop codons cluster tightly**: stop-stop distance 0.055 vs stop-sense 0.085 (1.55× ratio)
- Amino acid property groups (hydrophobic/polar/charged): NO clustering (silhouette -0.159)
- Synonymous codon proximity is driven by SEQUENCE SIMILARITY (shared bases), not amino acid identity

**Significance**: The model has learned:
- The SYNTAX of the genetic code: triplet structure (98.5%), stop signals (1.55× clustering)
- But NOT the SEMANTICS: which amino acid a codon encodes, or amino acid biochemical properties
- This is consistent with training on DNA only — the model learns DNA patterns, not protein biology

**Embedding Atlas UMAP exploration (2026-03-22)**:
- amino_acid coloring: no clustering — colors completely interleaved
- is_stop coloring: stop codons (3 points) clearly separated from sense codons
- property coloring: no biochemical property clustering (hydrophobic/polar/charged interleaved)
- first_base coloring: partial clustering — first nucleotide drives some spatial organization
- **gc_content: STRONG gradient** — the dominant organizing axis. 0% GC codons cluster together, 100% GC cluster together. Composition, not biology, drives the embedding space.
- Stop codon separation is likely composition-driven (TAA/TAG/TGA are AT-rich) not function-driven

**Status**: CONFIRMED. The codon embedding space is organized by nucleotide composition, not amino acid identity.

## Finding 17: Exon-Intron Boundary Detection Across Eukaryotes (MAJOR)

**What**: The offset-3 cosine inversion flips at splice sites in eukaryotic genes — positive in exons (coding), negative in introns (non-coding). This enables database-free eukaryotic gene structure prediction.

**Evidence**:
- **Human HBB** (beta-globin, 1.6kb): 3 exons perfectly resolved as positive peaks, 2 introns as negative dips, UTRs negative
- **Human TP53** (tumor suppressor, 15kb): 11 exons visible, large intronic regions correctly negative
- **Human BRCA1** (16kb region): exon peaks align with annotations
- **Drosophila Adh** (alcohol dehydrogenase, 4kb): exon-intron oscillation visible
- **C. elegans unc-54** (myosin heavy chain, 6kb): multiple exons detected
- **Arabidopsis AGAMOUS** (floral development, 5kb): exon peaks align, large intron negative
- **Yeast ACT1** (1.4kb): single intron context
- Tested across 5 species spanning 4 kingdoms (Mammalia, Insecta, Nematoda, Plantae, Fungi)

**Significance**:
- Fills a known gap: Arc Institute GitHub Issue #72 asked for Evo2 exon/intron classification — still not released as of March 2026
- **No splice site model, no RNA-seq, no reference genome needed**
- Works across all eukaryotic kingdoms tested
- Same signal (offset-3 inversion) that detects prokaryotic coding regions also detects eukaryotic exon structure
- This transforms the finding from "codon periodicity in prokaryotes" to "universal gene structure detection across all life"

**Quantified accuracy** (per-position, 50bp smoothing, inversion > 0 = coding):

| Gene | Species | Accuracy | Precision | Recall | F1 | Coding% | Exon regions |
|------|---------|----------|-----------|--------|-----|---------|-------------|
| HBB | Human | 76.1% | 53.6% | **100%** | 0.698 | 27.6% | 3 |
| TP53 (part 1) | Human | 89.6% | 40.3% | 92.5% | 0.561 | 7.2% | 57 (CDS parts) |
| TP53 (part 2) | Human | 91.0% | 28.9% | 85.3% | 0.432 | 4.0% | 36 (CDS parts) |
| BRCA1 | Human | 89.7% | 57.4% | **98.9%** | 0.726 | 13.9% | 51 (CDS parts) |
| unc-54 | C. elegans | **90.1%** | 63.0% | **96.4%** | **0.762** | 16.4% | 7 |
| AGAMOUS | Arabidopsis | 76.1% | 49.5% | **94.8%** | 0.650 | 23.4% | 10 |

**Key observations**:
- **Recall is very high** (85-100%) — the inversion detects almost all coding regions
- **Precision is lower** (29-63%) — some intronic regions are falsely called as coding (the smoothing window bleeds signal across boundaries)
- **Accuracy scales with gene density** — genes with more intron (lower coding %) have higher accuracy because there's more intron to correctly call
- Drosophila Adh had 0% coding annotation (annotation didn't map to this genomic region)
- Yeast ACT1 had 100% coding (entire region is CDS, no contrast to measure)

**Smoothing optimization** (tested 10, 25, 50, 75, 100, 150, 200, 300bp):
- Overall best: **100bp** (mean F1 = 0.689 across 6 genes)
- Sweet spot: 75-150bp range
- Recall consistently 86-100% across all windows — signal is robust
- Precision is the bottleneck (boundary blurring)
- Short-exon genes prefer 50-75bp; long-intron genes prefer 150bp

**Caveats**:
- Precision limited by smoothing — boundaries blurred by ~50bp on each side at optimal window
- Annotation quality varies — CDS parts from GenBank may not perfectly match extraction coordinates
- Small exons (<100bp) may be masked by smoothing
- N=6 quantified genes — expanding to 20-30 (in progress)

**Comprehensive quantification (36 genes, 13 species, 9 kingdoms)**:
- Mean recall: **98.0%** ± 3.6% — virtually every exon detected
- Recall > 80%: **36/36 (100%)**
- Recall > 90%: 33/36 (92%)
- Mean F1: 0.703 ± 0.132
- Mean precision: 56.3% (limited by 100bp smoothing at boundaries)
- Works across: Mammalia, Insecta, Nematoda, Aves, Fish, Amphibia, Plantae, Fungi, Protista

**Status**: CONFIRMED at scale. Major paper finding. Fills Arc Institute Issue #72.

## Finding 18: Embedding Atlas Interactive Exploration (3 Datasets)

**What**: Systematic UMAP exploration of three embedding datasets using Apple's Embedding Atlas.

**64 codons (40B blocks.10)**:
- GC content is the dominant organizing axis in codon embedding space
- Stop codons separate, but likely driven by AT-rich composition (TAA/TAG/TGA) not function
- Amino acid identity: no clustering. Property groups: no clustering.
- First nucleotide shows partial structure (compositional)

**287 gene families (40B blocks.10)**:
- Gene families completely interleaved — confirms negative functional clustering result
- Species shows some spatial organization — composition (species-level GC/codon usage) drives structure
- Outlier clusters correspond to compositionally extreme species (Plasmodium = AT-rich)

**13,417 benchmark (40B blocks.28.mlp.l3)**:
- **Category separation is clear**: RNA viruses form isolated island, phage dominates central cloud, plasmid/chromosome overlap
- The RNA virus island explains why dark matter detection works (95.2%)
- Plasmid/chromosome overlap explains lower plasmid specificity (81.5%)
- Sequence length does NOT drive structure — all length bins interleaved

**Significance**: The 13,417-sequence UMAP colored by category is publication-worthy for Paper 2. The codon and gene family explorations reinforce Paper 1's negative results about amino acid encoding.

**Status**: DOCUMENTED. Screenshots saved. Benchmark visualization → Paper 2 supplementary or main figure.

---

## Open Questions

1. **Does the inversion detect overlapping reading frames?** Some viral genomes have genes in +1 or +2 frames overlapping. The periodicity at offset-3 should detect the dominant frame — but what about the secondary frame?

2. **What happens at programmed frameshifts?** Some genes (e.g., HIV gag-pol) use -1 ribosomal frameshifts. Does the per-position signal change at the frameshift site?

3. **Can we detect promoters/terminators?** The coding/non-coding inversion detects gene bodies. Do regulatory elements have their own signature?

4. **Does the signal work on single-stranded RNA genomes?** We tested cDNA copies. Do RNA genomes (if converted to DNA) show the same pattern?

5. **What about pseudogenes?** They have coding-like sequence but are non-functional. Does the inversion persist in pseudogenes? (ydbA test was inconclusive.)

6. **Is the comb filter pattern identical across all Evo2 layers?** We tested blocks.10. Does the mod-3 vs non-mod-3 pattern change at other layers?

7. **Do other DNA foundation models (Nucleotide Transformer, DNABERT-2, Caduceus) show the same pattern?** If universal across models, it's a property of DNA language models, not just Evo2.
