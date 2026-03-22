# Per-position DNA foundation model embeddings encode gene structure from codons to splice sites across all domains of life

## Abstract

DNA foundation models trained on nucleotide sequences learn representations that capture biological structure, but the specific features encoded in per-position embeddings remain poorly characterized. Here we show that frozen per-position embeddings from Evo2, a 40-billion parameter DNA model trained by next-nucleotide prediction, encode gene structure at multiple scales. We identify an offset-3 cosine inversion — a geometric signature in which nucleotides separated by one codon are more similar in embedding space than adjacent nucleotides in coding regions, with this relationship inverting in non-coding regions. This signal is universal: validated on 459 protein-coding sequences spanning all three domains of life (55 phyla, 10 conserved gene families, GC content 9.8–78.8%), it achieves 98.5% sensitivity overall and 100% for sequences above 500 bp. The same signal detects eukaryotic exon-intron boundaries: across 36 genes in 13 species spanning 9 eukaryotic kingdoms, the inversion achieves 98% recall for exon detection without splice site models, RNA-seq data, or reference genomes. We further show that the model learned the structural syntax of the genetic code — triplet organization and stop codon boundaries — but not the semantic mapping from codons to amino acids, delineating the boundary between what can be learned from DNA sequence statistics alone versus what requires protein-level evolutionary selection. Trinucleotide frequency classifiers achieve comparable performance for binary coding detection at 1,527× the speed, but cannot replicate the per-position capabilities that enable exon-intron detection and gene boundary resolution. These findings reveal that next-nucleotide prediction on DNA is sufficient for a model to discover gene structure across all life, from prokaryotic operons to eukaryotic splice sites.

## Introduction

The central dogma of molecular biology — DNA encodes RNA encodes protein — implies that DNA sequences contain information about gene structure at multiple scales. Protein-coding regions are organized in triplet codons, delineated by start and stop signals, and in eukaryotes interrupted by non-coding introns that must be precisely excised during RNA processing. These structural features have been exploited for decades by gene-finding algorithms that use statistical models of codon periodicity, splice site motifs, and compositional signatures to predict genes in genomic sequences^1–3^.

DNA foundation models — large neural networks trained on nucleotide sequences by next-token prediction — have recently achieved remarkable performance across genomic tasks^4–7^. Evo2, a 40-billion parameter model trained on DNA from all domains of life, generates functional genes, predicts variant effects, and exhibits emergent understanding of genome organization^5^. Interpretability analyses using sparse autoencoders (SAEs) have revealed that Evo2 learns features corresponding to coding regions, protein secondary structure, and prophage sequences^8,9^. However, these analyses require training additional interpretability layers, and the specific geometric properties of per-position embeddings — the raw outputs at each nucleotide position — remain poorly characterized.

Here we show that per-position Evo2 embeddings encode gene structure as a simple, measurable geometric property: the offset-3 cosine inversion. In protein-coding DNA, the cosine similarity between embeddings at positions separated by 3 nucleotides (one codon) exceeds that between adjacent positions. In non-coding DNA, this relationship inverts. This signal requires no training, no database, and no reference genome — it is an intrinsic property of the model's learned representations, detectable through elementary vector operations.

We validate this signal across 459 coding sequences spanning all three domains of life, demonstrate that it detects eukaryotic exon-intron boundaries across 36 genes in 13 species, and characterize the boundaries of what the model learned: triplet structure and stop codon identity, but not the codon-to-amino-acid mapping. These findings establish per-position embedding analysis as a practical approach to database-free gene structure prediction and reveal fundamental properties of what DNA foundation models learn from sequence prediction alone.

## Results

### Per-position embeddings encode the triplet genetic code

We extracted per-position embeddings from block 10 of Evo2 40B (8,192 dimensions per nucleotide position) and computed the cosine similarity between embeddings at various offsets along protein-coding and non-coding DNA sequences (Fig. 1A). In coding regions, we observed a striking pattern: cosine similarity at offset-3 (nucleotides separated by one codon length) consistently exceeded similarity at offset-1 (adjacent nucleotides), while this relationship inverted in non-coding regions. We term this the offset-3 cosine inversion.

To characterize the specificity of this periodicity, we computed mean cosine similarity at offsets 1 through 15 across 36 coding and 14 non-coding sequences (Fig. 1B). Coding sequences exhibited a 3-periodic "comb filter" pattern: offsets that are multiples of 3 (3, 6, 9, 12, 15) showed cosine similarity of 0.41–0.49, while all other offsets were 0.29–0.31 — a consistent ~0.17 elevation at every codon-length spacing. Non-coding sequences showed smooth monotonic decay with no periodicity. This pattern confirms that the signal is specifically triplet-periodic, consistent with the codon structure of protein-coding DNA.

The inversion is visible along individual genes. When plotted along the E. coli K12 lac operon (6,001 bp containing four genes — cynX, lacA, lacY, and lacZ), the offset-3 cosine clearly exceeds offset-1 within each coding region and drops below offset-1 in intergenic regions (Fig. 1C). The transitions align with annotated gene boundaries without any gene-calling tool.

We selected block 10 (of 32 total Evo2 blocks) based on systematic profiling showing the strongest coding/intergenic contrast at intermediate layers (Supplementary Fig. S1). Late layers (blocks 25–31) exhibited saturated residual streams with embedding norms exceeding 10^16, rendering MLP sub-layer outputs negligible.

### The offset-3 inversion is universal across all domains of life

To test universality, we assembled a comprehensive panel of 510 sequences: 459 coding sequences spanning 55 phyla from all three domains of life (Archaea, Bacteria, Eukarya), 10 conserved gene families with ~29 orthologs each, and 30 non-coding controls (Fig. 1D,E; Supplementary Fig. S2).

The inversion was present in 452 of 459 coding sequences (98.5%). Detection was 100% for sequences above 500 bp (316/316) and 85.3% for sequences below 300 bp (29/34), with the 7 failures all occurring in short sequences where cosine estimates are noisier. Every major taxonomic group showed ≥96% detection: Archaea (56/56, 100%), Bacteria (96/100, 96%), Vertebrata (67/68, 98.5%), Invertebrata (62/62, 100%), Plantae (46/46, 100%), Fungi (31/31, 100%), Protista (49/50, 98%), Algae (26/26, 100%), and Virus (10/10, 100%) (Fig. 1D).

The signal was independent of GC content (Pearson r = 0.26 across the full range of 9.8% to 78.8% GC; Fig. 1E), independent of genetic code variant (human mitochondrial CO1 with UGA=Trp showed strong inversion), and present in synthetic codon-optimized sequences (eGFP). A full E. coli K12 genome scan (4.6 Mb, 386 overlapping windows) confirmed that the inversion tracked all 4,651 annotated genes (Supplementary Fig. S4).

Non-coding specificity was category-dependent (Supplementary Fig. S3). Ribosomal RNA, long non-coding RNA, intronic sequences, and repetitive elements showed 76.5% correct classification (no inversion). tRNA sequences showed the inversion in all 5 cases tested, consistent with their inherent codon-anticodon structure. Compact prokaryotic intergenic regions frequently showed inversion, likely reflecting unannotated coding elements in these gene-dense genomes.

### The inversion detects eukaryotic exon-intron boundaries

We next asked whether the offset-3 inversion could detect gene structure within eukaryotic genes containing introns. We extracted per-position embeddings for 36 genes across 13 species spanning 9 eukaryotic kingdoms and computed the smoothed inversion signal (cos3 − cos1, 100 bp uniform filter) along each gene (Fig. 2).

In the human beta-globin gene (HBB), all three exons were perfectly resolved as positive inversion peaks, with both introns and UTR regions showing negative signal (Fig. 2A). The same pattern was observed across kingdoms: Drosophila melanogaster even-skipped (2 exons), Arabidopsis thaliana AGAMOUS (7 exons), and Saccharomyces cerevisiae ACT1, where the single 309-bp intron was detected as a clear negative dip in an otherwise positive signal (Fig. 2B).

Quantification across all 36 genes showed mean exon recall of 98.0% ± 3.6%, with 33 of 36 genes above 90% recall (Fig. 2C). Recall exceeded 80% in every gene tested. Mean precision was 56.3% ± 15.8%, limited by the 100 bp smoothing window which blurs signal across exon-intron boundaries. The optimal smoothing window was 100 bp (mean F1 = 0.689), with the 75–150 bp range representing the precision-recall sweet spot (Supplementary Fig. S6).

This exon-intron detection requires no splice site model, no RNA-seq data, and no reference genome. It fills a known gap: Arc Institute's GitHub Issue #72 requested Evo2-based exon/intron classification in March 2025, which remains unaddressed as of this writing.

### The model learned DNA syntax, not protein semantics

To probe the depth of the model's understanding of the genetic code, we generated 64 sequences each consisting of 100 repeats of a single codon (300 bp) and extracted mean-pooled embeddings for all 61 sense codons and 3 stop codons (Fig. 3).

Stop codons (TAA, TAG, TGA) clustered together in embedding space with 1.55× tighter within-group distance compared to stop-versus-sense distance (Fig. 3A). This indicates the model learned that stop codons represent a distinct functional category — gene boundaries are detectable from DNA statistics alone.

However, synonymous codons (encoding the same amino acid) showed only weakly elevated similarity (1.32× ratio), driven primarily by shared nucleotide composition rather than amino acid identity. Per-amino-acid analysis revealed no clustering by biochemical property: hydrophobic, polar, charged, and aromatic amino acid codons were interleaved in embedding space (Fig. 3B,D). The silhouette score for amino acid grouping was -0.40, indicating no meaningful clustering.

We further tested whether mean-pooled gene embeddings clustered by protein identity across species. Using 287 sequences from 10 conserved gene families (actin, GAPDH, EF-Tu, HSP70, COI, tubulin, histone H3, rpoB, atpA, rpsL), we found no protein-identity clustering across three model configurations (Evo2 40B blocks.10, 40B blocks.28, 7B layer.10): silhouette scores ranged from -0.064 to -0.165, and nearest-neighbor gene-family accuracy was 13–20% (Fig. 3C). Notably, UMAP visualization of these embeddings had initially suggested functional clustering, but quantitative analysis revealed this to be a visualization artifact — a cautionary example for dimensionality reduction interpretation in high-dimensional biological data.

These results delineate the boundary of DNA-level learning: Evo2 learned the structural syntax of the genetic code (triplet organization, stop codon identity, exon-intron boundaries) but not the semantic mapping from codons to amino acids or from genes to protein function, which requires protein-level evolutionary selection pressure.

### Practical context: per-position analysis versus k-mer approaches

Trinucleotide frequency classifiers achieved 93% accuracy for binary coding detection at 1,527× the speed of Evo2 embedding extraction (Fig. 4D). This establishes that simple compositional features capture the majority of the coding signal for binary classification tasks.

However, k-mer features cannot replicate the per-position capabilities of foundation model embeddings. Exon-intron boundary detection, gene boundary resolution at nucleotide scale, and the positional structure revealed by the comb filter pattern are fundamentally inaccessible to bag-of-k-mer representations, which discard all positional information. The unique contribution of per-position embedding analysis lies in this spatial resolution — the ability to trace gene structure along a sequence rather than classify it as a whole.

Detection was robust across all domains of life tested (Fig. 4A), with reduced sensitivity only for very short sequences below 300 bp (Fig. 4B). Non-coding specificity was high for ribosomal RNA, long non-coding RNA, intronic, and repetitive sequences, with biologically explainable exceptions for tRNA (which contains codon-related structure) and compact prokaryotic intergenic regions (which frequently contain unannotated coding elements) (Fig. 4C).

## Discussion

We have shown that a DNA foundation model trained solely on next-nucleotide prediction discovers gene structure at multiple scales — from the triplet codon code to eukaryotic splice sites — encoded as a measurable geometric property of per-position embeddings. This offset-3 cosine inversion is universal across all domains of life, independent of GC content and genetic code variant, and exploitable for database-free gene annotation without any supervised training.

**What the model learned — and what it didn't.** The offset-3 inversion reflects the fundamental periodicity of protein-coding DNA: every third nucleotide occupies the same codon position, creating a statistical regularity that next-nucleotide prediction captures. Stop codons, which mark gene boundaries, are also learnable from DNA statistics — they create predictable sequence transitions. In contrast, the mapping from codons to amino acids operates at the protein level: the "meaning" of a codon is determined by tRNA-mediated translation and protein-level selection, neither of which is visible in DNA sequence prediction. This explains why the model learned the syntax (triplet structure, stop signals, splice boundaries) but not the semantics (amino acid identity, protein function) of the genetic code.

**Relation to prior work.** The 3-base periodicity of protein-coding DNA has been known since Fickett's 1982 observation^1^ and has been exploited in gene-finding algorithms for decades^2,3^. Recent work on DNA foundation models has shown, using sparse autoencoder interpretability^8,9^, that models like Evo2 learn features corresponding to coding regions and exon-intron boundaries. Merchant et al.^10^ demonstrated that Evo 1 learns functional gene context through semantic design. Our contribution is distinct: we show that raw per-position embeddings — without SAE training or fine-tuning — encode exploitable codon periodicity as a simple geometric signal, and we provide the first comprehensive universality validation (459 sequences, 55 phyla) and the first quantified exon-intron detection across 36 genes in 13 species.

**Database-free gene annotation.** Reference genomes exist for fewer than 0.1% of estimated eukaryotic species. For the vast majority of organisms, gene annotation relies on transferring models from distant relatives — an approach that degrades rapidly with evolutionary distance. The offset-3 inversion provides a complementary signal: coding regions can be identified from raw DNA alone, without any reference, training data, or even knowledge of the organism's phylogenetic placement. While the boundary resolution (~100 bp) and precision (~56%) of our current approach are insufficient to replace established gene finders for well-studied organisms, the method's universality and database-independence make it valuable for initial gene structure characterization in newly sequenced genomes.

**Exon-intron detection without RNA-seq.** For many organisms — unculturable microeukaryotes, organisms known only from environmental DNA, extinct species from ancient DNA — RNA-seq data for splice site validation is unavailable. The offset-3 inversion provides evidence of exon structure from genomic DNA alone. With 98% recall across 36 genes spanning mammals, insects, nematodes, plants, fungi, and protists, the signal reliably identifies coding exons. The main limitation is precision at boundaries, set by the smoothing window (~100 bp). Future work could improve boundary resolution through derivative-based peak detection or learned thresholding.

**Generalizability.** The offset-3 inversion should be testable in any DNA foundation model that processes sequences at nucleotide resolution (Nucleotide Transformer^6^, DNABERT-2^11^, Caduceus^12^, HyenaDNA^13^). We predict it will be present in any model that achieves sufficient next-nucleotide prediction accuracy on coding DNA, as the triplet periodicity is a dominant statistical feature of the training data. Comparative studies across models could reveal how model architecture and training data composition influence the strength and specificity of the signal.

**Limitations.** (1) The signal is extracted from block 10 of 32 Evo2 layers, selected empirically; the optimal layer may differ for other models. (2) Sequences below 300 bp have reduced sensitivity (85%). (3) Non-coding specificity is imperfect: tRNA sequences show the inversion due to codon-anticodon structure, and compact prokaryotic intergenic regions may contain unannotated coding elements. (4) Boundary resolution is limited to ~100 bp by the smoothing window. (5) The approach requires GPU access for embedding extraction, though the embedding need only be computed once per sequence.

## Methods

### Embedding extraction

Per-position embeddings were extracted from block 10 of Evo2 40B (8,192 dimensions per position) via the NVIDIA NIM API using the `output_layers: [blocks.10]` parameter. Block 10 was selected based on systematic profiling of all 32 blocks, which showed maximal coding/intergenic contrast at intermediate layers. For self-hosted extraction on GPU clusters (NVIDIA L40S), the equivalent layer `decoder.layers.10` (4,096 dimensions) was used via the self-hosted NIM Evo2 7B container.

### Offset-3 cosine inversion

For each sequence, cosine similarity was computed between embeddings at positions i and i+k for offsets k = 1, 2, ..., 15. The inversion signal was defined as cos(offset-3) − cos(offset-1), averaged across all valid position pairs. A sequence was classified as "coding" if this value was positive. For genome-scale analysis, the inversion signal was smoothed with a uniform filter (window size as specified, typically 100 bp).

### Comprehensive validation panel

A panel of 510 sequences was assembled: 180 for universality testing (Component A: diverse species and genes across all domains), 300 for functional clustering analysis (Component B: 10 gene families × 30 orthologs), and 30 negative controls (Component C: diverse genes not in the 10 families). Coding sequences from NCBI RefSeq were selected to maximize taxonomic diversity, spanning 55 phyla and GC content from 9.8% to 78.8%. Non-coding controls included 5 each of ribosomal RNA, tRNA, long non-coding RNA, intronic, intergenic, and repetitive element sequences.

### Exon-intron detection

Genomic sequences for 36 eukaryotic genes were downloaded from NCBI with gene annotations. Per-position embeddings were extracted at blocks.10 (40B cloud NIM). The inversion signal was smoothed (100 bp uniform filter) and thresholded at zero to classify each position as exon (positive) or intron (negative). Precision, recall, F1, and accuracy were computed against NCBI CDS annotations as ground truth.

### Codon table analysis

64 synthetic sequences (61 sense codons + 3 stop codons, each consisting of 100 repeats = 300 bp) were embedded at blocks.10 and mean-pooled. Pairwise cosine distances were computed between all codons. Within-amino-acid versus between-amino-acid distances were compared, and PCA was used to visualize codon embedding space colored by amino acid biochemical property.

### Functional clustering analysis

Mean-pooled embeddings for 287 sequences from 10 gene families were extracted at three configurations: Evo2 40B blocks.10, 40B blocks.28, and 7B decoder.layers.10. Silhouette scores and nearest-neighbor gene-family accuracy were computed using cosine distance in PCA-reduced space (50 components).

### K-mer baseline

Trinucleotide frequency vectors (64 features) were computed for each sequence in the comprehensive panel. Binary coding detection was performed using a Random Forest classifier (200 trees, 5-fold stratified cross-validation).

## References

1. Fickett, J. W. Recognition of protein coding regions in DNA sequences. *Nucleic Acids Research* **10**, 5303–5318 (1982).
2. Borodovsky, M. & McIninch, J. GENMARK: parallel gene recognition for both DNA strands. *Computers & Chemistry* **17**, 123–133 (1993).
3. Delcher, A. L. et al. Improved microbial gene identification with GLIMMER. *Nucleic Acids Research* **27**, 4636–4641 (1999).
4. Dalla-Torre, H. et al. The Nucleotide Transformer: building and evaluating robust foundation models for human genomics. *Nature Methods* **21**, 1–11 (2024).
5. Brixi, G. et al. Genome modeling and design across all domains of life with Evo 2. *Nature* (2026).
6. Fishman, V. et al. GENA-LM: a family of open-source foundational DNA language models. *Bioinformatics* **39**, btad650 (2023).
7. Nguyen, E. et al. Sequence modeling and design from molecular to genome scale with Evo. *Science* **386**, eado9336 (2024).
8. Goodfire AI. Interpreting Evo 2. Blog post, https://www.goodfire.ai/research/interpreting-evo-2 (2025).
9. Arc Institute. Evo2 mechanistic interpretability. https://arcinstitute.org/tools/evo/evo-mech-interp (2026).
10. Merchant, A., King, N., Nguyen, E. & Hie, B. Semantic design of functional de novo genes from a genomic language model. *Nature* (2025).
11. Zhou, Z. et al. DNABERT-2: efficient foundation model and benchmark for multi-species genome. *arXiv* 2306.15006 (2023).
12. Schiff, Y. et al. Caduceus: bi-directional equivariant long-range DNA sequence modeling. *arXiv* 2403.03234 (2024).
13. Nguyen, E. et al. HyenaDNA: long-range genomic sequence modeling at single nucleotide resolution. *arXiv* 2306.15794 (2023).
