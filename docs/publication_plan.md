# ViroSense Publication Plan

**Working title**: "Per-position DNA foundation model embeddings encode the triplet genetic code across all domains of life"

**Alternative titles**:
- "A DNA foundation model discovers the genetic code through unsupervised learning"
- "The genetic code emerges in per-position embeddings of a nucleotide-resolution foundation model"

**Target journal**: Nature Methods (with Nature as stretch goal pending functional clustering validation)

Last updated: 2026-03-20

---

## Central Thesis

A DNA foundation model trained solely on next-nucleotide prediction **discovers the triplet genetic code** — and this discovery is universal across all domains of life, exploitable for novel applications, and reveals what foundation models actually compute about DNA.

Key contributions (ranked by novelty and impact):

1. **The offset-3 cosine inversion** — a novel, quantitative signature of codon structure in per-position embeddings. **No prior work has identified this signal.**
2. **Universality across all life** — validated on 459 coding sequences spanning 55 phyla from all three domains, GC 9.8–78.8%. 98.5% sensitivity overall, **100% for sequences >500 bp**. Independent of GC content, genetic code variant, and genome size. Every major taxonomic group at 100%.
3. **Non-coding specificity** — absent from rRNA, lncRNA, intronic, and repetitive DNA (76.5%). tRNA shows inversion due to codon-anticodon structure. Compact prokaryotic intergenic regions are unreliable negative controls.
4. **Novel applications** — 94.7% coding detection without gene calling, 97.5% database-free RNA virus identification from periodicity alone.
5. **Honest positioning** — k-mer baselines achieve 93% for binary classification at 1,527× the speed. Foundation models add per-position analysis, zero-shot generalization, and characterization that k-mers cannot replicate.
6. **Negative result** — mean-pooled embeddings do NOT cluster by protein identity despite visual appearance in UMAP (tested 40B + 7B, two layers, N=287, 10 gene families). Mean-pooled representations support classification but not functional annotation via embedding similarity.

### Negative Result: Functional Clustering (Definitively Tested)

Mean-pooled Evo2 embeddings do **NOT** cluster by protein identity at publication quality. Tested exhaustively:
- **40B blocks.10** (8192-D): Silhouette -0.064, NN accuracy 19.9%
- **40B blocks.28** (8192-D): Silhouette -0.140, NN accuracy 12.9%
- **7B layer.10** (4096-D): Silhouette -0.165, NN accuracy 20.2%
- N=287 sequences, 10 gene families × ~29 orthologs each
- Within/between distance ratios 0.9-1.2× (no meaningful cohesion)
- The Embedding Atlas UMAP visualization was a **visualization artifact**, not a real signal.

The model does not encode "which protein" a gene codes for in mean-pooled representations — at least not accessible via cosine similarity. Mean-pooled embeddings support classification and clustering tasks (95.4% viral detection, ARI=0.903) but this reflects learned compositional features, not protein identity.

### Relation to Prior Work

| Prior work | What they showed | What we add |
|-----------|-----------------|-------------|
| **Merchant et al. (Nature 2025)** — Evo1 "semantic design" | Functional *context*: gene neighbors predict function (generative) | Quantitative codon periodicity; functional clustering signal (pending validation) |
| **Goodfire/Arc (2025)** — Evo2 SAE interpretability | Per-position features: coding, secondary structure, prophage | Offset-3 cosine inversion as exploitable signal; 98.5% universality validation |
| **FANTASIA (Comm Bio 2025)** — protein LM functional annotation | Protein embedding similarity transfers GO terms across species | Per-position codon structure from DNA alone (complementary finding) |
| **gLM (Nat Comms 2024)** — genomic language model | Gene function clustering using **ESM protein embeddings** as input | DNA-only functional tendency (weaker than protein LMs but present) |
| **Nucleotide Transformer (Nat Methods 2024)** | Coding/non-coding separation in embeddings | Quantitative codon periodicity + universality across all domains |
| **cdsFM/CodonFM (2024)** | Codon-level models learn genetic code | Nucleotide-resolution model discovers codons without codon tokenization |

---

## What's Changed Since the Original Plan

| Phase | Framing |
|-------|---------|
| **v1** (Feb 2026) | "Viral detection tool that beats geNomad" |
| **v2** (mid-March) | "Universal characterization framework with viral detection as one application" |
| **v3** (March 18) | "Fundamental discovery about what DNA foundation models compute — codon periodicity" |
| **v4** (current) | **"The model learned the central dogma — codon structure AND protein identity from DNA alone"** |

| v3 framing | v4 framing (current) |
|------------|---------------------|
| Codon periodicity is the primary finding | **Two orthogonal findings**: codon structure (per-position) + protein identity (mean-pooled) |
| One main result + applications | **Two main results** that together show the model learned multi-scale biology |
| Target: Nature Methods | **Target: Nature** — this is a fundamental insight about foundation models + biology |
| Functional clustering is a supplementary observation | Functional clustering is a **co-primary finding** (potentially more novel than periodicity) |
| "Embed once, analyze many ways" is a practical framework | "Embed once, analyze many ways" reflects the model having **learned the central dogma** |
| Prior art comparison focuses on periodicity | Prior art comparison includes **Merchant et al. (Nature 2025), FANTASIA, gLM** |

---

## Key Findings (ranked by novelty, not by order discovered)

### 1. The Genetic Code in Embeddings (most novel)
- 3bp codon periodicity is the **dominant FFT frequency** in Evo2 per-position representations (for high-coding-density sequences)
- Lag-3 autocorrelation: 0.635 across 40 diverse metagenomic sequences, universal across all categories
- Offset-3 cosine inversion: 94.7% coding detection accuracy in mixed coding/intergenic fragments (40 seqs)
- Evo2 learned the triplet code from unsupervised next-nucleotide prediction
- **Universal across all domains of life**: 72 sequences spanning 25+ phyla, GC content 20–67%. Offset-3 cosine inversion present in **64/65 coding sequences (98.5%)** and **6/6 non-coding controls correctly lack it (100%)**. One apparent "intergenic" false positive turned out to be mislabeled — 95% of the sequence was lacY+lacZ CDS, which the model correctly identified. Tested at Evo2 block 10 (empirically optimal layer, see `docs/nim_api_layer_investigation.md`).
- **Every taxonomic group shows 100% inversion**: Archaea (10/10), Mammals (5/5), Fish (3/3), Insects (5/5), Plants (8/8), Fungi (4/4), Protists (7/7), Algae (3/3), Organellar genes (3/3), Viruses (4/4).
- **Key edge cases**: Plasmodium falciparum (24.8% GC, extreme AT-bias) — inversion present (gap +0.184). Human mitochondrial CO1 (non-standard genetic code) — inversion present. Synthetic eGFP (codon-optimized) — inversion present. Signal is NOT a GC artifact, NOT dependent on standard genetic code, and present in engineered sequences.
- **Original 40-seq metagenomic panel**: 94.7% coding detection accuracy using the inversion as a binary classifier (mixed coding/intergenic fragments).
- **No prior work has shown this.** This is a fundamental insight about DNA foundation models.

### 2. RNA Dark Matter Detection (high novelty + high impact)
- 97.5% accuracy, 0.990 AUC distinguishing eukaryotic RNA viruses from all other categories
- From per-position periodicity features alone — **no database, no homology, no gene calling**
- cos3 (offset-3 cosine) is the dominant feature (Cohen's d = 2.83)
- Zero Firmicutes phage false positives (0/25)
- 203 sequences validated across 4 categories
- **First database-free RNA virus identifier from DNA composition**

### 3. Universal Characterization Framework (architectural contribution)
- `virosense characterize` produces multi-dimensional "DNA passports"
- Identity (nearest category + anomaly score), Origin (viral/RNA/mobile signatures), Structure (coding, periodicity), Novelty (percentile against reference)
- 100% RNA origin interpretation accuracy on 25 diverse test sequences
- Anomaly scoring flags novel elements (Obelisk-like discoveries)
- **One Evo2 forward pass → comprehensive biological profile**

### 4. K-mer Baseline Comparison (honest, strengthens the paper)
- Trinucleotide frequencies achieve **93% viral detection** at **1527× the speed** of Evo2
- Gap analysis: 6.3% of sequences need Evo2 (host-adapted phages, low-GC false positives, short fragments)
- The gap is **biologically meaningful** — Evo2 captures contextual composition beyond k-mer bags
- Two-tier pipeline: k-mers screen everything, Evo2 characterizes the borderline 15%
- **This is the honest comparison most papers don't include**

### 5. Viral Detection Benchmark (validation, not main contribution)
- 40B: 99.7% phage sensitivity, 93.0% RNA virus recall, 95.4% overall accuracy
- vs geNomad: ViroSense wins on short fragments (99.7% vs 51.2% at 1-3kb), geNomad wins on plasmid specificity (99.3% vs 81.5%) and speed (500×)
- Bootstrap CIs: all key comparisons non-overlapping
- 13,417 sequences, complete benchmark

### 6. Additional Validated Capabilities
- **3-class contig typing**: 94.5% CV accuracy (virus/plasmid/chromosome)
- **Unsupervised clustering**: HDBSCAN ARI=0.903, RNA viruses separate from dsDNA phages (99% pure cluster)
- **Alignment-free phylogenomics**: Spearman r=0.504 between embedding distance and taxonomic distance
- **Prophage amelioration**: Coarse-pass viral scores reveal evolutionary age gradient (DLP12 active → e14 invisible)
- **Gene boundary detection**: 91.7% coding accuracy, 73.2% boundary recall, 1.72× norm ratio across 41 sequences

### 7. Preprocessing Insights
- L2-normalization: essential for 7B (63%→93% RNA recall), counterproductive for 40B
- Training data composition: adding plasmids improves RNA virus recall 4.7%
- Model scale: preprocessing must match model tier
- **Transferable findings for anyone building classifiers on foundation model embeddings**

---

## Manuscript Outline (Universality-First Framing)

### Title
"Per-position DNA foundation model embeddings encode the triplet genetic code across all domains of life"

### Alternative titles
- "A DNA foundation model discovers the genetic code through unsupervised learning"
- "The genetic code emerges in per-position embeddings of a nucleotide-resolution foundation model"

### Prior Art Context (see docs/prior_art.md for full analysis)
- 3bp periodicity in coding DNA: known since 1982 (Fickett), NOT novel
- Foundation models learn gene structure: shown qualitatively (Goodfire blog 2025, NOT peer-reviewed; Nucleotide Transformer, Nature Methods 2024)
- Codon-level foundation models: cdsFM/CodonFM (2024) trains ON codons, learns genetic code — but given the reading frame by tokenization
- "What Do Biological Foundation Models Compute?" (bioRxiv March 2026): SAE framework, mentions coding features but NOT codon periodicity in embeddings
- **Our novelty**: (a) the *offset-3 cosine inversion* as a specific, exploitable binary signal — no prior art; (b) universality validation across 72 sequences from all domains of life (98.5% sensitivity, 100% non-coding specificity); (c) novel applications: 94.7% coding detection without gene calling, 97.5% database-free RNA virus identification from periodicity alone; (d) systematic layer profiling identifying optimal representation depth. The model discovers codon structure without being told about it — at single-nucleotide resolution, without codon tokenization.

### Abstract
DNA foundation models trained on nucleotide sequences learn representations that capture biological structure, but the specific features encoded in per-position embeddings remain poorly characterized. Here we show that frozen per-position embeddings from Evo2, a 40-billion parameter DNA model trained by next-nucleotide prediction, encode the triplet genetic code as a measurable structural feature. We identify an offset-3 cosine inversion — a geometric signature in which nucleotides separated by one codon are more similar in embedding space than adjacent nucleotides in coding regions, with this relationship inverting in non-coding regions. This signal is universal: validated on 459 protein-coding sequences spanning all three domains of life (55 phyla, 10 conserved gene families, GC content 9.8–78.8%), it achieves 98.5% sensitivity overall and 100% for sequences above 500 bp. The inversion is absent from ribosomal RNA, long non-coding RNA, intronic sequences, and repetitive elements, though present in tRNA (which has inherent codon-related structure). The signal is independent of GC content, genetic code variant (standard and mitochondrial), and genome size, and is present in synthetic codon-optimized sequences. We demonstrate that the per-position periodicity signal enables 94.7% coding region detection without gene calling and 97.5% database-free identification of RNA viral sequences from periodicity features alone. Mean-pooled embeddings from the same extraction support viral detection (99.7% phage sensitivity), contig typing (94.5%), unsupervised clustering (ARI=0.903), and alignment-free phylogenomics (r=0.504), though we find that mean-pooled representations do not encode protein identity — genes for the same protein from different species do not cluster in embedding space despite visual appearance in dimensionality-reduced projections. Trinucleotide frequency classifiers achieve 93% of Evo2's accuracy for binary classification at 1,527× the speed, but cannot replicate the per-position capabilities. These findings reveal that next-nucleotide prediction on DNA is sufficient for a model to discover the triplet organization of protein-coding regions across all life.

### Results

**1. Per-position embeddings encode the triplet genetic code** (Figure 1)
- Offset-3 cosine inversion: in coding regions, cosine similarity between positions i and i+3 exceeds that between i and i+1; in non-coding regions, the relationship inverts
- This is a binary signal: coding regions have cos3 > cos1, non-coding have cos1 > cos3
- Achieves 94.7% ± 4.9% coding region detection accuracy on metagenomic fragments without gene calling, ORF analysis, or training
- Norm signal: coding 1.72× ± 0.23 intergenic across 41 diverse sequences
- FFT confirms 3bp as dominant frequency in high-coding-density sequences
- RNA viruses show strongest periodicity (lag-3 = 0.822) vs dsDNA phages (0.624) — relates to coding density and codon usage bias
- Layer profiling: signal strongest at block 10 of 32 (systematic profiling of all layers)

**2. The offset-3 inversion is universal across all domains of life** (Figure 2)
- 72 sequences spanning 25+ phyla, GC content 20–67%
- **64/65 coding sequences (98.5%)** show inversion; **6/6 non-coding controls (100%)** correctly lack it
- 100% inversion rate in every taxonomic group: Archaea (10/10), Mammals (5/5), Fish (3/3), Insects (5/5), Plants (8/8), Fungi (4/4), Protists (7/7), Algae (3/3), Organellar (3/3), Viruses (4/4)
- GC-independent: no correlation between GC content and inversion strength
- Genetic code-independent: works with mitochondrial code (UGA=Trp), not just standard code
- Works on synthetic sequences: codon-optimized eGFP shows inversion
- Works on minimal genomes: Mycoplasma genitalium (580 kb genome) shows inversion
- **Mislabeled control validation**: an E. coli "intergenic" region that was 95% lacY+lacZ CDS was correctly identified as coding — the model caught our annotation error
- Single failure: alligator hemoglobin (gap = -0.010, borderline, 429 bp)

**3. Mean-pooled embeddings support multi-task analysis** (Figure 3)
- "Embed once, analyze many ways" — a single Evo2 forward pass enables:
- Viral detection: 99.7% phage sensitivity, 93.0% RNA virus recall, 95.4% overall (13,417 sequences)
- 3-class contig typing: virus/plasmid/chromosome at 94.5% cross-validated accuracy
- Unsupervised clustering: HDBSCAN ARI=0.903, RNA viruses form 99%-pure cluster
- Alignment-free phylogenomics: Spearman r=0.504 between embedding distance and taxonomic distance
- **Negative result**: mean-pooled embeddings do NOT cluster by protein identity (tested 40B + 7B, blocks.10 + blocks.28, N=287, 10 families, NN accuracy 13-20%, silhouette negative). UMAP visualizations are misleading for this task.

**4. Foundation models vs k-mer baselines** (Figure 4)
- **Per-position applications**: RNA dark matter detection (97.5%, 0.990 AUC from periodicity alone), coding detection (94.7%), gene boundary detection (73.2% recall)
- **Mean-pooled applications**: viral detection (99.7% phage, 95.4% overall), contig typing (94.5%), unsupervised clustering (ARI=0.903), phylogenomics (r=0.504)
- **Combined**: DNA passport characterization (identity + origin + structure + novelty from one embedding), prophage amelioration gradient
- Implications: RNA dark matter discovery, alignment-free functional annotation from DNA, novel element detection

**6. Foundation models vs k-mer baselines: complementary, not competing** (Figure 6 or supplementary)
- Trinucleotide frequency classifiers achieve 93% viral detection at 1,527× the speed
- The 2.4% accuracy gap is biologically meaningful: Evo2 captures contextual composition beyond k-mer bags
- Gap analysis: 6.3% of sequences need Evo2 — host-adapted phages, short fragments, low-GC false positives
- Foundation model unique capabilities: per-position analysis, zero-shot generalization to unseen types, compositional characterization, anomaly detection
- Two-tier architecture: k-mers screen everything at scale, Evo2 characterizes the borderline fraction
- **Honest framing**: this is the comparison most papers don't include

### Discussion
- **What the model learned — and what it didn't.** Evo2 learned the structural syntax of the genetic code (triplet organization, stop codon boundaries, 3-periodic comb filter) but not the functional semantics (which amino acid a codon encodes, amino acid biochemical properties). This delineates the boundary between what can be learned from DNA sequence statistics alone versus what requires protein-level evolutionary pressure. Stop codons are learnable from DNA (they mark gene boundaries, a sequence-level pattern); the codon→amino acid mapping is not (it manifests at the protein level, invisible to DNA prediction).
- **Relation to prior work**: Fickett (1982) showed 3bp periodicity in DNA sequences. Nucleotide Transformer showed coding/non-coding separation. cdsFM trained on codons. Merchant et al. (Nature 2025) showed Evo1 learns functional context. We provide the first quantitative per-position measurements (offset-3 inversion, comb filter) and the most comprehensive universality validation (459 sequences, 55 phyla).
- **Why per-position matters**: mean-pooled embeddings support classification (95.4%) and clustering (ARI=0.903) but discard the per-position structure that encodes the genetic code. The offset-3 inversion, comb filter pattern, and coding/non-coding transitions are invisible in mean-pooled representations. Per-position analysis enables capabilities (RNA dark matter detection, coding detection, gene boundary identification) that neither mean-pooled embeddings nor k-mer features can replicate.
- **The functional clustering negative result** is important: despite visual appearance in UMAP, mean-pooled embeddings do not cluster by protein identity (tested 40B + 7B, two layers, N=287, 10 gene families). UMAP visualization can be misleading for high-dimensional biological data — proper quantification with silhouette scores and nearest-neighbor accuracy is essential.
- **Generalizability**: the offset-3 inversion should be testable in any DNA foundation model (Nucleotide Transformer, DNABERT-2, Caduceus, HyenaDNA). We predict it will be present in any model that achieves good next-nucleotide prediction accuracy on coding DNA.
- **Limitations**: (1) Speed — Evo2 is ~1,500× slower than k-mers; GPU required. (2) Layer dependence — signal strongest at intermediate layers (block 10 of 32). (3) Short sequences (<300bp) have 85% sensitivity vs 100% for >500bp. (4) Non-coding specificity is category-dependent: tRNA shows inversion (codon-anticodon structure), compact prokaryotic intergenic regions are unreliable controls. (5) The model does not encode amino acid identity or properties.
- **Future**: knowledge distillation for planetary-scale screening, extension to RNA foundation models, multi-model comparison, per-position analysis of eukaryotic genes with introns (exon-intron boundary detection), full-genome scans for gene annotation

### Methods
- Evo2 40B embedding extraction (NVIDIA NIM API, per-position via `output_layers: [blocks.10]`)
- Layer selection: systematic profiling of all 32 blocks; block 10 selected for strongest coding/intergenic contrast
- Cross-domain validation panel: 72 sequences (65 coding + 7 non-coding controls) from NCBI RefSeq, CDS-only extractions (no UTRs), spanning all 3 domains of life
- Offset-3 cosine inversion: cosine similarity between embeddings at positions i and i+1 (cos1) vs i and i+3 (cos3); inversion = cos3 > cos1
- Autocorrelation: normalized lag-k autocorrelation of per-position L2 norms
- FFT: power spectrum of per-position norms, dominant frequency identification
- Gene boundary detection: peaks in norm first-derivative as boundary candidates
- RNA dark matter classifier: Random Forest on 6 periodicity features (cos1, cos3, lag1, lag3, norm_cv, coding_fraction)
- Mean-pooled classifier: sklearn MLPClassifier(512, 128) with Platt calibration on frozen 8,192-D embeddings
- K-mer baseline: 84-feature vector (64 trinucleotide + 16 dinucleotide + GC + frame entropy), Random Forest
- Benchmark: Gauge Your Phage (13,417 sequences) + RNA Virus Database (1,000 sequences)
- Bootstrap CIs: 10,000 resamples, BCa method
- geNomad v1.11.2 comparison on shared benchmark manifest
- HDBSCAN clustering with PCA preprocessing (auto ~90% variance)
- Alignment-free phylogenomics: pairwise cosine distance vs ICTV taxonomic distance

---

## Figure Set (5 main + supplementary)

### Figure 1: Per-Position Embeddings Encode the Triplet Genetic Code
*Discovery + universality of the structural signal*
- A: **The signal** — Smoothed offset-3 vs offset-1 cosine along E. coli lac operon (coding + intergenic). Inversion visible by eye.
- B: **The triplet fingerprint** — Bar chart of mean cosine similarity at offsets 1-15 for coding vs non-coding sequences. Coding shows a 3-periodic "comb filter" pattern (multiples of 3 are 0.44-0.51, all others 0.32-0.34). Non-coding shows smooth decay with no periodicity. This is the most direct proof the model learned triplet structure.
- C: **Universal across all domains** — Box plot of inversion signal (cos3-cos1) by taxonomic group (N=459, 55 phyla). Clean coding/non-coding separation.
- D: **GC-independent** — Scatter: GC (9.8–78.8%) vs inversion signal. No correlation. Edge cases labeled.
- E: **Summary** — 452/459 (98.5%) coding, 100% above 500bp, 7 failures all in short sequences.

### Figure 2: Applications of Per-Position Periodicity
*What the codon structure discovery enables*
- A: RNA dark matter detection — 97.5% accuracy, 0.990 AUC from periodicity features alone. No database, no homology.
- B: Feature importance — cos3 dominates (50.9%, Cohen's d = 2.83). RNA viruses have strongest periodicity.
- C: Coding region detection — 94.7% from inversion threshold, no gene calling needed.
- D: Prophage amelioration gradient — viral scores as evolutionary age proxy.

### Figure 3: Multi-Task DNA Analysis from Mean-Pooled Embeddings
*One extraction supports classification, clustering, and phylogenomics*
- A: Viral detection (99.7% phage, 95.4% overall) + 3-class contig typing (94.5%).
- B: HDBSCAN clustering (ARI=0.903, RNA viruses form 99%-pure cluster).
- C: Alignment-free phylogenomics (embedding distance hierarchy, r=0.504).
- D: Negative result — mean-pooled embeddings do NOT cluster by protein identity (silhouette -0.06 to -0.21, NN 13-20%, tested 40B + 7B, 2 layers, N=287, 10 gene families). UMAP visualization was misleading.

### Figure 4: Foundation Models vs K-mer Baselines
*The honest comparison — complementary, not competing*
- A: K-mer baseline: 93% viral detection at 1,527× speed.
- B: Gap analysis: where per-position Evo2 adds unique value (host-adapted phages, short fragments).
- C: Capability matrix: k-mers vs per-position vs mean-pooled (detection, periodicity, gene structure, zero-shot, characterization).
- D: Two-tier pipeline: k-mers screen, Evo2 characterizes.

### Supplementary
- S1: Layer profiling — Evo2 blocks 0-31 profiled for periodicity signal (block 10 optimal)
- S2: Comprehensive validation data (489 sequences, all metrics, by domain/phylum/family/GC/length)
- S3: Non-coding specificity by category (rRNA, lncRNA, tRNA, intron, intergenic, repeats)
- S4: Functional clustering negative result (40B blocks.10/28 + 7B layer.10, N=287, 10 families)
- S5: L2-normalization analysis (7B vs 40B) — essential for 7B, counterproductive for 40B
- S6: ViroSense vs geNomad detailed head-to-head with bootstrap CIs
- S7: E. coli K12 prophage amelioration scores (9 known cryptic prophages)
- S8: Mislabeled intergenic control — 95% lacY+lacZ CDS correctly identified by model

---

## What's Ready vs What's Needed

| Component | Status | Action needed |
|-----------|--------|--------------|
| All benchmark data | ✅ Complete | None |
| K-mer baseline comparison | ✅ Complete | None |
| RNA dark matter (203 seqs) | ✅ Complete | None |
| Codon periodicity (40 seqs) | ✅ Complete | Superseded by comprehensive |
| Comprehensive validation (489 seqs) | ✅ Complete | 98.5%, 55 phyla |
| Multi-offset comb filter (50 seqs) | ✅ Complete | 3-periodic pattern confirmed |
| Functional clustering (287 seqs, 3 configs) | ✅ Complete | **NEGATIVE** — not a finding |
| Codon table in embedding space (64 codons) | ✅ Complete | Stop codons cluster; AA identity not encoded |
| Gene boundaries (41 seqs) | ✅ Complete | None |
| Phylogenomics pilot | ✅ Complete | None |
| Prophage amelioration | ✅ Complete (coarse) | Full scan nice-to-have |
| Taxonomy pilot (cached data) | ✅ Complete | 4.34× within-genome ratio |
| Forensics pilot | ✅ Complete | Supplementary |
| Characterize framework | ✅ Complete | None |
| Two-tier --fast pipeline | ✅ Complete | None |
| **E. coli genome scan** | ⏳ Running on HTCF | Job 37862760, ~4 hrs remaining |
| **Figure 1** (v5) | ✅ Complete | Schematic + comb + trajectory + boxplot + GC |
| **Figures 2-4** | ❌ Not started | Data ready, need visualization |
| **Manuscript text** | ❌ Not started | After figures |

---

## Timeline

| Task | Effort | Priority |
|------|--------|----------|
| Check genome scan results | 10 min | **NEXT** |
| **Generate figures 2-4** | 2-3 days | **NOW** |
| **Write manuscript** | 2-3 weeks | After figures |
| Real metagenome validation | 3-5 days | If time permits |
| Revisions | Ongoing | After submission |
