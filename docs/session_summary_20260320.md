# Session Summary: March 19-20, 2026

## What We Accomplished

### Major Experiments Completed

1. **Comprehensive validation (489 sequences)**
   - 510-sequence panel designed (180 universality + 300 functional clustering + 30 controls)
   - 489 downloaded, extracted on HTCF (7B, decoder.layers.10)
   - Codon periodicity: **452/459 (98.5%)**, 100% above 500bp, 55 phyla
   - Non-coding specificity: 76.5% for valid controls; tRNA and bacterial intergenic are biological exceptions
   - 7 failures all in short sequences (<300bp)

2. **Functional clustering — definitively negative**
   - Tested 40B blocks.10, 40B blocks.28, 7B layer.10 (N=287, 10 gene families)
   - Silhouette: -0.064 to -0.165, NN accuracy: 13-20%
   - Embedding Atlas UMAP was a visualization artifact
   - Mean-pooled embeddings support classification but NOT functional annotation

3. **Multi-offset cosine analysis (50 sequences)**
   - Offsets 1-15 reveal a 3-periodic "comb filter" in coding DNA
   - Mod-3 offsets: 0.41-0.49; non-mod-3: 0.29-0.31
   - Non-coding: smooth decay, no periodicity
   - Strongest single visualization of the finding

4. **Genetic code table in embedding space (64 codons)**
   - Stop codons cluster tightly (1.55× ratio) — model learned stop signals
   - Synonymous codons weakly closer (1.32×) — driven by sequence similarity, not amino acid identity
   - Amino acid properties NOT encoded (silhouette -0.159)
   - The model learned SYNTAX (triplets, stops) but not SEMANTICS (amino acid identity)

5. **Intra-codon structure**
   - Offset-1 > offset-2 in 100% of coding sequences — but also 100% of non-coding
   - This is a general sequential property, not wobble-specific
   - Mod-3 residue classes (class 1 vs class 2) are indistinguishable (diff = 0.0001)

6. **Taxonomy pilot (cached data, no new API calls)**
   - Same-genome fragments: 4.34× closer than between-genome (6,526 phage fragments)
   - Top-1 NN same-genome: 13.5% (135× random)
   - 5-class KNN: 85.3% from embedding geometry alone

7. **NIM API investigation**
   - API changed format: `output_layers` list, NPZ response, float64
   - Late blocks (25+) MLP returns near-zero; block 10 optimal for per-position
   - Self-hosted 7B returns (seq_len, 1, 4096) vs cloud 40B (1, seq_len, 8192)

### Paper Framing (Final)

**Title**: "Per-position DNA foundation model embeddings encode the triplet genetic code across all domains of life"

**Target**: Nature Methods

**Primary finding**: Offset-3 cosine inversion — 98.5% at N=459, 100% >500bp

**Key figures**:
- Fig 1: Schematic + comb filter + trajectory + box plot + GC scatter (v5 generated)
- Fig 2: Applications (RNA dark matter, coding detection, prophage)
- Fig 3: Multi-task (detection, clustering, phylogenomics + negative functional clustering)
- Fig 4: K-mer baseline comparison

**Important negative results**:
- Functional clustering does not exist in mean-pooled embeddings
- Amino acid identity/properties not encoded
- The model learned DNA syntax (triplets, stops) but not protein semantics

### Running on HTCF (check when back)

- **E. coli K12 full genome scan** (job 37862760): 386 windows × ~1 min each = ~6.5 hours
  - Check: `ssh shandley@login.htcf.wustl.edu "ls -lh /scratch/sahlab/shandley/virosense/ViroSense/results/genome_scan/"`

### Disk Usage

- Cleaned ~60 GB of per-position .npy files earlier
- Current local results/: ~1 GB
- Large files on HTCF scratch (2 TB available)

### Files Created/Modified This Session

**New scripts:**
- `scripts/comprehensive_validation.py` — panel-driven extraction + analysis pipeline
- `scripts/codon_periodicity_panel.py` — 510-sequence panel definition + download/extract/analyze
- `scripts/build_functional_panel.py` — NCBI ortholog search for gene families
- `scripts/analyze_functional_clustering_40b.py` — 40B vs 7B functional clustering comparison
- `scripts/generate_fig1_v4.py` — Figure 1 with multi-offset comb filter
- `scripts/generate_fig1_v5.py` — Figure 1 with schematic + trajectory + comb filter
- `scripts/genome_scan.py` — Full genome per-position scan

**New HTCF jobs:**
- `htcf/comprehensive_extraction.sbatch`
- `htcf/genome_scan.sbatch`

**New documentation:**
- `docs/comprehensive_validation_results.md` — full N=489 results
- `docs/functional_clustering_validation.md` — experiment design + results
- `docs/embedding_atlas_findings.md` — UMAP visualization analysis
- `docs/nim_api_layer_investigation.md` — layer profiling
- `docs/findings_log.md` — running log of all 16 findings
- `docs/session_summary_20260320.md` — this file

**Updated documentation:**
- `docs/publication_plan.md` — v4 framing, revised figures, negative results
- `docs/prior_art.md` — added Merchant et al., FANTASIA, gLM references
- `results/figures/fig1_caption.md` — updated for v5
- `results/PAPER_ASSETS.md` — updated for new figure structure
- Memory files updated

**New results:**
- `results/comprehensive/panel.json` — 510-sequence panel definition
- `results/codon_periodicity_panel/embeddings/` — 489 × metrics + mean-pooled (7B)
- `results/codon_periodicity_panel/embeddings_40b/` — 287 × mean-pooled at blocks.10 + blocks.28 (40B)
- `results/codon_periodicity_panel/multi_offset_expanded.json` — 50-sequence multi-offset data
- `results/codon_periodicity_panel/multi_offset_cosine.json` — 12-sequence multi-offset data
- `results/codon_periodicity_panel/functional_clustering_comparison.json` — 40B vs 7B
- `results/codon_table_embeddings/codon_embeddings.json` — 64 codon-repeat embeddings
- `results/figures/fig1_v5.png/pdf` — current Figure 1
- `results/genome_scan/` — E. coli genome + annotations (metrics pending HTCF)
