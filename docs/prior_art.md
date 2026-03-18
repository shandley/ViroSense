# Prior Art and Novelty Assessment

Last updated: 2026-03-18

## Our Key Claims and Their Novelty

### Claim 1: Per-position embedding norms distinguish coding from intergenic (1.72× ratio)

**Prior art**:
- Goodfire/Arc (Feb 2025, blog post, NOT peer-reviewed): SAE on Evo2 layer 26 found features for coding regions, prophage, exon-intron boundaries. Demonstrated via feature activation maps, not quantified as a norm ratio.
- Nucleotide Transformer (Nature Methods 2024): Showed embeddings separate coding/intronic/intergenic at the classification level, not per-position norm analysis.
- Evo2 paper (Nature 2026): Mentions coding region features in SAE analysis (section 2.4) but does not quantify per-position norm ratios.

**Our contribution**: Quantified the norm ratio (1.72× ± 0.23) across 41 diverse sequences in 5 categories. Showed it enables 91.7% coding detection from a simple threshold. This is a quantitative extension of known qualitative observations.

**Novelty level**: INCREMENTAL. The observation is expected; the quantification and application are new.

### Claim 2: 3bp codon periodicity in per-position embedding autocorrelation

**Prior art**:
- 3-base periodicity in DNA sequences: Known since Fickett 1982. Extensively used for gene finding (GeneMark, Glimmer). Based on nucleotide frequencies, NOT embeddings.
- cdsFM/CodonFM (Arc/NVIDIA, 2024, PMC): Foundation models trained ON codons as tokens learn the genetic code structure. But these models are GIVEN the reading frame — they tokenize at codon level.
- Goodfire SAE (blog, 2025): Found coding region features but did NOT report periodicity, autocorrelation, or FFT analysis of per-position embeddings.
- Nucleotide Transformer (Nature Methods 2024): Third codon position predicted with lower confidence than first two. This implies the model knows about codon structure, but it was observed in prediction confidence, not in embedding periodicity.
- "What Do Biological Foundation Models Compute?" (bioRxiv March 2026): SAE framework for biological models, mentions coding features but NOT codon periodicity in embeddings.

**Our contribution**: First demonstration that a nucleotide-resolution DNA model's per-position embeddings exhibit measurable 3bp autocorrelation. The lag-3 peak (0.635), FFT dominance, and offset-3 cosine inversion are novel measurements. The model discovers codon structure without being tokenized at codon level.

**Novelty level**: MODERATE-HIGH. The biological phenomenon is known; showing it emerges in nucleotide-level foundation model embeddings and can be exploited for coding detection (94.7%) is new.

**Key distinction from cdsFM**: cdsFM trains on codons (knows the frame). We show a nucleotide-level model discovers the frame from raw sequence. Different finding.

**Key distinction from Goodfire**: They used SAE feature visualization. We use autocorrelation/FFT on raw embedding norms — a simpler, more quantitative approach that produces directly exploitable signals.

### Claim 3: Offset-3 cosine inversion as universal coding signature

**Prior art**: None found. No prior work has measured cosine similarity between embeddings at different offsets and shown the inversion between coding and intergenic regions.

**Novelty level**: HIGH. Novel measurement and novel application (94.7% coding detection).

### Claim 4: RNA viruses have strongest codon periodicity (0.822 vs 0.624)

**Prior art**:
- RNA virus codon usage bias is well-studied (Jenkins & Holmes, PLOS ONE 2003; multiple reviews). The literature shows complex patterns — not simply "RNA > DNA."
- No prior work has compared foundation model embedding periodicity across viral categories.

**Our contribution**: Empirical observation from Evo2 embeddings. The mechanism (why RNA virus embeddings show stronger periodicity) is not fully explained. May relate to coding density, genome compactness, or RNA-specific compositional constraints.

**Novelty level**: MODERATE. The observation is new; the biological explanation needs more work.

### Claim 5: Database-free RNA dark matter detection (97.5%)

**Prior art**: No prior tool identifies RNA viruses from DNA composition alone without reference databases. geNomad and other tools use marker genes/protein domains.

**Novelty level**: HIGH. Novel capability, validated on 203 sequences.

### Claim 6: DNA passport characterization framework

**Prior art**: No prior tool produces multi-dimensional biological profiles from foundation model embeddings combining identity, origin, structure, and novelty scoring.

**Novelty level**: HIGH. Novel framework.

### Claim 7: Prophage amelioration gradient from embedding scores

**Prior art**:
- Prophage amelioration is well-studied (Lawrence & Ochman 1997, others). Measured via GC content, dinucleotide frequencies, codon usage comparison to host.
- No prior work uses foundation model embeddings to measure amelioration state.

**Novelty level**: MODERATE-HIGH. Novel method for a known phenomenon. The e14 invisibility finding is a nice contribution.

### Claim 8: K-mer baselines achieve 93% for viral detection

**Prior art**: DeepVirFinder uses CNNs on k-mer features (~80-90% accuracy). VirFinder uses k-mer signatures. The specific comparison of trinucleotide RF vs Evo2 embeddings is new.

**Novelty level**: LOW (the comparison), but HIGH (the honest framing of when foundation models add value).

## Key References

### Peer-Reviewed
| Reference | Year | Relevance | Status |
|-----------|------|-----------|--------|
| Evo2 (Nature) | 2026 | Base model, SAE mentions coding features | Peer-reviewed |
| Nucleotide Transformer (Nature Methods) | 2024 | DNA foundation model, coding/non-coding separation | Peer-reviewed |
| cdsFM/CodonFM (PMC) | 2024 | Codon-level foundation models learn genetic code | Peer-reviewed |
| geNomad (Nature Biotechnology) | 2024 | Marker-gene viral detection, our comparison target | Peer-reviewed |
| DNA Foundation Models Benchmark (Nature Comms) | 2025 | Benchmarks 5 DNA models on genomic tasks | Peer-reviewed |
| Jenkins & Holmes (PLOS ONE) | 2003 | RNA virus codon usage bias | Peer-reviewed |
| Lawrence & Ochman (Genetics) | 1997 | Amelioration of laterally transferred genes | Peer-reviewed |

### NOT Peer-Reviewed
| Reference | Year | Relevance | Status |
|-----------|------|-----------|--------|
| Goodfire Evo2 Interpretability | 2025 | SAE features for coding, prophage | **Blog post only** |
| SAE Bio Foundation Models (bioRxiv) | 2026 | SAE framework, mentions coding features | Preprint |

## Implications for the Paper

### What to emphasize
- The **offset-3 cosine inversion** (Claim 3) — genuinely novel, no prior art
- The **exploitability** of the periodicity (94.7% coding detection, 97.5% RNA dark matter) — novel applications
- The **quantitative characterization** across sequence types — new data
- The **characterization framework** (DNA passports) — novel concept
- The **honest k-mer comparison** — novel framing

### What to be careful about
- Don't overclaim the coding/intergenic norm difference (Claim 1) — Goodfire showed this qualitatively
- Don't overclaim "discovery of codon periodicity" — 3bp periodicity in DNA is 50-year-old knowledge
- Acknowledge cdsFM — they showed codon-level models learn the genetic code, but from codon tokens
- Acknowledge Nucleotide Transformer — they showed coding/non-coding separation in embeddings

### Suggested framing
> "The 3-base periodicity of protein-coding DNA has been known for decades and exploited in gene-finding algorithms. We show that this fundamental property manifests as a measurable signal in nucleotide-resolution foundation model per-position embeddings — specifically as a lag-3 autocorrelation peak and an offset-3 cosine inversion between coding and intergenic regions. While prior work has shown that DNA foundation models learn to distinguish coding from non-coding sequences (Dalla-Torre et al. 2024, Goodfire 2025), we quantify this for the first time through per-position embedding analysis and demonstrate novel applications: 94.7% coding region detection without gene calling, and 97.5% database-free RNA virus identification from periodicity features alone."
