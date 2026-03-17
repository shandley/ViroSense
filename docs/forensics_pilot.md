# Perplexity Forensics Pilot — Detecting Codon-Optimized Sequences

Last updated: 2026-03-17

## Concept

DNA foundation models (Evo2) encode per-position information about sequence composition. Engineered sequences — particularly codon-optimized genes — have distinctive embedding signatures that differ from natural genes encoding the same protein. This enables database-free detection of synthetic sequences.

## Proof-of-Concept (Single Gene)

Compared per-position Evo2 40B embeddings for one protein (422 aa, Erwinia phage gene) encoded three ways:
1. **Natural** — original phage codons
2. **Codon-optimized** — all E. coli preferred synonymous codons
3. **Chimera** — first half natural, second half optimized

### Key finding: Lag-1 autocorrelation is a smoking gun

| Feature | Natural | Codon-optimized | Effect |
|---------|---------|-----------------|--------|
| **Lag-1 autocorrelation** | **0.328** | **0.781** | 2.4× higher |
| Lag-3 autocorrelation | 0.730 | 0.842 | 1.15× |
| Offset-3 cosine | 0.489 | 0.620 | 1.27× |
| Position-wise cosine (nat vs opt) | — | 0.337 mean | Very different |

**Why it works**: Codon optimization replaces diverse synonymous codons with a single preferred codon per amino acid. This creates repetitive dinucleotide patterns → high adjacent-position similarity → high lag-1 autocorrelation. Natural genes use varied codons → diverse adjacent nucleotides → low lag-1.

## Expanded Pilot (20 Gene Pairs)

### Design
- 20 natural phage genes from GYP benchmark (diverse hosts, 300-2000+ bp)
- 20 codon-optimized versions (same proteins, E. coli preferred codons)
- Per-position Evo2 40B embeddings for all 40 sequences
- Features: lag-1, lag-3, cos1, cos3 autocorrelation/cosine metrics

### Status
- Sequences generated ✅
- Per-position extraction running (~20 min)
- Classification analysis: pending extraction completion

### Results (2026-03-17)

| Feature | Natural (n=20) | Optimized (n=20) | Cohen's d |
|---------|---------------|------------------|-----------|
| **norm_cv** | 0.239 ± 0.034 | **0.208 ± 0.035** | **-0.92** |
| lag1 | 0.311 ± 0.201 | 0.419 ± 0.249 | +0.48 |
| lag2 | 0.264 ± 0.188 | 0.377 ± 0.257 | +0.50 |

Random Forest (5-fold CV): **75% accuracy** (natural vs codon-optimized).

Best feature: norm_cv (coefficient of variation of embedding norms) — optimized genes have more uniform norms because preferred codons create more regular patterns.

**Assessment**: 75% is above chance but not publication-ready alone. The signal is real but variable — depends on how different the natural codon usage is from the optimization target. Needs real engineered sequences (NCBI synthetic constructs) for proper validation.

## Future Directions

### Phase B: Real Engineered Sequences (NCBI)
- Download synthetic constructs from NCBI (taxid:32630)
- Compare against matched natural genes from the same organisms
- Validates on real engineered data, not just synthetic codon-optimization

### Phase C: Addgene Dataset
- Bulk download via Addgene Developers Portal API
- Largest collection of experimentally validated engineered sequences
- Requires API access approval (~5 business days)

### Chimera Junction Detection
- The PoC chimera test did not detect the junction via simple norm derivatives
- Sliding-window lag-1 computation (local lag-1 in 100bp windows) should detect the transition from natural (low lag-1) to optimized (high lag-1) pattern
- This would enable detection of chimeric/mosaic engineered constructs

### Integration with ViroSense
- Could become `virosense forensics` subcommand
- Uses existing `embed --per-position` + `scan` infrastructure
- Output: per-gene forensic scores (natural vs engineered probability)

## Files

| File | Contents |
|------|----------|
| `results/forensics_pilot/forensics_genes.fasta` | 40 sequences (20 natural + 20 optimized) |
| `results/forensics_pilot/forensics_metadata.csv` | Sequence labels and metadata |
| `results/forensics_pilot/per_position/` | Per-position embedding .npy files |
| `/tmp/forensics_*.npy` | PoC single-gene embeddings |

## Sources

- [Addgene Developers Portal](https://developers.addgene.org/)
- [Addgene: Browse Plasmids](https://www.addgene.org/browse/)
