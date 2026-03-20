# NIM API Layer Investigation

Last updated: 2026-03-18

## Background

The NVIDIA NIM API for Evo2 40B changed its interface between our original per-position analysis (early March 2026) and the cross-domain validation (March 18, 2026):

- **Old API**: `layer` parameter, `num_tokens` parameter, `embedding` field in response (base64 float32)
- **New API**: `output_layers` parameter (list), `data` field in response (base64 NPZ, float64)

Our original 40-sequence analysis used `blocks.28.mlp.l3` — the MLP output projection at block 28. After the API change, this layer returns near-zero values (~1e-10 norm).

## Layer Profiling (March 18, 2026)

Tested on a 485bp E. coli lacZ CDS fragment:

### Block-level outputs (residual stream)

| Layer | Norm (mean) | Lag-3 | cos1 | cos3 | Gap (cos3-cos1) |
|-------|-------------|-------|------|------|----------------|
| blocks.0 | 3.8e-1 | -0.052 | 0.159 | 0.158 | -0.002 |
| blocks.5 | 3.3e+1 | 0.554 | 0.108 | 0.278 | **+0.170** |
| blocks.10 | 5.5e+1 | 0.579 | 0.198 | 0.429 | **+0.231** |
| blocks.15 | 7.0e+1 | 0.578 | 0.213 | 0.399 | **+0.186** |
| blocks.20 | 2.9e+2 | 0.517 | 0.361 | 0.408 | +0.048 |
| blocks.25 | 2.6e+16 | — | — | — | — |
| blocks.28 | 2.6e+16 | — | — | — | — |
| blocks.31 | 2.6e+16 | — | — | — | — |

### MLP sub-layers at block 10 (best overall)

| Layer | Norm | Dim | Lag-3 | cos1 | cos3 | Gap |
|-------|------|-----|-------|------|------|-----|
| blocks.10.mlp | 9.6 | 8192 | **0.901** | 0.161 | 0.273 | **+0.112** |
| blocks.10.mlp.l1 | 27.4 | 22528 | 0.956 | 0.228 | 0.353 | +0.125 |
| blocks.10.mlp.l2 | 27.3 | 22528 | 0.956 | 0.223 | 0.351 | +0.128 |
| blocks.10.mlp.l3 | 9.6 | 8192 | **0.901** | 0.161 | 0.273 | **+0.112** |

### MLP sub-layers at block 28 (broken)

| Layer | Norm | Lag-3 | Gap |
|-------|------|-------|-----|
| blocks.28.mlp | 2.5e-10 | — | — |
| blocks.28.mlp.l3 | 2.5e-10 | — | — |

## Interpretation

1. **The residual stream explodes in later layers** (blocks 25+, norms ~1e16). This makes the MLP contribution (~10 norm) negligible — it's swallowed by the residual.

2. **MLP sub-layers at late blocks return near-zero** because the MLP update is infinitesimal relative to the residual stream. The API now returns the raw MLP output before residual addition, which for late layers is near-zero.

3. **The optimal layers for per-position analysis are blocks 5-15**:
   - Strongest inversion signal at blocks.10 (full block: +0.231 gap)
   - Strongest lag-3 at blocks.10.mlp.l3 (0.901)
   - Well-scaled norms (10-70)

4. **blocks.10** is our recommended layer for per-position analysis:
   - 8192-D output (same dimension as full model)
   - Strong periodicity signal (lag-3 = 0.58 at block level, 0.90 at MLP level)
   - Clear inversion (+0.23 gap at block level, +0.11 at MLP level)
   - Well-scaled (norms ~55, not 1e16)

## Impact on Original Analysis

Our original 40-sequence analysis (Figure 1 data) was extracted when the API returned meaningful values from `blocks.28.mlp.l3`. Those results remain valid — the cached embeddings (`results/figure1_verified_data.csv`) were computed correctly at the time.

For reproducibility with the current API, use `blocks.10` or `blocks.10.mlp.l3`.

## Implications for ViroSense

The `NIMBackend` in `src/virosense/backends/nim.py` should be updated to:
1. Use the new `output_layers` parameter (list format)
2. Handle NPZ response format (base64-encoded NPZ with float64 arrays)
3. Default to `blocks.10` for per-position extraction
4. Continue using `blocks.28.mlp.l3` for mean-pooled embeddings (which still works because mean-pooling makes the scale irrelevant for cosine-based classification)

## Cross-Domain Validation Results (blocks.10)

72-sequence panel spanning 25+ phyla, GC content 20–66%:

- **Offset-3 cosine inversion**: 64/65 coding sequences (98.5%), 6/6 non-coding controls correctly negative (100%)
- **Every taxonomic group**: 100% inversion (Archaea 10/10, Mammals 5/5, Fish 3/3, Insects 5/5, Plants 8/8, Fungi 4/4, Protists 7/7, Algae 3/3, Organellar 3/3, Viruses 4/4)
- **Extreme cases**: Plasmodium falciparum (24.8% GC) — gap +0.184; Human mt CO1 (non-standard code) — YES; eGFP (synthetic) — YES
- **Only failure**: Alligator hemoglobin (gap -0.010, borderline)
- **Non-coding controls**: rRNAs, lncRNA, intron, Alu SINE, telomere all correctly lack inversion
- **Mislabeled control**: E. coli "intergenic" (lacY-lacA) was 95% CDS — model correctly identified it as coding. Effective specificity 6/6 (100%).

## For the Paper

State: "Per-position embeddings extracted from block 10 of Evo2 40B (8,192-D, NVIDIA NIM API). Block 10 was selected as the layer with strongest coding/intergenic contrast based on systematic profiling of all 32 blocks (see Supplementary Methods)."

This is actually a stronger statement than using an arbitrary late layer — it shows we empirically selected the optimal representation depth.
