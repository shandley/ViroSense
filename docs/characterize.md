# ViroSense Characterize — DNA Passport

Last updated: 2026-03-17

## Overview

`virosense characterize` generates comprehensive biological profiles ("DNA passports") for DNA sequences using Evo2 foundation model embeddings. Rather than assigning binary labels (viral/non-viral), it produces a multi-dimensional characterization that reveals identity, origin, structure, and novelty — all from a single Evo2 forward pass.

## How It Works

### Step 1: Embedding Extraction

Evo2 reads the DNA sequence nucleotide by nucleotide and produces a hidden-state vector at each position. These are mean-pooled into a single 8,192-D vector (40B model) or 4,096-D (7B model) — the sequence's compositional "fingerprint."

This fingerprint encodes everything Evo2 learned from training on 9 trillion nucleotides: GC content, dinucleotide frequencies, codon usage, gene density, taxonomic signatures, and many other features — all compressed into one vector.

### Step 2: Identity — Cosine Similarity to Reference Centroids

We pre-compute the average embedding for each known category (phage, RNA virus, chromosome, plasmid, cellular) from thousands of labeled sequences. For a new sequence, we compute cosine similarity to each centroid.

Cosine similarity measures the angle between vectors: 1.0 = identical composition, 0.0 = completely different.

Crucially, we report **all similarities**, not just the top match. A plasmid scoring 0.94 vs phage AND 0.88 vs chromosome tells you "mobile element with mixed composition" — far more informative than a binary label.

### Step 3: Origin Scores — Derived Ratios

Ratios of the similarity scores capture specific biological questions:

| Score | Formula | Interpretation |
|-------|---------|---------------|
| **Viral signature** | max(phage, rna_virus) / max(chromosome, cellular) | >1.0 = more viral than cellular |
| **RNA origin** | rna_virus_sim / phage_sim | >1.0 = RNA-origin (cDNA), <1.0 = DNA-origin |
| **Mobile element** | (phage + plasmid) / (2 × chromosome) | >1.0 = mobile, <1.0 = chromosomal |

These are continuous scores, not binary calls. A viral signature of 1.27 means "somewhat viral" — the user can apply their own threshold based on their tolerance for false positives.

### Step 4: Anomaly Scoring — Nearest Neighbor Distance

A nearest-neighbor index on the reference panel (2,500 labeled sequences) computes each query's cosine distance to its 10th nearest neighbor. Large distance = sparse region of embedding space = compositionally unusual.

The anomaly **percentile** contextualizes this: "more unusual than X% of known sequences." A sequence at the 99th percentile is in the top 1% most unusual — the "unknown unknown" signal.

This is the feature that would flag an Obelisk-like novel element: high anomaly score, low similarity to all known categories, yet clearly biological (coding periodicity present).

### Step 5: Per-Position Structure (Optional)

With `--per-position` and pre-computed per-position embeddings:

| Feature | How computed | What it reveals |
|---------|-------------|----------------|
| **Coding density** | Fraction of positions with above-median norm | How much of the sequence is protein-coding |
| **Codon periodicity** | Lag-3 autocorrelation of norm signal | Strength of triplet code structure (>0.7 = strong) |
| **Offset-3 inversion** | cos3 > cos1 | Universal binary coding signature |
| **Norm CV** | std(norm) / mean(norm) | Compositional uniformity (low = uniform = possibly engineered) |
| **Compositional shifts** | Max cosine distance between adjacent windows | Internal boundaries (chimera, prophage integration) |

### Step 6: Interpretation

Rule-based translation of scores to human-readable assessments:

| Score range | Interpretation |
|-------------|---------------|
| Viral signature > 1.3 | "likely viral" |
| Viral signature 1.0–1.3 | "possibly viral" |
| Viral signature 0.7–1.0 | "ambiguous" |
| Viral signature < 0.7 | "likely non-viral" |
| RNA origin > 1.1 | "RNA-origin" |
| Anomaly percentile > 99% | "highly novel (top 1%)" |
| Lag-3 autocorr > 0.8 | "strong codon structure" |

## Why Characterization > Classification

| Classification | Characterization |
|---------------|-----------------|
| "This is viral" (might be wrong) | "This is 0.94 similar to phage, 0.62 to chromosome, viral signature 1.27" |
| Binary: viral / not viral | Multi-dimensional: identity + origin + structure + novelty |
| Wrong answer = useless | Every dimension is informative even when nearest category is wrong |
| Requires training on the exact categories | Works with any reference panel, zero-shot for novel categories |

## Usage

```bash
# Basic characterization (mean-pooled embeddings only)
virosense characterize -i contigs.fasta -o profiles/ \
    --cache-dir embeddings/ \
    --reference-panel reference_panel.npz

# With per-position structural analysis
virosense embed -i contigs.fasta -o embeddings/ --per-position
virosense characterize -i contigs.fasta -o profiles/ \
    --cache-dir embeddings/ \
    --reference-panel reference_panel.npz \
    --per-position
```

## Output

**characterization.json** — Full detail per sequence:
```json
{
  "sequence_id": "contig_001",
  "category_similarities": {"phage": 0.94, "chromosome": 0.62, ...},
  "viral_signature": 1.48,
  "rna_origin_score": 1.12,
  "anomaly_percentile": 0.95,
  "interpretation": {
    "viral": "likely viral",
    "origin": "RNA-origin",
    "novelty": "unusual (top 5%)"
  }
}
```

**characterization.tsv** — Flat summary for spreadsheet/downstream analysis.

## Validation (25 diverse sequences)

| Metric | Accuracy |
|--------|---------|
| RNA origin interpretation | **100%** (25/25) |
| Viral interpretation | **76%** (19/25) |
| Nearest category (strict) | 52% (13/25) |

The viral interpretation (76%) outperforms strict category matching (52%) because the continuous scores capture ambiguity that binary matching misses. The 100% RNA origin accuracy reflects the strong compositional distinction between RNA-origin and DNA-origin sequences in Evo2 embeddings.

## Reference Panel

The reference panel is an NPZ file with:
- `embeddings`: (N, hidden_dim) array of reference sequence embeddings
- `labels`: (N,) array of category strings

Build from labeled data:
```python
np.savez('reference_panel.npz', embeddings=emb_array, labels=label_array)
```

A pre-built panel from the GYP benchmark (2,500 sequences, 5 categories) is available at `data/reference/reference_panel_40b.npz`.
