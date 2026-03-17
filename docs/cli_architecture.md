# ViroSense CLI Architecture

Last updated: 2026-03-17

## Design Principle: "Embed Once, Analyze Many Ways"

ViroSense uses Evo2 DNA foundation model embeddings as a general-purpose sequence representation. Embedding extraction is computationally expensive (~3.3s/seq on 7B, ~50s/seq on 40B), but all downstream analyses on cached embeddings are instant (~37,000 seq/s). The CLI architecture separates the expensive embedding step from cheap downstream analysis.

```
virosense embed     ← expensive (one-time)
    ↓ (cached embeddings)
virosense detect    ← instant (binary viral detection)
virosense classify  ← instant (multi-class classification)
virosense cluster   ← instant (unsupervised clustering)
virosense scan      ← instant (per-position gene/codon analysis)
virosense prophage  ← instant (prophage region detection)
virosense run       ← orchestrator (embed → detect → scan → annotate)
```

## Commands

### `virosense embed` (NEW — to implement)

**Purpose**: Extract and cache Evo2 embeddings for a set of sequences.

**Rationale**: Currently every command embeds independently. Separating embedding extraction:
- Makes the expensive step explicit to users
- Allows embedding once and analyzing many ways without re-extraction
- Supports both mean-pooled (for classification) and per-position (for scan) output
- Enables the "one embedding, many heads" paradigm

**Interface**:
```bash
virosense embed -i contigs.fasta -o cache/ \
    --backend nim --model evo2_40b \
    --per-position  # also save per-position embeddings (large)
```

**Output**:
- `cache/{model}_{layer}_embeddings.npz` — mean-pooled (N × hidden_dim)
- `cache/per_position/{seq_id}.npy` — per-position (seq_len × hidden_dim), one per sequence (optional)

**Status**: TO IMPLEMENT

---

### `virosense detect`

**Purpose**: Binary viral/cellular classification.

**Pipeline**: FASTA → embeddings (cached) → MLP classifier → TSV with viral scores.

**Validation**: Publication-ready. 13,417 sequences, head-to-head vs geNomad, bootstrap CIs. 99.7% phage sensitivity, 93.0% RNA virus recall on 40B.

**Status**: COMPLETE — no changes needed.

---

### `virosense classify`

**Purpose**: Train or apply a multi-class classifier on frozen Evo2 embeddings. Supports arbitrary label schemes from TSV input.

**Pipeline**:
- Training: FASTA + labels TSV → embeddings → train MLP → save model + metrics
- Prediction: FASTA + trained model → embeddings → predictions TSV

**Subsumes**: `build-reference` (add `--normalize-l2` and `--install` flags to classify)

**Changes made (2026-03-17)**:
- `--task` now accepts any string (not restricted to 3 hardcoded options)
- Docstring updated for general multi-class description
- matmul overflow warnings suppressed

**Validation**: Tested on real data in training + prediction modes. 3-class contig typing (virus/plasmid/chromosome) validated with cross-validation.

**Status**: WORKING. Consider adding `--normalize-l2` and `--install` flags to fully subsume build-reference.

---

### `virosense cluster`

**Purpose**: Unsupervised clustering of sequences based on Evo2 embeddings.

**Pipeline**: FASTA → embeddings → PCA → HDBSCAN/KMeans/Leiden → cluster assignments TSV.

**Validation**: HDBSCAN ARI=0.903 on 13K sequences. RNA virus/phage separation (99% pure cluster). Host-taxonomy analysis. Genome binning analysis.

**Status**: WORKING. Tested end-to-end via CLI on real data.

---

### `virosense scan` (NEW — to implement)

**Purpose**: Per-position embedding analysis — gene boundaries, codon periodicity, coding region detection, RNA virus identification.

**Rationale**: Our most novel finding. Per-position Evo2 embeddings encode:
- Gene structure (coding norm 1.72× intergenic, 91.7% coding accuracy)
- The genetic code (3bp codon periodicity, lag-3 autocorr 0.635)
- RNA virus signatures (strongest periodicity, Cohen's d = 2.83)

Currently exists only as scripts. Needs a proper CLI command.

**Interface**:
```bash
virosense scan -i contigs.fasta --cache-dir cache/ -o scan/ \
    --coding           # detect coding regions via norm + periodicity
    --periodicity      # compute codon periodicity features
    --boundaries       # detect gene boundaries
```

**Output**:
- `scan/coding_predictions.tsv` — per-position coding/non-coding calls
- `scan/periodicity_features.tsv` — per-sequence lag-3, cos3, FFT features
- `scan/gene_boundaries.tsv` — predicted gene start/end positions
- `scan/summary.json` — aggregate statistics

**Status**: TO IMPLEMENT

---

### `virosense prophage`

**Purpose**: Detect integrated prophage regions in bacterial chromosomes via sliding-window embedding analysis.

**Pipeline**: Bacterial FASTA → sliding windows → embeddings → MLP classifier → region calls.

**Features**: Adaptive coarse→fine scanning (5× fewer API calls).

**Enhancement opportunity**: Per-position norm signal could refine prophage boundaries from ~2kb precision (sliding windows) to ~50bp precision (norm change points).

**Validation**: 36 unit tests, no real-data validation yet. Needs curated prophage benchmark.

**Status**: WORKING (unit tests). Needs real-data validation.

---

### `virosense run` (FUTURE)

**Purpose**: End-to-end pipeline — embed → detect → gene call → annotate → export.

**Pipeline**: Raw FASTA → embed → detect viral → Pyrodigal-gv gene calling → structural annotation → functional classification → export (GFF3, anvi'o, DRAM-v).

**Dependencies**: Requires annotate module completion (gene calling, export formats).

**Status**: FUTURE — not for initial paper.

---

### `virosense build-reference` (DEPRECATE)

**Purpose**: Train a binary viral/cellular classifier from labeled FASTA.

**Status**: Subsumed by `virosense classify --task viral_vs_cellular`. Keep as backward-compatible alias. Unique features to migrate to classify: `--normalize-l2`, `--install`.

---

### `virosense context` (DEPRECATE)

**Purpose**: ORF annotation with Evo2 genomic context windows.

**Status**: Superseded by `virosense scan`. The windowed mean-pooled approach is strictly less informative than per-position analysis. The vHold protein embedding fusion was the original motivation, but vHold is sunset. Remove or redesign around per-position embeddings.

---

## Implementation Priority

| Command | Priority | Effort | Blocked by |
|---------|----------|--------|-----------|
| `embed` | **HIGH** | Medium | Nothing — implement first |
| `scan` | **HIGH** | Medium | `embed` (needs per-position cache) |
| `classify` updates | LOW | Small | Nothing — add --normalize-l2, --install |
| `prophage` validation | MEDIUM | Medium | Need prophage benchmark dataset |
| `run` | LOW | Large | annotate module completion |
| `build-reference` deprecation | LOW | Small | classify absorbing its flags |
| `context` deprecation | LOW | Small | scan replacing it |

## Backward Compatibility

- `build-reference` keeps working as-is (alias to classify internals)
- `context` keeps working but prints deprecation warning pointing to `scan`
- All existing `--cache-dir` flags continue to work with the new `embed` output format
- Per-position embeddings are opt-in (large storage) — mean-pooled remains the default
