# ViroSense Architecture

## Scientific Motivation

### The Problem
Viral metagenomics generates enormous volumes of sequence data, much of which is "viral dark matter" — sequences with no detectable homology to known viruses. Current tools either:
- Focus only on phages (e.g., pHold)
- Rely solely on sequence homology (missing divergent viruses)
- Use a single modality (DNA or protein, not both)

### The Solution
ViroSense combines two complementary modalities:
- **DNA-level**: Evo2 foundation model embeddings capture genomic patterns independent of coding potential
- **Protein-level**: ProstT5 structural embeddings (via vHold) capture protein fold information

This multi-modal approach enables detection, annotation, and classification of viral sequences that would be missed by either modality alone.

## Four Application Modules

### 1. detect — Viral Sequence Detection
**Question**: "Is this contig viral or cellular?"

Classify metagenomic contigs as viral vs cellular using Evo2 DNA embeddings fed into a trained classifier head. Operates on raw DNA — no gene calling required.

**Pipeline**: FASTA → length filter → Evo2 embeddings → classifier → scored TSV

### 2. context — Genomic Context Annotation
**Question**: "What does the DNA neighborhood tell us about this protein?"

Extract Evo2 embeddings from genomic windows flanking each ORF, providing DNA-level context that complements vHold's protein structural annotations. Useful for annotating proteins in operonic context.

**Pipeline**: Contigs + ORF predictions → Evo2 window embeddings → merge with vHold annotations → enhanced annotation TSV

### 3. cluster — Viral Dark Matter Clustering
**Question**: "How do unclassified viral sequences relate to each other?"

Fuse DNA (Evo2) and protein (ProstT5) embeddings, then cluster using HDBSCAN/Leiden/k-means to organize viral dark matter into putative families.

**Pipeline**: FASTA → Evo2 + ProstT5 embeddings → fusion → clustering → cluster assignments + quality metrics

### 4. classify — Discriminative Viral Classifier
**Question**: "What viral family is this? What's its host range?"

Train lightweight classification heads on frozen Evo2 embeddings for specific prediction tasks. Discriminative only — no sequence generation.

**Pipeline**: FASTA + labels → Evo2 embeddings → train classifier → predictions + metrics

## Biosecurity Design

Evo2 deliberately excludes eukaryotic viral sequences from its training data for biosecurity. ViroSense uses Evo2 only for **discriminative** tasks (classification, clustering, embedding extraction) — never for sequence generation. This is a safe, defensive application focused on detection and characterization of existing sequences.

An adversarial fine-tuning study showed Evo2's viral exclusion can be bypassed with ~110 viral genomes and 4 H100 GPUs, so the exclusion "raises the bar but remains susceptible to circumvention." Our discriminative-only use avoids this concern entirely.

## Backend Abstraction

Evo2 requires NVIDIA GPU (H100/Ada+ with CUDA 12.1+ and FP8). Developer machine is Apple M4. Solution: backend abstraction.

```
Evo2Backend (ABC)
├── NIMBackend   — NVIDIA NIM cloud API (default, works anywhere)
├── LocalBackend — Direct evo2 Python package (needs CUDA GPU)
└── ModalBackend — Modal.com serverless GPU (future)
```

## NVIDIA NIM API Reference

### Endpoints

| Endpoint | URL | Purpose |
|----------|-----|---------|
| Forward | `POST /v1/biology/arc/evo2-40b/forward` | Embedding extraction |
| Generate | `POST /v1/biology/arc/evo2-40b/generate` | DNA sequence generation (not used by ViroSense) |

Base URL: `https://health.api.nvidia.com`

### Authentication

- **Header**: `Authorization: Bearer $NVIDIA_API_KEY`
- Key obtained from [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys)

### Forward Endpoint (Embedding Extraction)

This is ViroSense's primary API call.

**Request**:
```json
{
    "sequence": "ACTGTCGATGCATCA...",
    "output_layers": ["decoder.layers.20.mlp.linear_fc2"]
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `sequence` | string | Yes | DNA sequence (A, C, T, G), max 16,000 bp on cloud API |
| `output_layers` | array[string] | Yes | Layer names for embedding extraction (max 100) |

**Response**:
```json
{
    "data": "<base64-encoded NPZ>",
    "elapsed_ms": 1234
}
```

The `data` field is a base64-encoded NumPy NPZ file. Keys are `<layer_name>.output`.

**Decoding example**:
```python
import base64, io, numpy as np

decoded = base64.b64decode(response["data"].encode("ascii"))
npz = np.load(io.BytesIO(decoded))
embeddings = npz["decoder.layers.20.mlp.linear_fc2.output"]
# Shape: (1, seq_len, hidden_dim)

# Mean-pool for sequence-level representation:
seq_embedding = np.mean(embeddings, axis=1).squeeze()
```

### Layer Naming

**Critical**: NIM API uses `decoder.layers.[n].*` naming. The native Evo2 Python library uses `blocks.[n].*`. These refer to the same layers but different interfaces.

**40B model** (50 layers, 0-49):
- Most layers are **HyenaLayers**: `decoder.layers.[n].mixer`, `decoder.layers.[n].mlp`
- 8 layers are **TransformerLayers** (3, 10, 17, 24, 31, 35, 42, 49): `decoder.layers.[n].self_attention`, `decoder.layers.[n].mlp`
- Special: `embedding` (NOT contextual — raw token embeddings), `decoder.final_norm`, `output_layer`

**7B model** (32 layers, 0-31):
- Available self-hosted only (`NIM_VARIANT=7b`)

**Recommended layers for embeddings**: Mid-network MLP outputs (e.g., `decoder.layers.20.mlp.linear_fc2`). Do NOT use `embedding` layer — it returns static per-token embeddings with no context.

### Constraints

| Constraint | Value |
|------------|-------|
| Max sequence length (cloud) | 16,000 bp |
| Rate limit (cloud) | ~40 requests/minute |
| Max output_layers per request | 100 |
| Model context window (theoretical) | Up to 1M bp |

### Error Codes

| HTTP Status | Meaning |
|-------------|---------|
| 200 | Success |
| 422 | Sequence too long (>16,000 bp) |
| 429 | Rate limit exceeded |
| 503 | Model not ready |

### Generate Endpoint (Reference Only)

ViroSense does NOT use this endpoint (biosecurity design decision).

```json
{
    "sequence": "ACTGACTG...",
    "num_tokens": 100,
    "temperature": 0.7,
    "top_k": 3,
    "top_p": 0.0,
    "enable_sampled_probs": false
}
```

Response includes `sequence` (generated DNA), `elapsed_ms`, and optional `logits`/`sampled_probs`.

## Evo2 Model Specifications

| Model | Parameters | Layers | Embed Dim | Max Context | GPU Requirements |
|-------|-----------|--------|-----------|-------------|-----------------|
| evo2_7b | 7B | 32 | 4,096 | 1M bp | Self-hosted only |
| evo2_40b | 40B | 50 | — | 1M bp | Cloud API or 2x H100 |

Training data: ~9.3 trillion nucleotides across prokaryotic, eukaryotic, and metagenomic sequences. Eukaryotic viruses deliberately excluded for biosecurity.

Architecture: StripedHyena (mostly Hyena layers with sparse Transformer attention layers).

## Key Design Decisions

1. **NIM layer naming in code**: The `constants.py` currently uses `blocks.28.mlp.l3` (native Evo2 naming). The NIM backend must translate to `decoder.layers.28.mlp.linear_fc2`. This mapping is a Phase 2 task.

2. **Mean pooling**: Sequence-level embeddings obtained by mean-pooling over the sequence dimension of per-position embeddings.

3. **NPZ caching**: All embeddings cached as NPZ files to avoid re-extraction. Cache keyed by model + layer + sequence ID.

4. **scikit-learn classifiers**: Classification heads use scikit-learn (not PyTorch) on frozen embeddings — keeps core install lightweight.

5. **File-based vHold integration**: Exchange data via TSV/NPZ files, not tight API coupling. vHold is an optional dependency.
