# Speed Optimization Roadmap

Last updated: 2026-03-17

## The Problem

Embedding extraction is the fundamental bottleneck. Everything else is instant:

| Step | Time per sequence | 100K contigs |
|------|------------------|-------------|
| **Evo2 7B embedding** | **3.3s** | **92 hours** |
| **Evo2 40B embedding** | **50s** | **58 days** |
| Classification (cached) | 0.00003s | 3 seconds |
| Characterization (cached) | 0.03s | 50 minutes |
| Clustering (cached) | <0.001s | <1 second |

ViroSense's "embed once, analyze many ways" architecture amortizes the cost over all downstream analyses. But the first embedding run is expensive.

## Algorithmic Speedups

### 1. K-mer Pre-filtering (50-80% volume reduction)

Most metagenomic contigs are obviously bacterial. A fast tetranucleotide frequency classifier (microseconds/sequence) triages:
- Obviously bacterial → label as cellular, skip Evo2
- Obviously viral → label as viral, skip Evo2
- Ambiguous → send to Evo2 for full characterization

If 70% of contigs are trivially classified, ViroSense only embeds the 30% that need it. Estimated speedup: **3×** with no accuracy loss on hard cases.

### 2. Layer Early-Exit

We extract from layer 28/32. Earlier layers produce embeddings that may be sufficient for binary detection:
- Layer 28: current, full model depth
- Layer 14: ~2× faster (half the layers)
- Layer 7: ~4× faster (quarter)

Testable with cached data: re-extract from earlier layers, measure classification accuracy vs layer depth. If layer 14 preserves >95% accuracy for detection, this is a free 2× speedup.

### 3. Adaptive Scanning

Already implemented for prophage (coarse→fine), applicable to all commands:
- Coarse pass: large windows, big steps → identify interesting regions
- Fine pass: small windows, small steps → only on candidates
- Typical reduction: 5× fewer API calls

### 4. Sequence Batching

Short contigs (1-3kb) have disproportionate per-request overhead. Padding multiple short sequences into one 16kb request could improve throughput for fragmented assemblies.

### 5. Approximate Embeddings

PCA from 8,192-D to 256-D loses <10% variance. All downstream operations (classification, clustering, anomaly detection) run faster on reduced embeddings. The PCA can be pre-computed from the reference panel.

## Hardware Speedups

### Multi-GPU Parallelism (Most Immediate Impact)

Embedding extraction is embarrassingly parallel:

| Setup | Throughput | 100K contigs |
|-------|-----------|-------------|
| 1× L40S (7B) | 0.3 seq/s | 92 hours |
| 4× L40S (7B) | 1.2 seq/s | 23 hours |
| 1× H100 (7B) | ~1 seq/s | 28 hours |
| 4× H100 (7B) | ~4 seq/s | 7 hours |
| **8× H100 (7B)** | **~8 seq/s** | **3.5 hours** |

### WashU RIS (Planned Upgrade)

- Docker-native (no Apptainer workarounds)
- H100 SXM GPUs (faster than L40S)
- Multi-GPU jobs supported
- 16 PB storage for embedding caches
- No rate limits (self-hosted NIM)

Priority: onboard ASAP. Single biggest practical speedup.

### Apple Silicon (MLX Backend)

- Evo2 7B forward pass implemented in MLX
- Estimated: 10-20s per 5kb on M4 Max 128GB (untested)
- Local, no internet, no rate limits, no rental costs
- Democratizes access — every Mac user can run ViroSense

### Cloud GPU Burst

For one-off large analyses:
- Lambda Labs / RunPod: H100 at ~$2/hr
- 8× H100 for 4 hours = $64 for a full metagenome
- Cheaper than most sequencing runs

## Knowledge Distillation (1000× Speedup)

Train a lightweight student model to approximate Evo2 embeddings:

| Model | Speed | Accuracy (est) | Hardware |
|-------|-------|----------------|----------|
| Evo2 40B (NIM) | 50s/seq | 100% (reference) | H100 |
| Evo2 7B (NIM) | 3.3s/seq | ~95% | L40S |
| Evo2 7B (MLX) | ~15s/seq | ~95% | M4 Max |
| **Distilled CNN** | **0.001s/seq** | **~85-90%** | **CPU** |
| k-mer baseline | 0.0001s/seq | ~70-80% | CPU |

Approach:
- Teacher: Evo2 40B embeddings for reference genome fragments
- Student: 1D CNN on hexanucleotide frequencies → 256-D embedding
- Loss: contrastive (align student to teacher in embedding space)
- Training data: 5M teacher embeddings from fragmented RefSeq

The distilled model processes 100K contigs in ~2 minutes on CPU. Combined with Evo2 for the hard cases → near-Evo2 accuracy at near-k-mer speed.

## Two-Tier Pipeline (Recommended Architecture)

```
Input contigs (100K)
    │
    ▼
Tier 1: Distilled CNN (2 min, CPU)
    ├── 70K obviously bacterial → cellular
    ├── 20K obviously viral → viral (high confidence)
    └── 10K ambiguous → Tier 2
    │
    ▼
Tier 2: Evo2 7B (9 hours → 1 hour with 8 GPUs)
    ├── Full characterization (DNA passport)
    ├── Per-position analysis (gene structure)
    └── Novel element flagging (anomaly scoring)
```

Total: ~1-2 hours for a full metagenome with comprehensive characterization of the interesting fraction.

## Recommended Roadmap

### Immediate (this month)
1. Onboard to WashU RIS — multi-GPU H100 access
2. Test layer early-exit (does layer 14 work for detection?)
3. Implement k-mer pre-filter in ViroSense pipeline

### Short-term (1-2 months)
4. MLX backend validation + weight download
5. Multi-GPU NIM deployment on RIS
6. Pre-compute reference database (all RefSeq viral genomes)

### Medium-term (3-6 months)
7. Knowledge distillation (train student model)
8. Integrated two-tier pipeline
9. Pre-computed embedding databases for common organisms

### Long-term
10. Custom optimized inference kernels (MLX Metal, TensorRT)
11. Streaming embedding extraction (process reads directly)
12. Federated embedding databases (community-shared caches)
