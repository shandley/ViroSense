# Speed Benchmarking

Last updated: 2026-03-16

## Benchmark Dataset
13,417 sequences from Gauge Your Phage + RNA virus + cellular controls (500 bp – 16 kb fragments).

## End-to-End Wall Clock

| Tool | Time | Per seq | Hardware | Notes |
|------|------|---------|----------|-------|
| **geNomad v1.11.2** | **22 min** | 0.10 s | 8 CPU cores | Gene calling + marker search + NN classification |
| **ViroSense 7B** (first run) | ~12 hours | 3.3 s | 1× L40S 48GB | Self-hosted NIM, embedding extraction |
| **ViroSense 7B** (cached) | <5 sec | 0.0004 s | CPU only | Classification from cached embeddings |
| **ViroSense 40B** (first run) | ~180 hours | ~50 s | Cloud NIM API | Rate-limited to 3 concurrent requests |
| **ViroSense 40B** (cached) | <5 sec | 0.0004 s | CPU only | Classification from cached embeddings |
| **DeepVirFinder** | ~4 hours (est) | ~1 s | 8 CPU cores | CNN inference on CPU |

## Classification-Only Throughput (Cached Embeddings)

ViroSense classification head on pre-computed embeddings:
- **37,072 sequences/second** (13,000 × 8,192-D, MLP inference)
- Any new classifier head (different task, different threshold) is instant
- No GPU required for classification

## Key Architectural Difference

**geNomad**: Monolithic pipeline — gene calling → marker search → NN classification. Every new analysis re-runs the full pipeline.

**ViroSense**: Two-phase — embedding extraction (expensive, one-time) → classification heads (instant, swappable).

| | geNomad | ViroSense |
|---|---|---|
| First analysis | 22 min | 12 hours (7B) |
| Re-classify with new threshold | 22 min | <5 sec |
| Add plasmid detection head | N/A (built-in) | <5 sec (train new MLP) |
| Add taxonomy head | Rerun pipeline | <5 sec |
| Add host prediction head | Not supported | <5 sec |

For workflows that iterate on classification (threshold tuning, multi-task, ensemble methods), ViroSense's cached embedding approach amortizes the upfront cost.

## Scaling Projections

| Dataset size | geNomad (8 CPU) | ViroSense 7B (L40S) | ViroSense 7B (cached) |
|-------------|-----------------|--------------------|-----------------------|
| 10K contigs | 16 min | 9 hours | <1 sec |
| 100K contigs | 2.7 hours | 92 hours (3.8 days) | 4 sec |
| 1M contigs | 27 hours | 920 hours (38 days) | 40 sec |

ViroSense 7B becomes practical with HPC batch processing (multiple GPUs) or by pre-computing embeddings for reference databases. For a lab running the same reference panel repeatedly, the one-time embedding cost is amortized over all future analyses.
