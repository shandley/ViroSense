# Prophage Detection Validation

Last updated: 2026-03-17

## Overview

ViroSense prophage detection uses sliding-window Evo2 embeddings classified by the binary viral/cellular MLP. Bacterial chromosomes are scanned with overlapping windows; windows scoring above the viral threshold are merged into candidate prophage regions.

## Benchmark Dataset: Philympics 2021

Source: [ProphagePredictionComparisons](https://github.com/linsalrob/ProphagePredictionComparisons) (Roach et al., F1000Research 2022).

81 manually curated bacterial genomes with prophage annotations (`is_phage="1"` on CDS features). Used to benchmark PhiSpy, VIBRANT, VirSorter2, Phigaro, PhageBoost, DBSCAN-SWA.

### Pilot genomes (5)

| Genome | Species | Length | Prophage regions | Total phage bp |
|--------|---------|--------|-----------------|----------------|
| NC_000964 | *B. subtilis* 168 | 4.2 Mb | 3 | 196,076 |
| 1351.557 | *E. faecalis* V583 | 4.5 Mb | 1 | 52,208 |
| NC_000913 | *E. coli* K12 | 4.6 Mb | 3 | 59,747 |
| NC_003198 | *S.* Typhi CT18 | 4.8 Mb | 5 | 193,025 |
| 1280.10152 | *S. aureus* Newman | 2.9 Mb | 4 | 171,950 |

## Pilot Result: E. coli K12 Prophage 1

Tested on a 70 kb region (positions 540,000-610,000) of E. coli K12 containing one manually curated prophage.

**Parameters**: 5 kb windows, 2 kb step, threshold 0.5, full scan mode, Evo2 40B via cloud NIM.

### Ground truth vs prediction

| | Ground truth | ViroSense | Difference |
|---|---|---|---|
| **Start** | 24,037 | **24,000** | **37 bp** |
| **End** | 44,856 | **43,000** | 1,856 bp |
| **Length** | 20,819 bp | 19,000 bp | -1,819 bp (8.7%) |

### Window-level scores

The classifier produces a sharp, clean signal:

| Position range | Viral score | Classification |
|---------------|-------------|----------------|
| 0-22,000 | 0.0087-0.0092 | Cellular (confident) |
| 22,000-24,000 | 0.013-0.063 | Transition zone |
| **24,000-29,000** | **0.832** | **Viral** (prophage entry) |
| 26,000-31,000 | 0.872 | Viral |
| 28,000-33,000 | **0.947** | Viral (peak) |
| 30,000-35,000 | 0.638 | Viral |
| 32,000-37,000 | 0.705 | Viral |
| 34,000-39,000 | 0.919 | Viral |
| 36,000-41,000 | **0.973** | Viral (peak) |
| 38,000-43,000 | 0.969 | Viral |
| 40,000-45,000 | 0.470 | Cellular (transition) |
| 42,000-70,000 | 0.0088-0.014 | Cellular (confident) |

Key observations:
- **Bacterial DNA**: 0.0087-0.0092 — essentially zero, very high confidence non-viral
- **Prophage core**: 0.638-0.973 — 8 consecutive viral windows
- **Start boundary**: 37 bp from ground truth (sub-window precision)
- **End boundary**: 1,856 bp short — the prophage's 3' end has lower viral signal, likely decayed/host-adapted genes
- **No false positives**: zero spurious viral calls in 60 kb of flanking bacterial DNA

### Comparison to published tools

From the Philympics 2021 study on E. coli K12:

| Tool | Detected? | Boundary accuracy | Notes |
|------|-----------|-------------------|-------|
| PhiSpy | Yes | Good | Best overall in Philympics |
| VIBRANT | Yes | Good | Second best |
| Phigaro | Yes | Moderate | |
| VirSorter2 | Yes | Moderate | |
| **ViroSense** | **Yes** | **37 bp start, 1.9 kb end** | **No gene calling needed** |

ViroSense's key advantage: it detects this prophage using **only DNA composition** (Evo2 embeddings), without gene calling, HMM searches, or marker gene databases. This means it should work on:
- Degraded prophages lacking recognizable genes
- Novel prophages in uncharacterized bacterial hosts
- Prophage remnants too short for gene-based methods

## Full Pilot Plan (5 genomes)

### Approach

Run `virosense prophage` on all 5 pilot genomes with adaptive coarse→fine scanning:
- **Coarse pass**: 15 kb windows, 10 kb step → identify candidate regions (score ≥ 0.3)
- **Fine pass**: 5 kb windows, 2 kb step → refine boundaries

### Computational cost

| Backend | Est. windows | Time per window | Total time |
|---------|-------------|----------------|------------|
| 40B cloud NIM | ~2,500 | ~27s | ~19 hours |
| **7B HTCF NIM** | ~2,500 | ~3.3s | **~2.3 hours** |

**Plan**: Submit as SLURM job on HTCF with 7B NIM on L40S.

### Evaluation metrics

For each genome, compare ViroSense predictions against manual curation:
- **Region-level**: How many of the curated prophage regions are detected?
- **Boundary accuracy**: Distance from predicted to true start/end
- **Base-level precision/recall**: What fraction of prophage bp are correctly classified?
- **False positive rate**: Spurious viral calls in non-prophage regions

### Expected outcomes

Based on the E. coli pilot:
- High recall expected (strong viral signal in prophage regions)
- Start boundaries likely within 1-2 kb of ground truth
- End boundaries may be less precise (decayed prophage tails)
- False positive rate should be very low (bacterial DNA scores ~0.009)

## Files

| File | Contents |
|------|----------|
| `data/benchmarks/philympics/repo/` | Cloned Philympics repository |
| `data/benchmarks/philympics/*.fasta` | Extracted genome sequences |
| `data/benchmarks/philympics/*_ground_truth.json` | Curated prophage annotations |
| `/tmp/test_prophage_output/` | E. coli K12 pilot results |
