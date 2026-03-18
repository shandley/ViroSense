# Prophage Module Reassessment

Last updated: 2026-03-18

## What Evo2 Already Showed

The [Evo2 paper](https://www.nature.com/articles/s41586-026-10176-5) and the [Goodfire interpretability study](https://www.goodfire.ai/research/interpreting-evo-2) demonstrated that Evo2 **autonomously learns to identify prophage sequences**. Specifically:

- A sparse autoencoder (SAE) on layer 26 extracted a "phage-associated feature" (f/19746)
- This feature activates preferentially on RefSeq-annotated prophages in E. coli K12 MG1655
- It also fires on phage-derived CRISPR spacer sequences
- This emerged without any explicit prophage training signal

**This is prior art.** Our sliding-window prophage detection is a practical application of this known property, not a novel discovery. The 37bp boundary accuracy on E. coli K12 is a nice engineering result but doesn't fundamentally advance the science.

## Our Prophage Pilot Results

- E. coli K12 (local, 40B): Detected prophage at 24,000-43,000 vs ground truth 24,037-44,856 (37bp start accuracy)
- HTCF benchmark (7B): Coarse scan identified 4 candidate regions (Philympics curation has 3). Fine pass failed due to NIM connectivity.
- Bug: prophage module crashes on empty embeddings (fine pass failure → 0-length array)

## The Reframing Question

Given our shift from "viral detection tool" to "universal DNA characterization framework," is prophage detection the best use of effort?

### What prophage detection offers:
- Practical tool for a real bioinformatics need
- Demonstrates the sliding-window architecture works
- The 4th candidate region on E. coli could be a known cryptic prophage not in Philympics — a nice validation story

### What it DOESN'T offer:
- Scientific novelty (Evo2 paper already showed this)
- Unique capability (PHASTER, PhiSpy, VirSorter already do this)
- Advancement of the "characterization" thesis

## What Would Be More Interesting About Prophage DNA

Rather than just **detecting** prophages (where we compete with many existing tools), we should focus on **characterizing** them — things only our framework can do:

### 1. Prophage Lifecycle State (novel)
Can per-position embeddings distinguish:
- **Active prophages** (recently integrated, strong phage composition)
- **Degraded remnants** (ancient, partially deleted, ameliorated)
- **Domesticated elements** (phage genes co-opted for host function)

The embedding norm transition at prophage boundaries should be sharper for recent integrations and smoother for ancient ones. The codon periodicity within the prophage should reflect the degree of amelioration to host composition. **No existing tool can assess prophage age/state from composition alone.**

### 2. Prophage-Host Compatibility (novel)
The characterize framework gives both a viral signature and a host-similarity profile. For a prophage:
- High viral signature + low host similarity → recent integration from divergent phage
- Moderate viral signature + high host similarity → ancient, ameliorated
- Low viral signature + high host similarity → fully domesticated or false positive

This is a **characterization**, not a detection. The DNA passport for a prophage region tells you about its evolutionary history.

### 3. Prophage Boundary Refinement (incremental but valuable)
Per-position norm analysis gives ~50bp boundary resolution vs sliding-window's ~2-5kb. The norm transition profile at the integration site encodes:
- Sharpness of transition → clean integration vs gradual boundary
- Presence of att sites → signature motifs at exact junction
- Flanking host gene disruption → was a gene interrupted?

### 4. Cryptic Prophage Census (high impact, uses existing tools)
The E. coli K12 case study is compelling: 9 known cryptic prophages, only 3 in Philympics curation. Our coarse scan found 4 candidates. A systematic census of cryptic prophages across all well-studied bacterial genomes using the characterize framework would be novel and impactful.

## Recommended Approach

**For the ViroSense methods paper:**
- Include the E. coli K12 pilot as a demonstration (prophage at 24-43kb, 37bp accuracy)
- Frame as: "the same embeddings that detect viruses also detect integrated viral DNA"
- Brief — not a main result, more a proof of generality

**For a separate paper or follow-up:**
- Prophage lifecycle characterization (active vs degraded vs domesticated)
- Cryptic prophage census using characterize anomaly scoring
- This is where the novelty lies — not in detection, but in characterization

**For the codebase:**
- Fix the empty-embedding bug (graceful fallback)
- Keep prophage module as-is (working, useful)
- Don't invest in expanding the Philympics benchmark — the detection angle is not our differentiator

## Prophage Amelioration — Validated from Coarse Pass (2026-03-18)

The HTCF coarse-pass window scores for E. coli K12 reveal a **compositional age gradient** across the 9 known cryptic prophages:

| Prophage | Size | Max viral score | Mean score | State |
|----------|------|----------------|-----------|-------|
| DLP12 | 21 kb | **0.957** | 0.538 | Strong phage composition |
| rac | 23 kb | **0.957** | 0.538 | Strong phage composition |
| Qin | 20 kb | **0.957** | 0.538 | Strong phage composition |
| CP4-6 | 34 kb | **0.957** (1/5 windows) | 0.287 | **Mosaic** — partly ameliorated |
| **e14** | 15 kb | **0.120** | 0.119 | **Fully ameliorated** — invisible |
| Background | — | 0.119 | 0.119 | Chromosomal baseline |

CP4-44, CPS-53, CPZ-55, CP4-57 were outside the scanned range (coarse pass covered 0-1.7 Mb of the 4.6 Mb genome before the NIM 8-hour wall time expired).

### Key findings

1. **CP4-6 and Qin detected — not in Philympics curation.** ViroSense found 2 prophages that the manual benchmark curation missed. These are well-documented cryptic prophages (Wang et al. 2010, Nature Comms).

2. **e14 is invisible** (score 0.120, indistinguishable from background). e14 is an ancient lambdoid prophage whose composition has fully converged with the host chromosome. The embedding cannot distinguish it from E. coli chromosomal DNA. **This is the amelioration signal** — fully ameliorated prophages lose their compositional signature.

3. **CP4-6 is mosaic** — only 1 of 5 overlapping windows scores as viral (0.957), the other 4 score as chromosomal (0.119). This means CP4-6 retains phage-like composition in some regions but has been ameliorated in others. **The per-window score variance is a characterization feature**: high variance = mosaic/partially ameliorated.

4. **DLP12, rac, Qin are compositionally similar** — all score max 0.957, mean 0.538. These are at a similar intermediate amelioration state (strong phage signal but not in all windows).

5. **Background is flat** — 0.119 ± 0.000 across 153 non-prophage windows. The classifier is extremely confident about chromosomal DNA.

### Amelioration Gradient

The viral score serves as a **proxy for prophage evolutionary age**:

```
Recently integrated → Partially ameliorated → Fully ameliorated → Domesticated
     max 0.95+            mix of 0.95/0.12          max 0.12           max 0.12
   (all windows viral)   (mosaic pattern)       (invisible)      (invisible, host function)
    DLP12, rac, Qin          CP4-6                 e14              (gene islands)
```

This gradient is detectable from embedding scores alone, without any gene annotation or phylogenetic analysis. **No existing tool measures prophage amelioration state from DNA composition.**

### Implications

- **Viral score** = compositional foreignness (high = recent/foreign, low = ameliorated/native)
- **Score variance across windows** = mosaicism (high variance = partially ameliorated)
- **e14 invisibility** = fully ameliorated prophages are a fundamental limit of composition-based detection
- Per-position analysis (norm transitions, periodicity shifts at boundaries) could provide finer resolution than window-level scores

## Recommended Next Steps

### For the ViroSense methods paper (immediate)
1. Include the E. coli K12 coarse-pass results as a proof-of-concept for prophage characterization
2. Frame as: "embedding scores reveal a compositional amelioration gradient across prophages of different evolutionary ages"
3. Note CP4-6 and Qin detection beyond the Philympics curation (demonstrates finding uncurated elements)
4. Note e14 invisibility (honest limitation of composition-based approaches)
5. Brief — 1-2 paragraphs + 1 figure panel, not a main result

### For a dedicated prophage characterization paper (follow-up)
1. **Full E. coli K12 scan** — complete the coarse pass for the entire genome (remaining ~3 Mb)
2. **Per-position analysis** — extract per-position embeddings for all 9 prophage regions, characterize boundary sharpness and internal structure
3. **Multi-genome comparison** — same analysis on B. subtilis (3 prophages), S. aureus Newman (4 prophages), S. Typhi (5 prophages)
4. **Literature correlation** — compare embedding-derived amelioration state against published evolutionary age estimates for these prophages
5. **Develop an amelioration index** — formal metric combining (a) max viral score, (b) score variance across windows, (c) boundary sharpness from per-position norms, (d) codon periodicity similarity to host

### For the codebase (immediate)
1. Fix the empty-embedding bug in prophage module
2. Add amelioration scoring to `virosense characterize` — when a region is flagged as viral, compute score variance as a mosaicism/age indicator
3. Consider a `--characterize-prophages` flag that provides the full age/state assessment for detected prophage regions

## Bug Fix Needed

`virosense.models.prophage.score_windows()` crashes when `embeddings` is empty (shape (0, 4096)). Should return empty results instead. One-line fix in the score_windows function.

## Sources

- [Evo2 Nature paper](https://www.nature.com/articles/s41586-026-10176-5)
- [Goodfire: Interpreting Evo2](https://www.goodfire.ai/research/interpreting-evo-2)
- [Arc Institute Evo2 tools](https://arcinstitute.org/tools/evo)
