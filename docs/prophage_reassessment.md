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

## Bug Fix Needed

`virosense.models.prophage.score_windows()` crashes when `embeddings` is empty (shape (0, 4096)). Should return empty results instead. One-line fix in the score_windows function.

## Sources

- [Evo2 Nature paper](https://www.nature.com/articles/s41586-026-10176-5)
- [Goodfire: Interpreting Evo2](https://www.goodfire.ai/research/interpreting-evo-2)
- [Arc Institute Evo2 tools](https://arcinstitute.org/tools/evo)
