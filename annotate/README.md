# ViroSense Annotate Module — Integration Plan

Last updated: 2026-02-27

## Overview

The `annotate` module extends ViroSense from viral detection ("is this viral?") to functional annotation ("what do the proteins do?"). It replaces the ProstT5-dependent vHold integration with a ColabFold + BFVD + Foldseek + FoldMason pipeline.

## Pipeline

```
virosense detect (Evo2) → viral contigs
  │
  ├─ 1. Gene calling (Pyrodigal-gv)
  │     Input:  viral contig FASTA
  │     Output: protein ORFs (FASTA + GFF3)
  │
  ├─ 2. Structure acquisition
  │     a. AlphaFold DB lookup (known proteins → instant PDB)
  │     b. ColabFold prediction (novel proteins → minutes/protein)
  │     Output: PDB files for each ORF
  │
  ├─ 3. Structural search (Foldseek vs BFVD)
  │     Input:  query PDB files
  │     Output: structural hits with e-value, identity, coverage
  │     DB:     BFVD (351K AlphaFold2 viral protein structures)
  │
  ├─ 4. Structural alignment (FoldMason)
  │     Input:  query + hit structures
  │     Output: AA MSA, 3Di MSA, Newick guide tree, LDDT scores
  │
  ├─ 5. Functional classification
  │     Input:  Foldseek hit descriptions
  │     Output: functional category (11 categories) + confidence
  │
  └─ 6. Export
        Output: anvi'o, DRAM-v, vConTACT2/3, GFF3
```

## Components

### Built (109 tests passing)

| Component | File | Tests | Description |
|-----------|------|:-----:|-------------|
| Structure acquisition | `structure.py` | 27 | AlphaFold DB lookup + ColabFold prediction → PDB files |
| Foldseek PDB search | `foldseek.py` | 27 | PDB → `foldseek createdb` → search BFVD (with prob/LDDT/TM-score) |
| FoldMason alignment | `foldmason.py` | 13 | Structural MSA → AA MSA + 3Di MSA + guide tree |
| Functional classification | `categories.py` | 42 | Keyword/Pfam/GO/SUPERFAMILY → 11 functional categories |

### Remaining

| Component | vHold source | Status |
|-----------|-------------|--------|
| Gene calling (Pyrodigal-gv) | `vhold/features/genecall.py` | Pending — drop miniprot, keep Pyrodigal-gv only |
| Metagenomic export | `vhold/results/export.py` | Pending — anvi'o, DRAM-v, vConTACT2/3, GFF3 |
| GO term resolution | `vhold/results/go_terms.py` | Pending — bundled `go_term_map.json` |
| BFVD management | `vhold/databases/bfvd.py` | Pending — download/path logic |
| `annotate` CLI subcommand | — | Pending — wire pipeline together |

### NOT MIGRATING (ProstT5-dependent)

- ProstT5 encoder/decoder
- Embedding triage (436K embedding DB)
- MLP classifiers (ProstT5 embeddings)
- FANTASIA GO transfer (Swiss-Prot reference DB)
- ONNX backend
- LoRA / contrastive training
- Disorder pipeline (STARLING/metapredict)

## External Dependencies

| Tool | Purpose | Install |
|------|---------|---------|
| Pyrodigal | Gene calling | `pip install pyrodigal` (pure Python/Cython) |
| Foldseek | Structural search | `conda install -c bioconda foldseek` |
| FoldMason | Structural alignment | `conda install -c bioconda foldmason` |
| ColabFold | Structure prediction | `pip install colabfold` or conda |
| BFVD | Reference structures | `virosense install --bfvd` (download ~534 MB) |

## CLI Design (proposed)

```bash
# Full annotation pipeline
virosense annotate -i viral_contigs.fasta -o results/

# With pre-called ORFs (skip gene calling)
virosense annotate -i viral_contigs.fasta --orfs proteins.faa -o results/

# Skip ColabFold (sequence search only, no structural search for novel proteins)
virosense annotate -i viral_contigs.fasta --no-colabfold -o results/

# Export to specific formats
virosense annotate -i viral_contigs.fasta -o results/ --export anvio --export dramv

# End-to-end: detect + annotate
virosense run -i metagenome.fasta -o results/
```

## Key Design Decisions

1. **Two-pass structural search**: Fast MMseqs2 sequence search first (catches close homologs). ColabFold structure prediction only for proteins with no sequence match (the dark matter). Minimizes compute.

2. **AlphaFold DB before ColabFold**: Check if query protein already has a structure in AlphaFold Protein Structure Database (~200M proteins). Only run ColabFold prediction for truly novel proteins.

3. **BFVD as structure source**: 351K pre-computed AlphaFold2 viral protein structures in Foldseek format. No need to predict reference structures. From Steinegger lab (same group as Foldseek/FoldMason).

4. **FoldMason for phylogenetics**: Structural MSA + guide trees enable structure-based viral protein phylogenetics below the BLAST twilight zone. Full 3D coordinates from ColabFold/BFVD enable refinement and LDDT scoring (unlike ProstT5 3Di-only fastMode).

5. **Pyrodigal-gv only**: Drop miniprot splice-aware alignment (eukaryotic virus-specific). Pyrodigal-gv handles >97% of phage gene calling. Keeps dependencies minimal.

## Build Order

1. ~~**ColabFold wrapper + AlphaFold DB lookup**~~ — DONE (`structure.py`, 27 tests)
2. ~~**Foldseek PDB search wrapper**~~ — DONE (`foldseek.py`, 27 tests)
3. **Port gene calling** — Extract Pyrodigal-gv from vHold
4. ~~**Port FoldMason wrapper**~~ — DONE (`foldmason.py`, 13 tests)
5. ~~**Port functional classification**~~ — DONE (`categories.py`, 42 tests)
6. **Port metagenomic export** — anvi'o, DRAM-v, vConTACT2/3, GFF3
7. **Wire `annotate` CLI subcommand** — Integrate into virosense CLI
8. **Wire `run` CLI subcommand** — detect + annotate end-to-end

## Source References

- vHold codebase: `~/Code/tools/vHold/src/vhold/`
- BFVD: https://bfvd.steineggerlab.workers.dev/latest
- Foldseek: https://github.com/steineggerlab/foldseek
- FoldMason: https://github.com/steineggerlab/foldmason (Science, 2026)
- ColabFold: https://github.com/sokrypton/ColabFold
- AlphaFold DB: https://alphafold.ebi.ac.uk/
- Pyrodigal: https://github.com/althonos/pyrodigal
