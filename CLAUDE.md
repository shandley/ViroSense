# ViroSense - Claude Code Context

Last updated: 2026-02-07

## Project Overview

**ViroSense** is a multi-modal viral detection and characterization tool that combines DNA-level analysis (Evo2 foundation model) with protein-level analysis (ProstT5/vHold) for viral sequence detection, annotation, and classification.

### Relationship to vHold
- vHold: Viral protein annotation via structural homology (ProstT5 + Foldseek)
- ViroSense: DNA + protein multi-modal analysis using Evo2 + vHold integration
- vHold is an optional dependency (`[vhold]` extra), integrated via file-based data exchange

### Key Constraint
Evo2 requires NVIDIA GPU (H100/Ada+ with CUDA 12.1+ and FP8). Developer machine is Apple M4. Solution: backend abstraction with NVIDIA NIM API as default.

## Architecture

### Four Modules

1. **detect** — Classify metagenomic contigs as viral vs cellular using Evo2 DNA embeddings
2. **context** — Enhance ORF annotation with genomic context (Evo2 windows + vHold merge)
3. **cluster** — Organize viral dark matter using fused DNA + protein embeddings
4. **classify** — Train discriminative classifiers on frozen Evo2 embeddings

### Backend Abstraction

```
Evo2Backend (ABC)
├── NIMBackend   — NVIDIA NIM API (default, works anywhere, needs NVIDIA_API_KEY)
├── LocalBackend — Direct evo2 package (needs CUDA H100/Ada GPU)
└── ModalBackend — Modal.com serverless GPU (stub)
```

All backends implement: `extract_embeddings()`, `is_available()`, `max_context_length()`

### Dependencies
- **Core**: click, biopython, numpy, pandas, loguru, scikit-learn (no torch!)
- **[gpu]**: torch, evo2
- **[vhold]**: torch, transformers, sentencepiece
- **[nim]**: httpx
- **[dev]**: pytest, pytest-cov, ruff

## Commands

```bash
virosense detect -i contigs.fasta -o results/ --backend nim
virosense context -i viral.fasta --orfs orfs.gff3 -o results/
virosense cluster -i unknown.fasta -o clusters/ --mode multi
virosense classify -i seqs.fasta --labels labels.tsv -o model/
```

## Implementation Status

### Phase 1: Scaffold — COMPLETE
- Project structure, pyproject.toml, CLI with all 4 commands
- Utils: constants, logging, config, device
- All module stubs with NotImplementedError
- Tests: CLI, I/O, backends, clustering

### Phase 2: Backend Layer — NOT STARTED
- Implement NIM API client (HTTP calls to health.api.nvidia.com)
- Mocked HTTP tests

### Phase 3: Embedding Infrastructure — NOT STARTED
### Phase 4: detect module — NOT STARTED
### Phase 5: vHold bridge — NOT STARTED
### Phase 6: cluster module — NOT STARTED
### Phase 7: context module — NOT STARTED
### Phase 8: classify module — NOT STARTED

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `NVIDIA_API_KEY` | NIM API access |
| `VIROSENSE_DATA_DIR` | Override ~/.virosense |
| `VIROSENSE_CACHE_DIR` | Embedding cache location |

## Key Source Files

| File | Purpose |
|------|---------|
| `src/virosense/cli.py` | Click CLI with 4 subcommands |
| `src/virosense/backends/base.py` | Evo2Backend ABC + factory |
| `src/virosense/backends/nim.py` | NIM API client |
| `src/virosense/features/evo2_embeddings.py` | Embedding extraction + NPZ cache |
| `src/virosense/features/prostt5_bridge.py` | Optional vHold integration |
| `src/virosense/clustering/multimodal.py` | Multi-modal fusion + clustering |
| `src/virosense/models/detector.py` | Viral classifier head |
| `src/virosense/io/fasta.py` | DNA FASTA I/O |

## Development

```bash
# Install in dev mode
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Run CLI
uv run virosense --help
```

## Biosecurity Note

Evo2 deliberately excludes eukaryotic viral sequences from training for biosecurity. ViroSense uses Evo2 only for discriminative tasks (classification, clustering) — not sequence generation. This is a safe, defensive application focused on detection and characterization of existing sequences.
