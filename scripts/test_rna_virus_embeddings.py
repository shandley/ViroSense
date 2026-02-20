#!/usr/bin/env python3
"""Test whether Evo2 embeddings discriminate eukaryotic RNA viruses from cellular DNA.

Evo2 was trained on prokaryotic DNA + bacteriophages, with eukaryotic viruses
deliberately excluded (biosecurity). This experiment tests whether the
intermediate hidden states still capture discriminative signal for eukaryotic
RNA viruses presented as cDNA.

Experiment:
  1. Extract Evo2 embeddings for ~5kb fragments from:
     - Eukaryotic RNA viruses (SARS-CoV-2, HIV-1, Influenza A, HCV) as cDNA
     - DNA bacteriophages (T4, lambda, P22) — positive control (in-distribution)
     - Bacterial chromosomes (E. coli, B. subtilis) — negative control
     - Eukaryotic DNA viruses (HSV-1, vaccinia) — also excluded from training
  2. Compute pairwise cosine similarity matrix
  3. Visualize with PCA/UMAP to see clustering structure

Usage:
    NVIDIA_API_KEY=... python scripts/test_rna_virus_embeddings.py [--backend nim|mlx]

Sequences are fetched from NCBI or provided as inline 5kb fragments.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np
from loguru import logger

# 5kb representative fragments from well-characterized genomes.
# These are real sequences from NCBI RefSeq, truncated to ~5000 bp.
# For RNA viruses, these are the cDNA (T instead of U) as stored in GenBank.

SEQUENCES: dict[str, dict] = {
    # --- Eukaryotic RNA viruses (cDNA) — excluded from Evo2 training ---
    "SARS-CoV-2_ORF1a": {
        "category": "euk_rna_virus",
        "description": "SARS-CoV-2 ORF1a region (NC_045512.2:266-5266)",
        "accession": "NC_045512.2",
        "start": 266,
        "end": 5266,
    },
    "HIV1_gag_pol": {
        "category": "euk_rna_virus",
        "description": "HIV-1 HXB2 gag-pol region (K03455.1:790-5790)",
        "accession": "K03455.1",
        "start": 790,
        "end": 5790,
    },
    "InfluenzaA_PB2": {
        "category": "euk_rna_virus",
        "description": "Influenza A H1N1 PB2 segment (NC_026433.1:1-2341)",
        "accession": "NC_026433.1",
        "start": 0,
        "end": 2341,
    },
    "HCV_NS3_NS5": {
        "category": "euk_rna_virus",
        "description": "Hepatitis C polyprotein region (NC_004102.1:3420-8420)",
        "accession": "NC_004102.1",
        "start": 3420,
        "end": 8420,
    },
    # --- DNA bacteriophages — in Evo2 training (positive control) ---
    "T4_gene23": {
        "category": "dna_phage",
        "description": "T4 phage major capsid protein region (NC_000866.4:67000-72000)",
        "accession": "NC_000866.4",
        "start": 67000,
        "end": 72000,
    },
    "Lambda_CI_N": {
        "category": "dna_phage",
        "description": "Lambda phage CI-N region (NC_001416.1:33000-38000)",
        "accession": "NC_001416.1",
        "start": 33000,
        "end": 38000,
    },
    "P22_tailspike": {
        "category": "dna_phage",
        "description": "P22 phage tailspike region (NC_002371.2:25000-30000)",
        "accession": "NC_002371.2",
        "start": 25000,
        "end": 30000,
    },
    # --- Bacterial chromosomes — negative control ---
    "Ecoli_rpoB": {
        "category": "bacteria",
        "description": "E. coli K-12 rpoB region (NC_000913.3:4180000-4185000)",
        "accession": "NC_000913.3",
        "start": 4180000,
        "end": 4185000,
    },
    "Bsubtilis_sporulation": {
        "category": "bacteria",
        "description": "B. subtilis sporulation region (NC_000964.3:2500000-2505000)",
        "accession": "NC_000964.3",
        "start": 2500000,
        "end": 2505000,
    },
    # --- Eukaryotic DNA viruses — also excluded from training ---
    "HSV1_UL30": {
        "category": "euk_dna_virus",
        "description": "HSV-1 DNA polymerase region (NC_001806.2:63000-68000)",
        "accession": "NC_001806.2",
        "start": 63000,
        "end": 68000,
    },
    "Vaccinia_F13L": {
        "category": "euk_dna_virus",
        "description": "Vaccinia virus F13L region (NC_006998.1:40000-45000)",
        "accession": "NC_006998.1",
        "start": 40000,
        "end": 45000,
    },
}

CATEGORY_LABELS = {
    "euk_rna_virus": "Eukaryotic RNA virus (cDNA)",
    "dna_phage": "DNA bacteriophage (in-distribution)",
    "bacteria": "Bacterial chromosome (cellular)",
    "euk_dna_virus": "Eukaryotic DNA virus (excluded)",
}


def fetch_sequences() -> dict[str, str]:
    """Fetch sequences from NCBI Entrez or local cache.

    Returns dict of seq_id -> DNA sequence string.
    """
    from Bio import Entrez, SeqIO

    Entrez.email = "virosense@example.com"
    cache_dir = Path.home() / ".virosense" / "cache" / "rna_virus_test"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "test_sequences.npz"

    if cache_file.exists():
        logger.info(f"Loading cached sequences from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return dict(data["sequences"].item())

    sequences = {}
    # Group by accession to minimize Entrez calls
    accession_map: dict[str, list[str]] = {}
    for seq_id, info in SEQUENCES.items():
        acc = info["accession"]
        accession_map.setdefault(acc, []).append(seq_id)

    for acc, seq_ids in accession_map.items():
        logger.info(f"Fetching {acc} from NCBI...")
        try:
            handle = Entrez.efetch(db="nucleotide", id=acc, rettype="fasta", retmode="text")
            record = SeqIO.read(handle, "fasta")
            handle.close()
            full_seq = str(record.seq).upper()
            # Replace IUPAC ambiguity codes with random valid bases
            full_seq = re.sub(r"[^ACGT]", lambda _: random.choice("ACGT"), full_seq)

            for seq_id in seq_ids:
                info = SEQUENCES[seq_id]
                fragment = full_seq[info["start"]:info["end"]]
                if len(fragment) < 100:
                    logger.warning(f"Fragment {seq_id} too short ({len(fragment)} bp), skipping")
                    continue
                sequences[seq_id] = fragment
                logger.info(f"  {seq_id}: {len(fragment)} bp")
        except Exception as e:
            logger.error(f"Failed to fetch {acc}: {e}")

    # Cache
    np.savez(cache_file, sequences=sequences)
    logger.info(f"Cached {len(sequences)} sequences to {cache_file}")
    return sequences


def extract_and_analyze(sequences: dict[str, str], backend_name: str, nim_url: str | None) -> None:
    """Extract embeddings and analyze clustering structure."""
    from sklearn.metrics.pairwise import cosine_similarity

    from virosense.backends.base import EmbeddingRequest, get_backend

    # Get backend
    kwargs = {}
    if nim_url:
        kwargs["nim_url"] = nim_url
    backend = get_backend(backend_name, **kwargs)
    if not backend.is_available():
        logger.error(f"Backend '{backend_name}' is not available")
        sys.exit(1)

    logger.info(f"Using backend: {backend}")

    # Extract embeddings
    request = EmbeddingRequest(sequences=sequences)
    result = backend.extract_embeddings(request)

    embeddings = result.embeddings  # (N, D)
    seq_ids = result.sequence_ids
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Categories for each sequence
    categories = [SEQUENCES[sid]["category"] for sid in seq_ids]
    unique_cats = sorted(set(categories))

    # Cosine similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # Print similarity matrix grouped by category
    print("\n" + "=" * 80)
    print("COSINE SIMILARITY MATRIX")
    print("=" * 80)

    # Print header
    print(f"{'':>25s}", end="")
    for sid in seq_ids:
        print(f" {sid[:10]:>10s}", end="")
    print()

    for i, sid_i in enumerate(seq_ids):
        cat_i = SEQUENCES[sid_i]["category"]
        print(f"  {sid_i[:23]:>23s}", end="")
        for j in range(len(seq_ids)):
            val = sim_matrix[i, j]
            print(f" {val:>10.3f}", end="")
        print(f"  [{cat_i}]")

    # Within-category vs between-category similarities
    print("\n" + "=" * 80)
    print("CATEGORY ANALYSIS")
    print("=" * 80)

    for cat in unique_cats:
        cat_indices = [i for i, c in enumerate(categories) if c == cat]
        cat_sims = []
        for i in cat_indices:
            for j in cat_indices:
                if i < j:
                    cat_sims.append(sim_matrix[i, j])
        if cat_sims:
            print(f"\n  {CATEGORY_LABELS[cat]}:")
            print(f"    Within-group similarity: {np.mean(cat_sims):.3f} "
                  f"(min={np.min(cat_sims):.3f}, max={np.max(cat_sims):.3f})")

    # Cross-category similarities
    print("\n  Cross-category mean similarities:")
    for i_cat in unique_cats:
        for j_cat in unique_cats:
            if i_cat >= j_cat:
                continue
            i_idx = [k for k, c in enumerate(categories) if c == i_cat]
            j_idx = [k for k, c in enumerate(categories) if c == j_cat]
            cross_sims = [sim_matrix[a, b] for a in i_idx for b in j_idx]
            print(f"    {CATEGORY_LABELS[i_cat][:30]:>30s} vs "
                  f"{CATEGORY_LABELS[j_cat][:30]:<30s}: {np.mean(cross_sims):.3f}")

    # Key question: do RNA virus embeddings look more like phage or bacteria?
    print("\n" + "=" * 80)
    print("KEY QUESTION: Are eukaryotic RNA virus embeddings discriminative?")
    print("=" * 80)

    rna_idx = [i for i, c in enumerate(categories) if c == "euk_rna_virus"]
    phage_idx = [i for i, c in enumerate(categories) if c == "dna_phage"]
    bact_idx = [i for i, c in enumerate(categories) if c == "bacteria"]

    if rna_idx and phage_idx and bact_idx:
        rna_phage = np.mean([sim_matrix[a, b] for a in rna_idx for b in phage_idx])
        rna_bact = np.mean([sim_matrix[a, b] for a in rna_idx for b in bact_idx])
        phage_bact = np.mean([sim_matrix[a, b] for a in phage_idx for b in bact_idx])

        print(f"\n  RNA virus ↔ Phage similarity:    {rna_phage:.3f}")
        print(f"  RNA virus ↔ Bacteria similarity:  {rna_bact:.3f}")
        print(f"  Phage ↔ Bacteria similarity:      {phage_bact:.3f}")

        if rna_bact < rna_phage:
            print("\n  → RNA viruses are MORE similar to phages than bacteria")
            print("    Evo2 embeddings may carry discriminative signal for RNA viruses!")
        else:
            print("\n  → RNA viruses are MORE similar to bacteria than phages")
            print("    Evo2 embeddings may NOT distinguish RNA viruses from cellular DNA")

    # Save results
    output_dir = Path("data/test/rna_virus_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "embeddings.npz",
        embeddings=embeddings,
        sequence_ids=np.array(seq_ids),
        categories=np.array(categories),
        similarity_matrix=sim_matrix,
    )
    logger.info(f"Results saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Evo2 embeddings on RNA viruses")
    parser.add_argument("--backend", default="nim", choices=["nim", "mlx"],
                        help="Embedding backend (default: nim)")
    parser.add_argument("--nim-url", default=None, help="Self-hosted NIM URL")
    args = parser.parse_args()

    sequences = fetch_sequences()
    logger.info(f"Loaded {len(sequences)} test sequences")

    extract_and_analyze(sequences, args.backend, args.nim_url)


if __name__ == "__main__":
    main()
