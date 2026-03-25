#!/usr/bin/env python3
"""
Find gene-rich 500kb regions for non-model organism scans.

Downloads annotations for candidate windows across each chromosome and
selects the 500kb region with the highest CDS coverage.

Usage:
    uv run python scripts/find_gene_rich_regions.py
"""

import json
import time
from pathlib import Path
from io import StringIO

from Bio import Entrez, SeqIO

Entrez.email = "shandley@wustl.edu"

OUT_DIR = Path("results/experiments/nonmodel_genome")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORGANISMS = [
    {
        "name": "danaus_plexippus",
        "common": "Monarch butterfly",
        "accession": "NC_045808.1",
        "chr_length": 10_600_000,
        "gc_expected": 31.5,
    },
    {
        "name": "physcomitrium_patens",
        "common": "Spreading earthmoss",
        "accession": "NC_037253.2",
        "chr_length": 30_300_000,
        "gc_expected": 33.5,
    },
    {
        "name": "acropora_millepora",
        "common": "Staghorn coral",
        "accession": "NC_058066.1",
        "chr_length": 39_400_000,
        "gc_expected": 39.0,
    },
    {
        "name": "dictyostelium_discoideum",
        "common": "Social amoeba",
        "accession": "NC_007088.5",
        "chr_length": 8_470_286,
        "gc_expected": 22.5,
    },
    {
        "name": "magallana_gigas",
        "common": "Pacific oyster",
        "accession": "NC_088853.1",
        "chr_length": 76_100_000,
        "gc_expected": 33.5,
    },
]

REGION_SIZE = 500_000
# Sample windows spaced across the chromosome
N_CANDIDATES = 10


def count_cds_coverage(accession: str, start: int, end: int) -> tuple[int, int, float]:
    """Download annotations for a region and count CDS coverage."""
    for attempt in range(3):
        try:
            handle = Entrez.efetch(
                db="nucleotide", id=accession, rettype="gb", retmode="text",
                seq_start=start + 1, seq_stop=end,
            )
            record = next(SeqIO.parse(StringIO(handle.read()), "genbank"))
            handle.close()

            region_len = end - start
            cds_track = [0] * region_len
            n_genes = 0
            n_cds = 0

            for feat in record.features:
                if feat.type == "gene":
                    n_genes += 1
                if feat.type == "CDS":
                    for part in feat.location.parts:
                        s = max(0, int(part.start))
                        e = min(int(part.end), region_len)
                        for i in range(s, e):
                            cds_track[i] = 1
                        n_cds += 1

            coding_fraction = sum(cds_track) / region_len
            return n_genes, n_cds, coding_fraction

        except Exception as e:
            if attempt < 2:
                time.sleep(3)
            else:
                print(f"      FAILED: {e}")
                return 0, 0, 0.0


def main():
    results = {}

    for org in ORGANISMS:
        name = org["name"]
        acc = org["accession"]
        chr_len = org["chr_length"]

        print(f"\n{'='*60}")
        print(f"{org['common']} ({name})")
        print(f"  Chromosome: {acc}, ~{chr_len/1e6:.1f} Mb")
        print(f"  Sampling {N_CANDIDATES} candidate 500kb windows...")

        # Generate candidate start positions evenly spaced
        # Avoid very start and end (often gene-poor telomeric regions)
        margin = max(REGION_SIZE, int(chr_len * 0.05))
        step = (chr_len - 2 * margin) // (N_CANDIDATES - 1)
        candidates = []

        for i in range(N_CANDIDATES):
            start = margin + i * step
            end = start + REGION_SIZE
            if end > chr_len:
                break

            print(f"    Window {i+1}/{N_CANDIDATES}: {start:,}-{end:,}...", end=" ", flush=True)
            n_genes, n_cds, coding_frac = count_cds_coverage(acc, start, end)
            print(f"{n_genes} genes, {coding_frac:.1%} coding")

            candidates.append({
                "start": start,
                "end": end,
                "n_genes": n_genes,
                "n_cds": n_cds,
                "coding_fraction": coding_frac,
            })

            # Be nice to NCBI
            time.sleep(1)

        # Pick the best
        best = max(candidates, key=lambda c: c["coding_fraction"])
        print(f"\n  BEST: {best['start']:,}-{best['end']:,} — "
              f"{best['n_genes']} genes, {best['coding_fraction']:.1%} coding")

        results[name] = {
            "organism": org["common"],
            "accession": acc,
            "best_start": best["start"],
            "best_end": best["end"],
            "best_genes": best["n_genes"],
            "best_coding_fraction": best["coding_fraction"],
            "candidates": candidates,
        }

    # Print summary config for the scan script
    print(f"\n{'='*60}")
    print("RECOMMENDED CONFIGURATIONS:")
    print(f"{'='*60}")
    for name, r in results.items():
        print(f'    {{"name": "{name}",')
        print(f'     "start": {r["best_start"]},')
        print(f'     "region": {REGION_SIZE},')
        print(f'     # {r["organism"]}: {r["best_genes"]} genes, '
              f'{r["best_coding_fraction"]:.1%} coding}}')

    with open(OUT_DIR / "gene_rich_regions.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/gene_rich_regions.json")


if __name__ == "__main__":
    main()
