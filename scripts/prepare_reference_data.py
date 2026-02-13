#!/usr/bin/env python3
"""Download and prepare reference data for ViroSense viral classifier training.

Downloads DNA phage and cellular genomes from NCBI RefSeq via Entrez,
fragments them into contig-like pieces at various lengths, and outputs
a training FASTA + labels TSV for use with `virosense build-reference`.

TAXONOMIC SCOPE:
  Viral (label=1): DNA bacteriophages and archaeal viruses only.
    - Caudoviricetes (tailed dsDNA phage) — dominant in metagenomes
    - Microviridae (ssDNA phage) — common in gut/ocean metagenomes
    - Inoviridae (filamentous ssDNA phage) — prophage-associated
    - Archaeal DNA viruses (Lipothrixviridae, Rudiviridae, etc.)

  Cellular (label=0): Bacteria and archaea.
    - Diverse bacterial phyla from RefSeq
    - Archaeal genomes

  EXCLUDED: RNA viruses, eukaryotic viruses (Evo2 was not trained on
  eukaryotic viral sequences for biosecurity reasons; RNA virus cDNA
  is out-of-distribution for a DNA foundation model).

Usage:
    python scripts/prepare_reference_data.py \\
        --email your@email.com \\
        --output data/reference/ \\
        --n-phage-dsdna 600 \\
        --n-phage-ssdna 100 \\
        --n-archaeal-virus 30

Requires: biopython (already a virosense dependency)
"""

import argparse
import json
import random
import sys
import time
from io import StringIO
from pathlib import Path

from Bio import Entrez, SeqIO

# Fragment sizes (bp) with number of fragments per genome
FRAGMENT_SIZES = [
    (500, 3),    # 3 fragments of 500 bp per genome
    (1000, 2),   # 2 fragments of 1000 bp
    (2000, 2),   # 2 fragments of 2000 bp
    (3000, 1),   # 1 fragment of 3000 bp
    (5000, 1),   # 1 fragment of 5000 bp
]

# ---------------------------------------------------------------------------
# Entrez search queries — taxonomically explicit
# ---------------------------------------------------------------------------

# dsDNA tailed phages (Caudoviricetes) — by far the most common phage in
# metagenomes.  Includes former Myoviridae, Siphoviridae, Podoviridae
# (dissolved by ICTV 2022) plus newer families like Autographiviridae,
# Demerecviridae, Herelleviridae, Drexlerviridae, etc.
PHAGE_DSDNA_QUERY = (
    'Caudoviricetes[Organism] AND "complete genome"[Title] '
    "AND refseq[filter] AND 10000:500000[SLEN]"
)

# ssDNA phages — Microviridae (tailless icosahedral, common in gut/ocean)
PHAGE_SSDNA_MICRO_QUERY = (
    'Microviridae[Organism] AND "complete genome"[Title] '
    "AND refseq[filter] AND 3000:10000[SLEN]"
)

# ssDNA phages — Inoviridae (filamentous, often integrated as prophage)
PHAGE_SSDNA_INO_QUERY = (
    'Inoviridae[Organism] AND "complete genome"[Title] '
    "AND refseq[filter] AND 4000:15000[SLEN]"
)

# Archaeal DNA viruses — diverse morphotypes infecting extremophiles
# Lipothrixviridae (filamentous), Rudiviridae (rod-shaped),
# Fuselloviridae (spindle-shaped), Bicaudaviridae (two-tailed),
# Turriviridae (icosahedral)
ARCHAEAL_VIRUS_QUERY = (
    "(Lipothrixviridae[Organism] OR Rudiviridae[Organism] "
    "OR Fuselloviridae[Organism] OR Bicaudaviridae[Organism] "
    "OR Turriviridae[Organism] OR Globuloviridae[Organism] "
    "OR Ampullaviridae[Organism] OR Guttaviridae[Organism]) "
    'AND "complete genome"[Title] AND refseq[filter]'
)

# Cellular: bacteria (diverse phyla)
BACTERIAL_QUERY = (
    'bacteria[Organism] AND "complete genome"[Title] '
    "AND refseq[filter] AND 1000000:15000000[SLEN]"
)

# Cellular: archaea
ARCHAEAL_QUERY = (
    'archaea[Organism] AND "complete genome"[Title] '
    "AND refseq[filter] AND 500000:10000000[SLEN]"
)

ENTREZ_BATCH_SIZE = 100
ENTREZ_DELAY = 0.4  # seconds between requests (NCBI asks for <=3/sec without API key)


def search_ids(query: str, max_results: int) -> list[str]:
    """Search NCBI nucleotide database and return accession IDs."""
    print(f"  Searching: {query[:90]}...")
    handle = Entrez.esearch(
        db="nucleotide", term=query, retmax=max_results, usehistory="y"
    )
    results = Entrez.read(handle)
    handle.close()
    ids = results["IdList"]
    total = int(results["Count"])
    print(f"  Found {total} results, retrieved {len(ids)} IDs")
    return ids


def download_sequences(ids: list[str], label: str) -> list[tuple]:
    """Download sequences from NCBI in batches. Returns list of (id, seq, description)."""
    sequences = []
    for i in range(0, len(ids), ENTREZ_BATCH_SIZE):
        batch = ids[i : i + ENTREZ_BATCH_SIZE]
        batch_num = i // ENTREZ_BATCH_SIZE + 1
        total_batches = (len(ids) + ENTREZ_BATCH_SIZE - 1) // ENTREZ_BATCH_SIZE
        print(f"  Downloading {label} batch {batch_num}/{total_batches} ({len(batch)} sequences)...")

        try:
            handle = Entrez.efetch(
                db="nucleotide", id=batch, rettype="fasta", retmode="text"
            )
            text = handle.read()
            handle.close()

            for record in SeqIO.parse(StringIO(text), "fasta"):
                seq = str(record.seq).upper()
                if set(seq).issubset({"A", "C", "G", "T", "N"}):
                    sequences.append((record.id, seq, record.description))

        except Exception as e:
            print(f"    Warning: batch {batch_num} failed: {e}")

        time.sleep(ENTREZ_DELAY)

    print(f"  Downloaded {len(sequences)} valid {label} sequences")
    return sequences


def fragment_genome(
    seq_id: str, sequence: str, prefix: str
) -> list[tuple[str, str]]:
    """Fragment a genome into contig-like pieces at various lengths.

    Returns list of (fragment_id, fragment_sequence).
    """
    fragments = []
    seq_len = len(sequence)

    for frag_size, n_frags in FRAGMENT_SIZES:
        if seq_len < frag_size:
            continue

        for j in range(n_frags):
            max_start = seq_len - frag_size
            if max_start <= 0:
                start = 0
            else:
                start = random.randint(0, max_start)

            frag_seq = sequence[start : start + frag_size]

            # Skip fragments with too many Ns (>5%)
            n_count = frag_seq.count("N")
            if n_count / len(frag_seq) > 0.05:
                continue

            frag_id = f"{prefix}_{seq_id}_f{frag_size}_{j}"
            fragments.append((frag_id, frag_seq))

    return fragments


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare reference data for ViroSense",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Taxonomic scope:
  Viral:    DNA phages (Caudoviricetes, Microviridae, Inoviridae) and
            archaeal DNA viruses. NO RNA viruses or eukaryotic viruses.
  Cellular: Bacteria and archaea from NCBI RefSeq.
""",
    )
    parser.add_argument(
        "--email", required=True, help="Email for NCBI Entrez (required by NCBI)"
    )
    parser.add_argument(
        "--output", default="data/reference", help="Output directory"
    )
    parser.add_argument(
        "--n-phage-dsdna", type=int, default=600,
        help="Number of tailed dsDNA phage genomes (Caudoviricetes) (default: 600)",
    )
    parser.add_argument(
        "--n-phage-ssdna", type=int, default=100,
        help="Number of ssDNA phage genomes (Microviridae+Inoviridae) (default: 100)",
    )
    parser.add_argument(
        "--n-archaeal-virus", type=int, default=30,
        help="Number of archaeal DNA virus genomes (default: 30)",
    )
    parser.add_argument(
        "--n-bacterial", type=int, default=300,
        help="Number of bacterial genomes (default: 300)",
    )
    parser.add_argument(
        "--n-archaeal", type=int, default=50,
        help="Number of archaeal genomes (default: 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    Entrez.email = args.email

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Search and sample genome IDs ----
    print("\n=== Step 1: Searching NCBI for genome IDs ===")

    # -- Viral: dsDNA tailed phage (Caudoviricetes) --
    print("\n[VIRAL] Searching tailed dsDNA phage (Caudoviricetes)...")
    phage_dsdna_ids = search_ids(PHAGE_DSDNA_QUERY, max_results=20000)
    if len(phage_dsdna_ids) > args.n_phage_dsdna:
        phage_dsdna_ids = random.sample(phage_dsdna_ids, args.n_phage_dsdna)
    print(f"  Sampled {len(phage_dsdna_ids)} tailed phage IDs")

    # -- Viral: ssDNA phage (Microviridae) --
    n_micro = args.n_phage_ssdna * 2 // 3  # allocate 2/3 to Microviridae
    n_ino = args.n_phage_ssdna - n_micro    # allocate 1/3 to Inoviridae

    print("\n[VIRAL] Searching ssDNA phage (Microviridae)...")
    micro_ids = search_ids(PHAGE_SSDNA_MICRO_QUERY, max_results=2000)
    if len(micro_ids) > n_micro:
        micro_ids = random.sample(micro_ids, n_micro)
    print(f"  Sampled {len(micro_ids)} Microviridae IDs")

    print("\n[VIRAL] Searching ssDNA phage (Inoviridae)...")
    ino_ids = search_ids(PHAGE_SSDNA_INO_QUERY, max_results=1000)
    if len(ino_ids) > n_ino:
        ino_ids = random.sample(ino_ids, n_ino)
    print(f"  Sampled {len(ino_ids)} Inoviridae IDs")

    # -- Viral: archaeal DNA viruses --
    print("\n[VIRAL] Searching archaeal DNA viruses...")
    archaeal_virus_ids = search_ids(ARCHAEAL_VIRUS_QUERY, max_results=500)
    if len(archaeal_virus_ids) > args.n_archaeal_virus:
        archaeal_virus_ids = random.sample(archaeal_virus_ids, args.n_archaeal_virus)
    print(f"  Sampled {len(archaeal_virus_ids)} archaeal virus IDs")

    # -- Cellular: bacteria --
    print("\n[CELLULAR] Searching bacterial genomes...")
    bacterial_ids = search_ids(BACTERIAL_QUERY, max_results=10000)
    if len(bacterial_ids) > args.n_bacterial:
        bacterial_ids = random.sample(bacterial_ids, args.n_bacterial)
    print(f"  Sampled {len(bacterial_ids)} bacterial genome IDs")

    # -- Cellular: archaea --
    print("\n[CELLULAR] Searching archaeal genomes...")
    archaeal_ids = search_ids(ARCHAEAL_QUERY, max_results=2000)
    if len(archaeal_ids) > args.n_archaeal:
        archaeal_ids = random.sample(archaeal_ids, args.n_archaeal)
    print(f"  Sampled {len(archaeal_ids)} archaeal genome IDs")

    total_viral_ids = len(phage_dsdna_ids) + len(micro_ids) + len(ino_ids) + len(archaeal_virus_ids)
    total_cellular_ids = len(bacterial_ids) + len(archaeal_ids)
    print(f"\n  Total viral genome IDs: {total_viral_ids}")
    print(f"  Total cellular genome IDs: {total_cellular_ids}")

    # ---- 2. Download sequences ----
    print("\n=== Step 2: Downloading sequences from NCBI ===")

    print("\nDownloading tailed dsDNA phage...")
    phage_dsdna_seqs = download_sequences(phage_dsdna_ids, "dsDNA phage")

    print("\nDownloading Microviridae...")
    micro_seqs = download_sequences(micro_ids, "Microviridae")

    print("\nDownloading Inoviridae...")
    ino_seqs = download_sequences(ino_ids, "Inoviridae")

    print("\nDownloading archaeal viruses...")
    archaeal_virus_seqs = download_sequences(archaeal_virus_ids, "archaeal virus")

    viral_seqs = phage_dsdna_seqs + micro_seqs + ino_seqs + archaeal_virus_seqs

    print("\nDownloading bacterial genomes...")
    bacterial_seqs = download_sequences(bacterial_ids, "bacterial")

    print("\nDownloading archaeal genomes...")
    archaeal_seqs = download_sequences(archaeal_ids, "archaeal")

    cellular_seqs = bacterial_seqs + archaeal_seqs

    if not viral_seqs:
        print("ERROR: No viral sequences downloaded. Check your network/email.")
        sys.exit(1)
    if not cellular_seqs:
        print("ERROR: No cellular sequences downloaded. Check your network/email.")
        sys.exit(1)

    # ---- 3. Fragment genomes ----
    print("\n=== Step 3: Fragmenting genomes ===")

    viral_fragments = []
    viral_type_counts = {
        "dsDNA_phage": 0, "Microviridae": 0, "Inoviridae": 0, "archaeal_virus": 0,
    }
    for seq_id, seq, desc in phage_dsdna_seqs:
        frags = fragment_genome(seq_id, seq, "phage_dsdna")
        viral_fragments.extend(frags)
        viral_type_counts["dsDNA_phage"] += len(frags)

    for seq_id, seq, desc in micro_seqs:
        frags = fragment_genome(seq_id, seq, "phage_micro")
        viral_fragments.extend(frags)
        viral_type_counts["Microviridae"] += len(frags)

    for seq_id, seq, desc in ino_seqs:
        frags = fragment_genome(seq_id, seq, "phage_ino")
        viral_fragments.extend(frags)
        viral_type_counts["Inoviridae"] += len(frags)

    for seq_id, seq, desc in archaeal_virus_seqs:
        frags = fragment_genome(seq_id, seq, "archvir")
        viral_fragments.extend(frags)
        viral_type_counts["archaeal_virus"] += len(frags)

    print(f"  Generated {len(viral_fragments)} viral fragments:")
    for vtype, count in viral_type_counts.items():
        print(f"    {vtype}: {count}")

    cellular_fragments = []
    cellular_type_counts = {"bacterial": 0, "archaeal": 0}
    for seq_id, seq, desc in bacterial_seqs:
        frags = fragment_genome(seq_id, seq, "bact")
        cellular_fragments.extend(frags)
        cellular_type_counts["bacterial"] += len(frags)

    for seq_id, seq, desc in archaeal_seqs:
        frags = fragment_genome(seq_id, seq, "arch")
        cellular_fragments.extend(frags)
        cellular_type_counts["archaeal"] += len(frags)

    print(f"  Generated {len(cellular_fragments)} cellular fragments:")
    for ctype, count in cellular_type_counts.items():
        print(f"    {ctype}: {count}")

    # ---- 4. Balance classes ----
    min_count = min(len(viral_fragments), len(cellular_fragments))
    if len(viral_fragments) > min_count:
        viral_fragments = random.sample(viral_fragments, min_count)
    if len(cellular_fragments) > min_count:
        cellular_fragments = random.sample(cellular_fragments, min_count)

    print(f"\n  Balanced to {min_count} fragments per class ({min_count * 2} total)")

    # ---- 5. Write output ----
    print("\n=== Step 4: Writing output files ===")

    fasta_path = output_dir / "sequences.fasta"
    labels_path = output_dir / "labels.tsv"
    manifest_path = output_dir / "manifest.json"

    with open(fasta_path, "w") as fasta_f, open(labels_path, "w") as labels_f:
        labels_f.write("sequence_id\tlabel\n")

        for frag_id, frag_seq in viral_fragments:
            fasta_f.write(f">{frag_id}\n{frag_seq}\n")
            labels_f.write(f"{frag_id}\t1\n")

        for frag_id, frag_seq in cellular_fragments:
            fasta_f.write(f">{frag_id}\n{frag_seq}\n")
            labels_f.write(f"{frag_id}\t0\n")

    total = len(viral_fragments) + len(cellular_fragments)
    print(f"  Wrote {total} fragments to {fasta_path}")
    print(f"  Wrote labels to {labels_path}")

    # ---- 6. Write manifest (provenance) ----
    manifest = {
        "description": "ViroSense reference classifier training data",
        "taxonomic_scope": {
            "viral": [
                "Caudoviricetes (tailed dsDNA bacteriophages)",
                "Microviridae (ssDNA phage)",
                "Inoviridae (filamentous ssDNA phage)",
                "Archaeal DNA viruses (Lipothrixviridae, Rudiviridae, etc.)",
            ],
            "cellular": [
                "Bacteria (diverse phyla, RefSeq complete genomes)",
                "Archaea (RefSeq complete genomes)",
            ],
            "excluded": [
                "RNA viruses (out-of-distribution for Evo2 DNA model)",
                "Eukaryotic viruses (excluded from Evo2 training, biosecurity)",
            ],
        },
        "source": "NCBI RefSeq via Entrez",
        "seed": args.seed,
        "genomes_downloaded": {
            "dsDNA_phage": len(phage_dsdna_seqs),
            "Microviridae": len(micro_seqs),
            "Inoviridae": len(ino_seqs),
            "archaeal_virus": len(archaeal_virus_seqs),
            "total_viral": len(viral_seqs),
            "bacterial": len(bacterial_seqs),
            "archaeal": len(archaeal_seqs),
            "total_cellular": len(cellular_seqs),
        },
        "fragments_before_balancing": {
            "viral": viral_type_counts,
            "cellular": cellular_type_counts,
        },
        "fragments_after_balancing": {
            "viral": len(viral_fragments),
            "cellular": len(cellular_fragments),
            "total": total,
        },
        "fragment_sizes_bp": [s for s, _ in FRAGMENT_SIZES],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote provenance manifest to {manifest_path}")

    # ---- Summary ----
    fasta_size_mb = fasta_path.stat().st_size / (1024 * 1024)
    print(f"\n=== Summary ===")
    print(f"  Genomes downloaded:")
    print(f"    dsDNA phage (Caudoviricetes): {len(phage_dsdna_seqs)}")
    print(f"    ssDNA phage (Microviridae):   {len(micro_seqs)}")
    print(f"    ssDNA phage (Inoviridae):     {len(ino_seqs)}")
    print(f"    Archaeal DNA viruses:         {len(archaeal_virus_seqs)}")
    print(f"    --- Total viral:              {len(viral_seqs)}")
    print(f"    Bacterial:                    {len(bacterial_seqs)}")
    print(f"    Archaeal:                     {len(archaeal_seqs)}")
    print(f"    --- Total cellular:           {len(cellular_seqs)}")
    print(f"  Fragments (balanced): {total} ({len(viral_fragments)} viral + {len(cellular_fragments)} cellular)")
    print(f"  FASTA size: {fasta_size_mb:.1f} MB")
    print(f"\n  Next step:")
    print(f"  virosense build-reference \\")
    print(f"    -i {fasta_path} \\")
    print(f"    --labels {labels_path} \\")
    print(f"    -o data/reference/model/ \\")
    print(f"    --install")


if __name__ == "__main__":
    main()
