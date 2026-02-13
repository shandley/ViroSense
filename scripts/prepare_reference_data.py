#!/usr/bin/env python3
"""Download and prepare reference data for ViroSense viral classifier training.

Downloads viral and bacterial/archaeal genomes from NCBI RefSeq via Entrez,
fragments them into contig-like pieces at various lengths, and outputs
a training FASTA + labels TSV for use with `virosense build-reference`.

Usage:
    python scripts/prepare_reference_data.py \
        --email your@email.com \
        --output data/reference/ \
        --n-viral 300 \
        --n-cellular 100

Requires: biopython (already a virosense dependency)
"""

import argparse
import random
import sys
import time
from io import StringIO
from pathlib import Path

from Bio import Entrez, SeqIO

# Fragment sizes (bp) with sampling weights (more short contigs)
FRAGMENT_SIZES = [
    (500, 3),    # 3 fragments of 500 bp per genome
    (1000, 2),   # 2 fragments of 1000 bp
    (2000, 2),   # 2 fragments of 2000 bp
    (3000, 1),   # 1 fragment of 3000 bp
    (5000, 1),   # 1 fragment of 5000 bp
]

# Entrez search queries
VIRAL_QUERY = (
    'viruses[Organism] AND "complete genome"[Title] '
    "AND refseq[filter] AND 5000:300000[SLEN]"
)

BACTERIAL_QUERY = (
    'bacteria[Organism] AND "complete genome"[Title] '
    "AND refseq[filter] AND 1000000:15000000[SLEN]"
)

ARCHAEAL_QUERY = (
    'archaea[Organism] AND "complete genome"[Title] '
    "AND refseq[filter] AND 500000:10000000[SLEN]"
)

ENTREZ_BATCH_SIZE = 100
ENTREZ_DELAY = 0.4  # seconds between requests (NCBI asks for ≤3/sec without API key)


def search_ids(query: str, max_results: int) -> list[str]:
    """Search NCBI nucleotide database and return accession IDs."""
    print(f"  Searching: {query[:80]}...")
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
        description="Download and prepare reference data for ViroSense"
    )
    parser.add_argument(
        "--email", required=True, help="Email for NCBI Entrez (required by NCBI)"
    )
    parser.add_argument(
        "--output", default="data/reference", help="Output directory"
    )
    parser.add_argument(
        "--n-viral", type=int, default=300,
        help="Number of viral genomes to download (default: 300)",
    )
    parser.add_argument(
        "--n-cellular", type=int, default=100,
        help="Number of cellular genomes to download (default: 100)",
    )
    parser.add_argument(
        "--n-archaeal", type=int, default=20,
        help="Number of archaeal genomes to download (default: 20)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    Entrez.email = args.email

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Search and sample genome IDs ---
    print("\n=== Step 1: Searching NCBI for genome IDs ===")

    print("\nSearching viral genomes...")
    viral_ids = search_ids(VIRAL_QUERY, max_results=10000)
    if len(viral_ids) > args.n_viral:
        viral_ids = random.sample(viral_ids, args.n_viral)
    print(f"  Sampled {len(viral_ids)} viral genome IDs")

    print("\nSearching bacterial genomes...")
    bacterial_ids = search_ids(BACTERIAL_QUERY, max_results=5000)
    if len(bacterial_ids) > args.n_cellular:
        bacterial_ids = random.sample(bacterial_ids, args.n_cellular)
    print(f"  Sampled {len(bacterial_ids)} bacterial genome IDs")

    print("\nSearching archaeal genomes...")
    archaeal_ids = search_ids(ARCHAEAL_QUERY, max_results=1000)
    if len(archaeal_ids) > args.n_archaeal:
        archaeal_ids = random.sample(archaeal_ids, args.n_archaeal)
    print(f"  Sampled {len(archaeal_ids)} archaeal genome IDs")

    # --- 2. Download sequences ---
    print("\n=== Step 2: Downloading sequences from NCBI ===")

    print("\nDownloading viral genomes...")
    viral_seqs = download_sequences(viral_ids, "viral")

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

    # --- 3. Fragment genomes ---
    print("\n=== Step 3: Fragmenting genomes ===")

    viral_fragments = []
    for seq_id, seq, desc in viral_seqs:
        frags = fragment_genome(seq_id, seq, "vir")
        viral_fragments.extend(frags)
    print(f"  Generated {len(viral_fragments)} viral fragments")

    cellular_fragments = []
    for seq_id, seq, desc in cellular_seqs:
        frags = fragment_genome(seq_id, seq, "cel")
        cellular_fragments.extend(frags)
    print(f"  Generated {len(cellular_fragments)} cellular fragments")

    # --- 4. Balance classes ---
    min_count = min(len(viral_fragments), len(cellular_fragments))
    if len(viral_fragments) > min_count:
        viral_fragments = random.sample(viral_fragments, min_count)
    if len(cellular_fragments) > min_count:
        cellular_fragments = random.sample(cellular_fragments, min_count)

    print(f"\n  Balanced to {min_count} fragments per class ({min_count * 2} total)")

    # --- 5. Write output ---
    print("\n=== Step 4: Writing output files ===")

    fasta_path = output_dir / "sequences.fasta"
    labels_path = output_dir / "labels.tsv"

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

    # --- Summary ---
    fasta_size_mb = fasta_path.stat().st_size / (1024 * 1024)
    print(f"\n=== Summary ===")
    print(f"  Viral genomes downloaded: {len(viral_seqs)}")
    print(f"  Cellular genomes downloaded: {len(cellular_seqs)}")
    print(f"  Total fragments: {total}")
    print(f"  Viral fragments: {len(viral_fragments)}")
    print(f"  Cellular fragments: {len(cellular_fragments)}")
    print(f"  FASTA size: {fasta_size_mb:.1f} MB")
    print(f"\n  Next step:")
    print(f"  virosense build-reference \\")
    print(f"    -i {fasta_path} \\")
    print(f"    --labels {labels_path} \\")
    print(f"    -o data/reference/model/ \\")
    print(f"    --install")


if __name__ == "__main__":
    main()
