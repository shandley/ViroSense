#!/usr/bin/env python3
"""Prepare 3-class reference data for ViroSense (chromosome/plasmid/viral).

Downloads RefSeq complete plasmid sequences from NCBI, fragments them, and
combines with existing viral + cellular reference data to produce a balanced
3-class training dataset.

Usage:
    python scripts/prepare_reference_data_3class.py \
        --email your@email.com \
        --existing-fasta data/reference/cleaned/sequences.fasta \
        --existing-labels data/reference/cleaned/labels.tsv \
        --output data/reference/3class/

    # Then train:
    virosense build-reference \
        -i data/reference/3class/sequences.fasta \
        --labels data/reference/3class/labels.tsv \
        -o data/reference/3class/model/ \
        --install

Requires: biopython (already a virosense dependency)
"""

import argparse
import json
import random
import time
from io import StringIO
from pathlib import Path

from Bio import Entrez, SeqIO

# Fragment sizes (bp) with number of fragments per genome
# Matches prepare_reference_data.py
FRAGMENT_SIZES = [
    (500, 3),    # 3 fragments of 500 bp
    (1000, 2),   # 2 fragments of 1000 bp
    (2000, 2),   # 2 fragments of 2000 bp
    (3000, 1),   # 1 fragment of 3000 bp
    (5000, 1),   # 1 fragment of 5000 bp
]

# RefSeq complete plasmid sequences from bacteria
PLASMID_QUERY = (
    'plasmid[Title] AND "complete sequence"[Title] '
    "AND refseq[filter] AND 2000:500000[SLEN] "
    "AND bacteria[Organism]"
)

ENTREZ_BATCH_SIZE = 100
ENTREZ_DELAY = 0.4


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


def fragment_genome(seq_id: str, sequence: str, prefix: str) -> list[tuple[str, str]]:
    """Fragment a genome into contig-like pieces at various lengths."""
    fragments = []
    seq_len = len(sequence)

    for frag_size, n_frags in FRAGMENT_SIZES:
        if seq_len < frag_size:
            continue

        for j in range(n_frags):
            max_start = seq_len - frag_size
            start = random.randint(0, max_start) if max_start > 0 else 0
            frag_seq = sequence[start : start + frag_size]

            # Skip fragments with too many Ns (>5%)
            if frag_seq.count("N") / len(frag_seq) > 0.05:
                continue

            frag_id = f"{prefix}_{seq_id}_f{frag_size}_{j}"
            fragments.append((frag_id, frag_seq))

    return fragments


def load_existing_data(
    fasta_path: str, labels_path: str
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Load existing 2-class reference data, returning viral and cellular fragments.

    Returns:
        (viral_fragments, cellular_fragments) where each is a list of (id, sequence).
    """
    import pandas as pd

    labels_df = pd.read_csv(labels_path, sep="\t")
    id_col, label_col = labels_df.columns[0], labels_df.columns[1]
    label_map = dict(zip(labels_df[id_col], labels_df[label_col]))

    viral, cellular = [], []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_id = record.id
        seq = str(record.seq).upper()
        label = label_map.get(seq_id)
        if label == 1:
            viral.append((seq_id, seq))
        elif label == 0:
            cellular.append((seq_id, seq))

    print(f"  Loaded {len(viral)} viral + {len(cellular)} cellular from existing data")
    return viral, cellular


def main():
    parser = argparse.ArgumentParser(
        description="Prepare 3-class reference data (chromosome/plasmid/viral)",
    )
    parser.add_argument(
        "--email", required=True, help="Email for NCBI Entrez (required by NCBI)"
    )
    parser.add_argument(
        "--existing-fasta", required=True,
        help="Existing reference sequences.fasta (viral + cellular)",
    )
    parser.add_argument(
        "--existing-labels", required=True,
        help="Existing reference labels.tsv (0=cellular, 1=viral)",
    )
    parser.add_argument(
        "--output", default="data/reference/3class", help="Output directory"
    )
    parser.add_argument(
        "--n-plasmid", type=int, default=350,
        help="Number of plasmid genomes to download (default: 350)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    Entrez.email = args.email

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load existing reference data ----
    print("\n=== Step 1: Loading existing reference data ===")
    viral_fragments, cellular_fragments = load_existing_data(
        args.existing_fasta, args.existing_labels
    )

    # ---- 2. Download plasmid sequences ----
    print("\n=== Step 2: Downloading RefSeq plasmid sequences ===")
    plasmid_ids = search_ids(PLASMID_QUERY, max_results=10000)
    if len(plasmid_ids) > args.n_plasmid:
        plasmid_ids = random.sample(plasmid_ids, args.n_plasmid)
    print(f"  Sampled {len(plasmid_ids)} plasmid IDs")

    plasmid_seqs = download_sequences(plasmid_ids, "plasmid")

    # ---- 3. Fragment plasmid genomes ----
    print("\n=== Step 3: Fragmenting plasmid genomes ===")
    plasmid_fragments = []
    for seq_id, seq, desc in plasmid_seqs:
        frags = fragment_genome(seq_id, seq, "plasmid")
        plasmid_fragments.extend(frags)

    print(f"  Generated {len(plasmid_fragments)} plasmid fragments from {len(plasmid_seqs)} genomes")

    if not plasmid_fragments:
        print("ERROR: No plasmid fragments generated. Check network/email.")
        return

    # ---- 4. Balance classes ----
    print("\n=== Step 4: Balancing classes ===")
    min_count = min(len(viral_fragments), len(cellular_fragments), len(plasmid_fragments))
    if len(viral_fragments) > min_count:
        viral_fragments = random.sample(viral_fragments, min_count)
    if len(cellular_fragments) > min_count:
        cellular_fragments = random.sample(cellular_fragments, min_count)
    if len(plasmid_fragments) > min_count:
        plasmid_fragments = random.sample(plasmid_fragments, min_count)

    total = len(viral_fragments) + len(cellular_fragments) + len(plasmid_fragments)
    print(f"  Balanced to {min_count} fragments per class ({total} total)")
    print(f"    Viral: {len(viral_fragments)}")
    print(f"    Chromosome: {len(cellular_fragments)}")
    print(f"    Plasmid: {len(plasmid_fragments)}")

    # ---- 5. Write output with string labels ----
    print("\n=== Step 5: Writing output files ===")

    fasta_path = output_dir / "sequences.fasta"
    labels_path = output_dir / "labels.tsv"
    manifest_path = output_dir / "manifest.json"

    with open(fasta_path, "w") as fasta_f, open(labels_path, "w") as labels_f:
        labels_f.write("sequence_id\tlabel\n")

        for frag_id, frag_seq in viral_fragments:
            fasta_f.write(f">{frag_id}\n{frag_seq}\n")
            labels_f.write(f"{frag_id}\tviral\n")

        for frag_id, frag_seq in cellular_fragments:
            fasta_f.write(f">{frag_id}\n{frag_seq}\n")
            labels_f.write(f"{frag_id}\tchromosome\n")

        for frag_id, frag_seq in plasmid_fragments:
            fasta_f.write(f">{frag_id}\n{frag_seq}\n")
            labels_f.write(f"{frag_id}\tplasmid\n")

    print(f"  Wrote {total} fragments to {fasta_path}")
    print(f"  Wrote labels to {labels_path}")

    # ---- 6. Write manifest ----
    manifest = {
        "description": "ViroSense 3-class reference classifier training data",
        "classes": {
            "viral": f"{len(viral_fragments)} fragments (DNA phages + archaeal viruses)",
            "chromosome": f"{len(cellular_fragments)} fragments (bacteria + archaea)",
            "plasmid": f"{len(plasmid_fragments)} fragments (bacterial plasmids, RefSeq)",
        },
        "label_encoding": {
            "chromosome": 0,
            "plasmid": 1,
            "viral": 2,
        },
        "plasmid_source": {
            "query": PLASMID_QUERY,
            "n_genomes_downloaded": len(plasmid_seqs),
            "n_fragments_raw": sum(
                len(fragment_genome(s[0], s[1], "plasmid")) for s in plasmid_seqs
            ),
        },
        "viral_cellular_source": "Existing 2-class reference data (prophage-filtered)",
        "seed": args.seed,
        "total_fragments": total,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote manifest to {manifest_path}")

    # ---- Summary ----
    fasta_size_mb = fasta_path.stat().st_size / (1024 * 1024)
    print(f"\n=== Summary ===")
    print(f"  3-class dataset: {total} fragments ({min_count} per class)")
    print(f"  FASTA size: {fasta_size_mb:.1f} MB")
    print(f"  Labels: chromosome/plasmid/viral (string labels)")
    print(f"\n  Next step:")
    print(f"  virosense build-reference \\")
    print(f"    -i {fasta_path} \\")
    print(f"    --labels {labels_path} \\")
    print(f"    -o {output_dir / 'model'} \\")
    print(f"    --install")


if __name__ == "__main__":
    main()
