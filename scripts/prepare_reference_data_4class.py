#!/usr/bin/env python3
"""Prepare 4-class reference data for ViroSense (chromosome/phage/plasmid/rna_virus).

Extends the 3-class dataset by splitting the "viral" class (DNA phages) into
"phage" and adding RNA virus cDNA sequences from the curated RNA virus database.

The 4 classes:
  - chromosome: bacterial/archaeal chromosomes (from existing reference data)
  - phage: DNA bacteriophages + archaeal viruses (from existing reference data)
  - plasmid: bacterial plasmids (from existing 3-class data)
  - rna_virus: RNA virus cDNA sequences (from RNA_virus_database.fasta)

The RNA virus database (385K sequences) contains RefSeq NC_ accessions and RVMT
sequences. We use only NC_ RefSeq sequences for training quality, randomly
sampling and fragmenting to match the other class sizes.

Usage:
    python scripts/prepare_reference_data_4class.py \
        --existing-fasta data/reference/3class/sequences.fasta \
        --existing-labels data/reference/3class/labels.tsv \
        --rna-virus-db data/reference/rna_viruses/RNA_virus_database.fasta \
        --output data/reference/4class/

    # Then train:
    virosense build-reference \
        -i data/reference/4class/sequences.fasta \
        --labels data/reference/4class/labels.tsv \
        -o data/reference/4class/model/ \
        --install
"""

import argparse
import json
import random
from pathlib import Path

from Bio import SeqIO

# Fragment sizes (bp) with number of fragments per genome
# Matches prepare_reference_data.py
FRAGMENT_SIZES = [
    (500, 3),    # 3 fragments of 500 bp
    (1000, 2),   # 2 fragments of 1000 bp
    (2000, 2),   # 2 fragments of 2000 bp
    (3000, 1),   # 1 fragment of 3000 bp
    (5000, 1),   # 1 fragment of 5000 bp
]


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


def load_3class_data(
    fasta_path: str, labels_path: str
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Load existing 3-class reference data.

    Returns:
        (phage_fragments, chromosome_fragments, plasmid_fragments)
        where each is a list of (id, sequence).
    """
    import pandas as pd

    labels_df = pd.read_csv(labels_path, sep="\t")
    id_col, label_col = labels_df.columns[0], labels_df.columns[1]
    label_map = dict(zip(labels_df[id_col], labels_df[label_col]))

    phage, chromosome, plasmid = [], [], []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_id = record.id
        seq = str(record.seq).upper()
        label = label_map.get(seq_id)
        if label == "viral":
            phage.append((seq_id, seq))
        elif label == "chromosome":
            chromosome.append((seq_id, seq))
        elif label == "plasmid":
            plasmid.append((seq_id, seq))

    print(f"  Loaded from 3-class data:")
    print(f"    Phage (was 'viral'): {len(phage)}")
    print(f"    Chromosome:          {len(chromosome)}")
    print(f"    Plasmid:             {len(plasmid)}")
    return phage, chromosome, plasmid


def load_rna_virus_genomes(
    db_path: str, n_genomes: int, min_length: int = 500
) -> list[tuple[str, str]]:
    """Load and sample RefSeq RNA virus genomes from the database.

    Filters for NC_ accessions (RefSeq) and sequences >= min_length.
    """
    candidates = []
    for record in SeqIO.parse(db_path, "fasta"):
        if not record.id.startswith("NC_"):
            continue
        seq = str(record.seq).upper()
        # Convert U -> T for cDNA representation
        seq = seq.replace("U", "T")
        if len(seq) >= min_length and set(seq).issubset({"A", "C", "G", "T", "N"}):
            candidates.append((record.id, seq))

    print(f"  Found {len(candidates)} NC_ RefSeq RNA virus genomes >= {min_length}bp")

    if len(candidates) > n_genomes:
        candidates = random.sample(candidates, n_genomes)
        print(f"  Sampled {n_genomes} genomes")

    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Prepare 4-class reference data (chromosome/phage/plasmid/rna_virus)",
    )
    parser.add_argument(
        "--existing-fasta", required=True,
        help="Existing 3-class sequences.fasta (viral + chromosome + plasmid)",
    )
    parser.add_argument(
        "--existing-labels", required=True,
        help="Existing 3-class labels.tsv (string labels: viral/chromosome/plasmid)",
    )
    parser.add_argument(
        "--rna-virus-db", required=True,
        help="RNA virus database FASTA (data/reference/rna_viruses/RNA_virus_database.fasta)",
    )
    parser.add_argument(
        "--output", default="data/reference/4class", help="Output directory"
    )
    parser.add_argument(
        "--n-rna-virus-genomes", type=int, default=500,
        help="Number of RNA virus genomes to sample (default: 500)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load existing 3-class data ----
    print("\n=== Step 1: Loading existing 3-class reference data ===")
    phage_fragments, chromosome_fragments, plasmid_fragments = load_3class_data(
        args.existing_fasta, args.existing_labels
    )

    # ---- 2. Load and fragment RNA virus genomes ----
    print("\n=== Step 2: Loading RNA virus genomes ===")
    rna_virus_genomes = load_rna_virus_genomes(
        args.rna_virus_db, args.n_rna_virus_genomes
    )

    print("\n=== Step 3: Fragmenting RNA virus genomes ===")
    rna_virus_fragments = []
    for seq_id, seq in rna_virus_genomes:
        frags = fragment_genome(seq_id, seq, "rnavirus")
        rna_virus_fragments.extend(frags)

    print(f"  Generated {len(rna_virus_fragments)} RNA virus fragments from {len(rna_virus_genomes)} genomes")

    if not rna_virus_fragments:
        print("ERROR: No RNA virus fragments generated.")
        return

    # ---- 3. Balance classes ----
    print("\n=== Step 4: Balancing classes ===")
    min_count = min(
        len(phage_fragments),
        len(chromosome_fragments),
        len(plasmid_fragments),
        len(rna_virus_fragments),
    )
    print(f"  Class sizes before balancing:")
    print(f"    Phage:      {len(phage_fragments)}")
    print(f"    Chromosome: {len(chromosome_fragments)}")
    print(f"    Plasmid:    {len(plasmid_fragments)}")
    print(f"    RNA virus:  {len(rna_virus_fragments)}")
    print(f"  Balancing to {min_count} per class")

    if len(phage_fragments) > min_count:
        phage_fragments = random.sample(phage_fragments, min_count)
    if len(chromosome_fragments) > min_count:
        chromosome_fragments = random.sample(chromosome_fragments, min_count)
    if len(plasmid_fragments) > min_count:
        plasmid_fragments = random.sample(plasmid_fragments, min_count)
    if len(rna_virus_fragments) > min_count:
        rna_virus_fragments = random.sample(rna_virus_fragments, min_count)

    total = min_count * 4
    print(f"  Total: {total} fragments ({min_count} per class)")

    # ---- 4. Write output with string labels ----
    print("\n=== Step 5: Writing output files ===")

    fasta_path = output_dir / "sequences.fasta"
    labels_path = output_dir / "labels.tsv"
    manifest_path = output_dir / "manifest.json"

    # Write in sorted label order: chromosome, phage, plasmid, rna_virus
    # This ensures consistent label encoding across runs
    classes = [
        ("chromosome", chromosome_fragments),
        ("phage", phage_fragments),
        ("plasmid", plasmid_fragments),
        ("rna_virus", rna_virus_fragments),
    ]

    with open(fasta_path, "w") as fasta_f, open(labels_path, "w") as labels_f:
        labels_f.write("sequence_id\tlabel\n")
        for label, fragments in classes:
            for frag_id, frag_seq in fragments:
                fasta_f.write(f">{frag_id}\n{frag_seq}\n")
                labels_f.write(f"{frag_id}\t{label}\n")

    print(f"  Wrote {total} fragments to {fasta_path}")
    print(f"  Wrote labels to {labels_path}")

    # ---- 5. Write manifest ----
    manifest = {
        "description": "ViroSense 4-class reference classifier training data",
        "classes": {
            "chromosome": f"{len(chromosome_fragments)} fragments (bacteria + archaea)",
            "phage": f"{len(phage_fragments)} fragments (DNA phages + archaeal viruses)",
            "plasmid": f"{len(plasmid_fragments)} fragments (bacterial plasmids, RefSeq)",
            "rna_virus": f"{len(rna_virus_fragments)} fragments (RNA virus cDNA, RefSeq NC_)",
        },
        "label_encoding": {
            "chromosome": 0,
            "phage": 1,
            "plasmid": 2,
            "rna_virus": 3,
        },
        "rna_virus_source": {
            "database": "RNA_virus_database.fasta (385K sequences)",
            "filter": "NC_ RefSeq accessions only",
            "n_genomes_sampled": len(rna_virus_genomes),
            "n_fragments_raw": sum(
                len(fragment_genome(s[0], s[1], "rnavirus")) for s in rna_virus_genomes
            ),
        },
        "existing_3class_source": "data/reference/3class/ (viral renamed to phage)",
        "seed": args.seed,
        "total_fragments": total,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote manifest to {manifest_path}")

    # ---- Summary ----
    fasta_size_mb = fasta_path.stat().st_size / (1024 * 1024)
    print(f"\n=== Summary ===")
    print(f"  4-class dataset: {total} fragments ({min_count} per class)")
    print(f"  FASTA size: {fasta_size_mb:.1f} MB")
    print(f"  Labels: chromosome/phage/plasmid/rna_virus (string labels)")
    print(f"  Label encoding (sorted): chromosome=0, phage=1, plasmid=2, rna_virus=3")
    print(f"\n  Next step:")
    print(f"  virosense build-reference \\")
    print(f"    -i {fasta_path} \\")
    print(f"    --labels {labels_path} \\")
    print(f"    -o {output_dir / 'model'} \\")
    print(f"    --install")


if __name__ == "__main__":
    main()
