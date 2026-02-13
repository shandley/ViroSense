#!/usr/bin/env python3
"""Download and prepare reference data for ViroSense viral classifier training.

Downloads DNA phage and cellular genomes from NCBI RefSeq via Entrez,
fragments them into contig-like pieces at various lengths, and outputs
a training FASTA + labels TSV for use with `virosense build-reference`.

TAXONOMIC SCOPE:
  Viral (label=1): DNA bacteriophages and archaeal viruses only.
    - Caudoviricetes (tailed dsDNA phage) — dominant in metagenomes
    - Steigviridae (crAss-like phage) — most abundant phage in human gut
    - Microviridae (ssDNA phage) — common in gut/ocean metagenomes
    - Inoviridae (filamentous ssDNA phage) — prophage-associated
    - Archaeal DNA viruses (Lipothrixviridae, Rudiviridae, etc.)

  Cellular (label=0): Bacteria and archaea.
    - Diverse bacterial phyla from RefSeq
    - Archaeal genomes
    - Optional: prophage masking via geNomad (--mask-prophages)

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

    # v2: with prophage masking (requires geNomad)
    python scripts/prepare_reference_data.py \\
        --email your@email.com \\
        --output data/reference_v2/ \\
        --mask-prophages \\
        --genomad-db /path/to/genomad_db

Requires: biopython (already a virosense dependency)
Optional: geNomad (for --mask-prophages)
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
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

# crAss-like phages (Steigviridae) — the most abundant phage in the human
# gut microbiome.  Originally discovered by Dutilh et al. 2014.  ICTV now
# classifies them under the family Steigviridae within Caudoviricetes.
# Queried separately to ensure representation since they may be
# underrepresented in RefSeq "complete genome" entries.
PHAGE_CRASS_QUERY = (
    '(Steigviridae[Organism] OR "crAss-like"[All Fields] OR '
    '"crAssphage"[All Fields]) '
    'AND ("complete genome"[Title] OR "complete sequence"[Title]) '
    "AND refseq[filter] AND 80000:120000[SLEN]"
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
    seq_id: str,
    sequence: str,
    prefix: str,
    masked_intervals: list[tuple[int, int]] | None = None,
    mask_overlap_threshold: float = 0.5,
) -> list[tuple[str, str]]:
    """Fragment a genome into contig-like pieces at various lengths.

    Args:
        seq_id: Genome accession/identifier.
        sequence: Full genome DNA sequence.
        prefix: Prefix for fragment IDs (e.g., 'bact', 'phage_dsdna').
        masked_intervals: List of (start, end) intervals to avoid (e.g.,
            prophage regions from geNomad). Fragments overlapping these
            by more than mask_overlap_threshold are rejected.
        mask_overlap_threshold: Fraction of fragment that must overlap a
            masked region to be rejected (default: 0.5 = 50%).

    Returns:
        List of (fragment_id, fragment_sequence).
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

            end = start + frag_size
            frag_seq = sequence[start:end]

            # Skip fragments with too many Ns (>5%)
            n_count = frag_seq.count("N")
            if n_count / len(frag_seq) > 0.05:
                continue

            # Skip fragments overlapping masked regions (e.g., prophages)
            if masked_intervals and _overlaps_masked(
                start, end, masked_intervals, mask_overlap_threshold
            ):
                continue

            frag_id = f"{prefix}_{seq_id}_f{frag_size}_{j}"
            fragments.append((frag_id, frag_seq))

    return fragments


def _overlaps_masked(
    frag_start: int,
    frag_end: int,
    intervals: list[tuple[int, int]],
    threshold: float,
) -> bool:
    """Check if a fragment overlaps masked intervals above a threshold."""
    frag_len = frag_end - frag_start
    total_overlap = 0
    for m_start, m_end in intervals:
        overlap_start = max(frag_start, m_start)
        overlap_end = min(frag_end, m_end)
        if overlap_start < overlap_end:
            total_overlap += overlap_end - overlap_start
    return (total_overlap / frag_len) >= threshold


# ---------------------------------------------------------------------------
# geNomad prophage masking
# ---------------------------------------------------------------------------


def run_genomad_prophage_detection(
    genomes: list[tuple[str, str, str]],
    genomad_db: str,
    threads: int = 4,
) -> dict[str, list[tuple[int, int]]]:
    """Run geNomad on cellular genomes to identify prophage regions.

    Args:
        genomes: List of (seq_id, sequence, description) tuples.
        genomad_db: Path to geNomad database directory.
        threads: Number of threads for geNomad.

    Returns:
        Dict of seq_id -> list of (start, end) prophage intervals.
    """
    # Check geNomad is installed
    if not shutil.which("genomad"):
        print("ERROR: geNomad not found. Install with: pip install genomad")
        print("  Then download the database: genomad download-database /path/to/db")
        sys.exit(1)

    if not Path(genomad_db).exists():
        print(f"ERROR: geNomad database not found: {genomad_db}")
        print("  Download with: genomad download-database /path/to/db")
        sys.exit(1)

    prophage_regions = {}

    with tempfile.TemporaryDirectory(prefix="virosense_genomad_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Write genomes to a single FASTA for geNomad
        input_fasta = tmpdir / "cellular_genomes.fasta"
        with open(input_fasta, "w") as f:
            for seq_id, seq, _ in genomes:
                f.write(f">{seq_id}\n{seq}\n")

        output_dir = tmpdir / "genomad_output"

        print(f"  Running geNomad on {len(genomes)} genomes...")
        cmd = [
            "genomad", "end-to-end",
            str(input_fasta),
            str(output_dir),
            genomad_db,
            "--threads", str(threads),
            "--enable-score-calibration",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=7200,
            )
            if result.returncode != 0:
                print(f"  Warning: geNomad failed: {result.stderr[:500]}")
                return prophage_regions
        except subprocess.TimeoutExpired:
            print("  Warning: geNomad timed out (2h limit)")
            return prophage_regions

        # Parse geNomad provirus output
        provirus_tsv = output_dir / "cellular_genomes_find_proviruses" / "cellular_genomes_provirus.tsv"
        if not provirus_tsv.exists():
            # Try alternative path structure
            for tsv in output_dir.rglob("*provirus.tsv"):
                provirus_tsv = tsv
                break

        if provirus_tsv.exists():
            prophage_regions = _parse_genomad_proviruses(provirus_tsv)
            total_regions = sum(len(v) for v in prophage_regions.values())
            n_genomes_with = sum(1 for v in prophage_regions.values() if v)
            print(f"  geNomad found {total_regions} provirus regions in {n_genomes_with} genomes")
        else:
            print("  Warning: geNomad provirus output not found")

    return prophage_regions


def _parse_genomad_proviruses(tsv_path: Path) -> dict[str, list[tuple[int, int]]]:
    """Parse geNomad provirus TSV output.

    geNomad provirus TSV has columns including:
    source_seq, start, end, ...

    The source_seq is the contig/genome ID and start/end are the
    provirus coordinates within that contig.
    """
    import pandas as pd

    regions = {}
    try:
        df = pd.read_csv(tsv_path, sep="\t")

        # geNomad output column names vary by version
        # Common: 'source_seq' or 'seq_name', 'start'/'coordinates'
        seq_col = None
        for candidate in ["source_seq", "seq_name", "contig"]:
            if candidate in df.columns:
                seq_col = candidate
                break

        if seq_col is None or "start" not in df.columns or "end" not in df.columns:
            # Try parsing from 'coordinates' column (format: "start-end")
            if "coordinates" in df.columns and seq_col:
                for _, row in df.iterrows():
                    seq_id = str(row[seq_col])
                    coords = str(row["coordinates"]).split("-")
                    if len(coords) == 2:
                        start, end = int(coords[0]), int(coords[1])
                        regions.setdefault(seq_id, []).append((start, end))
                return regions

            print(f"  Warning: Unexpected geNomad column format: {list(df.columns)}")
            return regions

        for _, row in df.iterrows():
            seq_id = str(row[seq_col])
            start = int(row["start"])
            end = int(row["end"])
            regions.setdefault(seq_id, []).append((start, end))

    except Exception as e:
        print(f"  Warning: Failed to parse geNomad output: {e}")

    return regions


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare reference data for ViroSense",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Taxonomic scope:
  Viral:    DNA phages (Caudoviricetes, Steigviridae/crAss-like,
            Microviridae, Inoviridae) and archaeal DNA viruses.
            NO RNA viruses or eukaryotic viruses.
  Cellular: Bacteria and archaea from NCBI RefSeq.
            Use --mask-prophages to remove prophage-contaminated fragments.
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
        "--n-crass", type=int, default=50,
        help="Number of crAss-like phage genomes (Steigviridae) (default: 50)",
    )
    parser.add_argument(
        "--mask-prophages", action="store_true", default=False,
        help="Run geNomad to mask prophage regions in cellular genomes",
    )
    parser.add_argument(
        "--genomad-db", default=None,
        help="Path to geNomad database (required with --mask-prophages)",
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="Threads for geNomad (default: 4)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    if args.mask_prophages and not args.genomad_db:
        parser.error("--genomad-db is required when using --mask-prophages")

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

    # -- Viral: crAss-like phage (Steigviridae) --
    print("\n[VIRAL] Searching crAss-like phage (Steigviridae)...")
    crass_ids = search_ids(PHAGE_CRASS_QUERY, max_results=1000)
    if len(crass_ids) > args.n_crass:
        crass_ids = random.sample(crass_ids, args.n_crass)
    print(f"  Sampled {len(crass_ids)} crAss-like phage IDs")

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

    total_viral_ids = len(phage_dsdna_ids) + len(crass_ids) + len(micro_ids) + len(ino_ids) + len(archaeal_virus_ids)
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

    print("\nDownloading crAss-like phage...")
    crass_seqs = download_sequences(crass_ids, "crAss-like phage")

    print("\nDownloading archaeal viruses...")
    archaeal_virus_seqs = download_sequences(archaeal_virus_ids, "archaeal virus")

    viral_seqs = phage_dsdna_seqs + crass_seqs + micro_seqs + ino_seqs + archaeal_virus_seqs

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
        "dsDNA_phage": 0, "crAss_like": 0, "Microviridae": 0,
        "Inoviridae": 0, "archaeal_virus": 0,
    }
    for seq_id, seq, desc in phage_dsdna_seqs:
        frags = fragment_genome(seq_id, seq, "phage_dsdna")
        viral_fragments.extend(frags)
        viral_type_counts["dsDNA_phage"] += len(frags)

    for seq_id, seq, desc in crass_seqs:
        frags = fragment_genome(seq_id, seq, "phage_crass")
        viral_fragments.extend(frags)
        viral_type_counts["crAss_like"] += len(frags)

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

    # Optionally detect prophage regions with geNomad
    prophage_masks = {}
    if args.mask_prophages:
        print("\n=== Step 3b: Detecting prophage regions with geNomad ===")
        prophage_masks = run_genomad_prophage_detection(
            cellular_seqs, args.genomad_db, threads=args.threads,
        )
        if prophage_masks:
            total_masked = sum(len(v) for v in prophage_masks.values())
            print(f"  Will mask {total_masked} prophage regions during fragmentation")
        else:
            print("  No prophage regions detected (or geNomad failed)")

    cellular_fragments = []
    cellular_type_counts = {"bacterial": 0, "archaeal": 0}
    n_masked_fragments = 0
    for seq_id, seq, desc in bacterial_seqs:
        masks = prophage_masks.get(seq_id, None)
        frags_before = len(fragment_genome(seq_id, seq, "bact")) if masks else 0
        frags = fragment_genome(seq_id, seq, "bact", masked_intervals=masks)
        if masks:
            n_masked_fragments += frags_before - len(frags)
        cellular_fragments.extend(frags)
        cellular_type_counts["bacterial"] += len(frags)

    for seq_id, seq, desc in archaeal_seqs:
        masks = prophage_masks.get(seq_id, None)
        frags_before = len(fragment_genome(seq_id, seq, "arch")) if masks else 0
        frags = fragment_genome(seq_id, seq, "arch", masked_intervals=masks)
        if masks:
            n_masked_fragments += frags_before - len(frags)
        cellular_fragments.extend(frags)
        cellular_type_counts["archaeal"] += len(frags)

    if args.mask_prophages:
        print(f"  Prophage masking rejected {n_masked_fragments} fragments")

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
                "Steigviridae (crAss-like phages)",
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
        "prophage_masking": {
            "enabled": args.mask_prophages,
            "tool": "geNomad" if args.mask_prophages else None,
            "fragments_rejected": n_masked_fragments if args.mask_prophages else 0,
            "genomes_with_prophages": (
                sum(1 for v in prophage_masks.values() if v)
                if args.mask_prophages else 0
            ),
        },
        "source": "NCBI RefSeq via Entrez",
        "seed": args.seed,
        "genomes_downloaded": {
            "dsDNA_phage": len(phage_dsdna_seqs),
            "crAss_like": len(crass_seqs),
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
    print(f"    crAss-like (Steigviridae):    {len(crass_seqs)}")
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
