"""DNA FASTA I/O utilities."""

from pathlib import Path

from loguru import logger


def read_fasta(path: str | Path) -> dict[str, str]:
    """Read a FASTA file into a dict of id -> sequence.

    Args:
        path: Path to FASTA file.

    Returns:
        Dict mapping sequence ID to DNA sequence string.
    """
    from Bio import SeqIO

    sequences = {}
    for record in SeqIO.parse(str(path), "fasta"):
        sequences[record.id] = str(record.seq).upper()

    logger.info(f"Read {len(sequences)} sequences from {path}")
    return sequences


def filter_by_length(
    sequences: dict[str, str], min_length: int = 500
) -> dict[str, str]:
    """Filter sequences by minimum length.

    Args:
        sequences: Dict of id -> sequence.
        min_length: Minimum sequence length in bp.

    Returns:
        Filtered dict with only sequences >= min_length.
    """
    filtered = {k: v for k, v in sequences.items() if len(v) >= min_length}
    n_removed = len(sequences) - len(filtered)
    if n_removed:
        logger.info(f"Filtered {n_removed} sequences shorter than {min_length} bp")
    return filtered


def write_fasta(
    sequences: dict[str, str], path: str | Path, wrap: int = 80
) -> Path:
    """Write sequences to a FASTA file.

    Args:
        sequences: Dict mapping sequence ID to DNA sequence string.
        path: Output file path.
        wrap: Line width for sequence wrapping (0 = no wrapping).

    Returns:
        Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for seq_id, seq in sequences.items():
            f.write(f">{seq_id}\n")
            if wrap > 0:
                for i in range(0, len(seq), wrap):
                    f.write(seq[i : i + wrap] + "\n")
            else:
                f.write(seq + "\n")

    logger.info(f"Wrote {len(sequences)} sequences to {path}")
    return path
