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
