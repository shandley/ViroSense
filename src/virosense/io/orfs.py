"""ORF and GFF3 parsing utilities."""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class ORF:
    """Representation of an open reading frame."""

    orf_id: str
    contig_id: str
    start: int
    end: int
    strand: str  # "+" or "-"
    protein_sequence: str | None = None


def parse_orfs(path: str | Path) -> list[ORF]:
    """Parse ORF predictions from GFF3, prodigal output, or protein FASTA.

    Args:
        path: Path to ORF file (GFF3, prodigal, or FASTA).

    Returns:
        List of ORF objects.
    """
    raise NotImplementedError(
        "ORF parsing not yet implemented. See Phase 7 in the plan."
    )
