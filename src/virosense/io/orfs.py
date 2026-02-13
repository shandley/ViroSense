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

    Auto-detects format by file extension or content sniffing:
    - .gff, .gff3 → GFF3/prodigal GFF3 format
    - .faa, .fasta, .fa → prodigal protein FASTA format

    Args:
        path: Path to ORF file.

    Returns:
        List of ORF objects.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ORF file not found: {path}")

    # Auto-detect format
    suffix = path.suffix.lower()
    if suffix in (".gff", ".gff3"):
        return _parse_gff3(path)

    if suffix in (".faa", ".fasta", ".fa"):
        # Sniff first line to distinguish protein FASTA from DNA FASTA
        with open(path) as f:
            first_line = f.readline().strip()
        if first_line.startswith(">") and "#" in first_line:
            return _parse_prodigal_fasta(path)
        return _parse_protein_fasta(path)

    # Sniff content for unknown extensions
    with open(path) as f:
        first_line = f.readline().strip()
    if first_line.startswith("##gff") or "\t" in first_line:
        return _parse_gff3(path)
    if first_line.startswith(">"):
        if "#" in first_line:
            return _parse_prodigal_fasta(path)
        return _parse_protein_fasta(path)

    raise ValueError(
        f"Cannot detect ORF file format for {path}. "
        "Supported: GFF3 (.gff/.gff3) or protein FASTA (.faa/.fasta)"
    )


def _parse_gff3(path: Path) -> list[ORF]:
    """Parse GFF3/prodigal GFF3 format.

    Extracts CDS features. ORF ID comes from the ID= attribute,
    or is generated from contig_id + coordinates if absent.
    """
    orfs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 9:
                continue

            feature_type = parts[2]
            if feature_type not in ("CDS", "gene"):
                continue

            contig_id = parts[0]
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            attributes = parts[8]

            # Parse ID from attributes
            orf_id = None
            for attr in attributes.split(";"):
                attr = attr.strip()
                if attr.startswith("ID="):
                    orf_id = attr[3:]
                    break

            if orf_id is None:
                orf_id = f"{contig_id}_{start}_{end}"

            orfs.append(
                ORF(
                    orf_id=orf_id,
                    contig_id=contig_id,
                    start=start,
                    end=end,
                    strand=strand,
                )
            )

    logger.info(f"Parsed {len(orfs)} ORFs from GFF3: {path}")
    return orfs


def _parse_prodigal_fasta(path: Path) -> list[ORF]:
    """Parse prodigal protein FASTA output.

    Prodigal headers look like:
    >contig_1_1 # 3 # 1205 # 1 # ID=1_1;partial=00;...
    """
    orfs = []
    current_seq = []
    current_orf = None

    def _flush():
        if current_orf is not None:
            current_orf.protein_sequence = "".join(current_seq)
            orfs.append(current_orf)

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                _flush()
                current_seq = []
                current_orf = _parse_prodigal_header(line[1:])
            else:
                current_seq.append(line)

    _flush()
    logger.info(f"Parsed {len(orfs)} ORFs from prodigal FASTA: {path}")
    return orfs


def _parse_prodigal_header(header: str) -> ORF:
    """Parse a prodigal FASTA header line.

    Format: contig_1_1 # 3 # 1205 # 1 # ID=1_1;partial=00;...
    Fields separated by ' # ':
      [0] orf_id
      [1] start
      [2] end
      [3] strand (1 = +, -1 = -)
      [4] attributes (ID=...; etc)
    """
    parts = header.split(" # ")
    orf_id = parts[0].strip()

    start = int(parts[1]) if len(parts) > 1 else 0
    end = int(parts[2]) if len(parts) > 2 else 0
    strand_val = parts[3].strip() if len(parts) > 3 else "1"
    strand = "+" if strand_val == "1" else "-"

    # Extract contig_id: everything up to the last _N suffix
    # e.g., "contig_1_1" -> "contig_1"
    contig_id = "_".join(orf_id.rsplit("_", 1)[:-1]) if "_" in orf_id else orf_id

    return ORF(
        orf_id=orf_id,
        contig_id=contig_id,
        start=start,
        end=end,
        strand=strand,
    )


def _parse_protein_fasta(path: Path) -> list[ORF]:
    """Parse a simple protein FASTA (non-prodigal).

    Headers are just >protein_id with no coordinate info.
    Coordinates default to 0,0 since they're unknown.
    """
    orfs = []
    current_id = None
    current_seq = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    orfs.append(
                        ORF(
                            orf_id=current_id,
                            contig_id=current_id,
                            start=0,
                            end=0,
                            strand="+",
                            protein_sequence="".join(current_seq),
                        )
                    )
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        orfs.append(
            ORF(
                orf_id=current_id,
                contig_id=current_id,
                start=0,
                end=0,
                strand="+",
                protein_sequence="".join(current_seq),
            )
        )

    logger.info(f"Parsed {len(orfs)} ORFs from protein FASTA: {path}")
    return orfs
