"""Result output utilities (TSV/JSON)."""

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from loguru import logger


def write_tsv(results: list, output_path: Path, filename: str = "results.tsv") -> Path:
    """Write a list of dataclass results to TSV.

    Args:
        results: List of dataclass instances.
        output_path: Output directory.
        filename: Output filename.

    Returns:
        Path to the written file.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / filename

    records = [asdict(r) for r in results]
    df = pd.DataFrame(records)
    df.to_csv(filepath, sep="\t", index=False)

    logger.info(f"Wrote {len(results)} results to {filepath}")
    return filepath


def write_json(data: dict, output_path: Path, filename: str = "results.json") -> Path:
    """Write a dict to JSON.

    Args:
        data: Dictionary to serialize.
        output_path: Output directory.
        filename: Output filename.

    Returns:
        Path to the written file.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / filename

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Wrote results to {filepath}")
    return filepath


def write_bed(
    regions: list, output_path: Path, filename: str = "regions.bed"
) -> Path:
    """Write genomic regions to BED format for genome browser visualization.

    BED columns: chrom, start, end, name, score (0-1000), strand.

    Args:
        regions: List of dataclass instances with chromosome_id, start,
            end, region_id, and mean_score attributes.
        output_path: Output directory.
        filename: Output filename.

    Returns:
        Path to the written file.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / filename

    with open(filepath, "w") as f:
        for r in regions:
            score = min(1000, int(r.mean_score * 1000))
            f.write(
                f"{r.chromosome_id}\t{r.start}\t{r.end}\t"
                f"{r.region_id}\t{score}\t.\n"
            )

    logger.info(f"Wrote {len(regions)} regions to {filepath}")
    return filepath
