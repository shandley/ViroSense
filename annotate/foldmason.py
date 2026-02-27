"""FoldMason wrapper for multiple structural alignment.

Produces amino acid MSA, 3Di MSA, and Newick guide trees from a set of
protein structures. With PDB input (from ColabFold/AlphaFold DB), FoldMason
uses full 3D coordinates for alignment and LDDT scoring.

Reference: Gilchrist et al., Science (2026).
Designed for integration into ViroSense's annotate module.
"""

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FoldMasonResult:
    """Result from a FoldMason multiple structural alignment."""

    aa_msa: Path
    three_di_msa: Path
    guide_tree: Path
    num_sequences: int
    alignment_length: int


def check_foldmason() -> tuple[bool, str]:
    """Check if foldmason is available in PATH.

    Returns:
        (available, version_or_error)
    """
    try:
        result = subprocess.run(
            ["foldmason", "version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            version = result.stdout.strip() or "unknown"
            return True, version
        return False, f"foldmason returned {result.returncode}"
    except FileNotFoundError:
        return False, "foldmason not found in PATH"
    except subprocess.TimeoutExpired:
        return False, "foldmason timed out"


def run_foldmason_msa(
    query_db: Path,
    output_prefix: Path,
    threads: int = 4,
    refine_iters: int = 0,
) -> FoldMasonResult:
    """Run FoldMason structuremsa on a Foldseek-format database.

    With PDB-derived databases (full C-alpha coordinates), FoldMason can
    perform refinement and produce LDDT scores. This is a key advantage
    over the ProstT5-based 3Di-only pipeline.

    Args:
        query_db: Path to Foldseek-format database (from foldseek createdb).
        output_prefix: Output prefix for result files.
        threads: Number of CPU threads.
        refine_iters: Number of refinement iterations (requires coordinates).
            Set to 0 to skip refinement (faster).

    Returns:
        FoldMasonResult with paths to output files.
    """
    available, version = check_foldmason()
    if not available:
        raise FileNotFoundError(
            f"FoldMason not available: {version}. "
            "Install with: conda install -c bioconda foldmason"
        )
    logger.info(f"Using FoldMason {version}")

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "foldmason", "structuremsa",
        str(query_db),
        str(output_prefix),
        "--threads", str(threads),
    ]

    if refine_iters > 0:
        cmd.extend(["--refine-iters", str(refine_iters)])

    logger.info(f"Running FoldMason structuremsa with {threads} threads")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"foldmason structuremsa failed (exit {result.returncode}): "
            f"{result.stderr}"
        )

    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            if line.strip():
                logger.debug(f"foldmason: {line.strip()}")

    # Parse output files
    aa_msa_path = Path(str(output_prefix) + "_aa.fa")
    three_di_msa_path = Path(str(output_prefix) + "_3di.fa")
    tree_path = Path(str(output_prefix) + ".nw")

    if not aa_msa_path.exists():
        raise RuntimeError(
            f"FoldMason did not produce expected output: {aa_msa_path}"
        )

    # Count sequences and alignment length from AA MSA
    num_sequences = 0
    alignment_length = 0
    with open(aa_msa_path) as f:
        for line in f:
            if line.startswith(">"):
                num_sequences += 1
            elif num_sequences == 1:
                alignment_length += len(line.strip())

    logger.info(
        f"Alignment complete: {num_sequences} sequences, "
        f"{alignment_length} columns"
    )

    return FoldMasonResult(
        aa_msa=aa_msa_path,
        three_di_msa=three_di_msa_path,
        guide_tree=tree_path,
        num_sequences=num_sequences,
        alignment_length=alignment_length,
    )


def write_alignment_summary(
    result: FoldMasonResult,
    output_dir: Path,
) -> Path:
    """Write alignment summary as JSON.

    Args:
        result: FoldMasonResult from alignment.
        output_dir: Output directory.

    Returns:
        Path to summary JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "num_sequences": result.num_sequences,
        "alignment_length": result.alignment_length,
        "aa_msa": str(result.aa_msa),
        "three_di_msa": str(result.three_di_msa),
        "guide_tree": str(result.guide_tree),
    }

    summary_path = output_dir / "alignment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary_path
