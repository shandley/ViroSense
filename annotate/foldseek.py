"""Foldseek structural search for viral protein annotation.

Searches query protein structures (PDB files from ColabFold/AlphaFold DB)
against reference databases (BFVD, Viro3D) using Foldseek.

Key difference from vHold: uses real PDB structures as input instead of
ProstT5-predicted 3Di sequences. This enables:
- Native Foldseek createdb (no manual binary DB construction)
- Structural quality metrics (prob, LDDT, TM-score) in output
- FoldMason refinement with full 3D coordinates

Designed for integration into ViroSense's annotate module.
"""

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Foldseek output columns — with PDB input we get structural metrics
# that aren't available with 3Di-only input
FOLDSEEK_OUTPUT_COLUMNS = [
    "query",
    "target",
    "fident",
    "alnlen",
    "mismatch",
    "gapopen",
    "qstart",
    "qend",
    "tstart",
    "tend",
    "evalue",
    "bits",
    "prob",
    "lddt",
    "alntmscore",
    "qlen",
    "tlen",
    "qcov",
    "tcov",
]

FOLDSEEK_OUTPUT_FORMAT = (
    "query,target,fident,alnlen,mismatch,gapopen,"
    "qstart,qend,tstart,tend,evalue,bits,"
    "prob,lddt,alntmscore,"
    "qlen,tlen,qcov,tcov"
)

# Default search parameters
DEFAULT_EVALUE = 1e-3
DEFAULT_SENSITIVITY = 9.5
DEFAULT_MAX_SEQS = 1000


@dataclass
class FoldseekResult:
    """Result from a Foldseek structural search."""

    hits: pd.DataFrame
    query_db: Path | None = None
    n_queries: int = 0
    n_hits: int = 0
    database: str = ""

    @property
    def has_hits(self) -> bool:
        return self.n_hits > 0

    def summary(self) -> str:
        return (
            f"Foldseek {self.database}: "
            f"{self.n_hits} hits for {self.n_queries} queries"
        )


@dataclass
class SearchBatch:
    """Combined results from searching multiple databases."""

    results: dict[str, FoldseekResult] = field(default_factory=dict)

    @property
    def all_hits(self) -> pd.DataFrame:
        """Concatenate all hits with source_db column."""
        dfs = []
        for db_name, result in self.results.items():
            if result.has_hits:
                df = result.hits.copy()
                df["source_db"] = db_name
                dfs.append(df)
        if not dfs:
            return pd.DataFrame(columns=FOLDSEEK_OUTPUT_COLUMNS + ["source_db"])
        return pd.concat(dfs, ignore_index=True)

    @property
    def best_hits(self) -> pd.DataFrame:
        """Best hit per query across all databases (by e-value)."""
        merged = self.all_hits
        if merged.empty:
            return merged
        merged = merged.sort_values(["query", "evalue"])
        return merged.drop_duplicates(subset=["query"], keep="first")

    def summary(self) -> str:
        parts = [r.summary() for r in self.results.values()]
        return " | ".join(parts)


# ============================================================================
# Foldseek Availability
# ============================================================================


def check_foldseek() -> tuple[bool, str]:
    """Check if foldseek is available in PATH.

    Returns:
        (available, version_or_error)
    """
    try:
        result = subprocess.run(
            ["foldseek", "version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            version = result.stdout.strip() or "unknown"
            return True, version
        return False, f"foldseek returned {result.returncode}"
    except FileNotFoundError:
        return False, "foldseek not found in PATH"
    except subprocess.TimeoutExpired:
        return False, "foldseek timed out"


# ============================================================================
# Query Database Creation
# ============================================================================


def create_query_db_from_pdbs(
    pdb_dir: Path,
    output_db: Path,
    threads: int = 4,
) -> Path:
    """Create a Foldseek database from a directory of PDB files.

    Uses `foldseek createdb` which natively handles PDB/mmCIF input,
    extracting both sequence and structural features (3Di + coordinates).

    Args:
        pdb_dir: Directory containing PDB files
        output_db: Path for output database (without extension)
        threads: Number of threads

    Returns:
        Path to the created database
    """
    output_db = Path(output_db)
    output_db.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "foldseek", "createdb",
        str(pdb_dir),
        str(output_db),
        "--threads", str(threads),
    ]

    logger.info(f"Creating Foldseek query database from {pdb_dir}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,  # 10 min max for large sets
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"foldseek createdb failed (exit {result.returncode}): {result.stderr}"
        )

    logger.debug(f"Created query database at {output_db}")
    return output_db


# ============================================================================
# Structural Search
# ============================================================================


def run_foldseek_search(
    query_db: Path,
    target_db: Path,
    output_path: Path,
    threads: int = 4,
    evalue: float = DEFAULT_EVALUE,
    sensitivity: float = DEFAULT_SENSITIVITY,
    max_seqs: int = DEFAULT_MAX_SEQS,
    tmp_dir: Path | None = None,
) -> pd.DataFrame:
    """Run Foldseek search against a target database.

    Args:
        query_db: Path to Foldseek query database (from create_query_db_from_pdbs)
        target_db: Path to target Foldseek database (e.g., BFVD)
        output_path: Path for output TSV file
        threads: Number of threads
        evalue: E-value threshold
        sensitivity: Search sensitivity (default 9.5 = exhaustive)
        max_seqs: Maximum target sequences per query
        tmp_dir: Temporary directory for Foldseek

    Returns:
        DataFrame with search results
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="foldseek_"))
    else:
        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

    result_db = tmp_dir / "resultDB"

    # Run search
    search_cmd = [
        "foldseek", "search",
        str(query_db),
        str(target_db),
        str(result_db),
        str(tmp_dir / "search_tmp"),
        "--threads", str(threads),
        "-e", str(evalue),
        "-s", str(sensitivity),
        "--max-seqs", str(max_seqs),
        "--exhaustive-search", "1",
    ]

    logger.info(f"Running Foldseek search against {target_db.name}")
    result = subprocess.run(
        search_cmd,
        capture_output=True,
        text=True,
        timeout=3600,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"foldseek search failed (exit {result.returncode}): {result.stderr}"
        )

    # Convert results to tabular format
    convert_cmd = [
        "foldseek", "convertalis",
        str(query_db),
        str(target_db),
        str(result_db),
        str(output_path),
        "--threads", str(threads),
        "--format-output", FOLDSEEK_OUTPUT_FORMAT,
    ]

    result = subprocess.run(
        convert_cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"foldseek convertalis failed (exit {result.returncode}): {result.stderr}"
        )

    # Parse results
    if output_path.exists() and output_path.stat().st_size > 0:
        df = pd.read_csv(
            output_path,
            sep="\t",
            header=None,
            names=FOLDSEEK_OUTPUT_COLUMNS,
        )
        logger.info(f"Found {len(df)} hits")
        return df

    logger.warning("No hits found")
    return pd.DataFrame(columns=FOLDSEEK_OUTPUT_COLUMNS)


def search_pdb_against_db(
    pdb_dir: Path,
    target_db: Path,
    output_dir: Path,
    db_name: str = "bfvd",
    threads: int = 4,
    evalue: float = DEFAULT_EVALUE,
    sensitivity: float = DEFAULT_SENSITIVITY,
    max_seqs: int = DEFAULT_MAX_SEQS,
) -> FoldseekResult:
    """Search PDB files against a Foldseek database.

    Convenience function that creates a query database and runs the search.

    Args:
        pdb_dir: Directory containing query PDB files
        target_db: Path to target Foldseek database
        output_dir: Output directory for results
        db_name: Name for the database (used in logging/result)
        threads: Number of threads
        evalue: E-value threshold
        sensitivity: Search sensitivity
        max_seqs: Maximum sequences per query

    Returns:
        FoldseekResult with hits DataFrame
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = output_dir / f"tmp_{db_name}"
    query_db = tmp_dir / "queryDB"
    output_path = output_dir / f"{db_name}_hits.tsv"

    # Count query PDB files
    pdb_files = list(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        logger.warning(f"No PDB files found in {pdb_dir}")
        return FoldseekResult(
            hits=pd.DataFrame(columns=FOLDSEEK_OUTPUT_COLUMNS),
            n_queries=0,
            n_hits=0,
            database=db_name,
        )

    try:
        # Create query database
        create_query_db_from_pdbs(pdb_dir, query_db, threads)

        # Run search
        hits = run_foldseek_search(
            query_db=query_db,
            target_db=target_db,
            output_path=output_path,
            threads=threads,
            evalue=evalue,
            sensitivity=sensitivity,
            max_seqs=max_seqs,
            tmp_dir=tmp_dir,
        )

        return FoldseekResult(
            hits=hits,
            query_db=query_db,
            n_queries=len(pdb_files),
            n_hits=len(hits),
            database=db_name,
        )

    except (RuntimeError, subprocess.TimeoutExpired) as e:
        logger.error(f"Foldseek search against {db_name} failed: {e}")
        return FoldseekResult(
            hits=pd.DataFrame(columns=FOLDSEEK_OUTPUT_COLUMNS),
            n_queries=len(pdb_files),
            n_hits=0,
            database=db_name,
        )


def search_databases(
    pdb_dir: Path,
    output_dir: Path,
    databases: dict[str, Path],
    threads: int = 4,
    evalue: float = DEFAULT_EVALUE,
    sensitivity: float = DEFAULT_SENSITIVITY,
    max_seqs: int = DEFAULT_MAX_SEQS,
) -> SearchBatch:
    """Search PDB files against one or more Foldseek databases.

    Args:
        pdb_dir: Directory containing query PDB files
        output_dir: Output directory for results
        databases: Dict mapping database name to database path
            e.g., {"bfvd": Path("~/.virosense/databases/bfvd/bfvd")}
        threads: Number of threads
        evalue: E-value threshold
        sensitivity: Search sensitivity
        max_seqs: Maximum sequences per query

    Returns:
        SearchBatch with results for each database
    """
    batch = SearchBatch()

    for db_name, db_path in databases.items():
        if not db_path.parent.exists():
            logger.warning(f"Database {db_name} not found at {db_path}")
            batch.results[db_name] = FoldseekResult(
                hits=pd.DataFrame(columns=FOLDSEEK_OUTPUT_COLUMNS),
                database=db_name,
            )
            continue

        logger.info(f"Searching {db_name} database...")
        batch.results[db_name] = search_pdb_against_db(
            pdb_dir=pdb_dir,
            target_db=db_path,
            output_dir=output_dir,
            db_name=db_name,
            threads=threads,
            evalue=evalue,
            sensitivity=sensitivity,
            max_seqs=max_seqs,
        )

    logger.info(batch.summary())
    return batch


# ============================================================================
# Easy Search (single-step PDB → TSV)
# ============================================================================


def easy_search(
    pdb_dir: Path,
    target_db: Path,
    output_path: Path,
    threads: int = 4,
    evalue: float = DEFAULT_EVALUE,
    sensitivity: float = DEFAULT_SENSITIVITY,
    tmp_dir: Path | None = None,
) -> pd.DataFrame:
    """Run Foldseek easy-search directly from PDB files to TSV output.

    Simpler alternative to search_pdb_against_db for quick searches.
    Uses `foldseek easy-search` which handles createdb internally.

    Args:
        pdb_dir: Directory containing query PDB files
        target_db: Path to target Foldseek database
        output_path: Path for output TSV file
        threads: Number of threads
        evalue: E-value threshold
        sensitivity: Search sensitivity
        tmp_dir: Temporary directory

    Returns:
        DataFrame with search results
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="foldseek_"))

    cmd = [
        "foldseek", "easy-search",
        str(pdb_dir),
        str(target_db),
        str(output_path),
        str(tmp_dir),
        "--threads", str(threads),
        "-e", str(evalue),
        "-s", str(sensitivity),
        "--exhaustive-search", "1",
        "--format-output", FOLDSEEK_OUTPUT_FORMAT,
    ]

    logger.info(f"Running Foldseek easy-search: {pdb_dir} → {target_db.name}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"foldseek easy-search failed (exit {result.returncode}): {result.stderr}"
        )

    if output_path.exists() and output_path.stat().st_size > 0:
        df = pd.read_csv(
            output_path,
            sep="\t",
            header=None,
            names=FOLDSEEK_OUTPUT_COLUMNS,
        )
        logger.info(f"Found {len(df)} hits")
        return df

    logger.warning("No hits found")
    return pd.DataFrame(columns=FOLDSEEK_OUTPUT_COLUMNS)
