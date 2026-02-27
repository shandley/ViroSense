"""Structure acquisition for viral protein annotation.

Two-pass strategy for obtaining protein structures:
1. AlphaFold DB lookup — instant PDB download for proteins with UniProt matches
2. ColabFold prediction — ab initio structure prediction for novel proteins

Designed for integration into ViroSense's annotate module.
"""

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# AlphaFold DB API
ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/prediction"
ALPHAFOLD_FILES_URL = "https://alphafold.ebi.ac.uk/files"

# HTTP settings
REQUEST_TIMEOUT = 30.0
MAX_RETRIES = 3
USER_AGENT = "ViroSense/0.1 (https://github.com/shandley/virosense)"


@dataclass
class StructureResult:
    """Result of structure acquisition for a single protein."""

    protein_id: str
    pdb_path: Path | None = None
    source: str = ""  # "alphafold_db", "colabfold", "failed"
    uniprot_accession: str | None = None
    plddt_mean: float | None = None
    error: str | None = None


@dataclass
class StructureBatch:
    """Results from a batch structure acquisition run."""

    results: dict[str, StructureResult] = field(default_factory=dict)
    n_alphafold: int = 0
    n_colabfold: int = 0
    n_failed: int = 0

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_with_structure(self) -> int:
        return self.n_alphafold + self.n_colabfold

    def summary(self) -> str:
        return (
            f"Structures: {self.n_with_structure}/{self.n_total} "
            f"(AlphaFold DB: {self.n_alphafold}, "
            f"ColabFold: {self.n_colabfold}, "
            f"failed: {self.n_failed})"
        )


# ============================================================================
# AlphaFold DB Lookup
# ============================================================================


def fetch_alphafold_structure(
    uniprot_accession: str,
    output_dir: Path,
    timeout: float = REQUEST_TIMEOUT,
) -> StructureResult:
    """Download a predicted structure from AlphaFold DB by UniProt accession.

    Args:
        uniprot_accession: UniProt accession (e.g., "P00520")
        output_dir: Directory to save PDB file
        timeout: HTTP request timeout in seconds

    Returns:
        StructureResult with path to downloaded PDB file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            # Query API for prediction metadata
            api_url = f"{ALPHAFOLD_API_URL}/{uniprot_accession}"
            resp = client.get(api_url, headers=headers)

            if resp.status_code == 404:
                return StructureResult(
                    protein_id=uniprot_accession,
                    source="failed",
                    error=f"No AlphaFold prediction for {uniprot_accession}",
                )

            resp.raise_for_status()
            predictions = resp.json()

            if not predictions:
                return StructureResult(
                    protein_id=uniprot_accession,
                    source="failed",
                    error=f"Empty prediction response for {uniprot_accession}",
                )

            # Use first prediction (usually the only one)
            prediction = predictions[0] if isinstance(predictions, list) else predictions
            pdb_url = prediction.get("pdbUrl")

            if not pdb_url:
                return StructureResult(
                    protein_id=uniprot_accession,
                    source="failed",
                    error=f"No PDB URL in prediction for {uniprot_accession}",
                )

            # Download PDB file
            pdb_filename = f"AF-{uniprot_accession}-F1-model_v4.pdb"
            pdb_path = output_dir / pdb_filename

            if pdb_path.exists():
                logger.debug(f"Using cached structure: {pdb_path}")
            else:
                logger.info(f"Downloading AlphaFold structure for {uniprot_accession}")
                pdb_resp = client.get(pdb_url, headers={"User-Agent": USER_AGENT})
                pdb_resp.raise_for_status()
                pdb_path.write_bytes(pdb_resp.content)

            return StructureResult(
                protein_id=uniprot_accession,
                pdb_path=pdb_path,
                source="alphafold_db",
                uniprot_accession=uniprot_accession,
                plddt_mean=_extract_mean_plddt(pdb_path),
            )

    except httpx.HTTPStatusError as e:
        return StructureResult(
            protein_id=uniprot_accession,
            source="failed",
            error=f"HTTP {e.response.status_code}: {e}",
        )
    except httpx.RequestError as e:
        return StructureResult(
            protein_id=uniprot_accession,
            source="failed",
            error=f"Request error: {e}",
        )


def batch_fetch_alphafold(
    accessions: list[str],
    output_dir: Path,
) -> dict[str, StructureResult]:
    """Download AlphaFold structures for multiple UniProt accessions.

    Args:
        accessions: List of UniProt accessions
        output_dir: Directory to save PDB files

    Returns:
        Dict mapping accession to StructureResult
    """
    results: dict[str, StructureResult] = {}
    for acc in accessions:
        results[acc] = fetch_alphafold_structure(acc, output_dir)
        if results[acc].pdb_path:
            logger.debug(f"  {acc}: {results[acc].source}")
        else:
            logger.debug(f"  {acc}: {results[acc].error}")
    return results


# ============================================================================
# ColabFold Prediction
# ============================================================================


def check_colabfold() -> tuple[bool, str]:
    """Check if colabfold_batch is available.

    Returns:
        (available, version_or_error)
    """
    try:
        result = subprocess.run(
            ["colabfold_batch", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True, "colabfold_batch available"
        return False, f"colabfold_batch returned {result.returncode}"
    except FileNotFoundError:
        return False, "colabfold_batch not found in PATH"
    except subprocess.TimeoutExpired:
        return False, "colabfold_batch timed out"


def run_colabfold(
    fasta_path: Path,
    output_dir: Path,
    num_models: int = 1,
    num_recycles: int = 3,
    msa_mode: str = "mmseqs2_uniref_env",
    use_gpu: bool = True,
) -> dict[str, StructureResult]:
    """Run ColabFold batch prediction on a FASTA file.

    Args:
        fasta_path: Path to input FASTA with protein sequences
        output_dir: Directory for ColabFold output
        num_models: Number of AlphaFold2 models to use (1-5)
        num_recycles: Number of recycling iterations
        msa_mode: MSA generation mode
        use_gpu: Whether to use GPU acceleration

    Returns:
        Dict mapping protein_id to StructureResult
    """
    available, msg = check_colabfold()
    if not available:
        raise FileNotFoundError(
            f"ColabFold not available: {msg}. "
            "Install with: pip install colabfold[alphafold] or see "
            "https://github.com/YoshitakaMo/localcolabfold"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "colabfold_batch",
        str(fasta_path),
        str(output_dir),
        "--num-models", str(num_models),
        "--num-recycle", str(num_recycles),
        "--msa-mode", msa_mode,
    ]

    if not use_gpu:
        cmd.append("--cpu")

    logger.info(f"Running ColabFold: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour max for batch
    )

    if result.returncode != 0:
        logger.error(f"ColabFold failed: {result.stderr}")
        raise RuntimeError(f"ColabFold failed with return code {result.returncode}")

    # Parse output — ColabFold writes one PDB per sequence
    return _parse_colabfold_output(output_dir)


def _parse_colabfold_output(output_dir: Path) -> dict[str, StructureResult]:
    """Parse ColabFold output directory for PDB files.

    ColabFold names output files as:
    {protein_id}_relaxed_rank_001_alphafold2_*.pdb
    or {protein_id}_unrelaxed_rank_001_alphafold2_*.pdb
    """
    results: dict[str, StructureResult] = {}

    # Look for ranked PDB files (prefer relaxed, fall back to unrelaxed)
    for pdb_file in sorted(output_dir.glob("*.pdb")):
        name = pdb_file.stem

        # Extract protein ID from ColabFold naming convention
        # Format: {id}_relaxed_rank_001_... or {id}_unrelaxed_rank_001_...
        for suffix in ("_relaxed_rank_001", "_unrelaxed_rank_001"):
            idx = name.find(suffix)
            if idx > 0:
                protein_id = name[:idx]
                # Only keep rank 1 (best model)
                if protein_id not in results:
                    results[protein_id] = StructureResult(
                        protein_id=protein_id,
                        pdb_path=pdb_file,
                        source="colabfold",
                        plddt_mean=_extract_mean_plddt(pdb_file),
                    )
                break

    return results


# ============================================================================
# Two-Pass Structure Acquisition
# ============================================================================


def acquire_structures(
    sequences: dict[str, str],
    output_dir: Path,
    uniprot_map: dict[str, str] | None = None,
    skip_colabfold: bool = False,
    colabfold_kwargs: dict | None = None,
) -> StructureBatch:
    """Two-pass structure acquisition for a set of protein sequences.

    Pass 1: For proteins with known UniProt accessions, download pre-computed
             structures from AlphaFold DB (instant, free).
    Pass 2: For remaining proteins, predict structures with ColabFold
             (minutes/protein, needs GPU).

    Args:
        sequences: Dict mapping protein_id to amino acid sequence
        output_dir: Base output directory
        uniprot_map: Optional dict mapping protein_id to UniProt accession.
            Proteins with accessions get AlphaFold DB lookup first.
        skip_colabfold: If True, skip ColabFold prediction for novel proteins
        colabfold_kwargs: Extra kwargs for run_colabfold()

    Returns:
        StructureBatch with results for all proteins
    """
    if uniprot_map is None:
        uniprot_map = {}
    if colabfold_kwargs is None:
        colabfold_kwargs = {}

    batch = StructureBatch()
    alphafold_dir = output_dir / "alphafold_structures"
    colabfold_dir = output_dir / "colabfold_structures"
    remaining_ids: list[str] = []

    # Pass 1: AlphaFold DB lookup for proteins with UniProt accessions
    proteins_with_accessions = {
        pid: acc for pid, acc in uniprot_map.items() if pid in sequences
    }

    if proteins_with_accessions:
        logger.info(
            f"Pass 1: AlphaFold DB lookup for {len(proteins_with_accessions)} proteins"
        )
        for pid, acc in proteins_with_accessions.items():
            result = fetch_alphafold_structure(acc, alphafold_dir)
            # Re-key with original protein ID
            result.protein_id = pid
            result.uniprot_accession = acc

            if result.pdb_path:
                batch.results[pid] = result
                batch.n_alphafold += 1
                logger.debug(f"  {pid} ({acc}): AlphaFold structure downloaded")
            else:
                remaining_ids.append(pid)
                logger.debug(f"  {pid} ({acc}): not in AlphaFold DB")
    else:
        logger.info("Pass 1: No UniProt accessions provided, skipping AlphaFold DB lookup")

    # Add proteins without accessions to remaining list
    for pid in sequences:
        if pid not in batch.results and pid not in remaining_ids:
            remaining_ids.append(pid)

    # Pass 2: ColabFold prediction for remaining proteins
    if remaining_ids and not skip_colabfold:
        logger.info(f"Pass 2: ColabFold prediction for {len(remaining_ids)} proteins")

        # Write remaining sequences to FASTA
        colabfold_fasta = output_dir / "colabfold_input.fasta"
        _write_fasta(
            {pid: sequences[pid] for pid in remaining_ids},
            colabfold_fasta,
        )

        try:
            colabfold_results = run_colabfold(
                colabfold_fasta, colabfold_dir, **colabfold_kwargs
            )
            for pid, result in colabfold_results.items():
                if pid in sequences:  # sanity check
                    batch.results[pid] = result
                    batch.n_colabfold += 1
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning(f"ColabFold failed: {e}")
            for pid in remaining_ids:
                if pid not in batch.results:
                    batch.results[pid] = StructureResult(
                        protein_id=pid,
                        source="failed",
                        error=str(e),
                    )
                    batch.n_failed += 1
    elif remaining_ids:
        logger.info(
            f"Pass 2: Skipping ColabFold for {len(remaining_ids)} proteins "
            "(--no-colabfold)"
        )
        for pid in remaining_ids:
            batch.results[pid] = StructureResult(
                protein_id=pid,
                source="failed",
                error="ColabFold skipped",
            )
            batch.n_failed += 1

    logger.info(batch.summary())
    return batch


# ============================================================================
# Utilities
# ============================================================================


def _extract_mean_plddt(pdb_path: Path) -> float | None:
    """Extract mean pLDDT from B-factor column of a PDB file.

    AlphaFold stores pLDDT scores (0-100) in the B-factor column.
    """
    b_factors: list[float] = []
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith(("ATOM  ", "HETATM")):
                    # B-factor is columns 61-66 in PDB format
                    try:
                        b_factor = float(line[60:66].strip())
                        b_factors.append(b_factor)
                    except (ValueError, IndexError):
                        continue
    except OSError:
        return None

    if b_factors:
        return sum(b_factors) / len(b_factors)
    return None


def _write_fasta(sequences: dict[str, str], path: Path) -> None:
    """Write sequences to FASTA file."""
    with open(path, "w") as f:
        for pid, seq in sequences.items():
            f.write(f">{pid}\n{seq}\n")
