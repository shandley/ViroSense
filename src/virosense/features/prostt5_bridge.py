"""Bridge to vHold's ProstT5 protein embeddings.

Optional integration — requires vhold package to be installed.
All vHold imports are guarded by try/except ImportError.
"""

from pathlib import Path

import numpy as np
from loguru import logger


def is_vhold_available() -> bool:
    """Check if vHold is installed and importable."""
    try:
        import vhold  # noqa: F401
        return True
    except ImportError:
        return False


def load_vhold_embeddings(embeddings_path: str | Path) -> tuple[list[str], np.ndarray]:
    """Load pre-computed ProstT5 embeddings from vHold output.

    Args:
        embeddings_path: Path to NPZ file with protein embeddings.

    Returns:
        Tuple of (protein_ids, embeddings_matrix).
    """
    path = Path(embeddings_path)
    if not path.exists():
        raise FileNotFoundError(f"vHold embeddings not found: {path}")

    data = np.load(path, allow_pickle=True)
    protein_ids = list(data["sequence_ids"])
    embeddings = data["embeddings"]

    logger.info(f"Loaded {len(protein_ids)} ProstT5 embeddings from {path}")
    return protein_ids, embeddings


def load_vhold_annotations(output_path: str | Path) -> dict:
    """Load vHold annotation results from TSV output.

    Args:
        output_path: Path to vHold output directory or TSV file.

    Returns:
        Dict mapping protein_id to annotation info.
    """
    import pandas as pd

    path = Path(output_path)
    if path.is_dir():
        tsv_files = list(path.glob("*.tsv"))
        if not tsv_files:
            raise FileNotFoundError(f"No TSV files found in {path}")
        path = tsv_files[0]

    df = pd.read_csv(path, sep="\t")
    logger.info(f"Loaded {len(df)} vHold annotations from {path}")

    return df.set_index(df.columns[0]).to_dict(orient="index")
