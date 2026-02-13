"""Evo2 embedding extraction and caching."""

from pathlib import Path

import numpy as np
from loguru import logger

from virosense.backends.base import EmbeddingRequest, EmbeddingResult, Evo2Backend


def extract_embeddings(
    sequences: dict[str, str],
    backend: Evo2Backend,
    layer: str = "blocks.28.mlp.l3",
    model: str = "evo2_7b",
    batch_size: int = 16,
    cache_dir: Path | None = None,
) -> EmbeddingResult:
    """Extract Evo2 embeddings with optional NPZ caching.

    Args:
        sequences: Dict of sequence_id -> DNA sequence.
        backend: Evo2 backend to use for extraction.
        layer: Model layer for embedding extraction.
        model: Evo2 model name.
        batch_size: Number of sequences per batch.
        cache_dir: Directory for NPZ cache files. None disables caching.

    Returns:
        EmbeddingResult with sequence IDs and embedding matrix.
    """
    if cache_dir:
        cached = _load_cached(sequences, cache_dir, layer, model)
        if cached is not None:
            return cached

    request = EmbeddingRequest(
        sequences=sequences,
        layer=layer,
        model=model,
    )

    result = backend.extract_embeddings(request)

    if cache_dir:
        _save_cache(result, cache_dir, layer, model)

    return result


def _cache_path(cache_dir: Path, layer: str, model: str) -> Path:
    """Get the cache file path for given parameters."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{model}_{layer.replace('.', '_')}_embeddings.npz"


def _load_cached(
    sequences: dict[str, str],
    cache_dir: Path,
    layer: str,
    model: str,
) -> EmbeddingResult | None:
    """Load embeddings from cache if all sequences are present."""
    path = _cache_path(cache_dir, layer, model)
    if not path.exists():
        return None

    try:
        data = np.load(path, allow_pickle=True)
        cached_ids = list(data["sequence_ids"])
        if all(sid in cached_ids for sid in sequences):
            indices = [cached_ids.index(sid) for sid in sequences]
            logger.info(f"Loaded {len(indices)} embeddings from cache")
            return EmbeddingResult(
                sequence_ids=list(sequences.keys()),
                embeddings=data["embeddings"][indices],
                layer=layer,
                model=model,
            )
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")

    return None


def _save_cache(
    result: EmbeddingResult,
    cache_dir: Path,
    layer: str,
    model: str,
) -> None:
    """Save embeddings to NPZ cache."""
    path = _cache_path(cache_dir, layer, model)
    try:
        np.savez(
            path,
            sequence_ids=np.array(result.sequence_ids),
            embeddings=result.embeddings,
        )
        logger.info(f"Cached {len(result.sequence_ids)} embeddings to {path}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")
