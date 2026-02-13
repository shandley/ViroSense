"""Evo2 embedding extraction with incremental checkpointing."""

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
    checkpoint_every: int = 50,
) -> EmbeddingResult:
    """Extract Evo2 embeddings with incremental checkpointing.

    When cache_dir is set, embeddings are saved to disk every
    `checkpoint_every` sequences. On restart, cached embeddings are
    loaded and only uncached sequences are sent to the backend.

    Args:
        sequences: Dict of sequence_id -> DNA sequence.
        backend: Evo2 backend to use for extraction.
        layer: Model layer for embedding extraction.
        model: Evo2 model name.
        batch_size: Number of sequences per batch.
        cache_dir: Directory for NPZ cache files. None disables caching.
        checkpoint_every: Save checkpoint every N sequences.

    Returns:
        EmbeddingResult with sequence IDs and embedding matrix.
    """
    all_ids = list(sequences.keys())

    if not cache_dir:
        # No caching — extract everything in one call
        request = EmbeddingRequest(sequences=sequences, layer=layer, model=model)
        return backend.extract_embeddings(request)

    # Load partial cache and determine what's left to extract
    path = _cache_path(cache_dir, layer, model)
    cached_ids_set = set()
    if path.exists():
        try:
            data = np.load(path, allow_pickle=True)
            cached_ids_set = set(data["sequence_ids"])
            logger.info(f"Found {len(cached_ids_set)} cached embeddings in {path}")
        except Exception as e:
            logger.warning(f"Failed to load cache, starting fresh: {e}")

    uncached = {k: v for k, v in sequences.items() if k not in cached_ids_set}

    if not uncached:
        logger.info(f"All {len(all_ids)} embeddings loaded from cache")
        return _load_ordered(all_ids, path, layer, model)

    logger.info(
        f"{len(cached_ids_set)} cached, {len(uncached)} remaining "
        f"(checkpointing every {checkpoint_every})"
    )

    # Extract uncached sequences in checkpoint batches
    uncached_keys = list(uncached.keys())
    for start in range(0, len(uncached_keys), checkpoint_every):
        end = min(start + checkpoint_every, len(uncached_keys))
        batch_keys = uncached_keys[start:end]
        batch_seqs = {k: uncached[k] for k in batch_keys}

        request = EmbeddingRequest(sequences=batch_seqs, layer=layer, model=model)
        batch_result = backend.extract_embeddings(request)

        _append_cache(batch_result, path)
        done = len(cached_ids_set) + end
        logger.info(
            f"Checkpoint saved: {done}/{len(all_ids)} total embeddings"
        )

    return _load_ordered(all_ids, path, layer, model)


def _cache_path(cache_dir: Path, layer: str, model: str) -> Path:
    """Get the cache file path for given parameters."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{model}_{layer.replace('.', '_')}_embeddings.npz"


def _append_cache(result: EmbeddingResult, path: Path) -> None:
    """Append new embeddings to an existing cache file on disk."""
    try:
        if path.exists():
            existing = np.load(path, allow_pickle=True)
            old_ids = list(existing["sequence_ids"])
            old_embeddings = existing["embeddings"]
            new_ids = old_ids + result.sequence_ids
            new_embeddings = np.vstack([old_embeddings, result.embeddings])
        else:
            new_ids = result.sequence_ids
            new_embeddings = result.embeddings

        np.savez(
            path,
            sequence_ids=np.array(new_ids),
            embeddings=new_embeddings,
        )
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")


def _load_ordered(
    requested_ids: list[str],
    path: Path,
    layer: str,
    model: str,
) -> EmbeddingResult:
    """Load embeddings from cache in the requested ID order."""
    data = np.load(path, allow_pickle=True)
    cached_ids = list(data["sequence_ids"])
    cached_embeddings = data["embeddings"]

    id_to_idx = {sid: i for i, sid in enumerate(cached_ids)}
    indices = [id_to_idx[sid] for sid in requested_ids]

    return EmbeddingResult(
        sequence_ids=requested_ids,
        embeddings=cached_embeddings[indices],
        layer=layer,
        model=model,
    )
