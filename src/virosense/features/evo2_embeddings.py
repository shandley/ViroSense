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

    Uses shard-based checkpointing to avoid write amplification:
    each checkpoint writes a small shard file, and shards are
    consolidated into a single NPZ at the end.

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
        result = backend.extract_embeddings(request)
        # Upcast to float64 to avoid overflow in downstream matmul
        result.embeddings = result.embeddings.astype(np.float64)
        return result

    # Load partial cache and determine what's left to extract
    main_path = _cache_path(cache_dir, layer, model)
    cached_ids_set = _find_cached_ids(main_path)

    uncached = {k: v for k, v in sequences.items() if k not in cached_ids_set}

    if not uncached:
        logger.info(f"All {len(all_ids)} embeddings loaded from cache")
        return _load_ordered(all_ids, main_path, layer, model)

    logger.info(
        f"{len(cached_ids_set)} cached, {len(uncached)} remaining "
        f"(checkpointing every {checkpoint_every})"
    )

    # Determine next shard index
    shard_idx = _next_shard_index(main_path)

    # Extract uncached sequences in checkpoint batches
    uncached_keys = list(uncached.keys())
    for start in range(0, len(uncached_keys), checkpoint_every):
        end = min(start + checkpoint_every, len(uncached_keys))
        batch_keys = uncached_keys[start:end]
        batch_seqs = {k: uncached[k] for k in batch_keys}

        request = EmbeddingRequest(sequences=batch_seqs, layer=layer, model=model)
        batch_result = backend.extract_embeddings(request)

        _save_shard(batch_result, main_path, shard_idx)
        shard_idx += 1
        done = len(cached_ids_set) + end
        logger.info(
            f"Checkpoint saved: {done}/{len(all_ids)} total embeddings"
        )

    # Consolidate shards into a single file for clean final state
    _consolidate_shards(main_path)

    return _load_ordered(all_ids, main_path, layer, model)


def _cache_path(cache_dir: Path, layer: str, model: str) -> Path:
    """Get the cache file path for given parameters."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{model}_{layer.replace('.', '_')}_embeddings.npz"


def _shard_pattern(main_path: Path) -> str:
    """Get the glob pattern for shard files associated with a main cache path."""
    return f"{main_path.stem}_shard_*.npz"


def _shard_path(main_path: Path, shard_idx: int) -> Path:
    """Get the path for a specific shard file."""
    return main_path.parent / f"{main_path.stem}_shard_{shard_idx:04d}.npz"


def _find_shard_files(main_path: Path) -> list[Path]:
    """Find all shard files for a given main cache path, sorted by index."""
    return sorted(main_path.parent.glob(_shard_pattern(main_path)))


def _next_shard_index(main_path: Path) -> int:
    """Determine the next shard index based on existing shard files."""
    shards = _find_shard_files(main_path)
    if not shards:
        return 0
    # Parse the index from the last shard filename
    last = shards[-1].stem
    idx_str = last.rsplit("_", 1)[-1]
    return int(idx_str) + 1


def _find_cached_ids(main_path: Path) -> set[str]:
    """Find all sequence IDs cached in the main file and any shards."""
    cached = set()

    # Check main consolidated file
    if main_path.exists():
        try:
            data = np.load(main_path, allow_pickle=True)
            cached.update(data["sequence_ids"])
            logger.info(f"Found {len(cached)} cached embeddings in {main_path}")
        except Exception as e:
            logger.warning(f"Failed to load main cache: {e}")

    # Check shard files
    shards = _find_shard_files(main_path)
    for shard in shards:
        try:
            data = np.load(shard, allow_pickle=True)
            n = len(data["sequence_ids"])
            cached.update(data["sequence_ids"])
            logger.debug(f"Found {n} embeddings in shard {shard.name}")
        except Exception as e:
            logger.warning(f"Failed to load shard {shard}: {e}")

    if shards:
        logger.info(
            f"Found {len(shards)} shard files with "
            f"{len(cached)} total cached embeddings"
        )

    return cached


def _save_shard(result: EmbeddingResult, main_path: Path, shard_idx: int) -> None:
    """Save a single checkpoint batch as a shard file."""
    path = _shard_path(main_path, shard_idx)
    try:
        np.savez(
            path,
            sequence_ids=np.array(result.sequence_ids),
            embeddings=result.embeddings,
        )
    except Exception as e:
        logger.warning(f"Failed to save shard {path}: {e}")


def _consolidate_shards(main_path: Path) -> None:
    """Merge all shards (and existing main file) into a single NPZ file.

    After consolidation, shard files are deleted.
    """
    shards = _find_shard_files(main_path)
    if not shards:
        return

    all_ids: list[str] = []
    all_embeddings: list[np.ndarray] = []

    # Load existing main file first
    if main_path.exists():
        try:
            data = np.load(main_path, allow_pickle=True)
            all_ids.extend(data["sequence_ids"])
            all_embeddings.append(data["embeddings"])
        except Exception as e:
            logger.warning(f"Failed to load main cache during consolidation: {e}")

    # Load each shard
    for shard in shards:
        try:
            data = np.load(shard, allow_pickle=True)
            all_ids.extend(data["sequence_ids"])
            all_embeddings.append(data["embeddings"])
        except Exception as e:
            logger.warning(f"Failed to load shard during consolidation: {e}")

    if not all_embeddings:
        return

    # Write consolidated file
    merged_embeddings = np.vstack(all_embeddings)
    np.savez(
        main_path,
        sequence_ids=np.array(all_ids),
        embeddings=merged_embeddings,
    )
    logger.info(
        f"Consolidated {len(shards)} shards into {main_path} "
        f"({len(all_ids)} sequences)"
    )

    # Clean up shard files
    for shard in shards:
        shard.unlink()


def _load_ordered(
    requested_ids: list[str],
    main_path: Path,
    layer: str,
    model: str,
) -> EmbeddingResult:
    """Load embeddings from cache in the requested ID order.

    Loads from the consolidated main file and any remaining shards.
    Cache stores float32 for space efficiency; returned as float64
    to avoid overflow in downstream matmul operations (sklearn PCA,
    StandardScaler, etc.) with high-dimensional embeddings.
    """
    all_ids: list[str] = []
    all_embeddings: list[np.ndarray] = []

    # Load main file
    if main_path.exists():
        data = np.load(main_path, allow_pickle=True)
        all_ids.extend(data["sequence_ids"])
        all_embeddings.append(data["embeddings"])

    # Load any remaining shards (should be empty after consolidation,
    # but handles interrupted runs)
    for shard in _find_shard_files(main_path):
        data = np.load(shard, allow_pickle=True)
        all_ids.extend(data["sequence_ids"])
        all_embeddings.append(data["embeddings"])

    cached_embeddings = np.vstack(all_embeddings)
    id_to_idx = {sid: i for i, sid in enumerate(all_ids)}
    indices = [id_to_idx[sid] for sid in requested_ids]

    return EmbeddingResult(
        sequence_ids=requested_ids,
        embeddings=cached_embeddings[indices].astype(np.float64),
        layer=layer,
        model=model,
    )
