"""Extract and cache Evo2 embeddings for a set of sequences.

Separates the expensive embedding extraction step from downstream
analysis. Embeddings are cached to disk and can be reused by
detect, classify, cluster, scan, and prophage commands.
"""

from pathlib import Path

from loguru import logger


def run_embed(
    input_file: str,
    output_dir: str,
    backend: str = "nim",
    model: str = "evo2_7b",
    layer: str = "blocks.28.mlp.l3",
    batch_size: int = 25,
    nim_url: str | None = None,
    max_concurrent: int = 3,
    per_position: bool = False,
) -> None:
    """Extract Evo2 embeddings and write to cache.

    Mean-pooled embeddings are always produced (one vector per sequence).
    Per-position embeddings (one vector per nucleotide) are optionally
    saved for use by the scan command.

    Args:
        input_file: Path to input FASTA.
        output_dir: Directory to write cached embeddings.
        backend: Evo2 backend (nim, mlx, local).
        model: Evo2 model name.
        layer: Model layer for extraction.
        batch_size: Sequences per checkpoint batch.
        nim_url: Self-hosted NIM URL.
        max_concurrent: Max concurrent API requests.
        per_position: Also save per-position embeddings (large files).
    """
    from virosense.backends.base import get_backend
    from virosense.features.evo2_embeddings import extract_embeddings
    from virosense.io.fasta import read_fasta

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read sequences
    sequences = read_fasta(input_file)
    if not sequences:
        logger.warning("No sequences found in input file.")
        return

    logger.info(f"Embedding {len(sequences)} sequences")
    logger.info(f"Backend: {backend}, Model: {model}, Layer: {layer}")
    logger.info(f"Output: {output_path}")

    # Initialize backend
    evo2_backend = get_backend(
        backend, model=model, nim_url=nim_url, max_concurrent=max_concurrent
    )
    resolved_model = evo2_backend.model

    if not evo2_backend.is_available():
        raise RuntimeError(
            f"Backend {backend!r} is not available. "
            "Check your configuration (e.g., NVIDIA_API_KEY for nim)."
        )

    # Extract mean-pooled embeddings (with caching/checkpointing)
    result = extract_embeddings(
        sequences=sequences,
        backend=evo2_backend,
        layer=layer,
        model=resolved_model,
        batch_size=batch_size,
        cache_dir=output_path,
        checkpoint_every=batch_size,
    )

    logger.info(
        f"Mean-pooled embeddings: {result.embeddings.shape} "
        f"({len(result.sequence_ids)} sequences, {result.embeddings.shape[1]}-D)"
    )

    # Per-position embeddings (optional — significantly larger)
    if per_position:
        _extract_per_position(
            sequences=sequences,
            backend=evo2_backend,
            layer=layer,
            output_dir=output_path / "per_position",
        )

    logger.info(f"Embeddings cached to {output_path}/")


def _extract_per_position(
    sequences: dict[str, str],
    backend,
    layer: str,
    output_dir: Path,
) -> None:
    """Extract and save per-position embeddings (one file per sequence).

    Per-position embeddings are (seq_len, hidden_dim) arrays — too large
    to store in a single NPZ for many sequences. Each sequence gets its
    own .npy file.
    """
    import base64
    import io

    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    n_existing = len(list(output_dir.glob("*.npy")))
    if n_existing > 0:
        logger.info(f"Found {n_existing} existing per-position files, skipping those")

    n_extracted = 0
    n_total = len(sequences)

    for seq_id, sequence in sequences.items():
        # Safe filename
        safe_name = seq_id.replace("/", "_")[:80]
        npy_path = output_dir / f"{safe_name}.npy"

        if npy_path.exists():
            continue

        try:
            # Use the backend's low-level API to get raw per-position data
            # This mirrors poc_gene_boundaries.py extraction
            import httpx
            import os

            api_key = os.environ.get("NVIDIA_API_KEY")
            if not api_key:
                logger.warning("NVIDIA_API_KEY not set, skipping per-position extraction")
                return

            nim_layer = layer.replace("blocks.28", "blocks.20")
            url = getattr(backend, "base_url", "https://health.api.nvidia.com") + "/v1/biology/arc/evo2-40b/forward"

            payload = {"sequence": sequence, "output_layers": [nim_layer]}
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            with httpx.Client(timeout=300) as client:
                resp = client.post(url, json=payload, headers=headers)
                if resp.status_code == 302:
                    resp = client.get(resp.headers["Location"])
                resp.raise_for_status()
                data = resp.json()

            raw = base64.b64decode(data["data"].encode("ascii"))
            npz = np.load(io.BytesIO(raw))
            key = f"{nim_layer}.output"
            if key not in npz:
                key = list(npz.keys())[0]

            per_pos = npz[key]
            if per_pos.ndim == 3:
                per_pos = per_pos.squeeze(
                    axis=0 if per_pos.shape[0] == 1 else 1
                )

            np.save(npy_path, per_pos.astype(np.float32))
            n_extracted += 1

            if n_extracted % 10 == 0 or n_extracted == 1:
                logger.info(
                    f"Per-position: {n_extracted + n_existing}/{n_total} "
                    f"({per_pos.shape[0]} positions × {per_pos.shape[1]}-D)"
                )

        except Exception as e:
            logger.warning(f"Per-position extraction failed for {seq_id[:50]}: {e}")
            continue

    logger.info(
        f"Per-position extraction complete: {n_extracted} new + {n_existing} cached"
    )
