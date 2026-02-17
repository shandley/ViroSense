"""Cluster unclassified viral sequences using multi-modal embeddings."""

from pathlib import Path

import numpy as np
from loguru import logger


def run_cluster(
    input_file: str,
    output_dir: str,
    backend: str = "nim",
    model: str = "evo2_7b",
    mode: str = "multi",
    algorithm: str = "hdbscan",
    min_cluster_size: int = 5,
    n_clusters: int | None = None,
    threads: int = 4,
    vhold_embeddings: str | None = None,
    layer: str = "blocks.28.mlp.l3",
    cache_dir: str | None = None,
    pca_dims: int | None = 0,
) -> None:
    """Run multi-modal clustering pipeline.

    1. Read unclassified viral sequences
    2. Extract DNA embeddings (Evo2) and optionally protein embeddings (ProstT5)
    3. Fuse embeddings based on selected mode
    4. Cluster using selected algorithm
    5. Write cluster assignments and quality metrics
    """
    from virosense.backends.base import get_backend
    from virosense.clustering.metrics import cluster_summary, silhouette_score
    from virosense.clustering.multimodal import (
        cluster_sequences,
        fuse_embeddings,
    )
    from virosense.features.evo2_embeddings import extract_embeddings
    from virosense.io.fasta import read_fasta
    from virosense.io.results import write_json, write_tsv

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Clustering viral sequences from {input_file}")
    logger.info(f"Mode: {mode}, Algorithm: {algorithm}")
    logger.info(f"Backend: {backend}, Min cluster size: {min_cluster_size}")

    # 1. Read sequences
    sequences = read_fasta(input_file)
    if not sequences:
        logger.warning("No sequences found in input file.")
        return

    # 2. Extract DNA embeddings
    evo2_backend = get_backend(backend, api_key=None, model=model)
    if not evo2_backend.is_available():
        raise RuntimeError(
            f"Backend {backend!r} is not available. "
            "Check your configuration (e.g., NVIDIA_API_KEY for nim)."
        )

    cache_path = Path(cache_dir) if cache_dir else None
    dna_result = extract_embeddings(
        sequences=sequences,
        backend=evo2_backend,
        layer=layer,
        model=model,
        cache_dir=cache_path,
    )
    dna_embeddings = dna_result.embeddings
    sequence_ids = dna_result.sequence_ids

    # 3. Optionally load protein embeddings
    protein_embeddings = None
    if vhold_embeddings and mode in ("protein", "multi"):
        protein_embeddings = _load_protein_embeddings(
            vhold_embeddings, sequence_ids
        )

    # 4. Fuse embeddings
    fused = fuse_embeddings(dna_embeddings, protein_embeddings, mode=mode)
    logger.info(f"Fused embedding shape: {fused.shape}")

    # 5. Cluster
    assignments = cluster_sequences(
        embeddings=fused,
        sequence_ids=sequence_ids,
        algorithm=algorithm,
        min_cluster_size=min_cluster_size,
        n_clusters=n_clusters,
        pca_dims=pca_dims,
    )

    # 6. Compute quality metrics
    labels = np.array([a.cluster_id for a in assignments])
    sil_score = silhouette_score(fused, labels)
    summary = cluster_summary(labels)
    summary["silhouette_score"] = sil_score
    summary["algorithm"] = algorithm
    summary["mode"] = mode
    summary["n_sequences"] = len(sequence_ids)

    logger.info(
        f"Clustering complete: {summary['n_clusters']} clusters, "
        f"{summary['n_noise']} noise, silhouette={sil_score:.3f}"
    )

    # 7. Write results
    write_tsv(assignments, output_path, "cluster_assignments.tsv")
    write_json(summary, output_path, "cluster_metrics.json")
    logger.info(f"Cluster results written to {output_path}")


def _load_protein_embeddings(
    vhold_path: str, sequence_ids: list[str]
) -> np.ndarray | None:
    """Load and align protein embeddings with DNA sequence order.

    Protein IDs from vHold may differ from DNA sequence IDs (e.g.,
    ORF-level vs contig-level). This function attempts to match them
    by prefix. If fewer than 50% of sequences have matching protein
    embeddings, it falls back to DNA-only mode.
    """
    from virosense.features.prostt5_bridge import load_vhold_embeddings

    protein_ids, protein_matrix = load_vhold_embeddings(vhold_path)
    protein_lookup = dict(zip(protein_ids, range(len(protein_ids))))

    # Try exact match first, then prefix match
    matched_indices = []
    for seq_id in sequence_ids:
        if seq_id in protein_lookup:
            matched_indices.append(protein_lookup[seq_id])
        else:
            # Try matching by contig prefix (ORF IDs like "contig_1_orf_3")
            match = None
            for pid in protein_ids:
                if pid.startswith(seq_id) or seq_id.startswith(pid):
                    match = protein_lookup[pid]
                    break
            matched_indices.append(match)

    n_matched = sum(1 for idx in matched_indices if idx is not None)
    if n_matched < len(sequence_ids) * 0.5:
        logger.warning(
            f"Only {n_matched}/{len(sequence_ids)} sequences matched "
            f"protein embeddings. Falling back to DNA-only mode."
        )
        return None

    # Build aligned matrix, zero-filling unmatched
    prot_dim = protein_matrix.shape[1]
    aligned = np.zeros((len(sequence_ids), prot_dim), dtype=np.float32)
    for i, idx in enumerate(matched_indices):
        if idx is not None:
            aligned[i] = protein_matrix[idx]

    logger.info(
        f"Aligned {n_matched}/{len(sequence_ids)} protein embeddings "
        f"(dim={prot_dim})"
    )
    return aligned
