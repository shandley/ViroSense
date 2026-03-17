#!/usr/bin/env python3
"""Proof-of-concept: Evo2 per-position embeddings reveal gene boundaries.

Demonstrates that Evo2's hidden-state representations contain spatial
information about coding regions, gene boundaries, and intergenic regions —
without any gene-calling tool like Prodigal.

Usage:
    export NVIDIA_API_KEY=...
    uv run python scripts/poc_gene_boundaries.py \
        --fasta /tmp/poc_phage.fasta \
        --output results/poc_gene_boundaries/
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
from pathlib import Path

import numpy as np
from Bio import SeqIO
from loguru import logger


def extract_per_position_embeddings(
    sequence: str,
    seq_id: str,
    layer: str = "blocks.28.mlp.l3",
    model: str = "evo2_40b",
) -> np.ndarray:
    """Extract per-position embeddings from NIM API (single sequence).

    Returns:
        2D array of shape (seq_len, hidden_dim).
    """
    import httpx

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set")

    # Cloud NIM endpoint
    url = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward"

    # Cloud NIM uses different layer naming
    nim_layer = layer.replace("blocks.28", "blocks.20")

    payload = {
        "sequence": sequence,
        "output_layers": [nim_layer],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    logger.info(f"Requesting per-position embeddings for {seq_id} ({len(sequence)} bp)")

    with httpx.Client(timeout=300) as client:
        resp = client.post(url, json=payload, headers=headers)

        if resp.status_code == 302:
            # S3 redirect for long sequences
            redirect_url = resp.headers["Location"]
            resp = client.get(redirect_url)

        resp.raise_for_status()
        data = resp.json()

    # Decode NPZ response
    raw = base64.b64decode(data["data"].encode("ascii"))
    npz = np.load(io.BytesIO(raw))

    key = f"{nim_layer}.output"
    if key not in npz:
        key = list(npz.keys())[0]
        logger.debug(f"Using NPZ key {key!r}")

    per_position = npz[key]

    # Normalize shape: (1, seq_len, hidden_dim) → (seq_len, hidden_dim)
    if per_position.ndim == 3:
        per_position = per_position.squeeze(
            axis=0 if per_position.shape[0] == 1 else 1
        )

    logger.info(f"Per-position embeddings: {per_position.shape}")
    return per_position.astype(np.float32)


def call_genes_pyrodigal(sequence: str) -> list[dict]:
    """Call genes using pyrodigal-gv (ground truth for comparison)."""
    try:
        import pyrodigal_gv
    except ImportError:
        try:
            import pyrodigal
            gene_finder = pyrodigal.GeneFinder(meta=True)
            genes = gene_finder.find_genes(sequence.encode())
        except ImportError:
            logger.warning("Neither pyrodigal-gv nor pyrodigal installed, skipping gene calling")
            return []
    else:
        gene_finder = pyrodigal_gv.ViralGeneFinder(meta=True)
        genes = gene_finder.find_genes(sequence.encode())

    gene_list = []
    for gene in genes:
        gene_list.append({
            "start": gene.begin,
            "end": gene.end,
            "strand": gene.strand,
            "start_type": getattr(gene, "start_type", "?"),
        })

    return gene_list


def analyze_embedding_landscape(
    per_position: np.ndarray,
    genes: list[dict],
    output_dir: Path,
    seq_id: str,
    window: int = 50,
) -> dict:
    """Analyze per-position embeddings for gene boundary signatures.

    Analyses:
    1. Local embedding variance (sliding window)
    2. Cosine similarity between adjacent positions
    3. PCA trajectory along the sequence
    4. Embedding norm along the sequence
    5. Comparison of coding vs intergenic positions
    """
    from sklearn.decomposition import PCA

    seq_len, hidden_dim = per_position.shape
    logger.info(f"Analyzing {seq_len} positions × {hidden_dim} dimensions")

    results = {"seq_id": seq_id, "seq_len": seq_len, "n_genes": len(genes)}

    # 1. Cosine similarity between adjacent positions
    norms = np.linalg.norm(per_position, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = per_position / norms

    # Adjacent cosine similarity
    adj_cosine = np.sum(normalized[:-1] * normalized[1:], axis=1)
    results["adj_cosine_mean"] = float(adj_cosine.mean())
    results["adj_cosine_std"] = float(adj_cosine.std())

    # 2. Embedding norm along sequence
    position_norms = np.linalg.norm(per_position, axis=1)

    # 3. Sliding window variance
    window_var = np.array([
        per_position[max(0, i - window // 2):i + window // 2].var()
        for i in range(seq_len)
    ])

    # 4. PCA on per-position embeddings
    pca = PCA(n_components=5)
    pcs = pca.fit_transform(per_position)
    results["pca_variance_explained"] = pca.explained_variance_ratio_.tolist()

    # 5. PC1 derivative (rate of change) — spikes at boundaries
    pc1_deriv = np.abs(np.diff(pcs[:, 0]))
    pc2_deriv = np.abs(np.diff(pcs[:, 1]))

    # 6. Coding vs intergenic embedding comparison
    coding_mask = np.zeros(seq_len, dtype=bool)
    boundary_mask = np.zeros(seq_len, dtype=bool)
    boundary_width = 20  # positions around gene start/end

    for gene in genes:
        start = min(gene["start"], gene["end"]) - 1  # 0-indexed
        end = max(gene["start"], gene["end"])
        coding_mask[start:end] = True
        # Mark boundaries
        for bp in [start, end]:
            lo = max(0, bp - boundary_width)
            hi = min(seq_len, bp + boundary_width)
            boundary_mask[lo:hi] = True

    intergenic_mask = ~coding_mask
    interior_coding = coding_mask & ~boundary_mask

    if coding_mask.sum() > 0 and intergenic_mask.sum() > 0:
        coding_norm_mean = float(position_norms[coding_mask].mean())
        intergenic_norm_mean = float(position_norms[intergenic_mask].mean())
        results["coding_norm_mean"] = coding_norm_mean
        results["intergenic_norm_mean"] = intergenic_norm_mean
        results["norm_ratio"] = coding_norm_mean / intergenic_norm_mean

        # Cosine similarity at boundaries vs interior
        boundary_positions = np.where(boundary_mask[:-1])[0]
        interior_positions = np.where(interior_coding[:-1])[0]
        intergenic_positions = np.where(intergenic_mask[:-1] & ~boundary_mask[:-1])[0]

        if len(boundary_positions) > 0:
            results["adj_cosine_at_boundaries"] = float(adj_cosine[boundary_positions].mean())
        if len(interior_positions) > 0:
            results["adj_cosine_interior_coding"] = float(adj_cosine[interior_positions].mean())
        if len(intergenic_positions) > 0:
            results["adj_cosine_intergenic"] = float(adj_cosine[intergenic_positions].mean())

        # PC1 derivative at boundaries vs elsewhere
        if len(boundary_positions) > 0 and len(interior_positions) > 0:
            bp_valid = boundary_positions[boundary_positions < len(pc1_deriv)]
            ip_valid = interior_positions[interior_positions < len(pc1_deriv)]
            if len(bp_valid) > 0 and len(ip_valid) > 0:
                results["pc1_deriv_at_boundaries"] = float(pc1_deriv[bp_valid].mean())
                results["pc1_deriv_interior"] = float(pc1_deriv[ip_valid].mean())
                results["pc1_deriv_ratio"] = results["pc1_deriv_at_boundaries"] / results["pc1_deriv_interior"]

    # 7. Gene boundary detection peaks
    # Smooth the PC1 derivative and find peaks
    from scipy.ndimage import uniform_filter1d
    smooth_deriv = uniform_filter1d(pc1_deriv, size=20)

    # Find peaks above 2× mean
    threshold = smooth_deriv.mean() + 2 * smooth_deriv.std()
    peaks = np.where(smooth_deriv > threshold)[0]

    # Compare peak positions to actual gene boundaries
    actual_boundaries = []
    for gene in genes:
        actual_boundaries.append(min(gene["start"], gene["end"]) - 1)
        actual_boundaries.append(max(gene["start"], gene["end"]) - 1)
    actual_boundaries = sorted(set(actual_boundaries))

    # For each actual boundary, find closest peak
    hits = 0
    tolerance = 50  # positions
    for ab in actual_boundaries:
        if len(peaks) > 0:
            closest = min(abs(peaks - ab))
            if closest <= tolerance:
                hits += 1

    results["n_actual_boundaries"] = len(actual_boundaries)
    results["n_detected_peaks"] = len(peaks)
    results["boundary_hits"] = hits
    if len(actual_boundaries) > 0:
        results["boundary_recall"] = hits / len(actual_boundaries)

    # Save raw data for plotting
    np.savez_compressed(
        output_dir / "embedding_landscape.npz",
        adj_cosine=adj_cosine,
        position_norms=position_norms,
        window_var=window_var,
        pcs=pcs,
        pc1_deriv=pc1_deriv,
        smooth_deriv=smooth_deriv,
        coding_mask=coding_mask,
        boundary_mask=boundary_mask,
    )

    # Print results
    print("\n" + "=" * 70)
    print(f"GENE BOUNDARY DETECTION — {seq_id}")
    print(f"Sequence: {seq_len} bp, {len(genes)} genes")
    print("=" * 70)

    print(f"\n--- Embedding statistics ---")
    print(f"Adjacent cosine similarity: {results['adj_cosine_mean']:.4f} ± {results['adj_cosine_std']:.4f}")
    if "coding_norm_mean" in results:
        print(f"Embedding norm (coding):     {results['coding_norm_mean']:.1f}")
        print(f"Embedding norm (intergenic): {results['intergenic_norm_mean']:.1f}")
        print(f"Norm ratio (coding/inter):   {results['norm_ratio']:.3f}")

    print(f"\n--- Cosine similarity by region ---")
    for key, label in [
        ("adj_cosine_at_boundaries", "At gene boundaries (±20bp)"),
        ("adj_cosine_interior_coding", "Interior coding regions"),
        ("adj_cosine_intergenic", "Intergenic regions"),
    ]:
        if key in results:
            print(f"  {label}: {results[key]:.4f}")

    print(f"\n--- PC1 rate of change ---")
    if "pc1_deriv_at_boundaries" in results:
        print(f"  At boundaries: {results['pc1_deriv_at_boundaries']:.2f}")
        print(f"  Interior coding: {results['pc1_deriv_interior']:.2f}")
        print(f"  Ratio (boundary/interior): {results['pc1_deriv_ratio']:.2f}×")

    print(f"\n--- Boundary detection ---")
    print(f"  Actual gene boundaries: {results['n_actual_boundaries']}")
    print(f"  Embedding peaks detected: {results['n_detected_peaks']}")
    print(f"  Hits (within {tolerance}bp): {results['boundary_hits']}")
    if "boundary_recall" in results:
        print(f"  Recall: {results['boundary_recall']:.1%}")

    # Gene-by-gene detail
    print(f"\n--- Gene positions (Pyrodigal-gv) ---")
    for i, gene in enumerate(genes):
        strand = "+" if gene["strand"] == 1 else "-"
        print(f"  Gene {i + 1}: {gene['start']}-{gene['end']} ({strand})")

    print(f"\n--- Detected peaks (PC1 derivative) ---")
    for p in peaks[:20]:
        # Find closest gene boundary
        if actual_boundaries:
            closest_boundary = min(actual_boundaries, key=lambda b: abs(b - p))
            dist = p - closest_boundary
            print(f"  Position {p}: score={smooth_deriv[p]:.2f}, closest boundary={closest_boundary} (Δ={dist:+d})")
        else:
            print(f"  Position {p}: score={smooth_deriv[p]:.2f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="PoC: Evo2 gene boundary detection")
    parser.add_argument("--fasta", required=True, help="Input FASTA (single sequence)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--layer", default="blocks.28.mlp.l3")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sequence
    record = next(SeqIO.parse(args.fasta, "fasta"))
    sequence = str(record.seq).upper()
    seq_id = record.description
    logger.info(f"Loaded: {seq_id} ({len(sequence)} bp)")

    # Step 1: Extract per-position embeddings
    per_position = extract_per_position_embeddings(sequence, seq_id, args.layer)
    np.save(output_dir / "per_position_embeddings.npy", per_position)

    # Step 2: Call genes with Pyrodigal-gv
    genes = call_genes_pyrodigal(sequence)
    logger.info(f"Pyrodigal-gv found {len(genes)} genes")

    with open(output_dir / "genes.json", "w") as f:
        json.dump(genes, f, indent=2)

    # Step 3: Analyze embedding landscape
    results = analyze_embedding_landscape(per_position, genes, output_dir, seq_id)

    with open(output_dir / "analysis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
