#!/usr/bin/env python3
"""
Extract multi-offset cosine features for the linear probe experiment.

For each of the 36 exon-intron genes, extract per-position embeddings
and compute cosine similarity at offsets 1-15, giving a 15-D feature
vector per position. This is a much richer signal than the 2-D (cos1, cos3)
used in the lite version.

Usage:
    uv run python scripts/extract_multioffset_features.py --nim-url http://localhost:8000
"""

import argparse
import asyncio
import base64
import io
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path("results/experiments/exon_intron")
OUT_DIR = Path("results/experiments/boundary_resolution")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_OFFSET = 15


async def extract_multioffset(seq: str, nim_url: str) -> np.ndarray:
    """Extract per-position cosines at offsets 1-15."""
    import httpx

    url = f"{nim_url.rstrip('/')}/biology/arc/evo2/forward"
    layer = "decoder.layers.10"
    headers = {"Content-Type": "application/json"}

    payload = {"sequence": seq, "output_layers": [layer]}

    async with httpx.AsyncClient() as client:
        for attempt in range(5):
            try:
                resp = await client.post(url, json=payload, headers=headers,
                                         timeout=600, follow_redirects=True)
                if resp.status_code in (429, 503):
                    await asyncio.sleep(2 ** attempt * 10)
                    continue
                resp.raise_for_status()
                data = resp.json()

                raw = base64.b64decode(data["data"])
                npz = np.load(io.BytesIO(raw))
                emb = npz[f"{layer}.output"]
                if emb.ndim == 3:
                    if emb.shape[0] == 1:
                        emb = emb.squeeze(0)
                    elif emb.shape[1] == 1:
                        emb = emb.squeeze(1)

                seq_len = len(seq)
                norms = np.linalg.norm(emb[:seq_len], axis=1)

                # Compute cosines at offsets 1-15
                features = np.zeros((seq_len, MAX_OFFSET))
                for offset in range(1, MAX_OFFSET + 1):
                    for i in range(seq_len - offset):
                        ni = norms[i]
                        nj = norms[i + offset]
                        if ni > 0 and nj > 0:
                            features[i, offset - 1] = (
                                np.dot(emb[i], emb[i + offset]) / (ni * nj)
                            )

                return features

            except Exception as e:
                if attempt < 4:
                    await asyncio.sleep(2 ** attempt * 5)
                else:
                    print(f"  FAILED: {e}")
                    return np.zeros((len(seq), MAX_OFFSET))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nim-url", type=str, required=True)
    args = parser.parse_args()

    # Get all genes with per-position data
    gene_files = sorted(DATA_DIR.glob("metrics/*_perpos.json"))
    seq_dir = DATA_DIR / "sequences"

    print(f"Extracting multi-offset features for {len(gene_files)} genes")
    print(f"Offsets: 1-{MAX_OFFSET}")

    for gf in gene_files:
        gene_name = gf.stem.replace("_perpos", "")
        out_path = OUT_DIR / f"{gene_name}_multioffset.npz"

        if out_path.exists():
            print(f"  Cached: {gene_name}")
            continue

        # Load sequence
        fasta_path = seq_dir / f"{gene_name}.fasta"
        if not fasta_path.exists():
            print(f"  Skipped (no FASTA): {gene_name}")
            continue

        with open(fasta_path) as f:
            seq = "".join(l.strip() for l in f if not l.startswith(">"))

        if len(seq) > 16000:
            print(f"  Skipped (too long: {len(seq)}bp): {gene_name}")
            continue

        print(f"  Extracting {gene_name} ({len(seq)}bp)...", end=" ", flush=True)
        features = asyncio.run(extract_multioffset(seq, nim_url=args.nim_url))
        np.savez_compressed(out_path, features=features)
        print(f"saved ({features.shape})")

    print("\nDone. Run boundary_resolution_full.py for the full comparison.")


if __name__ == "__main__":
    main()
