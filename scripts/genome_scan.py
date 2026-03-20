#!/usr/bin/env python3
"""
Full genome scan: per-position offset-3 cosine inversion along E. coli K12.

Extracts per-position embeddings in sliding windows across the full genome,
computes cos1/cos3/norms on-the-fly, and saves only the 1D profiles.

Usage:
    NVIDIA_API_KEY=... uv run python scripts/genome_scan.py
"""

import asyncio
import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

# ── Config ──
GENOME_ACC = "U00096.3"  # E. coli K12 MG1655
WINDOW_SIZE = 16000
STEP_SIZE = 12000  # 4kb overlap
LAYER = "blocks.10"
OUT_DIR = Path("results/genome_scan")
SMOOTH_WINDOW = 200  # bp for final smoothing


def download_genome():
    """Download E. coli K12 genome from NCBI."""
    from Bio import Entrez, SeqIO
    from io import StringIO

    Entrez.email = "shandley@wustl.edu"
    fasta_path = OUT_DIR / "ecoli_k12.fasta"

    if fasta_path.exists():
        with open(fasta_path) as f:
            seq = "".join(l.strip() for l in f if not l.startswith(">"))
        print(f"Genome cached: {len(seq):,} bp")
        return seq

    print(f"Downloading {GENOME_ACC}...")
    handle = Entrez.efetch(db="nucleotide", id=GENOME_ACC, rettype="fasta", retmode="text")
    record = next(SeqIO.parse(StringIO(handle.read()), "fasta"))
    handle.close()
    seq = str(record.seq).upper()

    with open(fasta_path, "w") as f:
        f.write(f">ecoli_k12 {GENOME_ACC} {len(seq)}bp\n")
        for i in range(0, len(seq), 60):
            f.write(seq[i:i+60] + "\n")

    print(f"Downloaded: {len(seq):,} bp")
    return seq


def download_gene_annotations():
    """Download gene annotations for E. coli K12."""
    ann_path = OUT_DIR / "ecoli_k12_genes.json"
    if ann_path.exists():
        with open(ann_path) as f:
            return json.load(f)

    from Bio import Entrez, SeqIO
    from io import StringIO

    Entrez.email = "shandley@wustl.edu"
    print("Downloading gene annotations...")

    # Fetch in chunks to get all genes
    handle = Entrez.efetch(db="nucleotide", id=GENOME_ACC, rettype="gb", retmode="text")
    record = next(SeqIO.parse(StringIO(handle.read()), "genbank"))
    handle.close()

    genes = []
    for feat in record.features:
        if feat.type == "gene":
            name = feat.qualifiers.get("gene", feat.qualifiers.get("locus_tag", ["?"]))[0]
            start = int(feat.location.start)
            end = int(feat.location.end)
            strand = "+" if feat.location.strand == 1 else "-"
            genes.append({"name": name, "start": start, "end": end, "strand": strand})

    with open(ann_path, "w") as f:
        json.dump(genes, f)

    print(f"Found {len(genes)} genes")
    return genes


def compute_window_metrics(emb: np.ndarray) -> dict:
    """Compute per-position cos1, cos3, and norms from per-position embeddings."""
    n = len(emb)
    norms = np.linalg.norm(emb, axis=1)

    cos1 = np.zeros(n)
    cos3 = np.zeros(n)

    for i in range(n - 1):
        ni = norms[i]
        ni1 = norms[i + 1]
        if ni > 0 and ni1 > 0:
            cos1[i] = np.dot(emb[i], emb[i + 1]) / (ni * ni1)

    for i in range(n - 3):
        ni = norms[i]
        ni3 = norms[i + 3]
        if ni > 0 and ni3 > 0:
            cos3[i] = np.dot(emb[i], emb[i + 3]) / (ni * ni3)

    return {"cos1": cos1, "cos3": cos3, "norms": norms}


async def extract_all_windows(seq: str, nim_url: str | None = None):
    """Extract per-position embeddings for all windows and compute metrics."""
    import httpx

    api_key = os.environ.get("NVIDIA_API_KEY", "")

    genome_len = len(seq)
    n_windows = (genome_len - WINDOW_SIZE) // STEP_SIZE + 1
    print(f"Genome: {genome_len:,} bp")
    print(f"Windows: {n_windows} ({WINDOW_SIZE} bp, step {STEP_SIZE} bp)")

    # Check cache
    cache_path = OUT_DIR / "window_metrics.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        print(f"Loaded cached metrics: {len(data['cos1']):,} positions")
        return dict(data)

    # Determine endpoint
    if nim_url:
        url = f"{nim_url.rstrip('/')}/biology/arc/evo2/forward"
        layer = "decoder.layers.10"
        concurrency = 1
        print(f"Using self-hosted NIM: {url}")
    else:
        url = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward"
        layer = LAYER
        concurrency = 3
        if not api_key:
            print("ERROR: Set NVIDIA_API_KEY or use --nim-url")
            sys.exit(1)
        print(f"Using cloud NIM")

    # Accumulator arrays — weighted average for overlapping regions
    cos1_sum = np.zeros(genome_len, dtype=np.float64)
    cos3_sum = np.zeros(genome_len, dtype=np.float64)
    norm_sum = np.zeros(genome_len, dtype=np.float64)
    count = np.zeros(genome_len, dtype=np.float64)

    completed = 0
    failed = 0

    sem = asyncio.Semaphore(concurrency)

    async def process_window(client: httpx.AsyncClient, win_idx: int, start: int):
        nonlocal completed, failed
        end = min(start + WINDOW_SIZE, genome_len)
        window_seq = seq[start:end]

        async with sem:
            headers = {"Content-Type": "application/json"}
            if not nim_url:
                headers["Authorization"] = f"Bearer {api_key}"
            payload = {"sequence": window_seq, "output_layers": [layer]}

            for attempt in range(5):
                try:
                    resp = await client.post(url, json=payload, headers=headers,
                                             timeout=300, follow_redirects=True)
                    if resp.status_code == 429:
                        await asyncio.sleep(2 ** attempt * 10)
                        continue
                    resp.raise_for_status()
                    data = resp.json()

                    raw = base64.b64decode(data["data"])
                    npz_data = np.load(io.BytesIO(raw))
                    key = f"{layer}.output"
                    emb = npz_data[key]
                    # Handle both tensor shapes:
                    # Cloud 40B: (1, seq_len, 8192)
                    # Self-hosted 7B: (seq_len, 1, 4096)
                    if emb.ndim == 3:
                        if emb.shape[0] == 1:
                            emb = emb.squeeze(0)
                        elif emb.shape[1] == 1:
                            emb = emb.squeeze(1)

                    metrics = compute_window_metrics(emb)

                    # Add to accumulators
                    wlen = end - start
                    cos1_sum[start:end] += metrics["cos1"][:wlen]
                    cos3_sum[start:end] += metrics["cos3"][:wlen]
                    norm_sum[start:end] += metrics["norms"][:wlen]
                    count[start:end] += 1

                    completed += 1
                    if completed % 20 == 0:
                        print(f"  [{completed}/{n_windows}] {start:,}-{end:,}")
                    return

                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (429, 503):
                        await asyncio.sleep(2 ** attempt * 10)
                        continue
                    if attempt < 4:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    failed += 1
                    print(f"  Window {win_idx} ({start:,}): HTTP {e.response.status_code}")
                    return
                except Exception as e:
                    if attempt < 4:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    failed += 1
                    print(f"  Window {win_idx} ({start:,}): {str(e)[:50]}")
                    return

    async with httpx.AsyncClient() as client:
        tasks = []
        for i, start in enumerate(range(0, genome_len - WINDOW_SIZE + 1, STEP_SIZE)):
            tasks.append(process_window(client, i, start))

        # Also add the final window to cover the end
        last_start = genome_len - WINDOW_SIZE
        if last_start > (n_windows - 1) * STEP_SIZE:
            tasks.append(process_window(client, n_windows, last_start))

        await asyncio.gather(*tasks)

    print(f"\nCompleted: {completed}/{n_windows}, Failed: {failed}")

    # Average overlapping regions
    mask = count > 0
    cos1_avg = np.zeros(genome_len)
    cos3_avg = np.zeros(genome_len)
    norm_avg = np.zeros(genome_len)
    cos1_avg[mask] = cos1_sum[mask] / count[mask]
    cos3_avg[mask] = cos3_sum[mask] / count[mask]
    norm_avg[mask] = norm_sum[mask] / count[mask]

    # Save
    np.savez_compressed(cache_path,
                        cos1=cos1_avg, cos3=cos3_avg, norms=norm_avg,
                        count=count)
    print(f"Saved metrics: {cache_path} ({cache_path.stat().st_size / 1024 / 1024:.1f} MB)")

    return {"cos1": cos1_avg, "cos3": cos3_avg, "norms": norm_avg, "count": count}


def plot_genome(metrics: dict, genes: list):
    """Plot the full genome cos3-cos1 profile."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cos1 = metrics["cos1"]
    cos3 = metrics["cos3"]
    genome_len = len(cos1)

    # Smooth
    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    cos1_smooth = np.convolve(cos1, kernel, mode="same")
    cos3_smooth = np.convolve(cos3, kernel, mode="same")
    inversion = cos3_smooth - cos1_smooth

    # Gene density (coding fraction in 1kb windows)
    gene_density = np.zeros(genome_len)
    for g in genes:
        gene_density[g["start"]:g["end"]] = 1
    density_smooth = np.convolve(gene_density, np.ones(1000) / 1000, mode="same")

    # ── Plot ──
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1, 0.5], "hspace": 0.08})

    x = np.arange(genome_len) / 1e6  # Mb

    # Panel 1: cos3 vs cos1
    ax1 = axes[0]
    ax1.plot(x, cos1_smooth, color="#C62828", linewidth=0.3, alpha=0.6, label="cos(offset-1)")
    ax1.plot(x, cos3_smooth, color="#1565C0", linewidth=0.3, alpha=0.6, label="cos(offset-3)")
    ax1.fill_between(x, cos1_smooth, cos3_smooth, where=cos3_smooth > cos1_smooth,
                      alpha=0.15, color="#1565C0")
    ax1.fill_between(x, cos3_smooth, cos1_smooth, where=cos1_smooth > cos3_smooth,
                      alpha=0.15, color="#C62828")
    ax1.set_ylabel("Cosine similarity\n(200bp smoothed)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("E. coli K12 MG1655 — Full genome offset-3 cosine inversion profile",
                   fontsize=11, fontweight="bold")

    # Panel 2: Inversion signal (cos3 - cos1)
    ax2 = axes[1]
    ax2.fill_between(x, 0, inversion, where=inversion > 0, alpha=0.5, color="#1565C0", label="Coding (cos3 > cos1)")
    ax2.fill_between(x, 0, inversion, where=inversion < 0, alpha=0.5, color="#C62828", label="Non-coding")
    ax2.axhline(0, color="black", linewidth=0.3)
    ax2.set_ylabel("Inversion signal\n(cos3 - cos1)")
    ax2.legend(loc="upper right", fontsize=7)

    # Panel 3: Gene density
    ax3 = axes[2]
    ax3.fill_between(x, 0, density_smooth, color="#4CAF50", alpha=0.5)
    ax3.set_ylabel("Gene\ndensity")
    ax3.set_xlabel("Genome position (Mb)")
    ax3.set_ylim(0, 1.1)

    # Mark known features
    features = {
        0.255: "oriC", 2.3: "terminus",
        0.265: "rrnH", 0.997: "rrnG", 2.729: "rrnD",
        3.427: "rrnC", 3.938: "rrnA", 3.945: "rrnB", 4.166: "rrnE",
    }
    for pos, name in features.items():
        if "rrn" in name:
            ax2.axvline(pos, color="#FF9800", linewidth=0.5, alpha=0.5)
        elif name == "oriC":
            ax2.axvline(pos, color="#9C27B0", linewidth=1, alpha=0.7)
            ax2.text(pos + 0.02, ax2.get_ylim()[1] * 0.8, name, fontsize=6, color="#9C27B0")

    plt.savefig(OUT_DIR / "ecoli_k12_genome_scan.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.savefig(OUT_DIR / "ecoli_k12_genome_scan.pdf", bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"Saved to {OUT_DIR}/ecoli_k12_genome_scan.png")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nim-url", type=str, default=None,
                        help="Self-hosted NIM URL (e.g., http://localhost:8000)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Download genome ===")
    seq = download_genome()

    print("\n=== Step 2: Download gene annotations ===")
    genes = download_gene_annotations()

    layer_name = "decoder.layers.10" if args.nim_url else LAYER
    print(f"\n=== Step 3: Extract per-position embeddings ({layer_name}) ===")
    metrics = asyncio.run(extract_all_windows(seq, nim_url=args.nim_url))

    print("\n=== Step 4: Plot ===")
    plot_genome(metrics, genes)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
