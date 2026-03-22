#!/usr/bin/env python3
"""
Non-model organism gene annotation: Zymoseptoria tritici (wheat pathogen fungus).

Tests whether the offset-3 cosine inversion can detect genes in a non-model
organism without any species-specific training.

Downloads a region of Zt chromosome 1, extracts per-position embeddings,
computes inversion signal, and compares to NCBI gene annotations.

Usage:
    # Test on 100kb region (cloud NIM, ~10 min):
    NVIDIA_API_KEY=... uv run python scripts/nonmodel_genome_scan.py --region 100000

    # Full chromosome 1 (HTCF, ~6 hours):
    uv run python scripts/nonmodel_genome_scan.py --nim-url http://localhost:8000
"""

import argparse
import asyncio
import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT_DIR = Path("results/experiments/nonmodel_genome")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GENOME_ACC = "NC_018218"  # Zt chr1
GENOME_LEN = 6088797


def download_region(start: int, end: int):
    """Download a genomic region from NCBI."""
    from Bio import Entrez, SeqIO
    from io import StringIO

    Entrez.email = "shandley@wustl.edu"

    fasta_path = OUT_DIR / f"zt_chr1_{start}_{end}.fasta"
    ann_path = OUT_DIR / f"zt_chr1_{start}_{end}_genes.json"

    # Download sequence
    if not fasta_path.exists():
        print(f"Downloading {GENOME_ACC}:{start}-{end}...")
        handle = Entrez.efetch(db="nucleotide", id=GENOME_ACC, rettype="fasta", retmode="text",
                               seq_start=start, seq_stop=end)
        record = next(SeqIO.parse(StringIO(handle.read()), "fasta"))
        handle.close()
        seq = str(record.seq).upper()
        with open(fasta_path, "w") as f:
            f.write(f">zt_chr1_{start}_{end} Zymoseptoria tritici chr1 [{len(seq)}bp]\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")
        print(f"  Saved {len(seq):,} bp")
    else:
        with open(fasta_path) as f:
            seq = "".join(l.strip() for l in f if not l.startswith(">"))
        print(f"Cached: {len(seq):,} bp")

    # Download annotations
    if not ann_path.exists():
        print("Downloading gene annotations...")
        handle = Entrez.efetch(db="nucleotide", id=GENOME_ACC, rettype="gb", retmode="text",
                               seq_start=start, seq_stop=end)
        from Bio import SeqIO as SeqIO2
        record = next(SeqIO2.parse(StringIO(handle.read()), "genbank"))
        handle.close()

        genes = []
        cds_regions = []
        for feat in record.features:
            if feat.type == "gene":
                name = feat.qualifiers.get("gene", feat.qualifiers.get("locus_tag", ["?"]))[0]
                genes.append({
                    "name": name, "start": int(feat.location.start),
                    "end": int(feat.location.end),
                    "strand": "+" if feat.location.strand == 1 else "-"
                })
            if feat.type == "CDS":
                for part in feat.location.parts:
                    cds_regions.append({"start": int(part.start), "end": int(part.end)})
            if feat.type == "mRNA":
                for part in feat.location.parts:
                    cds_regions.append({"start": int(part.start), "end": int(part.end)})

        with open(ann_path, "w") as f:
            json.dump({"genes": genes, "cds": cds_regions, "region_start": start, "region_end": end}, f, indent=2)
        print(f"  {len(genes)} genes, {len(cds_regions)} CDS parts")
    else:
        with open(ann_path) as f:
            ann = json.load(f)
        print(f"Cached: {len(ann['genes'])} genes")

    return seq, ann_path


async def extract_and_analyze(seq: str, nim_url: str | None = None):
    """Extract per-position embeddings and compute inversion signal."""
    import httpx

    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if nim_url:
        url = f"{nim_url.rstrip('/')}/biology/arc/evo2/forward"
        layer = "decoder.layers.10"
    else:
        url = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward"
        layer = "blocks.10"

    window_size = 16000
    step_size = 12000
    seq_len = len(seq)

    cos1_sum = np.zeros(seq_len, dtype=np.float64)
    cos3_sum = np.zeros(seq_len, dtype=np.float64)
    count = np.zeros(seq_len, dtype=np.float64)

    n_windows = (seq_len - window_size) // step_size + 2
    completed = 0

    headers = {"Content-Type": "application/json"}
    if not nim_url:
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient() as client:
        for start in range(0, seq_len - window_size + 1, step_size):
            end = min(start + window_size, seq_len)
            window_seq = seq[start:end]

            payload = {"sequence": window_seq, "output_layers": [layer]}

            for attempt in range(5):
                try:
                    resp = await client.post(url, json=payload, headers=headers,
                                             timeout=600, follow_redirects=True)
                    if resp.status_code == 429:
                        await asyncio.sleep(2 ** attempt * 10)
                        continue
                    resp.raise_for_status()
                    data = resp.json()

                    raw = base64.b64decode(data["data"])
                    npz = np.load(io.BytesIO(raw))
                    emb = npz[f"{layer}.output"]
                    if emb.ndim == 3:
                        if emb.shape[0] == 1: emb = emb.squeeze(0)
                        elif emb.shape[1] == 1: emb = emb.squeeze(1)

                    # Compute per-position cosines
                    norms = np.linalg.norm(emb, axis=1)
                    wlen = end - start
                    c1 = np.zeros(wlen)
                    c3 = np.zeros(wlen)
                    for i in range(wlen - 1):
                        ni = norms[i]; ni1 = norms[i+1]
                        if ni > 0 and ni1 > 0:
                            c1[i] = np.dot(emb[i], emb[i+1]) / (ni * ni1)
                    for i in range(wlen - 3):
                        ni = norms[i]; ni3 = norms[i+3]
                        if ni > 0 and ni3 > 0:
                            c3[i] = np.dot(emb[i], emb[i+3]) / (ni * ni3)

                    cos1_sum[start:end] += c1[:wlen]
                    cos3_sum[start:end] += c3[:wlen]
                    count[start:end] += 1

                    completed += 1
                    if completed % 5 == 0:
                        print(f"  [{completed}/{n_windows}] {start:,}-{end:,}")
                    break
                except Exception as e:
                    if attempt < 4:
                        await asyncio.sleep(2 ** attempt * 5)
                    else:
                        print(f"  Window {start}: FAILED ({str(e)[:50]})")

    mask = count > 0
    cos1_avg = np.zeros(seq_len)
    cos3_avg = np.zeros(seq_len)
    cos1_avg[mask] = cos1_sum[mask] / count[mask]
    cos3_avg[mask] = cos3_sum[mask] / count[mask]

    return cos1_avg, cos3_avg


def plot_and_compare(cos1, cos3, ann_path, seq_len, region_start):
    """Plot inversion signal and compare to gene annotations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(ann_path) as f:
        ann = json.load(f)

    # Smooth
    window = 200
    kernel = np.ones(window) / window
    cos1_s = np.convolve(cos1, kernel, mode="same")
    cos3_s = np.convolve(cos3, kernel, mode="same")
    inversion = cos3_s - cos1_s

    # Gene density
    gene_track = np.zeros(seq_len)
    for cds in ann.get("cds", []):
        s = max(0, cds["start"])
        e = min(cds["end"], seq_len)
        gene_track[s:e] = 1
    gene_density = np.convolve(gene_track, np.ones(1000) / 1000, mode="same")

    # Quantify
    truth = gene_track
    predicted = (inversion > 0).astype(int)
    coding_fraction = truth.mean()

    if coding_fraction > 0.01:
        tp = ((predicted == 1) & (truth == 1)).sum()
        fp = ((predicted == 1) & (truth == 0)).sum()
        fn = ((predicted == 0) & (truth == 1)).sum()
        accuracy = (predicted == truth).sum() / seq_len
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"\nQuantification:")
        print(f"  Coding fraction: {coding_fraction:.1%}")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  Precision: {precision:.1%}")
        print(f"  Recall: {recall:.1%}")
        print(f"  F1: {f1:.3f}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1, 0.5], "hspace": 0.08})

    x = np.arange(seq_len) / 1000  # kb

    ax1 = axes[0]
    ax1.plot(x, cos1_s, color="#C62828", linewidth=0.3, alpha=0.6, label="cos(offset-1)")
    ax1.plot(x, cos3_s, color="#1565C0", linewidth=0.3, alpha=0.6, label="cos(offset-3)")
    ax1.fill_between(x, cos1_s, cos3_s, where=cos3_s > cos1_s, alpha=0.15, color="#1565C0")
    ax1.set_ylabel("Cosine similarity")
    ax1.legend(loc="upper right", fontsize=7)
    ax1.set_title(f"Zymoseptoria tritici chr1 — gene structure from unsupervised embedding analysis",
                   fontsize=10, fontweight="bold")

    ax2 = axes[1]
    ax2.fill_between(x, 0, inversion, where=inversion > 0, alpha=0.6, color="#1565C0", label="Predicted coding")
    ax2.fill_between(x, 0, inversion, where=inversion < 0, alpha=0.4, color="#C62828", label="Predicted non-coding")
    ax2.axhline(0, color="black", linewidth=0.3)
    ax2.set_ylabel("cos3 - cos1")
    ax2.legend(loc="upper right", fontsize=7)

    ax3 = axes[2]
    ax3.fill_between(x, 0, gene_density, color="#4CAF50", alpha=0.6, label="NCBI gene annotation")
    ax3.set_ylabel("Gene density")
    ax3.set_xlabel("Position (kb)")
    ax3.set_ylim(0, 1.1)
    ax3.legend(loc="upper right", fontsize=7)

    plt.savefig(OUT_DIR / "zt_chr1_scan.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_DIR / "zt_chr1_scan.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved to {OUT_DIR}/zt_chr1_scan.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nim-url", type=str, default=None)
    parser.add_argument("--region", type=int, default=100000,
                        help="Region size to scan (default 100kb for testing)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start position on chromosome")
    args = parser.parse_args()

    end = min(args.start + args.region, GENOME_LEN)

    print(f"=== Zymoseptoria tritici chr1 scan ===")
    print(f"Region: {args.start:,}-{end:,} ({(end-args.start)/1000:.0f} kb)")

    seq, ann_path = download_region(args.start, end)

    print(f"\nExtracting embeddings...")
    cos1, cos3 = asyncio.run(extract_and_analyze(seq, nim_url=args.nim_url))

    print(f"\nPlotting...")
    plot_and_compare(cos1, cos3, ann_path, len(seq), args.start)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
