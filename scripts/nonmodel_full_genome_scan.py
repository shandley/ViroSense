#!/usr/bin/env python3
"""
Full genome scan for non-model organisms.

Scans entire chromosomes using overlapping windows, computes offset-3
cosine inversion, and compares to NCBI gene annotations.

Usage:
    # Dictyostelium discoideum (34 Mb, 6 chromosomes):
    uv run python scripts/nonmodel_full_genome_scan.py \
        --nim-url http://localhost:8000 --organism dictyostelium

    # Single chromosome:
    uv run python scripts/nonmodel_full_genome_scan.py \
        --nim-url http://localhost:8000 --organism dictyostelium --chr 1
"""

import argparse
import asyncio
import base64
import io
import json
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path("results/experiments/nonmodel_genome/full")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORGANISMS = {
    "dictyostelium": {
        "name": "Dictyostelium discoideum",
        "common": "Social amoeba",
        "kingdom": "Amoebozoa",
        "gc_expected": 22.5,
        "chromosomes": [
            {"name": "chr1", "accession": "NC_007087.3", "length": 4_923_596},
            {"name": "chr2", "accession": "NC_007088.5", "length": 8_470_286},
            {"name": "chr3", "accession": "NC_007089.4", "length": 5_566_838},
            {"name": "chr4", "accession": "NC_007090.3", "length": 5_703_738},
            {"name": "chr5", "accession": "NC_007091.4", "length": 5_162_065},
            {"name": "chr6", "accession": "NC_007092.4", "length": 4_074_406},
        ],
    },
}


def download_chromosome(org_key: str, chrom: dict) -> tuple[str, Path]:
    """Download full chromosome sequence and annotations from NCBI."""
    from Bio import Entrez, SeqIO
    from io import StringIO

    Entrez.email = "shandley@wustl.edu"

    acc = chrom["accession"]
    chr_name = chrom["name"]
    org = ORGANISMS[org_key]

    chr_dir = OUT_DIR / org_key
    chr_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = chr_dir / f"{chr_name}.fasta"
    ann_path = chr_dir / f"{chr_name}_genes.json"

    # Download sequence
    if not fasta_path.exists():
        print(f"  Downloading {acc} ({chr_name}, ~{chrom['length']/1e6:.1f} Mb)...")
        handle = Entrez.efetch(db="nucleotide", id=acc, rettype="fasta", retmode="text")
        record = next(SeqIO.parse(StringIO(handle.read()), "fasta"))
        handle.close()
        seq = str(record.seq).upper()
        with open(fasta_path, "w") as f:
            f.write(f">{org_key}_{chr_name} {org['name']} {chr_name} [{len(seq)}bp]\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i + 60] + "\n")
        print(f"    Saved {len(seq):,} bp")
    else:
        with open(fasta_path) as f:
            seq = "".join(l.strip() for l in f if not l.startswith(">"))
        print(f"  Cached: {chr_name} {len(seq):,} bp")

    # Download annotations via GFF3 (GenBank format unreliable for full chromosomes)
    if not ann_path.exists():
        print(f"  Downloading {chr_name} annotations (GFF3)...")
        import urllib.request

        # NCBI provides GFF3 annotations via datasets or direct efetch
        # Use paginated GenBank in 500kb chunks for reliability
        genes = []
        cds_regions = []
        chunk_size = 500_000
        chr_len = len(seq)

        for chunk_start in range(0, chr_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, chr_len)
            for attempt in range(3):
                try:
                    handle = Entrez.efetch(
                        db="nucleotide", id=acc, rettype="gb", retmode="text",
                        seq_start=chunk_start + 1, seq_stop=chunk_end,
                    )
                    record = next(SeqIO.parse(StringIO(handle.read()), "genbank"))
                    handle.close()

                    for feat in record.features:
                        if feat.type == "gene":
                            feat_name = feat.qualifiers.get(
                                "gene", feat.qualifiers.get("locus_tag", ["?"])
                            )[0]
                            genes.append({
                                "name": feat_name,
                                "start": int(feat.location.start) + chunk_start,
                                "end": int(feat.location.end) + chunk_start,
                                "strand": "+" if feat.location.strand == 1 else "-",
                            })
                        if feat.type == "CDS":
                            for part in feat.location.parts:
                                cds_regions.append({
                                    "start": int(part.start) + chunk_start,
                                    "end": int(part.end) + chunk_start,
                                })
                    break
                except Exception as e:
                    if attempt < 2:
                        import time as _time
                        _time.sleep(5)
                    else:
                        print(f"    Chunk {chunk_start}-{chunk_end}: annotation failed ({e})")

            if chunk_start % 2_000_000 == 0 and chunk_start > 0:
                print(f"    ... {chunk_start/1e6:.1f} Mb annotated ({len(genes)} genes so far)")

        with open(ann_path, "w") as f:
            json.dump({"genes": genes, "cds": cds_regions}, f, indent=2)
        print(f"    {len(genes)} genes, {len(cds_regions)} CDS parts")
    else:
        with open(ann_path) as f:
            ann = json.load(f)
        print(f"  Cached: {chr_name} {len(ann['genes'])} genes")

    return seq, ann_path


async def extract_chromosome_cosines(
    seq: str, nim_url: str, chr_name: str, cache_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-position cosines for an entire chromosome."""
    import httpx

    if cache_path.exists():
        print(f"  Loading cached cosines for {chr_name}...")
        cached = np.load(cache_path)
        return cached["cos1"], cached["cos3"]

    url = f"{nim_url.rstrip('/')}/biology/arc/evo2/forward"
    layer = "decoder.layers.10"

    window_size = 16000
    step_size = 12000
    seq_len = len(seq)

    cos1_sum = np.zeros(seq_len, dtype=np.float64)
    cos3_sum = np.zeros(seq_len, dtype=np.float64)
    count = np.zeros(seq_len, dtype=np.float64)

    n_windows = max(1, (seq_len - window_size) // step_size + 2)
    completed = 0
    start_time = time.time()

    headers = {"Content-Type": "application/json"}

    async with httpx.AsyncClient() as client:
        for wstart in range(0, seq_len - window_size + 1, step_size):
            wend = min(wstart + window_size, seq_len)
            window_seq = seq[wstart:wend]
            payload = {"sequence": window_seq, "output_layers": [layer]}

            for attempt in range(5):
                try:
                    resp = await client.post(
                        url, json=payload, headers=headers,
                        timeout=600, follow_redirects=True,
                    )
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

                    norms = np.linalg.norm(emb, axis=1)
                    wlen = wend - wstart
                    c1 = np.zeros(wlen)
                    c3 = np.zeros(wlen)
                    for i in range(wlen - 1):
                        ni, ni1 = norms[i], norms[i + 1]
                        if ni > 0 and ni1 > 0:
                            c1[i] = np.dot(emb[i], emb[i + 1]) / (ni * ni1)
                    for i in range(wlen - 3):
                        ni, ni3 = norms[i], norms[i + 3]
                        if ni > 0 and ni3 > 0:
                            c3[i] = np.dot(emb[i], emb[i + 3]) / (ni * ni3)

                    cos1_sum[wstart:wend] += c1[:wlen]
                    cos3_sum[wstart:wend] += c3[:wlen]
                    count[wstart:wend] += 1

                    completed += 1
                    if completed % 20 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed * 60
                        eta = (n_windows - completed) / (completed / elapsed)
                        print(
                            f"    [{completed}/{n_windows}] {wstart:,}-{wend:,} "
                            f"({rate:.0f} win/min, ETA {eta/60:.0f}m)"
                        )
                    break
                except Exception as e:
                    if attempt < 4:
                        await asyncio.sleep(2 ** attempt * 5)
                    else:
                        print(f"    Window {wstart}: FAILED ({str(e)[:60]})")

    mask = count > 0
    cos1_avg = np.zeros(seq_len)
    cos3_avg = np.zeros(seq_len)
    cos1_avg[mask] = cos1_sum[mask] / count[mask]
    cos3_avg[mask] = cos3_sum[mask] / count[mask]

    np.savez_compressed(cache_path, cos1=cos1_avg, cos3=cos3_avg)
    print(f"    Cached {chr_name} cosines ({cache_path.stat().st_size / 1e6:.1f} MB)")

    return cos1_avg, cos3_avg


def quantify_chromosome(cos1: np.ndarray, cos3: np.ndarray, ann_path: Path) -> dict:
    """Compute per-position coding detection metrics."""
    with open(ann_path) as f:
        ann = json.load(f)

    seq_len = len(cos1)
    window = 200
    kernel = np.ones(window) / window
    cos1_s = np.convolve(cos1, kernel, mode="same")
    cos3_s = np.convolve(cos3, kernel, mode="same")
    inversion = cos3_s - cos1_s

    gene_track = np.zeros(seq_len)
    for cds in ann.get("cds", []):
        s = max(0, cds["start"])
        e = min(cds["end"], seq_len)
        gene_track[s:e] = 1

    coding_fraction = gene_track.mean()
    predicted = (inversion > 0).astype(int)
    truth = gene_track.astype(int)

    tp = float(((predicted == 1) & (truth == 1)).sum())
    fp = float(((predicted == 1) & (truth == 0)).sum())
    fn = float(((predicted == 0) & (truth == 1)).sum())

    accuracy = float((predicted == truth).sum()) / seq_len
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "n_genes": len(ann["genes"]),
        "n_cds": len(ann.get("cds", [])),
        "coding_fraction": coding_fraction,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "seq_len": seq_len,
    }


def plot_genome_overview(org_key: str, chr_results: list[dict]):
    """Plot whole-genome summary with per-chromosome metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    org = ORGANISMS[org_key]
    out_dir = OUT_DIR / org_key

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    chr_names = [r["chr_name"] for r in chr_results]
    recalls = [r["recall"] * 100 for r in chr_results]
    f1s = [r["f1"] for r in chr_results]
    sizes = [r["seq_len"] / 1e6 for r in chr_results]

    # Panel 1: Recall by chromosome
    colors = ["#1565C0" if rec >= 80 else "#FF9800" for rec in recalls]
    bars = ax1.bar(range(len(chr_names)), recalls, color=colors, alpha=0.7,
                   edgecolor="black", linewidth=0.5, width=0.6)
    ax1.set_xticks(range(len(chr_names)))
    ax1.set_xticklabels(chr_names, fontsize=8)
    ax1.set_ylabel("Exon recall (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title("Coding region recall by chromosome", fontweight="bold")
    for bar, val, n_genes in zip(bars, recalls, [r["n_genes"] for r in chr_results]):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                 f"{val:.1f}%\n({n_genes} genes)", ha="center", fontsize=7)

    # Panel 2: Chromosome size vs recall
    for i, r in enumerate(chr_results):
        ax2.scatter(sizes[i], recalls[i], s=100, c="#1565C0", alpha=0.7,
                    edgecolors="black", linewidth=0.5, zorder=3)
        ax2.annotate(r["chr_name"], (sizes[i], recalls[i]),
                     xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax2.set_xlabel("Chromosome size (Mb)")
    ax2.set_ylabel("Exon recall (%)")
    ax2.set_title("Recall vs chromosome size", fontweight="bold")
    ax2.set_ylim(0, 105)

    # Genome-wide stats
    total_genes = sum(r["n_genes"] for r in chr_results)
    total_bp = sum(r["seq_len"] for r in chr_results)
    total_tp = sum(r["recall"] * r["coding_fraction"] * r["seq_len"] for r in chr_results)
    total_coding = sum(r["coding_fraction"] * r["seq_len"] for r in chr_results)
    genome_recall = total_tp / total_coding if total_coding > 0 else 0

    fig.suptitle(
        f"{org['name']} ({org['common']}) — full genome scan\n"
        f"{total_bp/1e6:.1f} Mb, {total_genes:,} genes, "
        f"genome-wide recall: {genome_recall:.1%}",
        fontsize=11, fontweight="bold", y=1.04,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "genome_overview.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(out_dir / "genome_overview.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved genome_overview.png")


def plot_chromosome_detail(cos1: np.ndarray, cos3: np.ndarray, ann_path: Path,
                           chr_name: str, org_key: str, metrics: dict):
    """Plot detailed chromosome scan (like the Zymoseptoria figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(ann_path) as f:
        ann = json.load(f)

    seq_len = len(cos1)
    window = 200
    kernel = np.ones(window) / window
    cos1_s = np.convolve(cos1, kernel, mode="same")
    cos3_s = np.convolve(cos3, kernel, mode="same")
    inversion = cos3_s - cos1_s

    gene_track = np.zeros(seq_len)
    for cds in ann.get("cds", []):
        s = max(0, cds["start"])
        e = min(cds["end"], seq_len)
        gene_track[s:e] = 1
    gene_density = np.convolve(gene_track, np.ones(1000) / 1000, mode="same")

    fig, axes = plt.subplots(3, 1, figsize=(16, 7), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1, 0.5], "hspace": 0.08})

    x = np.arange(seq_len) / 1e6  # Mb

    axes[0].plot(x, cos1_s, color="#C62828", linewidth=0.2, alpha=0.5, label="cos(offset-1)")
    axes[0].plot(x, cos3_s, color="#1565C0", linewidth=0.2, alpha=0.5, label="cos(offset-3)")
    axes[0].fill_between(x, cos1_s, cos3_s, where=cos3_s > cos1_s, alpha=0.1, color="#1565C0")
    axes[0].set_ylabel("Cosine similarity")
    axes[0].legend(loc="upper right", fontsize=7)
    org = ORGANISMS[org_key]
    axes[0].set_title(
        f"{org['name']} {chr_name} ({seq_len/1e6:.1f} Mb) — "
        f"recall: {metrics['recall']:.1%}, {metrics['n_genes']} genes",
        fontsize=10, fontweight="bold",
    )

    axes[1].fill_between(x, 0, inversion, where=inversion > 0, alpha=0.5,
                         color="#1565C0", label="Predicted coding")
    axes[1].fill_between(x, 0, inversion, where=inversion < 0, alpha=0.3,
                         color="#C62828", label="Predicted non-coding")
    axes[1].axhline(0, color="black", linewidth=0.3)
    axes[1].set_ylabel("cos3 - cos1")
    axes[1].legend(loc="upper right", fontsize=7)

    axes[2].fill_between(x, 0, gene_density, color="#4CAF50", alpha=0.6,
                         label="NCBI gene annotation")
    axes[2].set_ylabel("Gene density")
    axes[2].set_xlabel("Position (Mb)")
    axes[2].set_ylim(0, 1.1)
    axes[2].legend(loc="upper right", fontsize=7)

    out_dir = OUT_DIR / org_key
    plt.savefig(out_dir / f"{chr_name}_scan.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig(out_dir / f"{chr_name}_scan.pdf", bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nim-url", type=str, required=True)
    parser.add_argument("--organism", type=str, required=True, choices=list(ORGANISMS.keys()))
    parser.add_argument("--chr", type=int, default=None,
                        help="Scan only this chromosome number (1-indexed)")
    args = parser.parse_args()

    org_key = args.organism
    org = ORGANISMS[org_key]
    chroms = org["chromosomes"]

    if args.chr is not None:
        chroms = [chroms[args.chr - 1]]

    print(f"{'='*70}")
    print(f"{org['name']} ({org['common']}) — full genome scan")
    print(f"{len(chroms)} chromosomes, ~{sum(c['length'] for c in chroms)/1e6:.1f} Mb total")
    print(f"{'='*70}")

    chr_results = []
    total_start = time.time()

    for chrom in chroms:
        chr_name = chrom["name"]
        print(f"\n--- {chr_name} ({chrom['length']/1e6:.1f} Mb) ---")

        # Download
        seq, ann_path = download_chromosome(org_key, chrom)

        # Extract cosines
        cache_path = OUT_DIR / org_key / f"{chr_name}_cosines.npz"
        cos1, cos3 = asyncio.run(
            extract_chromosome_cosines(seq, args.nim_url, chr_name, cache_path)
        )

        # Quantify
        metrics = quantify_chromosome(cos1, cos3, ann_path)
        metrics["chr_name"] = chr_name
        chr_results.append(metrics)

        print(f"  {chr_name}: recall={metrics['recall']:.1%}, F1={metrics['f1']:.3f}, "
              f"{metrics['n_genes']} genes, {metrics['coding_fraction']:.1%} coding")

        # Plot chromosome detail
        plot_chromosome_detail(cos1, cos3, ann_path, chr_name, org_key, metrics)
        print(f"  Saved {chr_name}_scan.png")

    # Summary
    total_elapsed = time.time() - total_start
    total_genes = sum(r["n_genes"] for r in chr_results)
    total_bp = sum(r["seq_len"] for r in chr_results)
    total_tp = sum(r["recall"] * r["coding_fraction"] * r["seq_len"] for r in chr_results)
    total_coding = sum(r["coding_fraction"] * r["seq_len"] for r in chr_results)
    genome_recall = total_tp / total_coding if total_coding > 0 else 0

    print(f"\n{'='*70}")
    print(f"GENOME SUMMARY: {org['name']}")
    print(f"{'='*70}")
    print(f"Total: {total_bp/1e6:.1f} Mb, {total_genes:,} genes")
    print(f"Genome-wide recall: {genome_recall:.1%}")
    print(f"Time: {total_elapsed/60:.0f} minutes")
    print(f"\n{'Chr':<8s} {'Size (Mb)':>10s} {'Genes':>7s} {'Coding%':>8s} "
          f"{'Recall':>8s} {'Precision':>10s} {'F1':>8s}")
    print(f"{'-'*62}")
    for r in chr_results:
        print(f"{r['chr_name']:<8s} {r['seq_len']/1e6:>10.1f} {r['n_genes']:>7d} "
              f"{r['coding_fraction']:>7.1%} {r['recall']:>7.1%} "
              f"{r['precision']:>9.1%} {r['f1']:>8.3f}")

    # Plot genome overview
    if len(chr_results) > 1:
        plot_genome_overview(org_key, chr_results)

    # Save results
    results = {
        "organism": org["name"],
        "common": org["common"],
        "kingdom": org["kingdom"],
        "gc_expected": org["gc_expected"],
        "total_bp": total_bp,
        "total_genes": total_genes,
        "genome_recall": genome_recall,
        "elapsed_minutes": total_elapsed / 60,
        "chromosomes": chr_results,
    }
    with open(OUT_DIR / org_key / "genome_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR / org_key}/genome_results.json")


if __name__ == "__main__":
    main()
