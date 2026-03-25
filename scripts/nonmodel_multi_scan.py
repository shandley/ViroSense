#!/usr/bin/env python3
"""
Multi-organism non-model genome scan.

Scans 500kb gene-rich regions from 5 diverse organisms that lack Augustus
gene finder models, testing whether the offset-3 cosine inversion detects
genes universally without any species-specific training.

Organisms:
  1. Danaus plexippus (monarch butterfly) — Arthropoda, GC 31.5%
  2. Physcomitrium patens (earthmoss) — Bryophyta, GC 33.5%
  3. Acropora millepora (staghorn coral) — Cnidaria, GC 39%
  4. Dictyostelium discoideum (social amoeba) — Amoebozoa, GC 22.5%
  5. Magallana gigas (Pacific oyster) — Mollusca, GC 33.5%

Usage:
    uv run python scripts/nonmodel_multi_scan.py --nim-url http://localhost:8000
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

# Organism configurations
ORGANISMS = [
    {
        "name": "danaus_plexippus",
        "common": "Monarch butterfly",
        "kingdom": "Arthropoda",
        "accession": "NC_045808.1",
        "start": 5_830_000,  # gene-rich: 50 genes, 13.9% coding
        "region": 500_000,
        "gc_expected": 31.5,
    },
    {
        "name": "physcomitrium_patens",
        "common": "Spreading earthmoss",
        "kingdom": "Bryophyta",
        "accession": "NC_037253.2",
        "start": 16_665_000,  # gene-rich: 42 genes, 12.4% coding
        "region": 500_000,
        "gc_expected": 33.5,
    },
    {
        "name": "acropora_millepora",
        "common": "Staghorn coral",
        "kingdom": "Cnidaria",
        "accession": "NC_058066.1",
        "start": 9_850_000,  # gene-rich: 47 genes, 18.3% coding
        "region": 500_000,
        "gc_expected": 39.0,
    },
    {
        "name": "dictyostelium_discoideum",
        "common": "Social amoeba",
        "kingdom": "Amoebozoa",
        "accession": "NC_007088.5",
        "start": 3_820_124,  # gene-rich: 203 genes, 65.9% coding
        "region": 500_000,
        "gc_expected": 22.5,
    },
    {
        "name": "magallana_gigas",
        "common": "Pacific oyster",
        "kingdom": "Mollusca",
        "accession": "NC_088853.1",
        "start": 41_855_000,  # gene-rich: 27 genes, 7.3% coding (sparse genome)
        "region": 500_000,
        "gc_expected": 33.5,
    },
]


def download_region(org: dict) -> tuple[str, Path]:
    """Download a genomic region and annotations from NCBI."""
    from Bio import Entrez, SeqIO
    from io import StringIO

    Entrez.email = "shandley@wustl.edu"

    name = org["name"]
    acc = org["accession"]
    start = org["start"]
    end = start + org["region"]

    fasta_path = OUT_DIR / f"{name}_{start}_{end}.fasta"
    ann_path = OUT_DIR / f"{name}_{start}_{end}_genes.json"

    # Download sequence
    if not fasta_path.exists():
        print(f"  Downloading {acc}:{start}-{end}...")
        handle = Entrez.efetch(
            db="nucleotide", id=acc, rettype="fasta", retmode="text",
            seq_start=start, seq_stop=end,
        )
        record = next(SeqIO.parse(StringIO(handle.read()), "fasta"))
        handle.close()
        seq = str(record.seq).upper()
        with open(fasta_path, "w") as f:
            f.write(f">{name}_{start}_{end} {org['common']} [{len(seq)}bp]\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i + 60] + "\n")
        print(f"    Saved {len(seq):,} bp")
    else:
        with open(fasta_path) as f:
            seq = "".join(l.strip() for l in f if not l.startswith(">"))
        print(f"  Cached: {len(seq):,} bp")

    # Download annotations
    if not ann_path.exists():
        print(f"  Downloading gene annotations...")
        handle = Entrez.efetch(
            db="nucleotide", id=acc, rettype="gb", retmode="text",
            seq_start=start, seq_stop=end,
        )
        record = next(SeqIO.parse(StringIO(handle.read()), "genbank"))
        handle.close()

        genes = []
        cds_regions = []
        for feat in record.features:
            if feat.type == "gene":
                feat_name = feat.qualifiers.get("gene", feat.qualifiers.get("locus_tag", ["?"]))[0]
                genes.append({
                    "name": feat_name,
                    "start": int(feat.location.start),
                    "end": int(feat.location.end),
                    "strand": "+" if feat.location.strand == 1 else "-",
                })
            if feat.type == "CDS":
                for part in feat.location.parts:
                    cds_regions.append({"start": int(part.start), "end": int(part.end)})
            if feat.type == "mRNA":
                for part in feat.location.parts:
                    cds_regions.append({"start": int(part.start), "end": int(part.end)})

        with open(ann_path, "w") as f:
            json.dump({
                "genes": genes, "cds": cds_regions,
                "region_start": start, "region_end": end,
            }, f, indent=2)
        print(f"    {len(genes)} genes, {len(cds_regions)} CDS parts")
    else:
        with open(ann_path) as f:
            ann = json.load(f)
        print(f"  Cached: {len(ann['genes'])} genes")

    return seq, ann_path


async def extract_cosines(seq: str, nim_url: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-position embeddings and compute cosine signals."""
    import httpx

    url = f"{nim_url.rstrip('/')}/biology/arc/evo2/forward"
    layer = "decoder.layers.10"

    window_size = 16000
    step_size = 12000
    seq_len = len(seq)

    cos1_sum = np.zeros(seq_len, dtype=np.float64)
    cos3_sum = np.zeros(seq_len, dtype=np.float64)
    count = np.zeros(seq_len, dtype=np.float64)

    n_windows = (seq_len - window_size) // step_size + 2
    completed = 0

    headers = {"Content-Type": "application/json"}

    async with httpx.AsyncClient() as client:
        for start in range(0, seq_len - window_size + 1, step_size):
            end = min(start + window_size, seq_len)
            window_seq = seq[start:end]
            payload = {"sequence": window_seq, "output_layers": [layer]}

            for attempt in range(5):
                try:
                    resp = await client.post(
                        url, json=payload, headers=headers,
                        timeout=600, follow_redirects=True,
                    )
                    if resp.status_code == 429 or resp.status_code == 503:
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
                    wlen = end - start
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

                    cos1_sum[start:end] += c1[:wlen]
                    cos3_sum[start:end] += c3[:wlen]
                    count[start:end] += 1

                    completed += 1
                    if completed % 10 == 0:
                        print(f"    [{completed}/{n_windows}] {start:,}-{end:,}")
                    break
                except Exception as e:
                    if attempt < 4:
                        await asyncio.sleep(2 ** attempt * 5)
                    else:
                        print(f"    Window {start}: FAILED ({str(e)[:60]})")

    mask = count > 0
    cos1_avg = np.zeros(seq_len)
    cos3_avg = np.zeros(seq_len)
    cos1_avg[mask] = cos1_sum[mask] / count[mask]
    cos3_avg[mask] = cos3_sum[mask] / count[mask]

    return cos1_avg, cos3_avg


def quantify(cos1: np.ndarray, cos3: np.ndarray, ann_path: Path, seq_len: int) -> dict:
    """Compute coding detection metrics against NCBI annotations."""
    with open(ann_path) as f:
        ann = json.load(f)

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
    tn = float(((predicted == 0) & (truth == 0)).sum())

    accuracy = float((predicted == truth).sum()) / seq_len
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Normalized metrics (comparable across coding densities)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # false positive rate
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0  # Matthews correlation

    # Per-gene detection rate: fraction of genes with >50% CDS detected
    genes_detected = 0
    genes_total = 0
    for gene in ann.get("genes", []):
        gs = max(0, gene["start"])
        ge = min(gene["end"], seq_len)
        if ge <= gs:
            continue
        # Find CDS parts within this gene
        gene_cds = np.zeros(ge - gs)
        for cds in ann.get("cds", []):
            cs = max(0, cds["start"] - gs)
            ce = min(cds["end"] - gs, ge - gs)
            if ce > cs:
                gene_cds[cs:ce] = 1
        if gene_cds.sum() < 10:  # skip genes with <10bp CDS
            continue
        gene_pred = predicted[gs:ge]
        gene_cds_mask = gene_cds.astype(bool)
        if gene_cds_mask.sum() > 0:
            gene_recall = (gene_pred[gene_cds_mask] == 1).sum() / gene_cds_mask.sum()
            genes_total += 1
            if gene_recall > 0.5:
                genes_detected += 1

    gene_detection_rate = genes_detected / genes_total if genes_total > 0 else 0

    return {
        "coding_fraction": coding_fraction,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "mcc": mcc,
        "gene_detection_rate": gene_detection_rate,
        "genes_detected": genes_detected,
        "genes_total": genes_total,
        "n_genes": len(ann["genes"]),
        "n_cds": len(ann.get("cds", [])),
    }


def plot_organism(cos1: np.ndarray, cos3: np.ndarray, ann_path: Path,
                  seq_len: int, org: dict, metrics: dict):
    """Plot single organism scan."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(ann_path) as f:
        ann = json.load(f)

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

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1, 0.5], "hspace": 0.08})

    x = np.arange(seq_len) / 1000

    ax1 = axes[0]
    ax1.plot(x, cos1_s, color="#C62828", linewidth=0.3, alpha=0.6, label="cos(offset-1)")
    ax1.plot(x, cos3_s, color="#1565C0", linewidth=0.3, alpha=0.6, label="cos(offset-3)")
    ax1.fill_between(x, cos1_s, cos3_s, where=cos3_s > cos1_s, alpha=0.15, color="#1565C0")
    ax1.set_ylabel("Cosine similarity")
    ax1.legend(loc="upper right", fontsize=7)
    ax1.set_title(
        f"{org['common']} ({org['name'].replace('_', ' ').title()}) — "
        f"unsupervised gene detection\n"
        f"Recall: {metrics['recall']:.1%} | F1: {metrics['f1']:.3f} | "
        f"{metrics['n_genes']} genes | GC: {org['gc_expected']}%",
        fontsize=10, fontweight="bold",
    )

    ax2 = axes[1]
    ax2.fill_between(x, 0, inversion, where=inversion > 0, alpha=0.6,
                     color="#1565C0", label="Predicted coding")
    ax2.fill_between(x, 0, inversion, where=inversion < 0, alpha=0.4,
                     color="#C62828", label="Predicted non-coding")
    ax2.axhline(0, color="black", linewidth=0.3)
    ax2.set_ylabel("cos3 - cos1")
    ax2.legend(loc="upper right", fontsize=7)

    ax3 = axes[2]
    ax3.fill_between(x, 0, gene_density, color="#4CAF50", alpha=0.6, label="NCBI gene annotation")
    ax3.set_ylabel("Gene density")
    ax3.set_xlabel("Position (kb)")
    ax3.set_ylim(0, 1.1)
    ax3.legend(loc="upper right", fontsize=7)

    out_name = f"{org['name']}_scan"
    plt.savefig(OUT_DIR / f"{out_name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_DIR / f"{out_name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close()


def plot_summary(all_results: list[dict]):
    """Plot combined summary figure for all organisms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = [r["common"] for r in all_results]
    kingdoms = [r["kingdom"] for r in all_results]

    kingdom_colors = {
        "Arthropoda": "#E91E63",
        "Bryophyta": "#4CAF50",
        "Cnidaria": "#FF9800",
        "Amoebozoa": "#9C27B0",
        "Mollusca": "#2196F3",
        "Fungi": "#795548",
    }
    colors = [kingdom_colors.get(k, "#666") for k in kingdoms]

    # Panel 1: Per-gene detection rate (most intuitive)
    gene_rates = [r["gene_detection_rate"] * 100 for r in all_results]
    bars = axes[0].barh(range(len(names)), gene_rates, color=colors, alpha=0.7,
                        edgecolor="black", linewidth=0.5, height=0.6)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels([f"{n}\n({k})" for n, k in zip(names, kingdoms)], fontsize=7)
    axes[0].set_xlabel("Gene detection rate (%)")
    axes[0].set_title("A   Genes detected (>50% CDS recall)", loc="left", fontweight="bold")
    axes[0].set_xlim(0, 105)
    axes[0].invert_yaxis()
    for bar, val, r in zip(bars, gene_rates, all_results):
        axes[0].text(val + 1, bar.get_y() + bar.get_height() / 2,
                     f"{val:.0f}% ({r['genes_detected']}/{r['genes_total']})",
                     va="center", fontsize=7, fontweight="bold")

    # Panel 2: MCC (class-imbalance-robust)
    mccs = [r["mcc"] for r in all_results]
    bars = axes[1].barh(range(len(names)), mccs, color=colors, alpha=0.7,
                        edgecolor="black", linewidth=0.5, height=0.6)
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Matthews Correlation Coefficient")
    axes[1].set_title("B   MCC (coding density-independent)", loc="left", fontweight="bold")
    axes[1].set_xlim(0, 1)
    axes[1].invert_yaxis()
    for bar, val in zip(bars, mccs):
        axes[1].text(val + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                     va="center", fontsize=8, fontweight="bold")

    # Panel 3: Recall vs FPR (ROC-like)
    recalls = [r["recall"] * 100 for r in all_results]
    fprs = [r["fpr"] * 100 for r in all_results]
    for i, r in enumerate(all_results):
        axes[2].scatter(fprs[i], recalls[i], c=colors[i], s=120,
                        edgecolors="black", linewidth=0.5, zorder=3)
        axes[2].annotate(r["common"], (fprs[i], recalls[i]),
                         xytext=(5, -8), textcoords="offset points", fontsize=6.5)
    axes[2].plot([0, 100], [0, 100], color="#CCC", linewidth=0.5, linestyle="--", zorder=1)
    axes[2].set_xlabel("False positive rate (%)")
    axes[2].set_ylabel("Recall / true positive rate (%)")
    axes[2].set_title("C   Recall vs FPR", loc="left", fontweight="bold")
    axes[2].set_xlim(0, 100)
    axes[2].set_ylim(0, 105)

    plt.suptitle(
        "Unsupervised gene detection across non-model organisms\n"
        "(no training, no species model, no reference genome)",
        fontsize=11, fontweight="bold", y=1.04,
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "multi_organism_summary.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_DIR / "multi_organism_summary.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved multi_organism_summary.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nim-url", type=str, required=True)
    parser.add_argument("--organisms", type=str, default="all",
                        help="Comma-separated organism indices (1-5) or 'all'")
    args = parser.parse_args()

    if args.organisms == "all":
        orgs = ORGANISMS
    else:
        indices = [int(i) - 1 for i in args.organisms.split(",")]
        orgs = [ORGANISMS[i] for i in indices]

    all_results = []

    for org in orgs:
        print(f"\n{'='*70}")
        print(f"{org['common']} ({org['name']}) — {org['kingdom']}")
        print(f"{'='*70}")

        # Download
        seq, ann_path = download_region(org)

        # Check cache
        cache_path = OUT_DIR / f"{org['name']}_{org['start']}_{org['start'] + org['region']}_cosines.npz"

        if cache_path.exists():
            print(f"  Loading cached cosines...")
            cached = np.load(cache_path)
            cos1, cos3 = cached["cos1"], cached["cos3"]
        else:
            print(f"  Extracting embeddings ({len(seq):,} bp)...")
            cos1, cos3 = asyncio.run(extract_cosines(seq, nim_url=args.nim_url))
            np.savez_compressed(cache_path, cos1=cos1, cos3=cos3)
            print(f"  Cached cosines to {cache_path}")

        # Quantify
        metrics = quantify(cos1, cos3, ann_path, len(seq))
        metrics.update({
            "name": org["name"],
            "common": org["common"],
            "kingdom": org["kingdom"],
            "gc_expected": org["gc_expected"],
        })

        print(f"\n  Results:")
        print(f"    Genes: {metrics['n_genes']}, CDS parts: {metrics['n_cds']}")
        print(f"    Coding fraction: {metrics['coding_fraction']:.1%}")
        print(f"    Recall: {metrics['recall']:.1%}  |  FPR: {metrics['fpr']:.1%}")
        print(f"    Precision: {metrics['precision']:.1%}  |  MCC: {metrics['mcc']:.3f}")
        print(f"    F1: {metrics['f1']:.3f}")
        print(f"    Gene detection: {metrics['genes_detected']}/{metrics['genes_total']} "
              f"({metrics['gene_detection_rate']:.1%})")

        # Plot individual
        plot_organism(cos1, cos3, ann_path, len(seq), org, metrics)
        print(f"  Saved {org['name']}_scan.png")

        all_results.append(metrics)

    # Save combined results
    with open(OUT_DIR / "multi_organism_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary plot
    # Include Zymoseptoria if available
    zt_cache = OUT_DIR / "zt_chr1_3000000_3500000_cosines.npz"
    zt_ann = OUT_DIR / "zt_chr1_3000000_3500000_genes.json"
    if zt_cache.exists() and zt_ann.exists():
        zt_cos = np.load(zt_cache)
        zt_fasta = OUT_DIR / "zt_chr1_3000000_3500000.fasta"
        if zt_fasta.exists():
            with open(zt_fasta) as f:
                zt_seq = "".join(l.strip() for l in f if not l.startswith(">"))
            zt_metrics = quantify(zt_cos["cos1"], zt_cos["cos3"], zt_ann, len(zt_seq))
            zt_metrics.update({
                "name": "zymoseptoria_tritici",
                "common": "Wheat pathogen fungus",
                "kingdom": "Fungi",
                "gc_expected": 52.0,
            })
            all_results.insert(0, zt_metrics)
            print(f"\n  Including Zymoseptoria tritici (cached)")

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'Organism':<25s} {'Kingdom':<13s} {'Coding%':>7s} {'Recall':>7s} "
          f"{'FPR':>6s} {'MCC':>6s} {'Genes found':>12s}")
    print(f"{'-'*80}")
    for r in all_results:
        print(f"{r['common']:<25s} {r['kingdom']:<13s} "
              f"{r['coding_fraction']:>6.1%} {r['recall']:>6.1%} "
              f"{r['fpr']:>5.1%} {r['mcc']:>6.3f} "
              f"{r['genes_detected']:>4d}/{r['genes_total']:<4d} "
              f"({r['gene_detection_rate']:.0%})")

    plot_summary(all_results)

    # Save final results with Zt
    with open(OUT_DIR / "multi_organism_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()
