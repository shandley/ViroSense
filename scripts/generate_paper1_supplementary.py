#!/usr/bin/env python3
"""
Paper 1 Supplementary Figures S1, S2, S3, S6

S1: Layer profiling — periodicity signal across Evo2 blocks
S2: Comprehensive validation — inversion by domain, phylum, gene family
S3: Non-coding specificity — detailed breakdown by control type
S6: Smoothing window optimization for exon-intron detection
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

OUT_DIR = Path("results/paper1/supplementary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "savefig.dpi": 200,
    "axes.linewidth": 0.5,
})


def get_domain(r):
    lin = r.get("lineage", "")
    if lin.startswith("Archaea"): return "Archaea"
    elif lin.startswith("Bacteria"): return "Bacteria"
    elif lin.startswith("Virus"): return "Virus"
    elif "mitochondrial" in lin or "chloroplast" in lin or "apicoplast" in lin: return "Organellar"
    elif lin.startswith("Eukarya; Chordata"): return "Vertebrata"
    elif any(x in lin for x in ["Arthropoda", "Nematoda", "Mollusca", "Cnidaria", "Echinoderm", "Annelida", "Tardigrada"]): return "Invertebrata"
    elif any(x in lin for x in ["Streptophyt", "Liliopsida", "Gymnosperm", "Polypodiopsida", "Bryophyta"]): return "Plantae"
    elif any(x in lin for x in ["mycota", "mycetes", "Fungi", "Chytridiomy"]): return "Fungi"
    elif any(x in lin for x in ["Apicomplexa", "Euglenozoa", "Amoebozoa", "Ciliophora", "Oomycota", "Rhizaria", "Haptophyta"]): return "Protista"
    elif any(x in lin for x in ["Chlorophyta", "Rhodophyta", "Bacillarioph", "Phaeophycea", "Dinophyceae"]): return "Algae"
    return "Other"


def generate_s1():
    """S1: Layer profiling — signal strength across Evo2 blocks."""
    print("Generating S1: Layer profiling...")

    # Data from docs/nim_api_layer_investigation.md
    # Tested on 485bp E. coli lacZ CDS
    layers = [0, 5, 10, 15, 20]
    lag3 = [-0.052, 0.554, 0.579, 0.578, 0.517]
    cos_gap = [-0.002, 0.170, 0.231, 0.186, 0.048]
    norm_mean = [0.38, 33.4, 54.7, 69.9, 291.2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: Inversion gap by layer
    ax1.bar(range(len(layers)), cos_gap, color=["#BDBDBD" if g < 0.1 else "#1565C0" for g in cos_gap],
            edgecolor="black", linewidth=0.5, width=0.6)
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels([f"Block {l}" for l in layers])
    ax1.set_ylabel("Inversion gap (cos3 - cos1)")
    ax1.set_title("Offset-3 inversion signal by layer", fontweight="bold")
    ax1.axhline(0, color="black", linewidth=0.3)

    for i, (l, g) in enumerate(zip(layers, cos_gap)):
        ax1.text(i, g + 0.008, f"{g:+.3f}", ha="center", fontsize=7, fontweight="bold")

    ax1.text(2, 0.20, "Block 10\n(optimal)", ha="center", fontsize=8, color="#1565C0", fontweight="bold")

    # Panel 2: Norm by layer (log scale)
    ax2.bar(range(len(layers)), norm_mean, color="#FF9800", alpha=0.7,
            edgecolor="black", linewidth=0.5, width=0.6)
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels([f"Block {l}" for l in layers])
    ax2.set_ylabel("Mean embedding norm")
    ax2.set_title("Embedding norm by layer", fontweight="bold")
    ax2.set_yscale("log")

    for i, n in enumerate(norm_mean):
        ax2.text(i, n * 1.3, f"{n:.1f}", ha="center", fontsize=7)

    ax2.text(0.5, 0.95, "Late blocks (25-31): norms ~10^16\n(residual stream saturated,\nMLP output near zero)",
             transform=ax2.transAxes, fontsize=7, ha="center", va="top",
             bbox=dict(boxstyle="round", facecolor="#FFF3E0", edgecolor="#FFB74D"))

    plt.tight_layout()
    plt.savefig(OUT_DIR / "s1_layer_profiling.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_DIR / "s1_layer_profiling.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved s1_layer_profiling.png")


def generate_s2():
    """S2: Comprehensive validation breakdown."""
    print("Generating S2: Comprehensive validation...")

    with open("results/experiments/comprehensive/panel.json") as f:
        panel = json.load(f)

    emb_dir = Path("results/experiments/codon_periodicity/embeddings")
    results = []
    for entry in panel:
        mp = emb_dir / f"{entry['name']}_metrics.json"
        if not mp.exists():
            continue
        with open(mp) as f:
            metrics = json.load(f)
        fp = Path("results/experiments/codon_periodicity/fasta") / f"{entry['name']}.fasta"
        gc, seq_len = 0, 0
        if fp.exists():
            with open(fp) as f2:
                seq = "".join(l.strip() for l in f2 if not l.startswith(">"))
            gc = sum(1 for c in seq.upper() if c in "GC") / max(len(seq), 1) * 100
            seq_len = len(seq)
        results.append({**entry, **metrics, "gc_content": round(gc, 1), "seq_len": seq_len})

    coding = [r for r in results if not r.get("noncoding") and not r.get("category", "").startswith("noncoding")]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: By domain
    ax = axes[0, 0]
    by_domain = defaultdict(list)
    for r in coding:
        by_domain[get_domain(r)].append(r)

    domains = sorted(by_domain.keys(), key=lambda d: -len(by_domain[d]))
    inv_rates = [100 * sum(1 for r in by_domain[d] if r.get("offset3_inversion")) / len(by_domain[d]) for d in domains]
    ns = [len(by_domain[d]) for d in domains]

    ax.barh(range(len(domains)), inv_rates, color="#1565C0", alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels([f"{d} (N={n})" for d, n in zip(domains, ns)], fontsize=7)
    ax.set_xlabel("Inversion detected (%)")
    ax.set_title("By domain", fontweight="bold")
    ax.set_xlim(80, 102)
    ax.invert_yaxis()

    # Panel 2: By gene family (Component B)
    ax = axes[0, 1]
    comp_b = [r for r in coding if r.get("component") == "B"]
    by_family = defaultdict(list)
    for r in comp_b:
        by_family[r.get("gene_family", r.get("category", "?"))].append(r)

    families = sorted(by_family.keys())
    fam_rates = [100 * sum(1 for r in by_family[f] if r.get("offset3_inversion")) / len(by_family[f]) for f in families]
    fam_ns = [len(by_family[f]) for f in families]

    ax.barh(range(len(families)), fam_rates, color="#4CAF50", alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels([f"{f.replace('family_','')} (N={n})" for f, n in zip(families, fam_ns)], fontsize=7)
    ax.set_xlabel("Inversion detected (%)")
    ax.set_title("By gene family (Component B)", fontweight="bold")
    ax.set_xlim(90, 102)
    ax.invert_yaxis()

    # Panel 3: By GC bin
    ax = axes[1, 0]
    gc_bins = [(0, 25), (25, 35), (35, 45), (45, 55), (55, 65), (65, 80)]
    gc_rates = []
    gc_labels = []
    gc_ns = []
    for lo, hi in gc_bins:
        seqs = [r for r in coding if lo <= r.get("gc_content", 0) < hi]
        if seqs:
            inv = sum(1 for r in seqs if r.get("offset3_inversion"))
            gc_rates.append(100 * inv / len(seqs))
            gc_labels.append(f"{lo}-{hi}%")
            gc_ns.append(len(seqs))

    ax.bar(range(len(gc_labels)), gc_rates, color="#FF9800", alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(gc_labels)))
    ax.set_xticklabels(gc_labels)
    ax.set_ylabel("Inversion detected (%)")
    ax.set_title("By GC content", fontweight="bold")
    ax.set_ylim(0, 108)
    for i, (rate, n) in enumerate(zip(gc_rates, gc_ns)):
        ax.text(i, rate + 1, f"{rate:.1f}%\nN={n}", ha="center", fontsize=6)

    # Panel 4: By length
    ax = axes[1, 1]
    len_bins = [(0, 300), (300, 500), (500, 800), (800, 1200), (1200, 2500)]
    len_rates = []
    len_labels = []
    len_ns = []
    for lo, hi in len_bins:
        seqs = [r for r in coding if lo <= r.get("seq_len", 0) < hi]
        if seqs:
            inv = sum(1 for r in seqs if r.get("offset3_inversion"))
            len_rates.append(100 * inv / len(seqs))
            len_labels.append(f"{lo}-{hi}")
            len_ns.append(len(seqs))

    colors = ["#FFCDD2" if r < 95 else "#C8E6C9" for r in len_rates]
    ax.bar(range(len(len_labels)), len_rates, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(len_labels)))
    ax.set_xticklabels(len_labels)
    ax.set_xlabel("Sequence length (bp)")
    ax.set_ylabel("Inversion detected (%)")
    ax.set_title("By sequence length", fontweight="bold")
    ax.set_ylim(0, 108)
    for i, (rate, n) in enumerate(zip(len_rates, len_ns)):
        ax.text(i, rate + 1, f"{rate:.1f}%\nN={n}", ha="center", fontsize=6)

    plt.suptitle("Supplementary Figure S2: Comprehensive validation (459 coding sequences)",
                  fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "s2_comprehensive_validation.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_DIR / "s2_comprehensive_validation.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved s2_comprehensive_validation.png")


def generate_s3():
    """S3: Non-coding specificity by category with individual sequences."""
    print("Generating S3: Non-coding specificity...")

    with open("results/experiments/comprehensive/panel.json") as f:
        panel = json.load(f)

    emb_dir = Path("results/experiments/codon_periodicity/embeddings")
    noncoding = []
    for entry in panel:
        if not (entry.get("noncoding") or entry.get("category", "").startswith("noncoding")):
            continue
        mp = emb_dir / f"{entry['name']}_metrics.json"
        if not mp.exists():
            continue
        with open(mp) as f:
            metrics = json.load(f)
        noncoding.append({**entry, **metrics})

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by category then by inversion gap
    noncoding.sort(key=lambda r: (r.get("category", ""), r.get("inversion_gap", r["cos3"] - r["cos1"])))

    names = [r["name"].replace("a_nc_", "") for r in noncoding]
    gaps = [r.get("inversion_gap", r["cos3"] - r["cos1"]) for r in noncoding]
    cats = [r.get("category", "").replace("noncoding_", "") for r in noncoding]

    cat_colors = {
        "rRNA": "#2196F3", "lncRNA": "#4CAF50", "intron": "#9C27B0",
        "repeat": "#FF9800", "repetitive": "#FF9800",
        "tRNA": "#F44336", "intergenic": "#E91E63",
    }

    colors = [cat_colors.get(c, "#666") for c in cats]
    bars = ax.barh(range(len(names)), gaps, color=colors, alpha=0.7,
                    edgecolor="black", linewidth=0.3, height=0.7)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Inversion gap (cos3 - cos1)")
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")

    # Category legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker="s", color="w", markerfacecolor=cat_colors[c],
                              markersize=8, label=c) for c in ["rRNA", "lncRNA", "intron", "repeat", "tRNA", "intergenic"]]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7)

    ax.set_title("Supplementary Figure S3: Non-coding control specificity (30 sequences)",
                  fontweight="bold", fontsize=10)

    # Annotations
    ax.text(0.25, 0.95, "Positive = false positive\n(model incorrectly calls coding)",
            transform=ax.transAxes, fontsize=7, va="top", color="#C62828")
    ax.text(-0.05, 0.95, "Negative = correct\n(model correctly calls non-coding)",
            transform=ax.transAxes, fontsize=7, va="top", color="#388E3C")

    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "s3_noncoding_specificity.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_DIR / "s3_noncoding_specificity.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved s3_noncoding_specificity.png")


def generate_s6():
    """S6: Smoothing window optimization for exon-intron detection."""
    print("Generating S6: Smoothing optimization...")

    with open("results/experiments/exon_intron/smoothing_optimization.json") as f:
        opt = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: F1 by window size per gene
    windows = opt["window_sizes"]
    genes = opt["genes"]
    results = opt["results"]

    gene_colors = [plt.cm.Set2(i / len(genes)) for i in range(len(genes))]

    for i, gene in enumerate(genes):
        f1s = [results[gene].get(str(w), {}).get("f1", 0) for w in windows]
        ax1.plot(windows, f1s, "o-", color=gene_colors[i], markersize=4, linewidth=1,
                 label=gene.replace("human_", "").replace("_part", " pt"), alpha=0.8)

    # Mean F1
    mean_f1 = [np.mean([results[g].get(str(w), {}).get("f1", 0) for g in genes]) for w in windows]
    ax1.plot(windows, mean_f1, "s-", color="black", markersize=6, linewidth=2,
             label="Mean", zorder=10)

    ax1.set_xlabel("Smoothing window (bp)")
    ax1.set_ylabel("F1 score")
    ax1.set_title("F1 by smoothing window", fontweight="bold")
    ax1.legend(fontsize=5.5, ncol=2, loc="lower right")
    ax1.axvline(100, color="#1565C0", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.text(105, max(mean_f1) * 0.98, "optimal\n(100bp)", fontsize=7, color="#1565C0")

    # Panel 2: Precision vs Recall tradeoff at different windows
    for i, gene in enumerate(genes):
        precisions = [results[gene].get(str(w), {}).get("precision", 0) for w in windows]
        recalls = [results[gene].get(str(w), {}).get("recall", 0) for w in windows]
        ax2.plot([r * 100 for r in recalls], [p * 100 for p in precisions],
                 "o-", color=gene_colors[i], markersize=3, linewidth=0.8, alpha=0.6)

    # Mean trajectory
    mean_prec = [np.mean([results[g].get(str(w), {}).get("precision", 0) for g in genes]) * 100 for w in windows]
    mean_rec = [np.mean([results[g].get(str(w), {}).get("recall", 0) for g in genes]) * 100 for w in windows]
    ax2.plot(mean_rec, mean_prec, "s-", color="black", markersize=6, linewidth=2, zorder=10, label="Mean")

    # Label window sizes on mean trajectory
    for w, mr, mp in zip(windows, mean_rec, mean_prec):
        ax2.annotate(f"{w}bp", xy=(mr, mp), xytext=(3, 3), textcoords="offset points",
                     fontsize=5, color="black")

    ax2.set_xlabel("Recall (%)")
    ax2.set_ylabel("Precision (%)")
    ax2.set_title("Precision-recall tradeoff by window size", fontweight="bold")
    ax2.legend(fontsize=7)

    plt.suptitle("Supplementary Figure S6: Smoothing window optimization for exon-intron detection",
                  fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "s6_smoothing_optimization.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_DIR / "s6_smoothing_optimization.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved s6_smoothing_optimization.png")


if __name__ == "__main__":
    generate_s1()
    generate_s2()
    generate_s3()
    generate_s6()
    print("\nAll supplementary figures generated.")
