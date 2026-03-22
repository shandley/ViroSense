#!/usr/bin/env python3
"""
Paper 1, Figure 2: Gene Structure Detection Across All Life

Panels:
A: Human HBB — gene structure diagram + inversion signal (star example)
B: Cross-kingdom montage — 3 diverse species (Drosophila, Arabidopsis, yeast)
C: Precision vs recall per gene, colored by kingdom (36 genes)
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

OUT_DIR = Path("results/paper1/figures")
DATA_DIR = Path("results/experiments/exon_intron")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "savefig.dpi": 300,
    "axes.linewidth": 0.5,
})

EXON_COLOR = "#1565C0"
INTRON_COLOR = "#C62828"
GENE_BAR_COLOR = "#4CAF50"

KINGDOM_COLORS = {
    "Mammalia": "#2196F3", "Insecta": "#8BC34A", "Nematoda": "#CDDC39",
    "Aves": "#03A9F4", "Fish": "#009688", "Amphibia": "#4CAF50",
    "Plantae": "#FF9800", "Fungi": "#9C27B0", "Protista": "#F44336",
}

KINGDOM_MAP = {
    "H. sapiens": "Mammalia", "M. musculus": "Mammalia", "D. rerio": "Fish",
    "G. gallus": "Aves", "X. tropicalis": "Amphibia", "D. melanogaster": "Insecta",
    "C. elegans": "Nematoda", "A. thaliana": "Plantae", "O. sativa": "Plantae",
    "Z. mays": "Plantae", "S. cerevisiae": "Fungi", "N. crassa": "Fungi",
    "T. gondii": "Protista",
}


def load_annotations():
    """Load all gene annotations."""
    ann = {}
    for ann_file in [DATA_DIR / "annotations_all.json", DATA_DIR / "annotations_fixed.json"]:
        if ann_file.exists():
            with open(ann_file) as f:
                ann.update(json.load(f))
    return ann


def load_gene_metrics(name):
    """Load per-position cosine data for a gene."""
    with open(DATA_DIR / "metrics" / f"{name}_perpos.json") as f:
        return json.load(f)


def compute_inversion(data, window=100):
    """Compute smoothed inversion signal."""
    cos1 = np.array(data["cos1"])
    cos3 = np.array(data["cos3"])
    kernel = np.ones(window) / window
    cos1_s = np.convolve(cos1, kernel, mode="same")
    cos3_s = np.convolve(cos3, kernel, mode="same")
    return cos1_s, cos3_s, cos3_s - cos1_s


def main():
    ann = load_annotations()

    fig = plt.figure(figsize=(7.2, 9.0))
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[0.4, 1.0, 1.0, 1.2],
                           hspace=0.35, wspace=0.30)

    # ══════════════════════════════════════════════════════════════
    # Panel A top: HBB gene structure diagram (schematic)
    # ══════════════════════════════════════════════════════════════
    ax_gene = fig.add_subplot(gs[0, :])
    ax_gene.axis("off")
    ax_gene.set_xlim(0, 1609)
    ax_gene.set_ylim(-0.5, 1.5)
    ax_gene.set_title("A   Human beta-globin (HBB): exon-intron detection from embedding geometry",
                       loc="left", fontweight="bold", fontsize=8)

    regions = ann.get("human_HBB", {}).get("cds", [])
    if not regions:
        regions = [{"start": 51, "end": 193}, {"start": 323, "end": 493}, {"start": 1349, "end": 1475}]

    # Draw intron line
    gene_start = regions[0]["start"]
    gene_end = regions[-1]["end"]
    ax_gene.plot([gene_start, gene_end], [0.5, 0.5], color="#999999", linewidth=1, zorder=1)

    # Draw exons as thick boxes
    for i, r in enumerate(regions):
        rect = Rectangle((r["start"], 0.15), r["end"] - r["start"], 0.7,
                          facecolor=GENE_BAR_COLOR, edgecolor="black", linewidth=0.8, zorder=3)
        ax_gene.add_patch(rect)
        mid = (r["start"] + r["end"]) / 2
        ax_gene.text(mid, -0.15, f"Exon {i+1}\n({r['end']-r['start']}bp)",
                     ha="center", va="top", fontsize=6, fontweight="bold")

    # Label introns
    for i in range(len(regions) - 1):
        mid = (regions[i]["end"] + regions[i+1]["start"]) / 2
        size = regions[i+1]["start"] - regions[i]["end"]
        ax_gene.text(mid, 1.1, f"Intron {i+1} ({size}bp)",
                     ha="center", fontsize=5.5, color="#888888", fontstyle="italic")

    # UTR labels
    ax_gene.text(25, 0.5, "5' UTR", ha="center", fontsize=5, color="#AAAAAA")
    ax_gene.text(1550, 0.5, "3' UTR", ha="center", fontsize=5, color="#AAAAAA")

    # ══════════════════════════════════════════════════════════════
    # Panel A bottom: HBB inversion signal
    # ══════════════════════════════════════════════════════════════
    ax_hbb = fig.add_subplot(gs[1, :])

    data = load_gene_metrics("human_HBB")
    cos1_s, cos3_s, inversion = compute_inversion(data)
    seq_len = data["length"]
    x = np.arange(seq_len)

    ax_hbb.fill_between(x, 0, inversion, where=inversion > 0,
                         alpha=0.7, color=EXON_COLOR, linewidth=0, label="Exon (cos3 > cos1)")
    ax_hbb.fill_between(x, 0, inversion, where=inversion <= 0,
                         alpha=0.5, color=INTRON_COLOR, linewidth=0, label="Intron/UTR (cos1 > cos3)")
    ax_hbb.axhline(0, color="black", linewidth=0.3)

    # Mark exon positions with subtle shading
    for r in regions:
        ax_hbb.axvspan(r["start"], r["end"], alpha=0.06, color=GENE_BAR_COLOR, zorder=0)

    ax_hbb.set_xlabel("Position in HBB gene (bp)")
    ax_hbb.set_ylabel("Inversion signal\n(cos3 - cos1)")
    ax_hbb.legend(loc="upper right", fontsize=6, framealpha=0.9)
    ax_hbb.set_xlim(0, seq_len)

    # ══════════════════════════════════════════════════════════════
    # Panel B: Cross-kingdom montage (3 species)
    # ══════════════════════════════════════════════════════════════
    montage = [
        ("drosophila_eve", "D. melanogaster — even-skipped (2 exons, insect)"),
        ("arabidopsis_AG", "A. thaliana — AGAMOUS (7 exons, plant)"),
        ("yeast_ACT1", "S. cerevisiae — ACT1 (1 intron, fungus)"),
    ]

    for i, (gene, title) in enumerate(montage):
        ax = fig.add_subplot(gs[2, 0]) if i == 0 else (
             fig.add_subplot(gs[2, 1]) if i == 1 else
             fig.add_subplot(gs[3, 0]))

        # Recalculate — need separate axes
        if i > 0:
            ax = fig.add_subplot(gs[2, 1]) if i == 1 else fig.add_subplot(gs[3, 0])

        data = load_gene_metrics(gene)
        _, _, inv = compute_inversion(data)
        sl = data["length"]
        xx = np.arange(sl)

        ax.fill_between(xx, 0, inv, where=inv > 0, alpha=0.7, color=EXON_COLOR, linewidth=0)
        ax.fill_between(xx, 0, inv, where=inv <= 0, alpha=0.5, color=INTRON_COLOR, linewidth=0)
        ax.axhline(0, color="black", linewidth=0.3)

        # Gene annotations
        gene_ann = ann.get(gene, {})
        gene_regions = gene_ann.get("cds", []) if gene_ann.get("cds") else gene_ann.get("exons", [])
        for r in gene_regions:
            s = max(0, r["start"])
            e = min(r["end"], sl)
            ax.axvspan(s, e, alpha=0.06, color=GENE_BAR_COLOR, zorder=0)

        ax.set_ylabel("cos3 - cos1", fontsize=6)
        ax.set_xlabel("Position (bp)", fontsize=6)
        ax.set_title(title, loc="left", fontsize=6.5, fontstyle="italic")
        ax.tick_params(labelsize=5.5)
        ax.set_xlim(0, sl)

    # Panel B label
    fig.text(0.02, 0.48, "B", fontsize=11, fontweight="bold", va="top")

    # ══════════════════════════════════════════════════════════════
    # Panel C: Precision vs Recall scatter (36 genes)
    # ══════════════════════════════════════════════════════════════
    ax_c = fig.add_subplot(gs[3, 1])

    with open(DATA_DIR / "quantification_all.json") as f:
        quant = json.load(f)

    for r in quant:
        kingdom = KINGDOM_MAP.get(r["species"], "Other")
        color = KINGDOM_COLORS.get(kingdom, "#666666")
        ax_c.scatter(r["recall"] * 100, r["precision"] * 100,
                     c=color, s=25, alpha=0.7, edgecolors="black", linewidth=0.3, zorder=3)

    ax_c.set_xlabel("Recall (%)")
    ax_c.set_ylabel("Precision (%)")
    ax_c.set_title("C   Precision vs recall (36 genes, 13 species)",
                    loc="left", fontweight="bold", fontsize=8)
    ax_c.set_xlim(82, 102)
    ax_c.set_ylim(20, 100)

    # Mean lines
    mean_rec = np.mean([r["recall"] for r in quant]) * 100
    mean_prec = np.mean([r["precision"] for r in quant]) * 100
    ax_c.axvline(mean_rec, color="#333", linewidth=0.5, linestyle="--", alpha=0.4)
    ax_c.axhline(mean_prec, color="#333", linewidth=0.5, linestyle="--", alpha=0.4)
    ax_c.text(83, mean_prec + 1, f"mean prec: {mean_prec:.0f}%", fontsize=5, color="#666")
    ax_c.text(mean_rec + 0.3, 22, f"mean rec:\n{mean_rec:.0f}%", fontsize=5, color="#666")

    # Kingdom legend
    from matplotlib.lines import Line2D
    legend_elements = []
    kingdoms_present = set()
    for r in quant:
        k = KINGDOM_MAP.get(r["species"], "Other")
        if k not in kingdoms_present:
            kingdoms_present.add(k)
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=KINGDOM_COLORS.get(k, "#666"),
                                          markersize=5, label=k))
    ax_c.legend(handles=legend_elements, loc="lower left", fontsize=5, ncol=2,
                framealpha=0.9, handletextpad=0.3, columnspacing=0.5)

    plt.savefig(OUT_DIR / "fig2.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_DIR / "fig2.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved to {OUT_DIR}/fig2.png and .pdf")


if __name__ == "__main__":
    main()
