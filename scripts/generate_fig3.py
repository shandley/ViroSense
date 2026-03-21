#!/usr/bin/env python3
"""
Generate Figure 3: Multi-Task DNA Analysis from Mean-Pooled Embeddings

4 panels:
A: Viral detection performance (confusion matrix or bar chart)
B: HDBSCAN clustering (category recovery summary)
C: Alignment-free phylogenomics (distance hierarchy)
D: Functional clustering negative result
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

OUT_DIR = Path("results/figures")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "savefig.dpi": 300,
    "axes.linewidth": 0.5,
})


def main():
    fig = plt.figure(figsize=(7.2, 7.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    _draw_detection(ax_a)

    ax_b = fig.add_subplot(gs[0, 1])
    _draw_clustering(ax_b)

    ax_c = fig.add_subplot(gs[1, 0])
    _draw_phylogenomics(ax_c)

    ax_d = fig.add_subplot(gs[1, 1])
    _draw_negative_clustering(ax_d)

    plt.savefig(OUT_DIR / "fig3_multitask.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_DIR / "fig3_multitask.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved to {OUT_DIR}/fig3_multitask.png and .pdf")


def _draw_detection(ax):
    """Panel A: Viral detection + contig typing performance."""
    # Binary detection metrics
    metrics = {
        "Phage\nsensitivity": 99.7,
        "RNA virus\nrecall": 93.0,
        "Chromosome\nspecificity": 99.2,
        "Plasmid\nspecificity": 81.5,
        "Overall\naccuracy": 95.4,
    }

    x = np.arange(len(metrics))
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#333333"]
    bars = ax.bar(x, list(metrics.values()), color=colors, alpha=0.7,
                   edgecolor="black", linewidth=0.3, width=0.65)

    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()), fontsize=6)
    ax.set_ylabel("Performance (%)")
    ax.set_ylim(70, 102)
    ax.set_title("A   Viral detection (40B, N=13,417)", loc="left", fontweight="bold", fontsize=8)

    # Value labels on bars
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=6, fontweight="bold")

    # Add 3-class typing annotation
    ax.text(0.95, 0.15, "3-class contig typing:\n94.5% CV accuracy\n(virus/plasmid/chr)",
            transform=ax.transAxes, fontsize=6, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", edgecolor="#CCCCCC"))


def _draw_clustering(ax):
    """Panel B: HDBSCAN clustering recovery."""
    # Cluster composition from docs/cluster_validation.md
    clusters = {
        "dsDNA phage": {"n": 4922, "purity": 99, "color": "#2196F3"},
        "RNA virus": {"n": 303, "purity": 99, "color": "#F44336"},
        "Chromosome": {"n": 1061, "purity": 88, "color": "#4CAF50"},
        "Mixed\n(plasmid+chr)": {"n": 685, "purity": 65, "color": "#FF9800"},
    }

    names = list(clusters.keys())
    sizes = [c["n"] for c in clusters.values()]
    purities = [c["purity"] for c in clusters.values()]
    colors = [c["color"] for c in clusters.values()]

    x = np.arange(len(names))
    width = 0.35

    # Size bars
    ax2 = ax.twinx()
    bars1 = ax.bar(x - width/2, purities, width, color=colors, alpha=0.7,
                    edgecolor="black", linewidth=0.3, label="Purity (%)")
    bars2 = ax2.bar(x + width/2, sizes, width, color=colors, alpha=0.3,
                     edgecolor="black", linewidth=0.3, label="Cluster size")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6)
    ax.set_ylabel("Cluster purity (%)")
    ax2.set_ylabel("Cluster size", color="#666666")
    ax.set_ylim(0, 110)
    ax.set_title("B   Unsupervised clustering (ARI=0.903)", loc="left", fontweight="bold", fontsize=8)

    # Value labels
    for bar, val in zip(bars1, purities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val}%", ha="center", va="bottom", fontsize=5.5, fontweight="bold")

    # Legend
    ax.text(0.95, 0.55, "HDBSCAN\n(no labels used)\n47% noise (expected)",
            transform=ax.transAxes, fontsize=5.5, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC"))


def _draw_phylogenomics(ax):
    """Panel C: Alignment-free phylogenomics distance hierarchy."""
    # Distance hierarchy from docs/applications_framework.md
    levels = ["Same\ngenome", "Same\ngenus", "Same\nphylum", "Different\nphylum"]
    distances = [0.069, 0.129, 0.175, 0.329]
    colors = ["#1565C0", "#42A5F5", "#90CAF9", "#E3F2FD"]

    x = np.arange(len(levels))
    bars = ax.bar(x, distances, color=colors, edgecolor="black", linewidth=0.3, width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=6.5)
    ax.set_ylabel("Mean cosine distance")
    ax.set_title("C   Alignment-free phylogenomics (r=0.504)",
                  loc="left", fontweight="bold", fontsize=8)

    # Value labels
    for bar, val in zip(bars, distances):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=6, fontweight="bold")

    # Ratio annotation
    ratio = distances[-1] / distances[0]
    ax.annotate("", xy=(0, distances[0] + 0.02), xytext=(3, distances[-1] + 0.02),
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=0.8))
    ax.text(1.5, 0.31, f"{ratio:.1f}x", ha="center", fontsize=7, fontweight="bold", color="#333333")

    ax.text(0.95, 0.95, "2,000 phage fragments\n774 source genomes\n6 host phyla",
            transform=ax.transAxes, fontsize=5.5, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC"))


def _draw_negative_clustering(ax):
    """Panel D: Functional clustering negative result."""
    # Data from functional_clustering_comparison.json
    configs = {
        "40B\nblocks.10": {"sil": -0.064, "nn": 19.9, "color": "#1565C0"},
        "40B\nblocks.28": {"sil": -0.140, "nn": 12.9, "color": "#42A5F5"},
        "7B\nlayer.10": {"sil": -0.165, "nn": 20.2, "color": "#FF9800"},
    }

    x = np.arange(len(configs))
    names = list(configs.keys())
    sils = [c["sil"] for c in configs.values()]
    nns = [c["nn"] for c in configs.values()]
    colors = [c["color"] for c in configs.values()]

    width = 0.35
    ax2 = ax.twinx()

    bars1 = ax.bar(x - width/2, sils, width, color=colors, alpha=0.7,
                    edgecolor="black", linewidth=0.3, label="Silhouette")
    bars2 = ax2.bar(x + width/2, nns, width, color=colors, alpha=0.3,
                     edgecolor="black", linewidth=0.3, label="NN accuracy (%)")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylabel("Silhouette score")
    ax2.set_ylabel("NN gene accuracy (%)", color="#666666")
    ax.axhline(0, color="black", linewidth=0.3, linestyle="--")
    ax2.axhline(10, color="#CCCCCC", linewidth=0.5, linestyle=":", label="Random (10%)")

    ax.set_title("D   No protein-identity clustering", loc="left", fontweight="bold", fontsize=8)

    # Annotation
    ax.text(0.5, 0.95,
            "Negative result\n"
            "N=287, 10 gene families\n"
            "Silhouette < 0 in all configs\n"
            "NN accuracy near random (10%)\n"
            "UMAP was misleading",
            transform=ax.transAxes, fontsize=5.5, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", edgecolor="#FFB74D"))

    ax.set_ylim(-0.25, 0.05)
    ax2.set_ylim(0, 35)


if __name__ == "__main__":
    main()
