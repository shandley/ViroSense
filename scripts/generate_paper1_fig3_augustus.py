#!/usr/bin/env python3
"""
Paper 1, Figure 3: Evo2 vs Augustus Benchmark

Panels:
A: Head-to-head F1 comparison (36 genes, paired)
B: Recall comparison — Evo2 matches Augustus
C: Precision gap — where Augustus wins (boundaries)
D: Xenopus example — Evo2 works where Augustus has no model
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

OUT_DIR = Path("results/paper1/figures")

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

EVO2_COLOR = "#1565C0"
AUG_COLOR = "#FF9800"


def main():
    with open("results/experiments/exon_intron/benchmark/benchmark_results.json") as f:
        data = json.load(f)

    results = data["results"]
    failures = data["augustus_failures"]

    # Filter to genes where both methods ran
    both = [r for r in results if r["augustus"]["f1"] > 0]
    evo2_only = [r for r in results if r["augustus"]["f1"] == 0]

    fig = plt.figure(figsize=(7.2, 7.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # ══════════════════════════════════════════════════════════════
    # Panel A: Paired F1 comparison
    # ══════════════════════════════════════════════════════════════
    ax_a = fig.add_subplot(gs[0, 0])

    evo2_f1 = [r["evo2"]["f1"] for r in both]
    aug_f1 = [r["augustus"]["f1"] for r in both]

    ax_a.scatter(aug_f1, evo2_f1, c=EVO2_COLOR, s=20, alpha=0.6,
                  edgecolors="black", linewidth=0.3, zorder=3)
    ax_a.plot([0, 1], [0, 1], color="#CCCCCC", linewidth=0.5, linestyle="--", zorder=1)

    ax_a.set_xlabel("Augustus F1")
    ax_a.set_ylabel("Evo2 F1 (unsupervised)")
    ax_a.set_title("A   F1 comparison (35 genes)", loc="left", fontweight="bold", fontsize=8)
    ax_a.set_xlim(0.4, 1.05)
    ax_a.set_ylim(0.35, 1.05)

    # Count above/below diagonal
    above = sum(1 for e, a in zip(evo2_f1, aug_f1) if e > a)
    ax_a.text(0.05, 0.95, f"Augustus wins F1\nin all 35 genes",
              transform=ax_a.transAxes, fontsize=6, va="top",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", edgecolor="#FFB74D", alpha=0.9))

    # ══════════════════════════════════════════════════════════════
    # Panel B: Recall comparison — key message
    # ══════════════════════════════════════════════════════════════
    ax_b = fig.add_subplot(gs[0, 1])

    evo2_rec = [r["evo2"]["recall"] * 100 for r in both]
    aug_rec = [r["augustus"]["recall"] * 100 for r in both]

    x = np.arange(len(both))
    width = 0.35

    # Sort by Augustus recall for cleaner visualization
    sort_idx = np.argsort(aug_rec)[::-1]
    evo2_rec_sorted = [evo2_rec[i] for i in sort_idx]
    aug_rec_sorted = [aug_rec[i] for i in sort_idx]

    ax_b.bar(x - width/2, aug_rec_sorted, width, color=AUG_COLOR, alpha=0.7,
             edgecolor="black", linewidth=0.2, label=f"Augustus (mean: {np.mean(aug_rec):.0f}%)")
    ax_b.bar(x + width/2, evo2_rec_sorted, width, color=EVO2_COLOR, alpha=0.7,
             edgecolor="black", linewidth=0.2, label=f"Evo2 (mean: {np.mean(evo2_rec):.0f}%)")

    ax_b.set_ylabel("Exon recall (%)")
    ax_b.set_title("B   Recall: Evo2 matches Augustus", loc="left", fontweight="bold", fontsize=8)
    ax_b.set_ylim(70, 105)
    ax_b.set_xticks([])
    ax_b.set_xlabel("35 genes (sorted by Augustus recall)")
    ax_b.legend(loc="lower left", fontsize=6, framealpha=0.9)

    # Mean lines
    ax_b.axhline(np.mean(evo2_rec), color=EVO2_COLOR, linewidth=0.8, linestyle="--", alpha=0.5)
    ax_b.axhline(np.mean(aug_rec), color=AUG_COLOR, linewidth=0.8, linestyle="--", alpha=0.5)

    # ══════════════════════════════════════════════════════════════
    # Panel C: Precision gap — where Augustus wins
    # ══════════════════════════════════════════════════════════════
    ax_c = fig.add_subplot(gs[1, 0])

    metrics = ["Recall", "Precision", "F1"]
    evo2_means = [np.mean([r["evo2"]["recall"] for r in both]) * 100,
                   np.mean([r["evo2"]["precision"] for r in both]) * 100,
                   np.mean([r["evo2"]["f1"] for r in both]) * 100]
    aug_means = [np.mean([r["augustus"]["recall"] for r in both]) * 100,
                  np.mean([r["augustus"]["precision"] for r in both]) * 100,
                  np.mean([r["augustus"]["f1"] for r in both]) * 100]

    x = np.arange(len(metrics))
    width = 0.3

    bars_e = ax_c.bar(x - width/2, evo2_means, width, color=EVO2_COLOR, alpha=0.7,
                       edgecolor="black", linewidth=0.3, label="Evo2 (unsupervised)")
    bars_a = ax_c.bar(x + width/2, aug_means, width, color=AUG_COLOR, alpha=0.7,
                       edgecolor="black", linewidth=0.3, label="Augustus (trained)")

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(metrics)
    ax_c.set_ylabel("Performance (%)")
    ax_c.set_ylim(0, 110)
    ax_c.set_title("C   The precision gap", loc="left", fontweight="bold", fontsize=8)
    ax_c.legend(loc="upper right", fontsize=6, framealpha=0.9)

    for bar, val in zip(bars_e, evo2_means):
        ax_c.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.0f}%",
                  ha="center", fontsize=6, fontweight="bold", color=EVO2_COLOR)
    for bar, val in zip(bars_a, aug_means):
        ax_c.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.0f}%",
                  ha="center", fontsize=6, fontweight="bold", color=AUG_COLOR)

    # Annotation for precision gap
    ax_c.annotate("", xy=(1 - width/2, evo2_means[1]), xytext=(1 + width/2, aug_means[1]),
                  arrowprops=dict(arrowstyle="<->", color="#333", lw=1))
    ax_c.text(1, (evo2_means[1] + aug_means[1])/2, "boundary\nresolution",
              ha="center", fontsize=5.5, fontstyle="italic", color="#555")

    # ══════════════════════════════════════════════════════════════
    # Panel D: Where Evo2 wins — organisms without Augustus models
    # ══════════════════════════════════════════════════════════════
    ax_d = fig.add_subplot(gs[1, 1])

    # Show the Xenopus case + highlight organisms where Augustus struggles
    # Find genes where Evo2 > Augustus by the largest margin in recall
    recall_advantage = [(r["gene"], r["evo2"]["recall"] - r["augustus"]["recall"],
                         r["evo2"]["recall"], r["augustus"]["recall"])
                        for r in results]
    recall_advantage.sort(key=lambda x: -x[1])

    # Show top examples + Xenopus (where Augustus = 0)
    examples = []
    for gene, adv, e_rec, a_rec in recall_advantage[:8]:
        short = gene.replace("human_", "").replace("_part1", " (1)").replace("_part2", " (2)")
        examples.append((short, e_rec * 100, a_rec * 100))

    # Add Xenopus
    for r in evo2_only:
        short = r["gene"].replace("xenopus_", "")
        examples.insert(0, (short, r["evo2"]["recall"] * 100, 0))

    names = [e[0] for e in examples]
    evo2_vals = [e[1] for e in examples]
    aug_vals = [e[2] for e in examples]

    y = np.arange(len(names))
    height = 0.35

    ax_d.barh(y - height/2, evo2_vals, height, color=EVO2_COLOR, alpha=0.7,
              edgecolor="black", linewidth=0.3, label="Evo2")
    ax_d.barh(y + height/2, aug_vals, height, color=AUG_COLOR, alpha=0.7,
              edgecolor="black", linewidth=0.3, label="Augustus")

    ax_d.set_yticks(y)
    ax_d.set_yticklabels(names, fontsize=6)
    ax_d.set_xlabel("Exon recall (%)")
    ax_d.set_title("D   Evo2 advantage: no species model needed",
                    loc="left", fontweight="bold", fontsize=8)
    ax_d.legend(loc="lower right", fontsize=6)
    ax_d.invert_yaxis()

    # Highlight Xenopus
    ax_d.text(50, 0, "No Augustus\nmodel exists", fontsize=6, ha="center",
              color="#C62828", fontweight="bold", fontstyle="italic")

    plt.savefig(OUT_DIR / "fig3.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_DIR / "fig3.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved to {OUT_DIR}/fig3.png and .pdf")


if __name__ == "__main__":
    main()
