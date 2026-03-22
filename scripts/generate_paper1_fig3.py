#!/usr/bin/env python3
"""
Paper 1, Figure 3: What the Model Learned — and What It Didn't
Boundaries of DNA-level learning

Panels:
A: Stop codon clustering (1.55x) — model learned gene boundaries
B: Amino acid identity NOT encoded — codon table doesn't cluster by AA
C: Protein identity clustering NEGATIVE — UMAP was misleading
D: Summary diagram: DNA syntax vs protein semantics
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
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
    "legend.fontsize": 6,
    "savefig.dpi": 300,
    "axes.linewidth": 0.5,
})

CODON_TABLE = {
    'TTT': 'Phe', 'TTC': 'Phe', 'TTA': 'Leu', 'TTG': 'Leu',
    'CTT': 'Leu', 'CTC': 'Leu', 'CTA': 'Leu', 'CTG': 'Leu',
    'ATT': 'Ile', 'ATC': 'Ile', 'ATA': 'Ile', 'ATG': 'Met',
    'GTT': 'Val', 'GTC': 'Val', 'GTA': 'Val', 'GTG': 'Val',
    'TCT': 'Ser', 'TCC': 'Ser', 'TCA': 'Ser', 'TCG': 'Ser',
    'CCT': 'Pro', 'CCC': 'Pro', 'CCA': 'Pro', 'CCG': 'Pro',
    'ACT': 'Thr', 'ACC': 'Thr', 'ACA': 'Thr', 'ACG': 'Thr',
    'GCT': 'Ala', 'GCC': 'Ala', 'GCA': 'Ala', 'GCG': 'Ala',
    'TAT': 'Tyr', 'TAC': 'Tyr', 'TAA': 'Stop', 'TAG': 'Stop',
    'CAT': 'His', 'CAC': 'His', 'CAA': 'Gln', 'CAG': 'Gln',
    'AAT': 'Asn', 'AAC': 'Asn', 'AAA': 'Lys', 'AAG': 'Lys',
    'GAT': 'Asp', 'GAC': 'Asp', 'GAA': 'Glu', 'GAG': 'Glu',
    'TGT': 'Cys', 'TGC': 'Cys', 'TGA': 'Stop', 'TGG': 'Trp',
    'CGT': 'Arg', 'CGC': 'Arg', 'CGA': 'Arg', 'CGG': 'Arg',
    'AGT': 'Ser', 'AGC': 'Ser', 'AGA': 'Arg', 'AGG': 'Arg',
    'GGT': 'Gly', 'GGC': 'Gly', 'GGA': 'Gly', 'GGG': 'Gly',
}

AA_PROPS = {
    'Ala': 'hydrophobic', 'Val': 'hydrophobic', 'Ile': 'hydrophobic', 'Leu': 'hydrophobic',
    'Met': 'hydrophobic', 'Phe': 'aromatic', 'Trp': 'aromatic', 'Tyr': 'aromatic',
    'Pro': 'special', 'Gly': 'special', 'Cys': 'special',
    'Ser': 'polar', 'Thr': 'polar', 'Asn': 'polar', 'Gln': 'polar',
    'Asp': 'negative', 'Glu': 'negative',
    'Lys': 'positive', 'Arg': 'positive', 'His': 'positive',
    'Stop': 'stop',
}

PROP_COLORS = {
    'hydrophobic': '#FF9800', 'aromatic': '#E91E63', 'special': '#9E9E9E',
    'polar': '#2196F3', 'negative': '#F44336', 'positive': '#4CAF50',
    'stop': '#000000',
}


def main():
    # Load codon embeddings
    with open("results/experiments/codon_table/codon_embeddings.json") as f:
        codon_data = json.load(f)
    embeddings = codon_data["embeddings"]

    # Load functional clustering results
    with open("results/experiments/codon_periodicity/functional_clustering_comparison.json") as f:
        fc_data = json.load(f)

    fig = plt.figure(figsize=(7.2, 7.5))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # ══════════════════════════════════════════════════════════════
    # Panel A: Stop codon clustering
    # ══════════════════════════════════════════════════════════════
    ax_a = fig.add_subplot(gs[0, 0])

    codons = sorted(embeddings.keys())
    emb_matrix = np.array([embeddings[c] for c in codons], dtype=np.float32)

    # L2 normalize
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_norm = emb_matrix / norms

    from scipy.spatial.distance import cosine

    # Compute within-group distances
    stop_codons = [c for c in codons if CODON_TABLE[c] == "Stop"]
    sense_codons = [c for c in codons if CODON_TABLE[c] != "Stop"]

    stop_idxs = [codons.index(c) for c in stop_codons]
    sense_idxs = [codons.index(c) for c in sense_codons]

    stop_within = [cosine(emb_norm[i], emb_norm[j])
                    for i in stop_idxs for j in stop_idxs if i < j]
    stop_vs_sense = [cosine(emb_norm[i], emb_norm[j])
                      for i in stop_idxs for j in sense_idxs[:20]]

    # Also compute synonymous codon distances
    from collections import defaultdict
    aa_groups = defaultdict(list)
    for c in sense_codons:
        aa_groups[CODON_TABLE[c]].append(codons.index(c))

    syn_within = []
    for aa, idxs in aa_groups.items():
        if len(idxs) >= 2:
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    syn_within.append(cosine(emb_norm[idxs[i]], emb_norm[idxs[j]]))

    between_all = [cosine(emb_norm[i], emb_norm[j])
                    for i in sense_idxs[:20] for j in sense_idxs[20:40]]

    # Bar chart
    categories = ["Stop-Stop", "Stop-Sense", "Synonymous\n(same AA)", "Between\n(diff AA)"]
    values = [np.mean(stop_within), np.mean(stop_vs_sense),
              np.mean(syn_within), np.mean(between_all)]
    colors = ["#000000", "#666666", "#2196F3", "#BDBDBD"]

    bars = ax_a.bar(range(len(categories)), values, color=colors, alpha=0.7,
                     edgecolor="black", linewidth=0.3, width=0.6)
    ax_a.set_xticks(range(len(categories)))
    ax_a.set_xticklabels(categories, fontsize=6)
    ax_a.set_ylabel("Cosine distance")
    ax_a.set_title("A   Stop codons cluster in embedding space",
                    loc="left", fontweight="bold", fontsize=8)

    for bar, val in zip(bars, values):
        ax_a.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                  f"{val:.3f}", ha="center", fontsize=5.5, fontweight="bold")

    # Ratio annotation
    ratio = np.mean(stop_vs_sense) / np.mean(stop_within)
    ax_a.annotate("", xy=(0, np.mean(stop_within) + 0.005),
                  xytext=(1, np.mean(stop_vs_sense) + 0.005),
                  arrowprops=dict(arrowstyle="<->", color="#333", lw=0.8))
    ax_a.text(0.5, (np.mean(stop_within) + np.mean(stop_vs_sense)) / 2 + 0.01,
              f"{ratio:.1f}x", ha="center", fontsize=7, fontweight="bold")

    # ══════════════════════════════════════════════════════════════
    # Panel B: Amino acid identity NOT encoded
    # ══════════════════════════════════════════════════════════════
    ax_b = fig.add_subplot(gs[0, 1])

    # Show per-AA synonymous distances — no clustering pattern
    aa_names = []
    aa_dists = []
    aa_colors = []
    for aa in sorted(aa_groups.keys()):
        idxs = aa_groups[aa]
        if len(idxs) < 2:
            continue
        dists = [cosine(emb_norm[i], emb_norm[j]) for i in idxs for j in idxs if i < j]
        aa_names.append(aa)
        aa_dists.append(np.mean(dists))
        aa_colors.append(PROP_COLORS.get(AA_PROPS.get(aa, "special"), "#666"))

    x_pos = np.arange(len(aa_names))
    bars = ax_b.bar(x_pos, aa_dists, color=aa_colors, alpha=0.7,
                     edgecolor="black", linewidth=0.3, width=0.7)
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(aa_names, fontsize=5, rotation=45, ha="right")
    ax_b.set_ylabel("Within-AA cosine distance")
    ax_b.set_title("B   Amino acid identity not encoded",
                    loc="left", fontweight="bold", fontsize=8)

    # Add between-AA reference line
    ax_b.axhline(np.mean(between_all), color="#333", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_b.text(len(aa_names) - 0.5, np.mean(between_all) + 0.003,
              "between-AA\nmean", fontsize=5, ha="right", color="#666")

    # Property legend
    from matplotlib.lines import Line2D
    prop_legend = [Line2D([0], [0], marker='s', color='w',
                          markerfacecolor=PROP_COLORS[p], markersize=5, label=p)
                   for p in ['hydrophobic', 'aromatic', 'polar', 'positive', 'negative']]
    ax_b.legend(handles=prop_legend, loc="upper right", fontsize=4.5, ncol=2,
                framealpha=0.9, handletextpad=0.2, columnspacing=0.5, title="AA property",
                title_fontsize=5)

    # ══════════════════════════════════════════════════════════════
    # Panel C: Functional clustering NEGATIVE
    # ══════════════════════════════════════════════════════════════
    ax_c = fig.add_subplot(gs[1, 0])

    configs = ["40B_blocks10", "40B_blocks28", "7B_layer10"]
    config_labels = ["40B\nblocks.10", "40B\nblocks.28", "7B\nlayer.10"]
    sils = [fc_data[c]["silhouette_pca"] for c in configs]
    nns = [fc_data[c]["nn_accuracy_pca"] * 100 for c in configs]
    config_colors = ["#1565C0", "#42A5F5", "#FF9800"]

    x = np.arange(len(configs))
    width = 0.35
    ax_c.bar(x - width / 2, sils, width, color=config_colors, alpha=0.7,
             edgecolor="black", linewidth=0.3, label="Silhouette")

    ax_c2 = ax_c.twinx()
    ax_c2.bar(x + width / 2, nns, width, color=config_colors, alpha=0.3,
              edgecolor="black", linewidth=0.3, label="NN accuracy (%)")

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(config_labels, fontsize=6.5)
    ax_c.set_ylabel("Silhouette score")
    ax_c2.set_ylabel("NN gene accuracy (%)", color="#666")
    ax_c.axhline(0, color="black", linewidth=0.3, linestyle="--")
    ax_c2.axhline(10, color="#CCCCCC", linewidth=0.5, linestyle=":")

    ax_c.set_title("C   Protein identity not clustered (N=287)",
                    loc="left", fontweight="bold", fontsize=8)
    ax_c.set_ylim(-0.25, 0.05)
    ax_c2.set_ylim(0, 30)

    ax_c.text(0.5, 0.95, "10 gene families\n3 model configs\nSilhouette < 0 everywhere\nUMAP was misleading",
              transform=ax_c.transAxes, fontsize=5.5, ha="center", va="top",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", edgecolor="#FFB74D", alpha=0.9))

    # ══════════════════════════════════════════════════════════════
    # Panel D: What the model learned (syntax) vs didn't (semantics)
    # ══════════════════════════════════════════════════════════════
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")
    ax_d.set_title("D   DNA syntax vs protein semantics",
                    loc="left", fontweight="bold", fontsize=8)

    rect = FancyBboxPatch((0.01, 0.02), 0.98, 0.92, transform=ax_d.transAxes,
                           boxstyle="round,pad=0.02", facecolor="#FAFAFA",
                           edgecolor="#BDBDBD", linewidth=0.5, zorder=0)
    ax_d.add_patch(rect)

    learned = [
        ("LEARNED FROM DNA", "#1565C0", True),
        ("Triplet codon structure (98.5%)", "#1565C0", False),
        ("Exon-intron boundaries (98% recall)", "#1565C0", False),
        ("Stop codon identity (1.55x)", "#1565C0", False),
        ("Gene boundaries in genomes", "#1565C0", False),
        ("GC-independent, code-independent", "#1565C0", False),
    ]

    not_learned = [
        ("NOT LEARNED", "#C62828", True),
        ("Codon-to-amino-acid mapping", "#C62828", False),
        ("Amino acid properties", "#C62828", False),
        ("Protein identity from DNA", "#C62828", False),
        ("Wobble position specificity", "#C62828", False),
    ]

    y = 0.88
    for text, color, is_header in learned:
        weight = "bold" if is_header else "normal"
        size = 7 if is_header else 6.5
        marker = "" if is_header else "  + "
        ax_d.text(0.05, y, f"{marker}{text}", fontsize=size, fontweight=weight,
                  color=color, transform=ax_d.transAxes, va="top")
        y -= 0.065

    y -= 0.03
    for text, color, is_header in not_learned:
        weight = "bold" if is_header else "normal"
        size = 7 if is_header else 6.5
        marker = "" if is_header else "  - "
        ax_d.text(0.05, y, f"{marker}{text}", fontsize=size, fontweight=weight,
                  color=color, transform=ax_d.transAxes, va="top")
        y -= 0.065

    y -= 0.03
    ax_d.text(0.05, y, "The model learned the syntax of\nthe genetic code from DNA alone,\nbut not the semantics that require\nprotein-level selection.",
              fontsize=6, color="#555", transform=ax_d.transAxes, va="top",
              fontstyle="italic", linespacing=1.4)

    plt.savefig(OUT_DIR / "fig3.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_DIR / "fig3.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved to {OUT_DIR}/fig3.png and .pdf")


if __name__ == "__main__":
    main()
