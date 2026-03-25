#!/usr/bin/env python3
"""
Paper 1 Supplementary Figures S1, S2, S3, S6, S7, S8, S9

S1: Layer profiling — periodicity signal across Evo2 blocks
S2: Comprehensive validation — inversion by domain, phylum, gene family
S3: Non-coding specificity — detailed breakdown by control type
S6: Smoothing window optimization for exon-intron detection
S7: Stop codon clustering + amino acid identity NOT encoded
S8: Protein identity clustering NEGATIVE (10 gene families, 3 configs)
S9: Syntax vs semantics summary diagram
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


def generate_s7():
    """S7: Stop codon clustering + amino acid identity NOT encoded."""
    print("Generating S7: Codon table analysis...")

    with open("results/experiments/codon_table/codon_embeddings.json") as f:
        data = json.load(f)

    embeddings = data["embeddings"]
    codon_table = data["codon_table"]
    aa_properties = data["aa_properties"]

    codons = sorted(embeddings.keys())
    emb_matrix = np.array([embeddings[c] for c in codons])
    aas = [codon_table[c] for c in codons]
    is_stop = [aa == "Stop" for aa in aas]

    # Cosine distance matrix (use float32 to avoid overflow in 8192-D)
    from numpy.linalg import norm
    emb_matrix = emb_matrix.astype(np.float32)
    norms = norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = emb_matrix / norms
    cos_sim = normed @ normed.T
    cos_dist = 1 - cos_sim

    # Stop codon analysis
    stop_idx = [i for i, s in enumerate(is_stop) if s]
    sense_idx = [i for i, s in enumerate(is_stop) if not s]

    within_stop = np.mean([cos_dist[i, j] for i in stop_idx for j in stop_idx if i < j])
    stop_to_sense = np.mean([cos_dist[i, j] for i in stop_idx for j in sense_idx])
    stop_ratio = stop_to_sense / within_stop if within_stop > 0 else 0

    # Synonymous codon analysis
    aa_groups = defaultdict(list)
    for i, aa in enumerate(aas):
        if aa != "Stop":
            aa_groups[aa].append(i)

    within_aa_dists = []
    between_aa_dists = []
    for aa, indices in aa_groups.items():
        if len(indices) > 1:
            for ii in range(len(indices)):
                for jj in range(ii + 1, len(indices)):
                    within_aa_dists.append(cos_dist[indices[ii], indices[jj]])
        for other_aa, other_indices in aa_groups.items():
            if other_aa != aa:
                for ii in indices:
                    for jj in other_indices:
                        between_aa_dists.append(cos_dist[ii, jj])

    syn_ratio = np.mean(between_aa_dists) / np.mean(within_aa_dists) if within_aa_dists else 0

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(normed)

    prop_colors = {
        "hydrophobic": "#1565C0",
        "polar": "#4CAF50",
        "charged": "#F44336",
        "aromatic": "#9C27B0",
        "special": "#FF9800",
    }

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    # Panel A: Stop codon clustering
    ax_a = fig.add_subplot(gs[0, 0])
    categories = ["Within stop\ncodons", "Stop vs\nsense codons", "Within\nsynonymous", "Between\namino acids"]
    values = [within_stop, stop_to_sense, np.mean(within_aa_dists), np.mean(between_aa_dists)]
    colors_bar = ["#F44336", "#FF9800", "#4CAF50", "#1565C0"]
    bars = ax_a.bar(range(len(categories)), values, color=colors_bar, alpha=0.7,
                    edgecolor="black", linewidth=0.5, width=0.6)
    ax_a.set_xticks(range(len(categories)))
    ax_a.set_xticklabels(categories, fontsize=7)
    ax_a.set_ylabel("Mean cosine distance")
    ax_a.set_title("A   Stop codons cluster in embedding space", loc="left", fontweight="bold")
    for bar, val in zip(bars, values):
        ax_a.text(bar.get_x() + bar.get_width() / 2, val + 0.002, f"{val:.3f}",
                  ha="center", fontsize=7, fontweight="bold")

    ax_a.annotate("", xy=(0, within_stop + 0.01), xytext=(1, stop_to_sense + 0.01),
                  arrowprops=dict(arrowstyle="<->", color="#333", lw=1.2))
    ax_a.text(0.5, (within_stop + stop_to_sense) / 2 + 0.015, f"{stop_ratio:.2f}×",
              ha="center", fontsize=9, fontweight="bold", color="#C62828")

    # Panel B: PCA colored by amino acid property
    ax_b = fig.add_subplot(gs[0, 1])
    for i, (c, aa) in enumerate(zip(codons, aas)):
        if aa == "Stop":
            color = "#333333"
            marker = "X"
            size = 80
        else:
            prop = aa_properties.get(aa, "special")
            color = prop_colors.get(prop, "#999")
            marker = "o"
            size = 40
        ax_b.scatter(coords[i, 0], coords[i, 1], c=color, s=size, marker=marker,
                     alpha=0.7, edgecolors="black", linewidth=0.3, zorder=3 if aa == "Stop" else 2)
        ax_b.annotate(c, (coords[i, 0], coords[i, 1]), fontsize=4, ha="center", va="bottom",
                      xytext=(0, 3), textcoords="offset points")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=prop_colors["hydrophobic"],
               markersize=7, label="Hydrophobic"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=prop_colors["polar"],
               markersize=7, label="Polar"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=prop_colors["charged"],
               markersize=7, label="Charged"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=prop_colors["aromatic"],
               markersize=7, label="Aromatic"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=prop_colors["special"],
               markersize=7, label="Special (Gly/Pro/Cys)"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#333",
               markersize=8, label="Stop codons"),
    ]
    ax_b.legend(handles=legend_elements, fontsize=6, loc="upper right")
    ax_b.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax_b.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax_b.set_title("B   PCA of codon embeddings (colored by AA property)", loc="left", fontweight="bold")

    # Panel C: PCA colored by GC content
    ax_c = fig.add_subplot(gs[1, 0])
    gc_vals = [sum(1 for b in c if b in "GC") / 3 for c in codons]
    sc = ax_c.scatter(coords[:, 0], coords[:, 1], c=gc_vals, cmap="RdYlBu_r",
                      s=40, alpha=0.8, edgecolors="black", linewidth=0.3, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax_c, label="GC fraction", shrink=0.8)
    for i, c in enumerate(codons):
        ax_c.annotate(c, (coords[i, 0], coords[i, 1]), fontsize=4, ha="center", va="bottom",
                      xytext=(0, 3), textcoords="offset points")
    ax_c.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax_c.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax_c.set_title("C   GC content dominates embedding space", loc="left", fontweight="bold")

    # Panel D: Silhouette scores
    ax_d = fig.add_subplot(gs[1, 1])

    # Compute silhouette by AA grouping
    from sklearn.metrics import silhouette_score
    sense_embs = normed[~np.array(is_stop)]
    sense_aas = [aa for aa, s in zip(aas, is_stop) if not s]
    aa_sil = silhouette_score(sense_embs, sense_aas, metric="cosine")

    # By property
    sense_props = [aa_properties.get(aa, "special") for aa in sense_aas]
    prop_sil = silhouette_score(sense_embs, sense_props, metric="cosine")

    # By GC
    gc_labels = ["high" if gc > 0.5 else "low" for gc, s in zip(gc_vals, is_stop) if not s]
    gc_sil = silhouette_score(sense_embs, gc_labels, metric="cosine")

    # By first base
    first_labels = [c[0] for c, s in zip(codons, is_stop) if not s]
    first_sil = silhouette_score(sense_embs, first_labels, metric="cosine")

    groupings = ["Amino acid\nidentity", "Biochemical\nproperty", "GC content\n(high/low)", "First\nnucleotide"]
    sils = [aa_sil, prop_sil, gc_sil, first_sil]
    bar_colors = ["#F44336" if s < 0 else "#4CAF50" for s in sils]

    bars = ax_d.bar(range(len(groupings)), sils, color=bar_colors, alpha=0.7,
                    edgecolor="black", linewidth=0.5, width=0.6)
    ax_d.set_xticks(range(len(groupings)))
    ax_d.set_xticklabels(groupings, fontsize=7)
    ax_d.set_ylabel("Silhouette score")
    ax_d.axhline(0, color="black", linewidth=0.5)
    ax_d.set_title("D   Clustering quality by grouping", loc="left", fontweight="bold")
    for bar, val in zip(bars, sils):
        ax_d.text(bar.get_x() + bar.get_width() / 2, val + 0.02 if val >= 0 else val - 0.04,
                  f"{val:.2f}", ha="center", fontsize=8, fontweight="bold")

    plt.savefig(OUT_DIR / "s7_codon_table.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_DIR / "s7_codon_table.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved s7_codon_table.png (stop ratio: {stop_ratio:.2f}×, syn ratio: {syn_ratio:.2f}×, AA sil: {aa_sil:.2f})")


def generate_s8():
    """S8: Protein identity clustering NEGATIVE across 3 model configs."""
    print("Generating S8: Protein identity clustering...")

    with open("results/experiments/codon_periodicity/functional_clustering_comparison.json") as f:
        data = json.load(f)

    configs = ["40B_blocks10", "40B_blocks28", "7B_layer10"]
    config_labels = ["Evo2 40B\nblock 10", "Evo2 40B\nblock 28", "Evo2 7B\nlayer 10"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    families = list(data["40B_blocks10"]["families"].keys())
    family_short = [f.replace("family_", "") for f in families]

    for idx, (config, label) in enumerate(zip(configs, config_labels)):
        ax = axes[idx]
        cfg = data[config]

        ratios = [cfg["families"][f]["ratio"] for f in families]
        ns = [int(cfg["families"][f]["n"]) for f in families]

        colors = ["#C8E6C9" if r < 1.0 else "#FFCDD2" for r in ratios]
        bars = ax.barh(range(len(families)), ratios, color=colors,
                       edgecolor="black", linewidth=0.3, height=0.6)
        ax.axvline(1.0, color="#C62828", linewidth=1, linestyle="--", alpha=0.7)
        ax.set_yticks(range(len(families)))
        ax.set_yticklabels([f"{f} (n={n})" for f, n in zip(family_short, ns)], fontsize=7)
        ax.set_xlabel("Between/within distance ratio")
        ax.invert_yaxis()

        sil = cfg["silhouette_pca"]
        nn = cfg["nn_accuracy_pca"]
        ax.set_title(f"{label}\nsilhouette: {sil:.3f}, NN acc: {nn:.1%}",
                     fontweight="bold", fontsize=8)

        # Annotation: ratio < 1.0 means within < between (good clustering)
        for i, (bar, r) in enumerate(zip(bars, ratios)):
            ax.text(r + 0.01, i, f"{r:.2f}", va="center", fontsize=6)

    fig.suptitle("Supplementary Figure S8: Protein identity clustering — NEGATIVE\n"
                 "(ratio > 1.0 = within-family MORE distant than between-family)",
                 fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "s8_protein_clustering.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_DIR / "s8_protein_clustering.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved s8_protein_clustering.png")


def generate_s9():
    """S9: Syntax vs semantics summary — what DNA models learn and don't."""
    print("Generating S9: Syntax vs semantics...")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(5, 9.5, "Supplementary Figure S9: What DNA foundation models learn",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(5, 9.0, "Delineating DNA syntax (learnable) from protein semantics (not learnable)",
            ha="center", fontsize=9, fontstyle="italic", color="#555")

    # Learned (syntax) box
    ax.add_patch(plt.Rectangle((0.3, 4.5), 4.4, 4.0, fill=True,
                                facecolor="#C8E6C9", edgecolor="#2E7D32", linewidth=1.5, zorder=1))
    ax.text(2.5, 8.2, "LEARNED (DNA syntax)", ha="center", fontsize=10,
            fontweight="bold", color="#2E7D32")

    learned = [
        ("Triplet periodicity", "98.5% detection, 459 seqs, 55 phyla"),
        ("3-periodic comb filter", "Offsets 3,6,9,12,15 elevated"),
        ("Stop codon boundaries", "1.55× clustering ratio"),
        ("Exon-intron structure", "98% recall, 36 genes, 13 species"),
        ("Coding vs non-coding", "94.7% per-position accuracy"),
    ]
    for i, (feat, evidence) in enumerate(learned):
        y = 7.6 - i * 0.6
        ax.text(0.6, y, f"+ {feat}", fontsize=8, fontweight="bold", color="#2E7D32")
        ax.text(0.8, y - 0.22, evidence, fontsize=6.5, color="#555")

    # NOT learned (semantics) box
    ax.add_patch(plt.Rectangle((5.3, 4.5), 4.4, 4.0, fill=True,
                                facecolor="#FFCDD2", edgecolor="#C62828", linewidth=1.5, zorder=1))
    ax.text(7.5, 8.2, "NOT LEARNED (protein semantics)", ha="center", fontsize=10,
            fontweight="bold", color="#C62828")

    not_learned = [
        ("Amino acid identity", "Silhouette -0.40, no AA clustering"),
        ("Biochemical properties", "Hydrophobic/polar/charged mixed"),
        ("Protein function", "NN accuracy 13-20% (10 families)"),
        ("Gene family identity", "Silhouette -0.06 to -0.21"),
        ("Wobble position specificity", "Offset-1 > offset-2 is sequential"),
    ]
    for i, (feat, evidence) in enumerate(not_learned):
        y = 7.6 - i * 0.6
        ax.text(5.6, y, f"- {feat}", fontsize=8, fontweight="bold", color="#C62828")
        ax.text(5.8, y - 0.22, evidence, fontsize=6.5, color="#555")

    # Bottom explanation
    ax.add_patch(plt.Rectangle((0.3, 0.5), 9.4, 3.5, fill=True,
                                facecolor="#FFF3E0", edgecolor="#FF9800", linewidth=1, zorder=1))
    ax.text(5, 3.7, "Interpretation", ha="center", fontsize=10, fontweight="bold", color="#E65100")

    explanations = [
        "Next-nucleotide prediction captures DNA-level statistical regularities:",
        "  -Codon triplets create 3bp periodicity in nucleotide transition probabilities",
        "  -Stop codons create predictable sequence transitions at gene boundaries",
        "  -Splice sites create compositional shifts between exons and introns",
        "",
        "But it cannot capture protein-level evolutionary selection:",
        "  -Synonymous codons (same AA) have different DNA statistics but identical protein effect",
        "  -Protein function depends on 3D structure, not DNA sequence composition",
        "  -The codon\u2192amino acid mapping is mediated by tRNA, invisible to DNA prediction",
    ]
    for i, line in enumerate(explanations):
        y = 3.2 - i * 0.3
        color = "#333" if not line.startswith("But") else "#C62828"
        weight = "bold" if (line and not line.startswith(" ")) else "normal"
        ax.text(0.6, y, line, fontsize=7, color=color, fontweight=weight)

    plt.savefig(OUT_DIR / "s9_syntax_vs_semantics.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_DIR / "s9_syntax_vs_semantics.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved s9_syntax_vs_semantics.png")


if __name__ == "__main__":
    generate_s1()
    generate_s2()
    generate_s3()
    generate_s6()
    generate_s7()
    generate_s8()
    # S9 removed — syntax vs semantics content is in manuscript Results text
    print("\nAll supplementary figures generated.")
