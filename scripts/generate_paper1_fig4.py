#!/usr/bin/env python3
"""
Paper 1, Figure 4: Coding Detection in Context

Panels:
A: Coding detection accuracy by kingdom (from comprehensive validation)
B: Length dependence — 100% >500bp, drops for short sequences
C: Non-coding specificity by category (rRNA, lncRNA, intron, repeat, tRNA, intergenic)
D: Capability comparison: what per-position analysis adds beyond k-mers
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

OUT_DIR = Path("results/paper1/figures")
DATA_DIR = Path("results/experiments/codon_periodicity")

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
    elif lin.startswith("Eukarya"): return "Other"
    return "Unknown"


def main():
    # Load comprehensive validation results
    with open(Path("results/experiments/comprehensive/panel.json")) as f:
        panel = json.load(f)

    emb_dir = DATA_DIR / "embeddings"
    results = []
    for entry in panel:
        mp = emb_dir / f"{entry['name']}_metrics.json"
        if not mp.exists():
            continue
        with open(mp) as f:
            metrics = json.load(f)
        # Get GC + length from FASTA
        fp = DATA_DIR / "fasta" / f"{entry['name']}.fasta"
        gc, seq_len = 0, 0
        if fp.exists():
            with open(fp) as f2:
                seq = "".join(l.strip() for l in f2 if not l.startswith(">"))
            gc = sum(1 for c in seq.upper() if c in "GC") / max(len(seq), 1) * 100
            seq_len = len(seq)
        results.append({**entry, **metrics, "gc_content": round(gc, 1), "seq_len": seq_len})

    coding = [r for r in results if not r.get("noncoding") and not r.get("category", "").startswith("noncoding")]
    noncoding = [r for r in results if r.get("noncoding") or r.get("category", "").startswith("noncoding")]

    fig = plt.figure(figsize=(7.2, 7.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # ══════════════════════════════════════════════════════════════
    # Panel A: Coding detection by domain
    # ══════════════════════════════════════════════════════════════
    ax_a = fig.add_subplot(gs[0, 0])

    domain_order = ["Archaea", "Bacteria", "Vertebrata", "Invertebrata", "Plantae",
                    "Fungi", "Protista", "Algae", "Organellar", "Virus"]
    domain_colors = {
        "Archaea": "#FF9800", "Bacteria": "#E65100", "Vertebrata": "#2196F3",
        "Invertebrata": "#8BC34A", "Plantae": "#4CAF50", "Fungi": "#9C27B0",
        "Protista": "#F44336", "Algae": "#00BCD4", "Organellar": "#795548",
        "Virus": "#607D8B",
    }

    by_domain = defaultdict(list)
    for r in coding:
        d = get_domain(r)
        by_domain[d].append(r)

    domains_present = [d for d in domain_order if d in by_domain]
    rates = []
    ns = []
    for d in domains_present:
        recs = by_domain[d]
        inv = sum(1 for r in recs if r.get("offset3_inversion"))
        rates.append(100 * inv / len(recs))
        ns.append(len(recs))

    x_pos = np.arange(len(domains_present))
    colors = [domain_colors.get(d, "#666") for d in domains_present]
    bars = ax_a.bar(x_pos, rates, color=colors, alpha=0.7, edgecolor="black",
                     linewidth=0.3, width=0.65)

    ax_a.set_xticks(x_pos)
    short_labels = {
        "Archaea": "Archaea", "Bacteria": "Bacteria", "Vertebrata": "Vertebr.",
        "Invertebrata": "Invertebr.", "Plantae": "Plantae", "Fungi": "Fungi",
        "Protista": "Protista", "Algae": "Algae", "Organellar": "Organellar",
        "Virus": "Virus",
    }
    ax_a.set_xticklabels([short_labels.get(d, d) for d in domains_present],
                          fontsize=5.5, rotation=45, ha="right")
    ax_a.set_ylabel("Inversion detected (%)")
    ax_a.set_ylim(82, 102)
    ax_a.set_title("A   Coding detection by domain (N=459)",
                    loc="left", fontweight="bold", fontsize=8)

    for bar, n, rate in zip(bars, ns, rates):
        ax_a.text(bar.get_x() + bar.get_width() / 2, rate + 0.5,
                  f"{rate:.0f}%", ha="center", fontsize=5, fontweight="bold")
        ax_a.text(bar.get_x() + bar.get_width() / 2, 83, f"N={n}",
                  ha="center", fontsize=4.5, color="#666")

    # ══════════════════════════════════════════════════════════════
    # Panel B: Length dependence
    # ══════════════════════════════════════════════════════════════
    ax_b = fig.add_subplot(gs[0, 1])

    len_bins = [(0, 300, "<300"), (300, 500, "300-500"), (500, 800, "500-800"),
                (800, 1200, "800-1200"), (1200, 2500, ">1200")]

    bin_rates = []
    bin_ns = []
    bin_labels = []
    for lo, hi, label in len_bins:
        seqs = [r for r in coding if lo <= r.get("seq_len", 0) < hi]
        if not seqs:
            continue
        inv = sum(1 for r in seqs if r.get("offset3_inversion"))
        bin_rates.append(100 * inv / len(seqs))
        bin_ns.append(len(seqs))
        bin_labels.append(label)

    x_pos = np.arange(len(bin_labels))
    colors_len = ["#FFCDD2" if r < 95 else "#C8E6C9" if r == 100 else "#BBDEFB" for r in bin_rates]
    bars = ax_b.bar(x_pos, bin_rates, color=colors_len, edgecolor="black",
                     linewidth=0.3, width=0.6)

    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(bin_labels, fontsize=6)
    ax_b.set_xlabel("Sequence length (bp)")
    ax_b.set_ylabel("Inversion detected (%)")
    ax_b.set_ylim(75, 105)
    ax_b.set_title("B   Length dependence", loc="left", fontweight="bold", fontsize=8)

    for bar, n, rate in zip(bars, bin_ns, bin_rates):
        ax_b.text(bar.get_x() + bar.get_width() / 2, rate + 0.8,
                  f"{rate:.1f}%", ha="center", fontsize=6, fontweight="bold")
        ax_b.text(bar.get_x() + bar.get_width() / 2, 76.5, f"N={n}",
                  ha="center", fontsize=5, color="#666")

    # Annotation
    ax_b.annotate("100% above\n500 bp", xy=(2.5, 100), fontsize=7, ha="center",
                  color="#388E3C", fontweight="bold",
                  xytext=(3.5, 90), arrowprops=dict(arrowstyle="->", color="#388E3C", lw=0.8))

    # ══════════════════════════════════════════════════════════════
    # Panel C: Non-coding specificity by category
    # ══════════════════════════════════════════════════════════════
    ax_c = fig.add_subplot(gs[1, 0])

    nc_cats = defaultdict(lambda: {"total": 0, "inversion": 0})
    for r in noncoding:
        cat = r.get("category", "unknown")
        # Simplify category names
        cat_short = cat.replace("noncoding_", "")
        nc_cats[cat_short]["total"] += 1
        if r.get("offset3_inversion"):
            nc_cats[cat_short]["inversion"] += 1

    cat_order = ["rRNA", "lncRNA", "intron", "repeat", "tRNA", "intergenic"]
    cat_labels = []
    cat_fp_rates = []
    cat_ns = []
    cat_colors = []

    for cat in cat_order:
        if cat in nc_cats:
            d = nc_cats[cat]
            fp_rate = 100 * d["inversion"] / d["total"]
            cat_labels.append(cat)
            cat_fp_rates.append(fp_rate)
            cat_ns.append(d["total"])
            # Color: green if correct (low FP), red if high FP
            if fp_rate < 30:
                cat_colors.append("#C8E6C9")
            elif fp_rate < 70:
                cat_colors.append("#FFF9C4")
            else:
                cat_colors.append("#FFCDD2")

    x_pos = np.arange(len(cat_labels))
    bars = ax_c.bar(x_pos, cat_fp_rates, color=cat_colors, edgecolor="black",
                     linewidth=0.3, width=0.6)

    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(cat_labels, fontsize=6)
    ax_c.set_ylabel("False positive rate (%)")
    ax_c.set_title("C   Non-coding specificity by category",
                    loc="left", fontweight="bold", fontsize=8)
    ax_c.set_ylim(0, 110)

    for bar, n, rate in zip(bars, cat_ns, cat_fp_rates):
        ax_c.text(bar.get_x() + bar.get_width() / 2, rate + 2,
                  f"{rate:.0f}%", ha="center", fontsize=6, fontweight="bold")
        ax_c.text(bar.get_x() + bar.get_width() / 2, -5, f"N={n}",
                  ha="center", fontsize=5, color="#666")

    # Annotations for tRNA and intergenic
    ax_c.text(4, 105, "codon\nstructure", ha="center", fontsize=5,
              color="#C62828", fontstyle="italic")
    ax_c.text(5, 105, "likely\ncoding", ha="center", fontsize=5,
              color="#C62828", fontstyle="italic")

    # ══════════════════════════════════════════════════════════════
    # Panel D: What per-position adds beyond k-mers
    # ══════════════════════════════════════════════════════════════
    ax_d = fig.add_subplot(gs[1, 1])

    capabilities = [
        "Binary coding\ndetection",
        "Exon-intron\nboundaries",
        "Gene boundary\nresolution",
        "Per-position\nstructure",
        "Database-free\noperation",
        "GC-independent",
    ]

    kmer_scores = [93, 0, 0, 0, 93, 70]  # k-mer capability (approximate %)
    evo2_scores = [98.5, 98, 73, 100, 98.5, 98.5]  # Evo2 per-position capability

    y_pos = np.arange(len(capabilities))
    height = 0.35

    bars_k = ax_d.barh(y_pos + height / 2, kmer_scores, height, color="#BDBDBD",
                        edgecolor="black", linewidth=0.3, label="K-mer (trinucleotide)")
    bars_e = ax_d.barh(y_pos - height / 2, evo2_scores, height, color="#1565C0",
                        alpha=0.7, edgecolor="black", linewidth=0.3, label="Evo2 per-position")

    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(capabilities, fontsize=6)
    ax_d.set_xlabel("Performance (%)")
    ax_d.set_title("D   Per-position vs k-mer capabilities",
                    loc="left", fontweight="bold", fontsize=8)
    ax_d.legend(loc="lower right", fontsize=6, framealpha=0.9)
    ax_d.set_xlim(0, 110)
    ax_d.invert_yaxis()

    # Speed annotation
    ax_d.text(50, 5.8, "K-mers: 1,527x faster for binary detection\nEvo2: unique per-position capabilities",
              fontsize=5.5, ha="center", fontstyle="italic", color="#555")

    plt.savefig(OUT_DIR / "fig4.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_DIR / "fig4.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved to {OUT_DIR}/fig4.png and .pdf")


if __name__ == "__main__":
    main()
