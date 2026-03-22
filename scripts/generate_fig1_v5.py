#!/usr/bin/env python3
"""
Generate Figure 1 v5: The Triplet Genetic Code in DNA Foundation Model Embeddings

6-panel figure (3×2 grid):
A: Conceptual schematic — what offset-3 cosine measures
B: Multi-offset "comb filter" (expanded N) + error bars
C: Genomic trajectory — cos3 vs cos1 along E. coli lac operon
D: Individual sequences from all 3 domains
E: Cross-domain box plot (N=459)
F: Summary statistics
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
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
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 0.5,
})

MOD3_COLOR = "#1565C0"
NONMOD3_COLOR = "#BDBDBD"
COS3_COLOR = "#1565C0"
COS1_COLOR = "#C62828"
NC_COLOR = "#E53935"
GENE_COLORS = ["#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]

DOMAIN_COLORS = {
    "Archaea": "#FF9800", "Bacteria": "#E65100",
    "Vertebrata": "#2196F3", "Invertebrata": "#8BC34A",
    "Plantae": "#4CAF50", "Fungi": "#9C27B0",
    "Protista": "#F44336", "Algae": "#00BCD4",
    "Organellar": "#795548", "Virus": "#607D8B",
    "Other_Eukarya": "#FF5722", "Non-coding": "#E53935",
}


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
    elif lin.startswith("Eukarya"): return "Other_Eukarya"
    return "Unknown"


def main():
    # ── Load all data ──
    # Lac operon trajectory
    emb = np.load("results/experiments/fig1_data/ecoli_lacz_emb_blocks10.npy")
    with open("results/experiments/fig1_data/ecoli_lacz_genes.json") as f:
        gene_data = json.load(f)

    # Multi-offset (expanded)
    expanded_path = Path("results/experiments/codon_periodicity/multi_offset_expanded.json")
    if expanded_path.exists():
        with open(expanded_path) as f:
            offset_data = json.load(f)
    else:
        with open("results/experiments/codon_periodicity/multi_offset_cosine.json") as f:
            offset_data = json.load(f)

    # Comprehensive validation
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
        gc = 0
        if fp.exists():
            with open(fp) as f2:
                seq = "".join(l.strip() for l in f2 if not l.startswith(">"))
            gc = sum(1 for c in seq.upper() if c in "GC") / max(len(seq), 1) * 100
        results.append({**entry, **metrics, "gc_content": round(gc, 1)})

    coding = [r for r in results if not r.get("noncoding") and not r.get("category", "").startswith("noncoding")]
    noncoding = [r for r in results if r.get("noncoding") or r.get("category", "").startswith("noncoding")]

    # ═══════════════════════════════════════════════════════════════
    # CREATE FIGURE — 3 rows × 2 columns
    # ═══════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(7.2, 10.5))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.1, 1.0],
                           hspace=0.38, wspace=0.35)

    # ── Panel A: Conceptual schematic ──
    ax_a = fig.add_subplot(gs[0, 0])
    _draw_schematic(ax_a)

    # ── Panel B: Multi-offset comb filter ──
    ax_b = fig.add_subplot(gs[0, 1])
    _draw_comb_filter(ax_b, offset_data)

    # ── Panel C: Genomic trajectory (full width) ──
    ax_c = fig.add_subplot(gs[1, :])
    _draw_trajectory(ax_c, emb, gene_data)

    # ── Panel D: Cross-domain box plot ──
    ax_d = fig.add_subplot(gs[2, 0])
    _draw_boxplot(ax_d, results)

    # ── Panel E: GC independence scatter ──
    ax_e = fig.add_subplot(gs[2, 1])
    _draw_gc_scatter(ax_e, coding, noncoding)

    plt.savefig(OUT_DIR / "fig1.png", dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.savefig(OUT_DIR / "fig1.pdf", bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved to {OUT_DIR}/fig1.png and .pdf")


def _draw_schematic(ax):
    """Panel A: Conceptual diagram — minimal, clear, intuitive."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title("A   The offset-3 cosine inversion", loc="left", fontweight="bold", fontsize=8)

    # Show just 6 bases (2 codons) — big and clear
    bases = ["A", "T", "G", "C", "A", "G"]
    y_dna = 3.2
    box_w = 1.1
    box_h = 0.8
    x_start = 1.6

    # Color: position 1 of each codon highlighted blue, others gray
    for i, base in enumerate(bases):
        x = x_start + i * box_w
        if i % 3 == 0:
            fc = MOD3_COLOR
            alpha = 0.25
            ec_width = 1.5
        else:
            fc = "#E0E0E0"
            alpha = 0.5
            ec_width = 0.5
        rect = Rectangle((x, y_dna), box_w - 0.08, box_h, facecolor=fc, alpha=alpha,
                          edgecolor="black", linewidth=ec_width)
        ax.add_patch(rect)
        ax.text(x + (box_w - 0.08) / 2, y_dna + box_h / 2, base, ha="center", va="center",
                fontsize=11, fontweight="bold", fontfamily="monospace")

    # Codon brackets
    for c in range(2):
        xl = x_start + c * 3 * box_w
        xr = xl + 3 * box_w - 0.08
        ax.plot([xl, xl, xr, xr], [y_dna - 0.08, y_dna - 0.22, y_dna - 0.22, y_dna - 0.08],
                color="black", linewidth=0.8)
        ax.text((xl + xr) / 2, y_dna - 0.45, f"codon {c+1}", ha="center", fontsize=7, color="#555555")

    # Offset-1 arrow: position 0 → position 1 (red, LOW similarity)
    x1 = x_start + 0 * box_w + (box_w - 0.08) / 2
    x2 = x_start + 1 * box_w + (box_w - 0.08) / 2
    y_arr1 = y_dna + box_h + 0.25
    ax.annotate("", xy=(x2, y_arr1), xytext=(x1, y_arr1),
                arrowprops=dict(arrowstyle="->, head_width=0.15", color=COS1_COLOR, lw=2))
    ax.text((x1 + x2) / 2, y_arr1 + 0.18, "offset-1", ha="center", fontsize=7, color=COS1_COLOR,
            fontweight="bold")

    # Offset-3 arrow: position 0 → position 3 (blue, HIGH similarity, same codon position)
    x3 = x_start + 3 * box_w + (box_w - 0.08) / 2
    y_arr3 = y_dna + box_h + 0.9
    ax.annotate("", xy=(x3, y_arr3), xytext=(x1, y_arr3),
                arrowprops=dict(arrowstyle="->, head_width=0.15", color=COS3_COLOR, lw=2,
                                connectionstyle="arc3,rad=-0.15"))
    ax.text((x1 + x3) / 2, y_arr3 + 0.25, "offset-3 (same codon position)",
            ha="center", fontsize=7, color=COS3_COLOR, fontweight="bold")

    # Key result text
    ax.text(5.0, 1.6, "In coding DNA:", fontsize=8, ha="center", fontweight="bold")
    ax.text(5.0, 0.85, "cos(offset-3)  >  cos(offset-1)", fontsize=9, ha="center",
            fontweight="bold", color=MOD3_COLOR)
    ax.text(5.0, 0.15, "In non-coding DNA: relationship inverts",
            fontsize=7, ha="center", color=NC_COLOR, fontstyle="italic")


def _draw_comb_filter(ax, offset_data):
    """Panel B: Multi-offset bar chart with error bars."""
    coding_r = [r for r in offset_data if r.get("is_coding", not r["name"].startswith("a_nc_"))]
    nc_r = [r for r in offset_data if not r.get("is_coding", not r["name"].startswith("a_nc_"))]

    offsets = range(1, 16)
    coding_means = [np.mean([r["offsets"][str(o)] for r in coding_r]) for o in offsets]
    # Use standard error (SE = SD/sqrt(N)) for cleaner bars
    coding_ses = [np.std([r["offsets"][str(o)] for r in coding_r]) / np.sqrt(len(coding_r)) for o in offsets]
    nc_means = [np.mean([r["offsets"][str(o)] for r in nc_r]) for o in offsets]
    nc_ses = [np.std([r["offsets"][str(o)] for r in nc_r]) / np.sqrt(len(nc_r)) for o in offsets]

    x = np.arange(len(offsets))
    width = 0.35

    colors_cod = [MOD3_COLOR if o % 3 == 0 else NONMOD3_COLOR for o in offsets]
    ax.bar(x - width / 2, coding_means, width, color=colors_cod,
           edgecolor="black", linewidth=0.3, yerr=coding_ses, capsize=1.5,
           error_kw={"linewidth": 0.5, "capthick": 0.5},
           label=f"Coding (N={len(coding_r)})")
    ax.bar(x + width / 2, nc_means, width, color=NC_COLOR, alpha=0.4,
           edgecolor="black", linewidth=0.3, yerr=nc_ses, capsize=1.5,
           error_kw={"linewidth": 0.5, "capthick": 0.5},
           label=f"Non-coding (N={len(nc_r)})")

    ax.set_xlabel("Offset (nucleotides)")
    ax.set_ylabel("Mean cosine similarity (± SE)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(o) for o in offsets])
    ax.set_title("B   Triplet periodicity in embedding space", loc="left", fontweight="bold", fontsize=8)
    ax.legend(loc="upper right", fontsize=5.5, framealpha=0.9)

    # More headroom
    ymax = max(max(nc_means), max(coding_means)) + 0.10
    ax.set_ylim(0, float(ymax))

    # Label mod-3 bars with values
    for i, o in enumerate(offsets):
        if o % 3 == 0:
            ax.text(i - width / 2, coding_means[i] + coding_ses[i] + 0.015,
                    f"{coding_means[i]:.2f}", ha="center", va="bottom",
                    fontsize=5.5, color=MOD3_COLOR, fontweight="bold")


def _draw_trajectory(ax, emb, gene_data):
    """Panel C: Cosine trajectory along E. coli lac operon."""
    genes = gene_data["genes"]
    norms = np.linalg.norm(emb, axis=1)

    # Compute smoothed cosine
    window = 100
    cos1_raw = np.zeros(len(emb) - 1)
    cos3_raw = np.zeros(len(emb) - 3)
    for i in range(len(emb) - 1):
        ni = norms[i]
        ni1 = norms[i + 1]
        if ni > 0 and ni1 > 0:
            cos1_raw[i] = np.dot(emb[i], emb[i + 1]) / (ni * ni1)
    for i in range(len(emb) - 3):
        ni = norms[i]
        ni3 = norms[i + 3]
        if ni > 0 and ni3 > 0:
            cos3_raw[i] = np.dot(emb[i], emb[i + 3]) / (ni * ni3)

    kernel = np.ones(window) / window
    cos1_smooth = np.convolve(cos1_raw, kernel, mode="same")
    cos3_smooth = np.convolve(cos3_raw, kernel, mode="same")

    x = np.arange(len(cos1_smooth))
    ax.plot(x, cos1_smooth, color=COS1_COLOR, linewidth=0.7, alpha=0.8, label="cos(offset-1)")
    ax.plot(x[:len(cos3_smooth)], cos3_smooth, color=COS3_COLOR, linewidth=0.7, alpha=0.8, label="cos(offset-3)")

    # Fill coding regions where cos3 > cos1
    for i in range(0, len(cos3_smooth) - 1, 5):
        end_i = min(i + 5, len(cos3_smooth))
        if cos3_smooth[i] > cos1_smooth[i]:
            ax.fill_between(range(i, end_i), cos1_smooth[i:end_i], cos3_smooth[i:end_i],
                            alpha=0.1, color=COS3_COLOR)

    # Mark intergenic
    sorted_genes = sorted(genes, key=lambda g: g["start"])
    for i in range(len(sorted_genes) - 1):
        ig_s = sorted_genes[i]["end"]
        ig_e = sorted_genes[i + 1]["start"]
        if ig_e - ig_s > 20:
            ax.axvspan(ig_s, ig_e, alpha=0.08, color="#888888", zorder=0)

    if sorted_genes[0]["start"] > 50:
        ax.axvspan(0, sorted_genes[0]["start"], alpha=0.08, color="#888888", zorder=0)

    # Gene annotations below
    ylim = ax.get_ylim()
    gene_y = ylim[0] - (ylim[1] - ylim[0]) * 0.10
    gene_h = (ylim[1] - ylim[0]) * 0.04
    for i, g in enumerate(genes):
        color = GENE_COLORS[i % len(GENE_COLORS)]
        rect = Rectangle((g["start"], gene_y), g["end"] - g["start"], gene_h,
                          facecolor=color, edgecolor="black", linewidth=0.3, clip_on=False, zorder=5)
        ax.add_patch(rect)
        mid = (g["start"] + g["end"]) / 2
        ax.text(mid, gene_y - (ylim[1] - ylim[0]) * 0.025, g["name"],
                ha="center", va="top", fontsize=7.5, fontstyle="italic", fontweight="bold", zorder=6)

    ax.set_ylim(ylim[0] - (ylim[1] - ylim[0]) * 0.2, ylim[1])
    ax.set_xlabel("Position in E. coli K12 lac operon (bp)")
    ax.set_ylabel("Cosine similarity (100bp smoothed)")
    ax.set_title("C   Offset-3 cosine inversion along a bacterial genome",
                  loc="left", fontweight="bold", fontsize=8)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.9)
    ax.set_xlim(0, len(cos1_smooth))

    # Add annotations
    ax.text(500, ax.get_ylim()[1] * 0.92, "intergenic", fontsize=7, color="#888888",
            fontstyle="italic", ha="center")
    ax.text(2500, ax.get_ylim()[1] * 0.92, "coding: cos3 > cos1", fontsize=7, color=COS3_COLOR,
            fontweight="bold", ha="center")


def _draw_boxplot(ax, results):
    """Panel D: Cross-domain box plot."""
    domain_data = defaultdict(list)
    for r in results:
        d = get_domain(r)
        if r.get("noncoding") or r.get("category", "").startswith("noncoding"):
            cat = r.get("category", "")
            if cat in ("noncoding_tRNA", "noncoding_intergenic"):
                continue
            d = "Non-coding"
        gap = r.get("inversion_gap", r["cos3"] - r["cos1"])
        domain_data[d].append(gap)

    domain_order = ["Archaea", "Bacteria", "Vertebrata", "Invertebrata", "Plantae", "Fungi",
                    "Protista", "Algae", "Organellar", "Virus", "Non-coding"]
    domain_order = [d for d in domain_order if d in domain_data]

    labels = {
        "Archaea": "Archaea", "Bacteria": "Bacteria", "Vertebrata": "Vertebr.",
        "Invertebrata": "Invertebr.", "Plantae": "Plantae", "Fungi": "Fungi",
        "Protista": "Protista", "Algae": "Algae", "Organellar": "Organellar",
        "Virus": "Virus", "Non-coding": "Non-coding",
    }
    short = [labels.get(d, d) for d in domain_order]

    bp = ax.boxplot([domain_data[d] for d in domain_order], tick_labels=short,
                     patch_artist=True, widths=0.6,
                     medianprops=dict(color="black", linewidth=0.8),
                     whiskerprops=dict(linewidth=0.5),
                     capprops=dict(linewidth=0.5),
                     flierprops=dict(markersize=2),
                     boxprops=dict(linewidth=0.5))

    for patch, d in zip(bp["boxes"], domain_order):
        patch.set_facecolor(DOMAIN_COLORS.get(d, "#666"))
        patch.set_alpha(0.6)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_ylabel("Inversion signal (cos3 - cos1)")
    ax.set_title("D   Universal across all domains (N=459, 55 phyla)",
                  loc="left", fontweight="bold", fontsize=8)
    ax.tick_params(axis="x", rotation=45, labelsize=5.5)


def _draw_gc_scatter(ax, coding, noncoding):
    """Panel E: GC content vs inversion signal scatter plot."""
    ax.set_title("E   GC-independent (459 coding sequences)", loc="left", fontweight="bold", fontsize=8)

    # Plot coding sequences colored by domain
    for r in coding:
        d = get_domain(r)
        color = DOMAIN_COLORS.get(d, "#666666")
        gap = r.get("inversion_gap", r["cos3"] - r["cos1"])
        gc = r.get("gc_content", 0)
        if gc > 0:
            ax.scatter(gc, gap, c=color, s=12, alpha=0.5, edgecolors="none", zorder=3)

    # Non-coding controls (excluding tRNA and intergenic)
    for r in noncoding:
        cat = r.get("category", "")
        if cat in ("noncoding_tRNA", "noncoding_intergenic"):
            continue
        gap = r.get("inversion_gap", r["cos3"] - r["cos1"])
        gc = r.get("gc_content", 0)
        if gc > 0:
            ax.scatter(gc, gap, c=NC_COLOR, s=30, marker="x", linewidth=1.2, zorder=5)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("GC content (%)")
    ax.set_ylabel("Inversion signal (cos3 - cos1)")

    # Compute and show correlation
    gcs = [r["gc_content"] for r in coding if r.get("gc_content", 0) > 0]
    gaps = [r.get("inversion_gap", r["cos3"] - r["cos1"]) for r in coding if r.get("gc_content", 0) > 0]
    r_val = float(np.corrcoef(gcs, gaps)[0, 1])
    ax.text(0.95, 0.05, f"r = {r_val:.2f}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="#666666", fontstyle="italic")

    # Label extreme edge cases
    extremes = {
        "a_toxoplasma_apicoplast_tufA": ("apicoplast\n9.8% GC", -12, 8),
        "a_streptomyces_rpoBII": ("Streptomyces\n78% GC", -12, -12),
        "a_human_mt_co1": ("mt CO1\n(UGA=Trp)", 8, -5),
    }
    for r in coding:
        if r["name"] in extremes:
            label, dx, dy = extremes[r["name"]]
            gap = r.get("inversion_gap", r["cos3"] - r["cos1"])
            gc = r["gc_content"]
            ax.annotate(label, xy=(gc, gap), xytext=(dx, dy), textcoords="offset points",
                        fontsize=5, ha="left", color="#444444",
                        arrowprops=dict(arrowstyle="->", color="#999999", lw=0.4))


if __name__ == "__main__":
    main()
