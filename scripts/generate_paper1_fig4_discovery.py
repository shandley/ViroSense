#!/usr/bin/env python3
"""
Paper 1, Figure 4: Non-Model Organism Gene Discovery

Panels:
A: Per-gene detection rate across 6 organisms (5 kingdoms)
B: RNA-seq transcription enrichment at FP positions (5 organisms)
C: Reannotation validation: 91% of novel genes detected (Z. tritici)
D: Example region showing FP predictions validated as novel genes
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

OUT_DIR = Path("results/paper1/figures")
DATA_DIR = Path("results/experiments/nonmodel_genome")

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

KINGDOM_COLORS = {
    "Amoebozoa": "#9C27B0",
    "Fungi": "#795548",
    "Arthropoda": "#E91E63",
    "Bryophyta": "#4CAF50",
    "Cnidaria": "#FF9800",
    "Mollusca": "#2196F3",
}


def load_organism_data():
    """Load cached cosines and annotations, compute all metrics."""
    organisms = [
        {"name": "dictyostelium_discoideum", "common": "Social amoeba",
         "kingdom": "Amoebozoa", "prefix": "dictyostelium_discoideum_3820124_4320124"},
        {"name": "zymoseptoria_tritici", "common": "Wheat pathogen fungus",
         "kingdom": "Fungi", "prefix": "zt_chr1_3000000_3500000"},
        {"name": "danaus_plexippus", "common": "Monarch butterfly",
         "kingdom": "Arthropoda", "prefix": "danaus_plexippus_5830000_6330000"},
        {"name": "physcomitrium_patens", "common": "Spreading earthmoss",
         "kingdom": "Bryophyta", "prefix": "physcomitrium_patens_16665000_17165000"},
        {"name": "acropora_millepora", "common": "Staghorn coral",
         "kingdom": "Cnidaria", "prefix": "acropora_millepora_9850000_10350000"},
        {"name": "magallana_gigas", "common": "Pacific oyster",
         "kingdom": "Mollusca", "prefix": "magallana_gigas_41855000_42355000"},
    ]

    results = []
    for org in organisms:
        cached = np.load(DATA_DIR / f"{org['prefix']}_cosines.npz")
        cos1, cos3 = cached["cos1"], cached["cos3"]
        with open(DATA_DIR / f"{org['prefix']}_genes.json") as f:
            ann = json.load(f)
        with open(DATA_DIR / f"{org['prefix']}.fasta") as f:
            seq = "".join(l.strip() for l in f if not l.startswith(">"))
        seq_len = len(seq)

        kernel = np.ones(200) / 200
        cos1_s = np.convolve(cos1[:seq_len], kernel, mode="same")
        cos3_s = np.convolve(cos3[:seq_len], kernel, mode="same")
        inversion = cos3_s - cos1_s

        gene_track = np.zeros(seq_len)
        for cds in ann.get("cds", []):
            s, e = max(0, cds["start"]), min(cds["end"], seq_len)
            gene_track[s:e] = 1

        predicted = (inversion > 0).astype(int)
        truth = gene_track.astype(int)

        tp = float(((predicted == 1) & (truth == 1)).sum())
        fp = float(((predicted == 1) & (truth == 0)).sum())
        fn = float(((predicted == 0) & (truth == 1)).sum())
        tn = float(((predicted == 0) & (truth == 0)).sum())

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0

        # Per-gene detection
        genes_detected = 0
        genes_total = 0
        for gene in ann.get("genes", []):
            gs, ge = max(0, gene["start"]), min(gene["end"], seq_len)
            if ge <= gs:
                continue
            gene_cds = np.zeros(ge - gs)
            for cds in ann.get("cds", []):
                cs = max(0, cds["start"] - gs)
                ce = min(cds["end"] - gs, ge - gs)
                if ce > cs:
                    gene_cds[cs:ce] = 1
            if gene_cds.sum() < 10:
                continue
            gene_pred = predicted[gs:ge]
            gene_recall = (gene_pred[gene_cds.astype(bool)] == 1).sum() / gene_cds.sum()
            genes_total += 1
            if gene_recall > 0.5:
                genes_detected += 1

        results.append({
            **org,
            "recall": recall,
            "fpr": fpr,
            "mcc": mcc,
            "coding_fraction": gene_track.mean(),
            "genes_detected": genes_detected,
            "genes_total": genes_total,
            "gene_detection_rate": genes_detected / genes_total if genes_total > 0 else 0,
        })

    return results


def load_reannotation_data():
    """Cross-reference Zt predictions against reannotation."""
    GFF = DATA_DIR / "zt_reannotation/zt_reannot.gff3"
    REGION_START = 3_000_000
    REGION_END = 3_500_000

    # Parse reannotation
    reannot_genes = []
    reannot_cds = []
    with open(GFF) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9 or parts[0] != "chr_1":
                continue
            start = int(parts[3]) - 1
            end = int(parts[4])
            if end < REGION_START or start > REGION_END:
                continue
            if parts[2] == "gene":
                gene_id = ""
                for attr in parts[8].split(";"):
                    if attr.startswith("ID="):
                        gene_id = attr.split("=")[1]
                reannot_genes.append({
                    "id": gene_id, "start": start, "end": end,
                    "local_start": max(0, start - REGION_START),
                    "local_end": min(end - REGION_START, REGION_END - REGION_START),
                })
            elif parts[2] == "CDS":
                reannot_cds.append({
                    "start": max(0, start - REGION_START),
                    "end": min(end - REGION_START, REGION_END - REGION_START),
                })

    # Load original annotation
    with open(DATA_DIR / "zt_chr1_3000000_3500000_genes.json") as f:
        orig_ann = json.load(f)

    # Find novel genes
    orig_gene_set = set()
    for g in orig_ann["genes"]:
        for pos in range(g["start"], g["end"]):
            orig_gene_set.add(pos)

    novel_genes = []
    for g in reannot_genes:
        ls, le = g["local_start"], g["local_end"]
        gene_len = le - ls
        if gene_len <= 0:
            continue
        overlap = sum(1 for p in range(ls, le) if p in orig_gene_set)
        if overlap / gene_len < 0.3:
            novel_genes.append(g)

    # Load predictions
    cached = np.load(DATA_DIR / "zt_chr1_3000000_3500000_cosines.npz")
    cos1, cos3 = cached["cos1"], cached["cos3"]
    with open(DATA_DIR / "zt_chr1_3000000_3500000.fasta") as f:
        seq = "".join(l.strip() for l in f if not l.startswith(">"))
    seq_len = len(seq)

    kernel = np.ones(200) / 200
    cos1_s = np.convolve(cos1[:seq_len], kernel, mode="same")
    cos3_s = np.convolve(cos3[:seq_len], kernel, mode="same")
    inversion = cos3_s - cos1_s
    predicted = (inversion > 0).astype(int)

    # Original and reannotation tracks
    orig_track = np.zeros(seq_len)
    for cds in orig_ann.get("cds", []):
        s, e = max(0, cds["start"]), min(cds["end"], seq_len)
        orig_track[s:e] = 1

    reannot_track = np.zeros(seq_len)
    for cds in reannot_cds:
        s, e = max(0, cds["start"]), min(cds["end"], seq_len)
        reannot_track[s:e] = 1

    # Novel gene detection
    novel_detected = 0
    novel_results = []
    for g in novel_genes:
        ls, le = g["local_start"], g["local_end"]
        gene_len = le - ls
        if gene_len < 10:
            continue
        gene_pred = predicted[ls:le]
        det_frac = gene_pred.mean()
        novel_results.append({"id": g["id"], "len": gene_len, "detected": det_frac})
        if det_frac > 0.5:
            novel_detected += 1

    # FP reclassification
    fp_mask = (predicted == 1) & (orig_track == 0)
    fp_now_coding = (fp_mask & (reannot_track == 1)).sum()

    return {
        "novel_genes": novel_results,
        "novel_detected": novel_detected,
        "novel_total": len(novel_results),
        "fp_reclassified_frac": fp_now_coding / fp_mask.sum() if fp_mask.sum() > 0 else 0,
        "inversion": inversion,
        "predicted": predicted,
        "orig_track": orig_track,
        "reannot_track": reannot_track,
        "seq_len": seq_len,
    }


def main():
    org_results = load_organism_data()
    reannot = load_reannotation_data()

    fig = plt.figure(figsize=(7.2, 7.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # ══════════════════════════════════════════════════════════════
    # Panel A: Per-gene detection rate
    # ══════════════════════════════════════════════════════════════
    ax_a = fig.add_subplot(gs[0, 0])

    names = [r["common"] for r in org_results]
    kingdoms = [r["kingdom"] for r in org_results]
    gene_rates = [r["gene_detection_rate"] * 100 for r in org_results]
    colors = [KINGDOM_COLORS[k] for k in kingdoms]

    bars = ax_a.barh(range(len(names)), gene_rates, color=colors, alpha=0.7,
                     edgecolor="black", linewidth=0.3, height=0.6)
    ax_a.set_yticks(range(len(names)))
    ax_a.set_yticklabels([f"{n}\n({k})" for n, k in zip(names, kingdoms)], fontsize=6)
    ax_a.set_xlabel("Gene detection rate (%)")
    ax_a.set_title("A   Per-gene detection (>50% CDS overlap)", loc="left",
                    fontweight="bold", fontsize=8)
    ax_a.set_xlim(0, 108)
    ax_a.invert_yaxis()

    for bar, val, r in zip(bars, gene_rates, org_results):
        ax_a.text(val + 1, bar.get_y() + bar.get_height() / 2,
                  f"{r['genes_detected']}/{r['genes_total']} ({val:.0f}%)",
                  va="center", fontsize=6, fontweight="bold")

    # ══════════════════════════════════════════════════════════════
    # Panel B: RNA-seq transcription enrichment at FP positions
    # ══════════════════════════════════════════════════════════════
    ax_b = fig.add_subplot(gs[0, 1])

    # Load RNA-seq validation results
    rnaseq_data = []
    rnaseq_orgs = [
        ("physcomitrium_patens", "Earthmoss", "Bryophyta"),
        ("dictyostelium_discoideum", "Social amoeba", "Amoebozoa"),
        ("acropora_millepora", "Staghorn coral", "Cnidaria"),
        ("magallana_gigas", "Pacific oyster", "Mollusca"),
        ("danaus_plexippus", "Monarch butterfly", "Arthropoda"),
    ]
    for org_name, common, kingdom in rnaseq_orgs:
        val_path = DATA_DIR / f"{org_name}_rnaseq_validation.json"
        if val_path.exists():
            with open(val_path) as f:
                val = json.load(f)
            rnaseq_data.append({
                "common": common, "kingdom": kingdom,
                "enrichment": val.get("fp_transcription_enrichment", 1.0),
                "fp_pct": val["categories"].get("False Positive", {}).get("pct_covered", 0),
                "tn_pct": val["categories"].get("True Negative", {}).get("pct_covered", 0),
            })

    if rnaseq_data:
        rna_names = [r["common"] for r in rnaseq_data]
        rna_kingdoms = [r["kingdom"] for r in rnaseq_data]
        enrichments = [r["enrichment"] for r in rnaseq_data]
        rna_colors = [KINGDOM_COLORS.get(k, "#666") for k in rna_kingdoms]

        bars_b = ax_b.barh(range(len(rna_names)), enrichments, color=rna_colors, alpha=0.7,
                           edgecolor="black", linewidth=0.3, height=0.6)
        ax_b.axvline(1.0, color="#999", linewidth=0.8, linestyle="--", zorder=1)
        ax_b.set_yticks(range(len(rna_names)))
        ax_b.set_yticklabels([f"{n}\n({k})" for n, k in zip(rna_names, rna_kingdoms)],
                             fontsize=6)
        ax_b.set_xlabel("FP / TN transcription enrichment")
        ax_b.set_title("B   RNA-seq validates 'false positives'", loc="left",
                        fontweight="bold", fontsize=8)
        ax_b.invert_yaxis()

        for bar, val_e in zip(bars_b, enrichments):
            ax_b.text(val_e + 0.02, bar.get_y() + bar.get_height() / 2,
                      f"{val_e:.2f}x",
                      va="center", fontsize=6, fontweight="bold")

        ax_b.text(0.97, 0.97, "All p < 1e-56",
                  transform=ax_b.transAxes, fontsize=6, ha="right", va="top",
                  fontstyle="italic", color="#555")

    # ══════════════════════════════════════════════════════════════
    # Panel C: Reannotation validation
    # ══════════════════════════════════════════════════════════════
    ax_c = fig.add_subplot(gs[1, 0])

    novel = reannot.get("novel_genes", [])
    novel_lens = [g["len"] for g in novel]
    novel_det = [g["detected"] * 100 for g in novel]
    novel_colors = ["#4CAF50" if d > 50 else "#F44336" for d in novel_det]

    ax_c.scatter(novel_lens, novel_det, c=novel_colors, s=30, alpha=0.7,
                 edgecolors="black", linewidth=0.3, zorder=3)
    ax_c.axhline(50, color="#999", linewidth=0.5, linestyle="--")
    ax_c.set_xlabel("Novel gene length (bp)")
    ax_c.set_ylabel("Detection (%)")
    ax_c.set_title("C   Novel gene detection (Z. tritici reannotation)",
                    loc="left", fontweight="bold", fontsize=8)
    ax_c.set_ylim(-5, 108)

    n_det = reannot["novel_detected"]
    n_tot = reannot["novel_total"]
    fp_pct = reannot["fp_reclassified_frac"] * 100
    ax_c.text(0.97, 0.05,
              f"{n_det}/{n_tot} novel genes detected ({n_det/n_tot:.0%})\n"
              f"{fp_pct:.0f}% of 'false positives' are\ncoding in reannotation",
              transform=ax_c.transAxes, fontsize=6, ha="right", va="bottom",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9",
                        edgecolor="#4CAF50", alpha=0.9))

    # ══════════════════════════════════════════════════════════════
    # Panel D: Example region showing validated predictions
    # ══════════════════════════════════════════════════════════════
    ax_d = fig.add_subplot(gs[1, 1])

    # Show a 50kb window with novel genes visible
    # Find a region with novel genes
    view_start = 25_000  # local coords (= 3,025,000 genomic)
    view_end = 75_000    # local coords (= 3,075,000 genomic)
    x = np.arange(view_start, view_end) / 1000  # kb

    inv = reannot["inversion"][view_start:view_end]
    orig = reannot["orig_track"][view_start:view_end]
    rean = reannot["reannot_track"][view_start:view_end]

    # Inversion signal
    ax_d.fill_between(x, 0, inv, where=inv > 0, alpha=0.4, color="#1565C0",
                      label="Predicted coding")
    ax_d.fill_between(x, 0, inv, where=inv < 0, alpha=0.3, color="#C62828")
    ax_d.axhline(0, color="black", linewidth=0.3)

    # Gene tracks as colored bars at bottom
    y_orig = -0.12
    y_rean = -0.17
    for i in range(view_start, view_end - 1):
        if orig[i - view_start]:
            ax_d.plot([i / 1000, (i + 1) / 1000], [y_orig, y_orig],
                      color="#4CAF50", linewidth=3, solid_capstyle="butt")
        if rean[i - view_start] and not orig[i - view_start]:
            ax_d.plot([i / 1000, (i + 1) / 1000], [y_rean, y_rean],
                      color="#FF6F00", linewidth=3, solid_capstyle="butt")
        elif rean[i - view_start]:
            ax_d.plot([i / 1000, (i + 1) / 1000], [y_rean, y_rean],
                      color="#4CAF50", linewidth=3, solid_capstyle="butt")

    ax_d.set_xlabel("Position (kb, local)")
    ax_d.set_ylabel("cos3 − cos1")
    ax_d.set_title("D   Example: novel genes in Z. tritici",
                    loc="left", fontweight="bold", fontsize=8)
    ax_d.set_ylim(-0.22, 0.5)

    # Manual legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#1565C0", linewidth=4, alpha=0.4, label="Evo2 prediction"),
        Line2D([0], [0], color="#4CAF50", linewidth=3, label="NCBI annotation"),
        Line2D([0], [0], color="#FF6F00", linewidth=3, label="Novel genes (reannotation)"),
    ]
    ax_d.legend(handles=legend_elements, fontsize=5.5, loc="upper right")

    plt.savefig(OUT_DIR / "fig4.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_DIR / "fig4.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved fig4.png and fig4.pdf")


if __name__ == "__main__":
    main()
