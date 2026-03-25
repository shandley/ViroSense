#!/usr/bin/env python3
"""Generate multi-organism summary figure with normalized metrics (MCC, per-gene detection)."""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────

BASE_DIR = Path("results/experiments/nonmodel_genome")
SMOOTH_W = 200

ORGANISMS = [
    {
        "prefix": "zt_chr1_3000000_3500000",
        "name": "Z. tritici",
        "common": "Wheat pathogen fungus",
        "kingdom": "Fungi",
    },
    {
        "prefix": "danaus_plexippus_5830000_6330000",
        "name": "D. plexippus",
        "common": "Monarch butterfly",
        "kingdom": "Arthropoda",
    },
    {
        "prefix": "physcomitrium_patens_16665000_17165000",
        "name": "P. patens",
        "common": "Spreading earthmoss",
        "kingdom": "Bryophyta",
    },
    {
        "prefix": "acropora_millepora_9850000_10350000",
        "name": "A. millepora",
        "common": "Staghorn coral",
        "kingdom": "Cnidaria",
    },
    {
        "prefix": "dictyostelium_discoideum_3820124_4320124",
        "name": "D. discoideum",
        "common": "Social amoeba",
        "kingdom": "Amoebozoa",
    },
    {
        "prefix": "magallana_gigas_41855000_42355000",
        "name": "M. gigas",
        "common": "Pacific oyster",
        "kingdom": "Mollusca",
    },
]

KINGDOM_COLORS = {
    "Arthropoda": "#E91E63",
    "Bryophyta": "#4CAF50",
    "Cnidaria": "#FF9800",
    "Amoebozoa": "#9C27B0",
    "Mollusca": "#2196F3",
    "Fungi": "#795548",
}

# ── Helpers ────────────────────────────────────────────────────────────────


def smooth(arr: np.ndarray, w: int) -> np.ndarray:
    """Uniform 1-D moving average."""
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


def build_coding_mask(cds_list: list[dict], length: int) -> np.ndarray:
    """Boolean mask: True = coding position."""
    mask = np.zeros(length, dtype=bool)
    for c in cds_list:
        mask[c["start"] : c["end"]] = True
    return mask


def compute_metrics(
    cos1: np.ndarray, cos3: np.ndarray, cds: list[dict], genes: list[dict]
) -> dict:
    """Compute per-position recall/FPR/MCC and per-gene detection rate."""
    n = len(cos1)
    inv = smooth(cos3 - cos1, SMOOTH_W)
    pred_coding = inv > 0
    true_coding = build_coding_mask(cds, n)

    tp = int(np.sum(pred_coding & true_coding))
    fp = int(np.sum(pred_coding & ~true_coding))
    tn = int(np.sum(~pred_coding & ~true_coding))
    fn = int(np.sum(~pred_coding & true_coding))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    # Per-gene detection: fraction of genes where >50% of CDS positions detected
    genes_detected = 0
    for g in genes:
        # Collect CDS intervals within this gene
        gstart, gend = g["start"], g["end"]
        gene_cds_mask = np.zeros(gend - gstart, dtype=bool)
        gene_pred_mask = pred_coding[gstart:gend]
        for c in cds:
            cs = max(c["start"], gstart) - gstart
            ce = min(c["end"], gend) - gstart
            if ce > cs:
                gene_cds_mask[cs:ce] = True
        cds_positions = np.sum(gene_cds_mask)
        if cds_positions > 0:
            detected = np.sum(gene_pred_mask & gene_cds_mask)
            if detected / cds_positions > 0.5:
                genes_detected += 1

    n_genes = len(genes)
    gene_detection_rate = genes_detected / n_genes if n_genes > 0 else 0.0

    return {
        "recall": recall,
        "fpr": fpr,
        "precision": precision,
        "mcc": mcc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "genes_detected": genes_detected,
        "n_genes": n_genes,
        "gene_detection_rate": gene_detection_rate,
    }


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    results = []
    for org in ORGANISMS:
        prefix = org["prefix"]
        cosines = np.load(BASE_DIR / f"{prefix}_cosines.npz")
        with open(BASE_DIR / f"{prefix}_genes.json") as f:
            ann = json.load(f)

        metrics = compute_metrics(
            cosines["cos1"], cosines["cos3"], ann["cds"], ann["genes"]
        )
        metrics.update(
            {
                "name": org["name"],
                "common": org["common"],
                "kingdom": org["kingdom"],
                "prefix": prefix,
            }
        )
        results.append(metrics)
        print(
            f"{org['name']:20s}  MCC={metrics['mcc']:.3f}  "
            f"Recall={metrics['recall']:.3f}  FPR={metrics['fpr']:.3f}  "
            f"GeneDetect={metrics['genes_detected']}/{metrics['n_genes']} "
            f"({metrics['gene_detection_rate']:.1%})"
        )

    # Save JSON
    out_json = BASE_DIR / "multi_organism_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics: {out_json}")

    # ── Figure ─────────────────────────────────────────────────────────────
    plt.rcParams.update(
        {
            "font.size": 8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "savefig.dpi": 200,
            "axes.linewidth": 0.5,
        }
    )

    fig, (ax_gene, ax_mcc, ax_scatter) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Unsupervised gene detection across non-model organisms\n"
        "(no training, no species model, no reference genome)",
        fontsize=10,
        fontweight="bold",
        y=0.98,
    )

    labels = [r["name"] for r in results]
    kingdoms = [r["kingdom"] for r in results]
    colors = [KINGDOM_COLORS[k] for k in kingdoms]

    # Panel A: Per-gene detection rate
    gene_rates = [r["gene_detection_rate"] * 100 for r in results]
    y_pos = np.arange(len(results))
    bars_a = ax_gene.barh(y_pos, gene_rates, color=colors, edgecolor="white", height=0.6)
    ax_gene.set_yticks(y_pos)
    ax_gene.set_yticklabels([f"{l} ({k})" for l, k in zip(labels, kingdoms)], fontsize=7)
    ax_gene.set_xlabel("Gene detection rate (%)")
    ax_gene.set_title("A. Per-gene detection (>50% CDS overlap)", fontsize=9, fontweight="bold")
    ax_gene.set_xlim(0, 110)
    ax_gene.invert_yaxis()
    for i, r in enumerate(results):
        pct = r["gene_detection_rate"] * 100
        txt = f"{r['genes_detected']}/{r['n_genes']} ({pct:.0f}%)"
        ax_gene.text(pct + 1.5, i, txt, va="center", fontsize=7)

    # Panel B: MCC
    mccs = [r["mcc"] for r in results]
    bars_b = ax_mcc.barh(y_pos, mccs, color=colors, edgecolor="white", height=0.6)
    ax_mcc.set_yticks(y_pos)
    ax_mcc.set_yticklabels([f"{l} ({k})" for l, k in zip(labels, kingdoms)], fontsize=7)
    ax_mcc.set_xlabel("MCC")
    ax_mcc.set_title("B. Matthews Correlation Coefficient", fontsize=9, fontweight="bold")
    ax_mcc.set_xlim(0, 1.0)
    ax_mcc.invert_yaxis()
    for i, m in enumerate(mccs):
        ax_mcc.text(m + 0.02, i, f"{m:.3f}", va="center", fontsize=7)

    # Panel C: Recall vs FPR scatter
    recalls = [r["recall"] for r in results]
    fprs = [r["fpr"] for r in results]
    for i, r in enumerate(results):
        ax_scatter.scatter(
            r["fpr"],
            r["recall"],
            c=colors[i],
            s=80,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
            label=f"{r['name']} ({r['kingdom']})",
        )
    ax_scatter.set_xlabel("False Positive Rate")
    ax_scatter.set_ylabel("Recall (Sensitivity)")
    ax_scatter.set_title("C. Recall vs FPR", fontsize=9, fontweight="bold")
    ax_scatter.set_xlim(-0.05, 1.05)
    ax_scatter.set_ylim(-0.05, 1.05)
    ax_scatter.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.5)
    ax_scatter.legend(fontsize=6, loc="lower right", framealpha=0.9)
    ax_scatter.set_aspect("equal")

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_png = BASE_DIR / "multi_organism_summary.png"
    out_pdf = BASE_DIR / "multi_organism_summary.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved figure: {out_png}")
    print(f"Saved figure: {out_pdf}")


if __name__ == "__main__":
    main()
