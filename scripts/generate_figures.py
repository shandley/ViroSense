"""Generate publication-quality figures from ViroSense benchmark results.

Reads structured results from results/ and produces PDF/PNG figures
suitable for manuscript inclusion.

Usage:
    # After running benchmarks:
    python scripts/generate_figures.py --results-dir results/

    # Single tier (if only one has run):
    python scripts/generate_figures.py --results-dir results/ --tier 40b

    # Classifier training only (no benchmark needed):
    python scripts/generate_figures.py --results-dir results/ --training-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# Publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Tier colors
COLORS = {
    "40b": "#2166AC",  # Blue
    "7b": "#B2182B",   # Red
}
TIER_LABELS = {
    "40b": "Evo2 40B (8,192-D)",
    "7b": "Evo2 7B (4,096-D)",
}

# Category colors
CAT_COLORS = {
    "phage": "#D32F2F",
    "rna_virus": "#F57C00",
    "chromosome": "#1976D2",
    "plasmid": "#388E3C",
    "cellular": "#1976D2",
}


def load_tier_data(results_dir: Path, tier: str) -> dict | None:
    """Load benchmark results for a tier, returns None if not available."""
    metrics_path = results_dir / "benchmark" / tier / "benchmark_results.json"
    details_path = results_dir / "benchmark" / tier / "detailed_results.tsv"
    if not metrics_path.exists():
        return None
    data = {"metrics": json.loads(metrics_path.read_text())}
    if details_path.exists():
        data["details"] = pd.read_csv(details_path, sep="\t")
    return data


def load_classifier_metrics(results_dir: Path, tier: str) -> dict | None:
    """Load classifier training metrics."""
    path = results_dir / "classifiers" / tier / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Figure 1: Classifier Training Summary
# ---------------------------------------------------------------------------

def fig_training_summary(results_dir: Path, output_dir: Path) -> None:
    """Confusion matrices and calibration for both tiers."""
    tiers = {}
    for tier in ["40b", "7b"]:
        m = load_classifier_metrics(results_dir, tier)
        if m:
            tiers[tier] = m

    if not tiers:
        print("No classifier metrics found, skipping training summary")
        return

    n_tiers = len(tiers)
    fig, axes = plt.subplots(1, n_tiers, figsize=(4 * n_tiers, 3.5))
    if n_tiers == 1:
        axes = [axes]

    for ax, (tier, m) in zip(axes, tiers.items()):
        cm = np.array(m["confusion_matrix"])
        labels = m["confusion_matrix_labels"]

        # Normalized confusion matrix
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")

        # Annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = cm[i, j]
                pct = cm_norm[i, j]
                color = "white" if pct > 0.5 else "black"
                ax.text(j, i, f"{val}\n({pct:.1%})", ha="center", va="center",
                        color=color, fontsize=9)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        acc = m["accuracy"]
        auc = m.get("auc", 0)
        ece = m.get("ece", m.get("ece_uncalibrated", 0))
        ax.set_title(f"{TIER_LABELS.get(tier, tier)}\nAcc={acc:.1%}  AUC={auc:.3f}  ECE={ece:.3f}")

    fig.suptitle("Classifier Training (held-out test set)", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "fig_training_summary.pdf")
    fig.savefig(output_dir / "fig_training_summary.png")
    plt.close(fig)
    print(f"  Saved: fig_training_summary.pdf")


# ---------------------------------------------------------------------------
# Figure 2: ROC Curves
# ---------------------------------------------------------------------------

def fig_roc_curves(results_dir: Path, output_dir: Path) -> None:
    """ROC curves for each dataset, both tiers overlaid."""
    from sklearn.metrics import roc_curve, auc

    datasets = ["gyp", "rna_virus"]
    dataset_titles = {"gyp": "GYP Benchmark (Phage Detection)", "rna_virus": "RNA Virus (Zero-Shot)"}

    tier_data = {}
    for tier in ["40b", "7b"]:
        d = load_tier_data(results_dir, tier)
        if d and "details" in d:
            tier_data[tier] = d

    if not tier_data:
        print("No benchmark results found, skipping ROC curves")
        return

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        for tier, d in tier_data.items():
            df = d["details"]

            if ds == "gyp":
                ds_df = df[df["dataset"] == "gyp"]
                y_true = (ds_df["true_category"] == "phage").astype(int).values
            elif ds == "rna_virus":
                ds_df = df[df["dataset"].isin(["rna_virus", "rna_cellular"])]
                y_true = (ds_df["true_category"] == "rna_virus").astype(int).values
            else:
                continue

            if len(ds_df) == 0:
                continue

            y_scores = ds_df["viral_score"].values
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, color=COLORS[tier], lw=2,
                    label=f"{TIER_LABELS[tier]} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(dataset_titles.get(ds, ds))
        ax.legend(loc="lower right")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    fig.savefig(output_dir / "fig_roc_curves.pdf")
    fig.savefig(output_dir / "fig_roc_curves.png")
    plt.close(fig)
    print(f"  Saved: fig_roc_curves.pdf")


# ---------------------------------------------------------------------------
# Figure 3: Score Distributions
# ---------------------------------------------------------------------------

def fig_score_distributions(results_dir: Path, output_dir: Path) -> None:
    """Viral score distributions by category for each tier."""
    tier_data = {}
    for tier in ["40b", "7b"]:
        d = load_tier_data(results_dir, tier)
        if d and "details" in d:
            tier_data[tier] = d

    if not tier_data:
        print("No benchmark results found, skipping score distributions")
        return

    n_tiers = len(tier_data)
    fig, axes = plt.subplots(1, n_tiers, figsize=(5 * n_tiers, 4))
    if n_tiers == 1:
        axes = [axes]

    categories_order = ["phage", "rna_virus", "chromosome", "plasmid", "cellular"]

    for ax, (tier, d) in zip(axes, tier_data.items()):
        df = d["details"]
        cats_present = [c for c in categories_order if c in df["true_category"].values]

        positions = []
        data = []
        colors = []
        for i, cat in enumerate(cats_present):
            scores = df[df["true_category"] == cat]["viral_score"].values
            data.append(scores)
            positions.append(i)
            colors.append(CAT_COLORS.get(cat, "#666666"))

        parts = ax.violinplot(data, positions=positions, showmedians=True, showextrema=False)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("black")

        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(cats_present, rotation=30, ha="right")
        ax.set_ylabel("Viral Score")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(TIER_LABELS.get(tier, tier))

    fig.suptitle("Viral Score Distributions by Category", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "fig_score_distributions.pdf")
    fig.savefig(output_dir / "fig_score_distributions.png")
    plt.close(fig)
    print(f"  Saved: fig_score_distributions.pdf")


# ---------------------------------------------------------------------------
# Figure 4: Sensitivity by Length
# ---------------------------------------------------------------------------

def fig_sensitivity_by_length(results_dir: Path, output_dir: Path) -> None:
    """Detection sensitivity across length bins for each dataset."""
    tier_data = {}
    for tier in ["40b", "7b"]:
        d = load_tier_data(results_dir, tier)
        if d and "details" in d:
            tier_data[tier] = d

    if not tier_data:
        print("No benchmark results found, skipping sensitivity by length")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # GYP phage sensitivity by length
    ax = axes[0]
    for tier, d in tier_data.items():
        df = d["details"]
        phage = df[df["true_category"] == "phage"]
        bins = sorted(phage["length_bin"].unique())
        sens = []
        for b in bins:
            bin_df = phage[phage["length_bin"] == b]
            sens.append(float((bin_df["viral_score"] >= 0.5).mean()))
        ax.plot(range(len(bins)), sens, "o-", color=COLORS[tier], lw=2,
                markersize=6, label=TIER_LABELS[tier])

    if bins:
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(bins)
    ax.set_xlabel("Fragment Length")
    ax.set_ylabel("Sensitivity")
    ax.set_ylim(0.8, 1.005)
    ax.set_title("Phage Sensitivity (GYP)")
    ax.legend()

    # RNA virus sensitivity by length
    ax = axes[1]
    for tier, d in tier_data.items():
        df = d["details"]
        rna = df[df["true_category"] == "rna_virus"]
        bins = sorted(rna["length_bin"].unique())
        sens = []
        for b in bins:
            bin_df = rna[rna["length_bin"] == b]
            sens.append(float((bin_df["viral_score"] >= 0.5).mean()))
        ax.plot(range(len(bins)), sens, "o-", color=COLORS[tier], lw=2,
                markersize=6, label=TIER_LABELS[tier])

    if bins:
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(bins, rotation=30, ha="right")
    ax.set_xlabel("Fragment Length")
    ax.set_ylabel("Sensitivity")
    ax.set_ylim(0.5, 1.05)
    ax.set_title("RNA Virus Sensitivity (Zero-Shot)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "fig_sensitivity_by_length.pdf")
    fig.savefig(output_dir / "fig_sensitivity_by_length.png")
    plt.close(fig)
    print(f"  Saved: fig_sensitivity_by_length.pdf")


# ---------------------------------------------------------------------------
# Figure 5: Tier Comparison Scatter
# ---------------------------------------------------------------------------

def fig_tier_comparison(results_dir: Path, output_dir: Path) -> None:
    """Scatter plot of 40B vs 7B viral scores per sequence."""
    a_data = load_tier_data(results_dir, "40b")
    b_data = load_tier_data(results_dir, "7b")
    if not a_data or not b_data or "details" not in a_data or "details" not in b_data:
        print("Need both tiers for comparison, skipping tier scatter")
        return

    merged = a_data["details"].merge(
        b_data["details"][["sequence_id", "viral_score"]],
        on="sequence_id", suffixes=("_40b", "_7b"),
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    for cat in ["phage", "rna_virus", "chromosome", "plasmid", "cellular"]:
        cat_df = merged[merged["true_category"] == cat]
        if len(cat_df) == 0:
            continue
        ax.scatter(
            cat_df["viral_score_40b"], cat_df["viral_score_7b"],
            c=CAT_COLORS.get(cat, "#666"), alpha=0.3, s=8,
            label=f"{cat} (n={len(cat_df)})", rasterized=True,
        )

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.axhline(0.5, color="gray", lw=0.5, alpha=0.3)
    ax.axvline(0.5, color="gray", lw=0.5, alpha=0.3)
    ax.set_xlabel(f"Viral Score — {TIER_LABELS['40b']}")
    ax.set_ylabel(f"Viral Score — {TIER_LABELS['7b']}")
    ax.set_title("Per-Sequence Score Comparison")
    ax.legend(loc="upper left", markerscale=3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    r = np.corrcoef(merged["viral_score_40b"], merged["viral_score_7b"])[0, 1]
    ax.text(0.95, 0.05, f"r = {r:.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_dir / "fig_tier_comparison.pdf")
    fig.savefig(output_dir / "fig_tier_comparison.png")
    plt.close(fig)
    print(f"  Saved: fig_tier_comparison.pdf")


# ---------------------------------------------------------------------------
# Figure 6: Threshold Sweep
# ---------------------------------------------------------------------------

def fig_threshold_sweep(results_dir: Path, output_dir: Path) -> None:
    """F1/precision/recall vs threshold for GYP dataset."""
    tier_data = {}
    for tier in ["40b", "7b"]:
        d = load_tier_data(results_dir, tier)
        if d:
            tier_data[tier] = d

    if not tier_data:
        print("No benchmark results found, skipping threshold sweep")
        return

    fig, axes = plt.subplots(1, len(tier_data), figsize=(5 * len(tier_data), 4))
    if len(tier_data) == 1:
        axes = [axes]

    for ax, (tier, d) in zip(axes, tier_data.items()):
        gyp = d["metrics"].get("datasets", {}).get("gyp", {})
        sweep = gyp.get("threshold_sweep", [])
        if not sweep:
            continue

        thresholds = [s["threshold"] for s in sweep]
        f1s = [s["f1"] for s in sweep]
        precs = [s["precision"] for s in sweep]
        recs = [s["recall"] for s in sweep]

        ax.plot(thresholds, f1s, "o-", color=COLORS[tier], lw=2, label="F1")
        ax.plot(thresholds, precs, "s--", color=COLORS[tier], lw=1.5, alpha=0.7, label="Precision")
        ax.plot(thresholds, recs, "^--", color=COLORS[tier], lw=1.5, alpha=0.7, label="Recall")
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"{TIER_LABELS[tier]} — GYP")
        ax.legend()
        ax.set_ylim(0.8, 1.005)

    plt.tight_layout()
    fig.savefig(output_dir / "fig_threshold_sweep.pdf")
    fig.savefig(output_dir / "fig_threshold_sweep.png")
    plt.close(fig)
    print(f"  Saved: fig_threshold_sweep.pdf")


# ---------------------------------------------------------------------------
# Figure 7: Plasmid Analysis
# ---------------------------------------------------------------------------

def fig_plasmid_analysis(results_dir: Path, output_dir: Path) -> None:
    """Plasmid false positive analysis — score histogram by length."""
    tier_data = {}
    for tier in ["40b", "7b"]:
        d = load_tier_data(results_dir, tier)
        if d and "details" in d:
            tier_data[tier] = d

    if not tier_data:
        print("No benchmark results found, skipping plasmid analysis")
        return

    n_tiers = len(tier_data)
    fig, axes = plt.subplots(1, n_tiers, figsize=(5 * n_tiers, 4))
    if n_tiers == 1:
        axes = [axes]

    for ax, (tier, d) in zip(axes, tier_data.items()):
        df = d["details"]
        plasmid = df[df["true_category"] == "plasmid"]
        if len(plasmid) == 0:
            continue

        scores = plasmid["viral_score"].values
        fp_rate = float((scores >= 0.5).mean())

        ax.hist(scores, bins=50, color=CAT_COLORS["plasmid"], alpha=0.7, edgecolor="white")
        ax.axvline(0.5, color="red", linestyle="--", linewidth=1.5, label=f"Threshold (FP={fp_rate:.1%})")
        ax.set_xlabel("Viral Score")
        ax.set_ylabel("Count")
        ax.set_title(f"{TIER_LABELS[tier]} — Plasmid Scores (n={len(plasmid)})")
        ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "fig_plasmid_analysis.pdf")
    fig.savefig(output_dir / "fig_plasmid_analysis.png")
    plt.close(fig)
    print(f"  Saved: fig_plasmid_analysis.pdf")


# ---------------------------------------------------------------------------
# Summary Table (Markdown)
# ---------------------------------------------------------------------------

def generate_summary_table(results_dir: Path, output_dir: Path) -> None:
    """Generate a markdown summary table of all metrics."""
    lines = ["# ViroSense Benchmark Summary", "", "Generated from `results/` directory.", ""]

    # Classifier training
    lines.extend(["## Classifier Training", ""])
    lines.append("| Metric | Evo2 40B | Evo2 7B |")
    lines.append("|--------|----------|---------|")

    m40 = load_classifier_metrics(results_dir, "40b")
    m7 = load_classifier_metrics(results_dir, "7b")
    metrics_keys = [
        ("Embedding dim", lambda m, t: "8,192" if t == "40b" else "4,096"),
        ("Test accuracy", lambda m, _: f"{m['accuracy']:.1%}" if m else "—"),
        ("Test F1", lambda m, _: f"{m['f1']:.3f}" if m else "—"),
        ("Test AUC", lambda m, _: f"{m['auc']:.3f}" if m else "—"),
        ("ECE (calibrated)", lambda m, _: f"{m.get('ece', 0):.3f}" if m else "—"),
        ("Brier score", lambda m, _: f"{m.get('brier_score', 0):.4f}" if m else "—"),
        ("N train", lambda m, _: str(m.get("n_train", "—")) if m else "—"),
        ("N test", lambda m, _: str(m.get("n_test", "—")) if m else "—"),
    ]
    for label, fn in metrics_keys:
        v40 = fn(m40, "40b")
        v7 = fn(m7, "7b")
        lines.append(f"| {label} | {v40} | {v7} |")

    # Benchmark results
    for tier in ["40b", "7b"]:
        d = load_tier_data(results_dir, tier)
        if not d:
            continue

        lines.extend(["", f"## Benchmark Results — {TIER_LABELS.get(tier, tier)}", ""])

        for ds_name in ["gyp", "rna_virus"]:
            ds = d["metrics"].get("datasets", {}).get(ds_name)
            if not ds:
                continue

            ds_title = {"gyp": "GYP Phage Detection", "rna_virus": "RNA Virus Zero-Shot"}
            lines.append(f"### {ds_title.get(ds_name, ds_name)}")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k in ["accuracy", "f1", "precision", "recall", "auc"]:
                v = ds.get(k)
                if v is not None:
                    lines.append(f"| {k.title()} | {v:.3f} |")

            # Per-category
            cats = ds.get("categories", {})
            if cats:
                lines.append("")
                lines.append("| Category | N | Rate | Mean Score |")
                lines.append("|----------|---|------|------------|")
                for cat, info in cats.items():
                    lines.append(f"| {cat} | {info['n']} | {info['rate']:.3f} | {info['mean_score']:.3f} |")

            lines.append("")

    (output_dir / "benchmark_summary.md").write_text("\n".join(lines) + "\n")
    print(f"  Saved: benchmark_summary.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate ViroSense manuscript figures")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--output", help="Output directory for figures (default: results/figures)")
    parser.add_argument("--tier", choices=["40b", "7b"], help="Generate for single tier only")
    parser.add_argument("--training-only", action="store_true",
                        help="Only generate training summary (no benchmark needed)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output) if args.output else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results: {results_dir}")
    print(f"Output:  {output_dir}")
    print()

    # Always generate training summary
    print("Generating figures...")
    fig_training_summary(results_dir, output_dir)

    if not args.training_only:
        fig_roc_curves(results_dir, output_dir)
        fig_score_distributions(results_dir, output_dir)
        fig_sensitivity_by_length(results_dir, output_dir)
        fig_tier_comparison(results_dir, output_dir)
        fig_threshold_sweep(results_dir, output_dir)
        fig_plasmid_analysis(results_dir, output_dir)

    # Always generate summary table
    generate_summary_table(results_dir, output_dir)

    print()
    print("Done. Figures saved to:", output_dir)


if __name__ == "__main__":
    main()
