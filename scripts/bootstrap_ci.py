#!/usr/bin/env python3
"""Compute bootstrap confidence intervals for viral detection tool benchmarks.

Produces 95% CIs for sensitivity, specificity, accuracy, F1, and AUC
across all tools and categories, suitable for publication.

Usage:
    uv run python scripts/bootstrap_ci.py \
        --manifest results/benchmark/manifest/manifest.tsv \
        --genomad results/benchmark/tool_comparison/genomad/sequences_summary/sequences_virus_summary.tsv \
        --virosense-7b results/benchmark/7b_16kb/detailed_results.tsv \
        --virosense-40b results/benchmark/40b/detailed_results.tsv \
        --output results/benchmark/comparison/bootstrap_ci.json \
        --n-bootstrap 10000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def load_manifest(path: str) -> pd.DataFrame:
    """Load benchmark manifest with ground truth."""
    return pd.read_csv(path, sep="\t")


def load_virosense(path: str) -> dict[str, float]:
    """Load ViroSense scores keyed by sequence_id."""
    df = pd.read_csv(path, sep="\t")
    return dict(zip(df["sequence_id"].astype(str), df["viral_score"].astype(float)))


def load_genomad(path: str) -> set[str]:
    """Load geNomad viral calls (presence = viral)."""
    df = pd.read_csv(path, sep="\t")
    viral_ids = set()
    for name in df["seq_name"]:
        base = name.rsplit("|provirus", 1)[0] if "|provirus" in str(name) else str(name)
        viral_ids.add(base)
    return viral_ids


def build_prediction_array(
    manifest: pd.DataFrame,
    scores: dict[str, float] | None = None,
    viral_ids: set[str] | None = None,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Build aligned arrays of (truth, prediction, score) for a tool.

    Only includes sequences that the tool actually evaluated.

    Returns:
        truth: 1=viral, 0=non-viral
        pred: 1=viral, 0=non-viral
        score: continuous score (0-1), or 1.0/0.0 for binary tools
        manifest_subset: filtered manifest rows (aligned with arrays)
    """
    viral_cats = {"phage", "rna_virus"}
    truth = []
    pred = []
    score_arr = []
    kept_indices = []

    for idx, row in manifest.iterrows():
        sid = str(row["sequence_id"])
        is_viral = row["category"] in viral_cats

        if scores is not None:
            if sid not in scores:
                continue  # skip missing sequences
            s = scores[sid]
            score_arr.append(s)
            pred.append(1 if s >= threshold else 0)
        elif viral_ids is not None:
            is_pred_viral = sid in viral_ids
            pred.append(1 if is_pred_viral else 0)
            score_arr.append(1.0 if is_pred_viral else 0.0)

        truth.append(1 if is_viral else 0)
        kept_indices.append(idx)

    manifest_subset = manifest.loc[kept_indices].reset_index(drop=True)
    return (
        np.array(truth),
        np.array(pred),
        np.array(score_arr),
        manifest_subset,
    )


def bootstrap_metric(
    truth: np.ndarray,
    pred: np.ndarray,
    score: np.ndarray,
    metric_fn: str,
    n_bootstrap: int = 10000,
    seed: int = 42,
    mask: np.ndarray | None = None,
) -> dict:
    """Compute a metric with bootstrap 95% CI.

    Args:
        truth: ground truth labels (0/1)
        pred: predicted labels (0/1)
        score: continuous scores
        metric_fn: one of 'accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'auc'
        n_bootstrap: number of bootstrap resamples
        seed: random seed
        mask: optional boolean mask to select a subset (e.g., specific category)
    """
    if mask is not None:
        truth = truth[mask]
        pred = pred[mask]
        score = score[mask]

    n = len(truth)
    if n == 0:
        return {"point": None, "ci_lower": None, "ci_upper": None, "n": 0}

    rng = np.random.RandomState(seed)
    point = _compute_metric(truth, pred, score, metric_fn)

    boot_values = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        t_boot = truth[idx]
        p_boot = pred[idx]
        s_boot = score[idx]
        val = _compute_metric(t_boot, p_boot, s_boot, metric_fn)
        if val is not None:
            boot_values.append(val)

    if not boot_values:
        return {"point": point, "ci_lower": None, "ci_upper": None, "n": n}

    boot_values = np.array(boot_values)
    ci_lower = float(np.percentile(boot_values, 2.5))
    ci_upper = float(np.percentile(boot_values, 97.5))

    return {
        "point": float(point) if point is not None else None,
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "n": int(n),
    }


def _compute_metric(
    truth: np.ndarray,
    pred: np.ndarray,
    score: np.ndarray,
    metric_fn: str,
) -> float | None:
    """Compute a single metric value."""
    tp = int(np.sum((truth == 1) & (pred == 1)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    tn = int(np.sum((truth == 0) & (pred == 0)))
    fn = int(np.sum((truth == 1) & (pred == 0)))

    if metric_fn == "accuracy":
        n = tp + fp + tn + fn
        return (tp + tn) / n if n > 0 else None
    elif metric_fn == "sensitivity":  # recall / TPR
        return tp / (tp + fn) if (tp + fn) > 0 else None
    elif metric_fn == "specificity":  # TNR
        return tn / (tn + fp) if (tn + fp) > 0 else None
    elif metric_fn == "precision":
        return tp / (tp + fp) if (tp + fp) > 0 else None
    elif metric_fn == "f1":
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else None
    elif metric_fn == "auc":
        try:
            from sklearn.metrics import roc_auc_score
            if len(set(truth)) < 2:
                return None
            return roc_auc_score(truth, score)
        except Exception:
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CIs for benchmark comparison")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--genomad", help="geNomad virus_summary.tsv")
    parser.add_argument("--virosense-7b", help="ViroSense 7B detailed_results.tsv")
    parser.add_argument("--virosense-40b", help="ViroSense 40B detailed_results.tsv")
    parser.add_argument("--virosense-40b-l2", help="ViroSense 40B+L2 detailed_results.tsv")
    parser.add_argument("--deepvirfinder", help="DVF prediction TSV")
    parser.add_argument("--output", required=True, help="Output JSON")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    logger.info(f"Manifest: {len(manifest)} sequences")

    viral_cats = {"phage", "rna_virus"}

    # Load tool predictions
    tools: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]] = {}

    if args.virosense_7b:
        scores = load_virosense(args.virosense_7b)
        tools["virosense_7b"] = build_prediction_array(manifest, scores=scores)
        logger.info(f"ViroSense 7B: {len(scores)} scores, {len(tools['virosense_7b'][0])} evaluated")

    if args.virosense_40b:
        scores = load_virosense(args.virosense_40b)
        tools["virosense_40b"] = build_prediction_array(manifest, scores=scores)
        logger.info(f"ViroSense 40B: {len(scores)} scores, {len(tools['virosense_40b'][0])} evaluated")

    if args.virosense_40b_l2:
        scores = load_virosense(args.virosense_40b_l2)
        tools["virosense_40b_l2"] = build_prediction_array(manifest, scores=scores)
        logger.info(f"ViroSense 40B+L2: {len(scores)} scores, {len(tools['virosense_40b_l2'][0])} evaluated")

    if args.genomad:
        viral_ids = load_genomad(args.genomad)
        tools["genomad"] = build_prediction_array(manifest, viral_ids=viral_ids)
        logger.info(f"geNomad: {len(viral_ids)} viral calls, {len(tools['genomad'][0])} evaluated")

    if args.deepvirfinder:
        dvf_df = pd.read_csv(args.deepvirfinder, sep="\t")
        dvf_scores = {}
        for _, row in dvf_df.iterrows():
            name = str(row["name"]).strip()
            s = float(row["score"])
            p = float(row["pvalue"])
            # DVF convention: score > 0.5 and pvalue < 0.05
            dvf_scores[name] = s if p < 0.05 else 0.0
        tools["deepvirfinder"] = build_prediction_array(manifest, scores=dvf_scores)
        logger.info(f"DeepVirFinder: {len(dvf_scores)} scores, {len(tools['deepvirfinder'][0])} evaluated")

    # Compute bootstrap CIs
    results: dict = {}
    metrics = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]

    for tool_name, (truth, pred, score, tool_manifest) in tools.items():
        logger.info(f"Bootstrapping {tool_name} ({len(truth)} sequences, {args.n_bootstrap} resamples)...")
        tool_results: dict = {}

        # Overall metrics
        tool_results["overall"] = {}
        for m in metrics:
            tool_results["overall"][m] = bootstrap_metric(
                truth, pred, score, m, args.n_bootstrap, args.seed
            )

        # Per-category metrics (using the aligned manifest subset)
        for cat in ["phage", "rna_virus", "chromosome", "plasmid", "cellular"]:
            cat_mask = (tool_manifest["category"] == cat).values
            if cat_mask.sum() == 0:
                continue

            key_metric = "sensitivity" if cat in viral_cats else "specificity"
            tool_results[cat] = {
                key_metric: bootstrap_metric(
                    truth, pred, score, key_metric, args.n_bootstrap, args.seed, mask=cat_mask
                )
            }

            # Length-binned sensitivity for phage and RNA virus
            if cat in {"phage", "rna_virus"}:
                cat_rows = tool_manifest[tool_manifest["category"] == cat]
                by_length = {}
                for lb in sorted(cat_rows["length_bin"].unique()):
                    lb_mask = (
                        (tool_manifest["category"] == cat) & (tool_manifest["length_bin"] == lb)
                    ).values
                    if lb_mask.sum() == 0:
                        continue
                    by_length[lb] = bootstrap_metric(
                        truth, pred, score, "sensitivity",
                        args.n_bootstrap, args.seed, mask=lb_mask
                    )
                tool_results[f"{cat}_by_length"] = by_length

        results[tool_name] = tool_results

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Written to {output_path}")

    # Print summary table
    print("\n" + "=" * 110)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS")
    print("=" * 110)

    tool_order = ["virosense_40b", "virosense_7b", "genomad", "virosense_40b_l2", "deepvirfinder"]
    tools_present = [t for t in tool_order if t in results]

    col_w = 24
    header = f"{'Metric':<28}" + "".join(f"{t:>{col_w}}" for t in tools_present)
    print(header)
    print("-" * len(header))

    def fmt_ci(d: dict) -> str:
        if d.get("point") is None:
            return "—"
        p = d["point"]
        lo = d.get("ci_lower")
        hi = d.get("ci_upper")
        if lo is not None and hi is not None:
            return f"{p:.1%} [{lo:.1%}-{hi:.1%}]"
        return f"{p:.1%}"

    # Overall
    for m in ["accuracy", "f1", "sensitivity", "specificity"]:
        row = f"{m:<28}"
        for t in tools_present:
            row += f"{fmt_ci(results[t]['overall'][m]):>{col_w}}"
        print(row)

    # Per-category
    for cat, label, key in [
        ("phage", "Phage sens.", "sensitivity"),
        ("rna_virus", "RNA virus sens.", "sensitivity"),
        ("chromosome", "Chr. spec.", "specificity"),
        ("plasmid", "Plasmid spec.", "specificity"),
    ]:
        row = f"{label:<28}"
        for t in tools_present:
            d = results[t].get(cat, {}).get(key, {})
            row += f"{fmt_ci(d):>{col_w}}" if d else f"{'—':>{col_w}}"
        print(row)

    # Length bins for phage and RNA virus
    for cat, label in [("phage", "Phage"), ("rna_virus", "RNA virus")]:
        all_bins: set[str] = set()
        for t in tools_present:
            if f"{cat}_by_length" in results[t]:
                all_bins.update(results[t][f"{cat}_by_length"].keys())

        if not all_bins:
            continue

        def bin_sort(b: str) -> float:
            return float(b.split("-")[0].replace("kb", "").replace("bp", "").replace("500", "0.5"))

        for lb in sorted(all_bins, key=bin_sort):
            row = f"  {label} {lb:<22}"
            for t in tools_present:
                d = results[t].get(f"{cat}_by_length", {}).get(lb, {})
                row += f"{fmt_ci(d):>{col_w}}" if d else f"{'—':>{col_w}}"
            print(row)

    print()


if __name__ == "__main__":
    main()
