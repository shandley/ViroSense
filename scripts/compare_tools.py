#!/usr/bin/env python3
"""Head-to-head comparison of ViroSense vs geNomad, DeepVirFinder, VirSorter2, VIBRANT.

Parses outputs from all tools and computes standardized metrics on the GYP + RNA
virus benchmark (13,417 sequences with known ground truth).

Usage:
    # After running htcf/benchmark_tools_run.sbatch:
    scp -r shandley@login.htcf.wustl.edu:/scratch/sahlab/shandley/benchmark_tools/results/ \
        results/benchmark/tool_comparison/

    # Analyze (local)
    uv run python scripts/compare_tools.py \
        --manifest results/benchmark/manifest/manifest.tsv \
        --tool-results results/benchmark/tool_comparison/ \
        --virosense-7b results/benchmark/7b_16kb/detailed_results.tsv \
        --virosense-7b-l2 results/benchmark/7b_l2norm/detailed_results.tsv \
        --virosense-40b results/benchmark/40b/detailed_results.tsv \
        --output results/benchmark/comparison/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


# ==========================================================================
# Tool output parsers
# ==========================================================================


def parse_genomad(result_dir: Path, all_ids: set[str]) -> dict[str, dict]:
    """Parse geNomad virus_summary.tsv.

    geNomad only outputs rows for sequences it classifies as viral.
    Sequences absent from the output are implicitly non-viral.
    """
    results = {}
    # geNomad output structure: result_dir/sequences_summary/sequences_virus_summary.tsv
    # The subdirectory name matches the input FASTA stem
    summary_files = list(result_dir.rglob("*virus_summary.tsv"))
    if not summary_files:
        logger.warning(f"No geNomad virus_summary.tsv found in {result_dir}")
        return results

    for summary_file in summary_files:
        with open(summary_file) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                seq_name = row["seq_name"]
                # geNomad may append "|provirus_N_N" for provirus calls
                # but only strip if it looks like a provirus suffix
                base_name = seq_name
                if "|provirus" in seq_name:
                    base_name = seq_name.rsplit("|provirus", 1)[0]
                results[base_name] = {
                    "tool": "genomad",
                    "viral": True,
                    "score": float(row["virus_score"]),
                    "n_hallmarks": int(row.get("n_hallmarks", 0)),
                    "taxonomy": row.get("taxonomy", ""),
                }

    # Non-viral sequences
    for sid in all_ids:
        if sid not in results:
            results[sid] = {"tool": "genomad", "viral": False, "score": 0.0}

    return results


def parse_deepvirfinder(result_dir: Path, all_ids: set[str]) -> dict[str, dict]:
    """Parse DeepVirFinder output TSV.

    DVF outputs: name, len, score, pvalue for all sequences.
    Convention: score > 0.5 and pvalue < 0.05 = viral.
    """
    results = {}
    dvf_files = list(result_dir.rglob("*dvfpred.txt"))
    if not dvf_files:
        logger.warning(f"No DeepVirFinder output found in {result_dir}")
        return results

    for dvf_file in dvf_files:
        df = pd.read_csv(dvf_file, sep="\t")
        for _, row in df.iterrows():
            name = str(row["name"]).strip()
            score = float(row["score"])
            pvalue = float(row["pvalue"])
            results[name] = {
                "tool": "deepvirfinder",
                "viral": score >= 0.5 and pvalue < 0.05,
                "score": score,
                "pvalue": pvalue,
            }

    # Fill missing
    for sid in all_ids:
        if sid not in results:
            results[sid] = {"tool": "deepvirfinder", "viral": False, "score": 0.0}

    return results


def parse_virsorter2(result_dir: Path, all_ids: set[str]) -> dict[str, dict]:
    """Parse VirSorter2 final-viral-score.tsv.

    VS2 outputs scores for sequences it considers potentially viral.
    Default: score >= 0.5 is viral.
    """
    results = {}
    score_files = list(result_dir.rglob("final-viral-score.tsv"))
    if not score_files:
        logger.warning(f"No VirSorter2 final-viral-score.tsv found in {result_dir}")
        return results

    for score_file in score_files:
        df = pd.read_csv(score_file, sep="\t")
        for _, row in df.iterrows():
            name = str(row["seqname"]).strip()
            # VS2 adds suffixes like ||full or ||lt2gene
            base_name = name.split("||")[0]
            score = float(row.get("max_score", row.get("max_score_group", 0)))
            results[base_name] = {
                "tool": "virsorter2",
                "viral": True,  # presence in output = viral call
                "score": score,
                "group": str(row.get("max_score_group", "")),
            }

    for sid in all_ids:
        if sid not in results:
            results[sid] = {"tool": "virsorter2", "viral": False, "score": 0.0}

    return results


def parse_vibrant(result_dir: Path, all_ids: set[str]) -> dict[str, dict]:
    """Parse VIBRANT output (phages detected).

    VIBRANT outputs FASTA files of detected phages.
    """
    results = {}
    # VIBRANT creates a directory structure with phage sequences
    phage_files = list(result_dir.rglob("*phages_combined.txt"))
    if not phage_files:
        # Try the FASTA output
        phage_files = list(result_dir.rglob("*phages_combined*.fasta"))
    if not phage_files:
        # Try genome quality TSV
        phage_files = list(result_dir.rglob("*genome_quality*.tsv"))

    detected = set()
    for pf in phage_files:
        if pf.suffix == ".tsv":
            df = pd.read_csv(pf, sep="\t")
            for _, row in df.iterrows():
                name = str(row.iloc[0]).strip()
                detected.add(name)
        elif pf.suffix == ".fasta":
            with open(pf) as f:
                for line in f:
                    if line.startswith(">"):
                        name = line[1:].strip().split()[0]
                        detected.add(name)
        else:
            with open(pf) as f:
                for line in f:
                    name = line.strip()
                    if name:
                        detected.add(name)

    for sid in all_ids:
        results[sid] = {
            "tool": "vibrant",
            "viral": sid in detected,
            "score": 1.0 if sid in detected else 0.0,
        }

    return results


def parse_virosense(results_tsv: Path, all_ids: set[str]) -> dict[str, dict]:
    """Parse ViroSense detailed_results.tsv from benchmark.

    Only includes sequences that were actually classified (present in results).
    Missing sequences (e.g., failed API extraction) are marked as missing,
    not as non-viral.
    """
    results = {}
    if not results_tsv.exists():
        logger.warning(f"ViroSense results not found: {results_tsv}")
        return results

    df = pd.read_csv(results_tsv, sep="\t")
    for _, row in df.iterrows():
        sid = str(row["sequence_id"]).strip()
        score = float(row["viral_score"])
        results[sid] = {
            "tool": "virosense",
            "viral": score >= 0.5,
            "score": score,
        }

    n_missing = len(all_ids - set(results.keys()))
    if n_missing > 0:
        logger.warning(
            f"ViroSense: {n_missing} sequences missing from results "
            f"(excluded from metrics, not counted as non-viral)"
        )

    return results


# ==========================================================================
# Metric computation
# ==========================================================================


def compute_metrics(
    truth: list[str],
    predictions: dict[str, dict],
    positive_labels: set[str],
) -> dict:
    """Compute binary classification metrics.

    Args:
        truth: list of (seq_id, ground_truth_category) tuples
        predictions: tool predictions keyed by seq_id
        positive_labels: categories that count as "viral" (e.g., {"phage", "rna_virus"})
    """
    tp = fp = tn = fn = 0
    scores = []
    labels = []

    for seq_id, category in truth:
        is_viral_truth = category in positive_labels
        pred = predictions.get(seq_id)
        if pred is None:
            continue  # Skip sequences not evaluated by this tool
        is_viral_pred = pred.get("viral", False)
        score = pred.get("score", 0.0)

        scores.append(score)
        labels.append(1 if is_viral_truth else 0)

        if is_viral_truth and is_viral_pred:
            tp += 1
        elif not is_viral_truth and is_viral_pred:
            fp += 1
        elif not is_viral_truth and not is_viral_pred:
            tn += 1
        elif is_viral_truth and not is_viral_pred:
            fn += 1

    n = tp + fp + tn + fn
    accuracy = (tp + tn) / n if n > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # AUC if scores are available
    auc = None
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(labels)) > 1:
            auc = roc_auc_score(labels, scores)
    except Exception:
        pass

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4) if auc is not None else None,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n": n,
    }


def compute_category_metrics(
    manifest: pd.DataFrame,
    predictions: dict[str, dict],
) -> dict:
    """Compute per-category metrics for a tool."""
    categories = manifest["category"].unique()
    viral_cats = {"phage", "rna_virus"}
    non_viral_cats = {"chromosome", "plasmid", "cellular"}

    results = {}

    # Overall binary: viral vs non-viral
    truth = [(row["sequence_id"], row["category"]) for _, row in manifest.iterrows()]
    results["overall"] = compute_metrics(truth, predictions, viral_cats)

    # Per-category sensitivity/specificity
    for cat in categories:
        cat_seqs = manifest[manifest["category"] == cat]
        truth_cat = [(row["sequence_id"], row["category"]) for _, row in cat_seqs.iterrows()]
        cat_metrics = compute_metrics(truth_cat, predictions, viral_cats)

        if cat in viral_cats:
            results[cat] = {
                "n": len(cat_seqs),
                "sensitivity": cat_metrics["recall"],
                "mean_score": np.mean([
                    predictions.get(row["sequence_id"], {}).get("score", 0.0)
                    for _, row in cat_seqs.iterrows()
                ]),
            }
        else:
            results[cat] = {
                "n": len(cat_seqs),
                "specificity": cat_metrics["specificity"],
                "false_positive_rate": 1 - cat_metrics["specificity"],
                "mean_score": np.mean([
                    predictions.get(row["sequence_id"], {}).get("score", 0.0)
                    for _, row in cat_seqs.iterrows()
                ]),
            }

    # Length-binned sensitivity for phage and RNA virus
    for cat in ["phage", "rna_virus"]:
        cat_df = manifest[manifest["category"] == cat]
        if len(cat_df) == 0:
            continue
        bins = sorted(cat_df["length_bin"].unique())
        by_length = {}
        for lb in bins:
            bin_seqs = cat_df[cat_df["length_bin"] == lb]
            # Only count sequences that this tool actually evaluated
            evaluated = [
                row for _, row in bin_seqs.iterrows()
                if row["sequence_id"] in predictions
            ]
            n_detected = sum(
                1 for row in evaluated
                if predictions[row["sequence_id"]].get("viral", False)
            )
            n_eval = len(evaluated)
            by_length[lb] = {
                "n": n_eval,
                "sensitivity": round(n_detected / n_eval, 4) if n_eval > 0 else 0,
            }
        results[f"{cat}_by_length"] = by_length

    return results


# ==========================================================================
# Main
# ==========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare viral detection tools on GYP + RNA benchmark"
    )
    parser.add_argument("--manifest", required=True, help="manifest.tsv with ground truth")
    parser.add_argument("--tool-results", required=True, help="Directory with tool outputs")
    parser.add_argument("--virosense-7b", help="ViroSense 7B detailed_results.tsv")
    parser.add_argument("--virosense-7b-l2", help="ViroSense 7B+L2 detailed_results.tsv")
    parser.add_argument("--virosense-40b", help="ViroSense 40B detailed_results.tsv")
    parser.add_argument("--virosense-40b-l2", help="ViroSense 40B+L2 detailed_results.tsv")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest = pd.read_csv(args.manifest, sep="\t")
    all_ids = set(manifest["sequence_id"])
    logger.info(f"Loaded manifest: {len(manifest)} sequences")
    logger.info(f"Categories: {dict(manifest['category'].value_counts())}")

    tool_dir = Path(args.tool_results)
    all_results = {}

    # Parse each tool
    parsers = {
        "genomad": (parse_genomad, tool_dir / "genomad"),
        "deepvirfinder": (parse_deepvirfinder, tool_dir / "deepvirfinder"),
        "virsorter2": (parse_virsorter2, tool_dir / "virsorter2"),
        "vibrant": (parse_vibrant, tool_dir / "vibrant"),
    }

    for tool_name, (parser_fn, result_path) in parsers.items():
        if result_path.exists():
            logger.info(f"Parsing {tool_name} results from {result_path}")
            preds = parser_fn(result_path, all_ids)
            n_viral = sum(1 for v in preds.values() if v.get("viral", False))
            logger.info(f"  {tool_name}: {n_viral}/{len(preds)} called viral")
            all_results[tool_name] = preds
        else:
            logger.warning(f"{tool_name} results not found at {result_path}")

    # Parse ViroSense results
    virosense_configs = {
        "virosense_7b": args.virosense_7b,
        "virosense_7b_l2": args.virosense_7b_l2,
        "virosense_40b": args.virosense_40b,
        "virosense_40b_l2": args.virosense_40b_l2,
    }
    for name, path in virosense_configs.items():
        if path:
            p = Path(path)
            if p.exists():
                logger.info(f"Parsing {name} from {p}")
                preds = parse_virosense(p, all_ids)
                n_viral = sum(1 for v in preds.values() if v.get("viral", False))
                logger.info(f"  {name}: {n_viral}/{len(preds)} called viral")
                all_results[name] = preds

    if not all_results:
        logger.error("No tool results found!")
        sys.exit(1)

    # Compute metrics for all tools
    comparison = {}
    for tool_name, preds in all_results.items():
        logger.info(f"Computing metrics for {tool_name}...")
        comparison[tool_name] = compute_category_metrics(manifest, preds)

    # Load timing data if available
    timing_file = tool_dir / "timing_summary.json"
    timing = {}
    if timing_file.exists():
        with open(timing_file) as f:
            timing = json.load(f).get("runtime_seconds", {})

    # ==========================================================================
    # Print comparison table
    # ==========================================================================

    tool_order = [
        "virosense_40b", "virosense_7b_l2", "virosense_7b",
        "genomad", "deepvirfinder", "virsorter2", "vibrant",
        "virosense_40b_l2",
    ]
    tools_present = [t for t in tool_order if t in comparison]

    print("\n" + "=" * 100)
    print("HEAD-TO-HEAD BENCHMARK COMPARISON")
    print("=" * 100)

    # Header
    col_width = 16
    header = f"{'Metric':<25}" + "".join(f"{t:>{col_width}}" for t in tools_present)
    print(header)
    print("-" * len(header))

    # Overall metrics
    for metric in ["accuracy", "precision", "recall", "specificity", "f1", "auc"]:
        row = f"{metric:<25}"
        for t in tools_present:
            val = comparison[t]["overall"].get(metric)
            if val is not None:
                row += f"{val:>{col_width}.1%}" if metric != "auc" else f"{val:>{col_width}.4f}"
            else:
                row += f"{'—':>{col_width}}"
        print(row)

    print()

    # Per-category
    for cat in ["phage", "rna_virus", "chromosome", "plasmid", "cellular"]:
        cat_present = [t for t in tools_present if cat in comparison[t]]
        if not cat_present:
            continue

        print(f"\n--- {cat.upper()} ---")
        if cat in {"phage", "rna_virus"}:
            metric_key = "sensitivity"
        else:
            metric_key = "specificity"

        row_n = f"{'n':<25}"
        row_metric = f"{metric_key:<25}"
        for t in tools_present:
            cat_data = comparison[t].get(cat, {})
            n = cat_data.get("n", 0)
            val = cat_data.get(metric_key, 0)
            row_n += f"{n:>{col_width}}"
            row_metric += f"{val:>{col_width}.1%}"
        print(row_n)
        print(row_metric)

        # Length bins
        length_key = f"{cat}_by_length"
        any_has_length = any(length_key in comparison[t] for t in tools_present)
        if any_has_length:
            # Get all bins
            all_bins = set()
            for t in tools_present:
                if length_key in comparison[t]:
                    all_bins.update(comparison[t][length_key].keys())

            # Sort bins by lower bound
            def bin_sort_key(b):
                num = b.split("-")[0].replace("kb", "").replace("bp", "")
                return float(num.replace("500", "0.5"))

            for lb in sorted(all_bins, key=bin_sort_key):
                row = f"  {lb:<23}"
                for t in tools_present:
                    bin_data = comparison[t].get(length_key, {}).get(lb, {})
                    sens = bin_data.get("sensitivity")
                    n = bin_data.get("n", 0)
                    if sens is not None:
                        row += f"{sens:>{col_width - 5}.1%} ({n:>3})"
                    else:
                        row += f"{'—':>{col_width}}"
                print(row)

    # Runtime
    if timing:
        print(f"\n--- RUNTIME ---")
        row = f"{'seconds':<25}"
        for t in tools_present:
            # Map tool names to timing keys
            timing_key = t.replace("virosense_", "").replace("_l2", "")
            val = timing.get(timing_key, timing.get(t, "—"))
            row += f"{str(val):>{col_width}}"
        print(row)

    # ==========================================================================
    # Save results
    # ==========================================================================

    output = {
        "benchmark": {
            "n_sequences": len(manifest),
            "categories": dict(manifest["category"].value_counts()),
        },
        "tools": comparison,
        "timing": timing,
    }

    output_file = output_dir / "comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results written to {output_file}")

    # Also write a LaTeX-ready table
    latex_file = output_dir / "comparison_table.tex"
    with open(latex_file, "w") as f:
        f.write("% Auto-generated by scripts/compare_tools.py\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Head-to-head comparison of viral detection tools on the ")
        f.write("Gauge Your Phage + RNA virus benchmark (N=13,417).}\n")
        f.write("\\label{tab:tool-comparison}\n")

        n_tools = len(tools_present)
        f.write("\\begin{tabular}{l" + "r" * n_tools + "}\n")
        f.write("\\toprule\n")

        # Header
        tool_labels = {
            "virosense_40b": "ViroSense 40B",
            "virosense_7b": "ViroSense 7B",
            "virosense_7b_l2": "ViroSense 7B+L2",
            "virosense_40b_l2": "ViroSense 40B+L2",
            "genomad": "geNomad",
            "deepvirfinder": "DeepVirFinder",
            "virsorter2": "VirSorter2",
            "vibrant": "VIBRANT",
        }
        header_cols = " & ".join(tool_labels.get(t, t) for t in tools_present)
        f.write(f"Metric & {header_cols} \\\\\n")
        f.write("\\midrule\n")

        # Metrics rows
        for metric, label in [
            ("accuracy", "Accuracy"),
            ("f1", "F1 Score"),
            ("recall", "Sensitivity"),
            ("specificity", "Specificity"),
        ]:
            vals = []
            for t in tools_present:
                v = comparison[t]["overall"].get(metric)
                vals.append(f"{v:.1%}" if v is not None else "—")
            f.write(f"{label} & {' & '.join(vals)} \\\\\n")

        f.write("\\midrule\n")

        # Per-category
        for cat, label, metric_key in [
            ("phage", "Phage sens.", "sensitivity"),
            ("rna_virus", "RNA virus sens.", "sensitivity"),
            ("chromosome", "Chromosome spec.", "specificity"),
            ("plasmid", "Plasmid spec.", "specificity"),
        ]:
            vals = []
            for t in tools_present:
                v = comparison[t].get(cat, {}).get(metric_key)
                vals.append(f"{v:.1%}" if v is not None else "—")
            f.write(f"{label} & {' & '.join(vals)} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    logger.info(f"LaTeX table written to {latex_file}")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
