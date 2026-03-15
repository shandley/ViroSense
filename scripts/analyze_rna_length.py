#!/usr/bin/env python3
"""Phase 1: Diagnose RNA virus length-dependent detection failure.

Runs entirely on cached data — no API calls needed.
Outputs numerical results as JSON + TSV for visualization.

Usage (on HTCF):
    .venv/bin/python scripts/analyze_rna_length.py \
        --results results/benchmark/7b_16kb/detailed_results.tsv \
        --cache-dir results/benchmark/cache_7b/ \
        --manifest results/benchmark/manifest/manifest.tsv \
        --classifier results/classifiers/7b_16kb/classifier.joblib \
        --output results/benchmark/7b_16kb/rna_analysis/
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def load_embeddings(cache_dir: str, layer: str = "blocks_28_mlp_l3") -> tuple:
    """Load all cached embeddings (main + shards)."""
    import glob

    prefix = None
    for f in os.listdir(cache_dir):
        if f.endswith("_embeddings.npz") and "shard" not in f:
            prefix = f.replace("_embeddings.npz", "")
            break

    if prefix is None:
        raise FileNotFoundError(f"No main cache file in {cache_dir}")

    main_path = os.path.join(cache_dir, f"{prefix}_embeddings.npz")
    d = np.load(main_path, allow_pickle=True)
    all_ids = list(d["sequence_ids"])
    all_embs = [d["embeddings"]]

    shards = sorted(glob.glob(os.path.join(cache_dir, f"{prefix}_embeddings_shard_*.npz")))
    for s in shards:
        sd = np.load(s, allow_pickle=True)
        all_ids.extend(sd["sequence_ids"])
        all_embs.append(sd["embeddings"])

    embeddings = np.vstack(all_embs).astype(np.float64)
    return np.array(all_ids), embeddings


def analyze_score_distributions(df: pd.DataFrame, output_dir: str):
    """1a: Score distribution stats by category and length bin."""
    results = {}

    for category in ["rna_virus", "phage", "chromosome", "plasmid"]:
        cat_df = df[df["true_category"] == category]
        if len(cat_df) == 0:
            continue
        cat_results = {}
        for lb in sorted(cat_df["length_bin"].unique()):
            sub = cat_df[cat_df["length_bin"] == lb]
            scores = sub["viral_score"].values
            cat_results[lb] = {
                "n": len(sub),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "median": float(np.median(scores)),
                "q25": float(np.percentile(scores, 25)),
                "q75": float(np.percentile(scores, 75)),
                "frac_below_03": float(np.mean(scores < 0.3)),
                "frac_below_05": float(np.mean(scores < 0.5)),
                "frac_above_09": float(np.mean(scores > 0.9)),
                "recall": float(np.mean(scores >= 0.5)) if category in ("rna_virus", "phage") else None,
                "specificity": float(np.mean(scores < 0.5)) if category in ("chromosome", "plasmid", "rna_cellular") else None,
            }
        results[category] = cat_results

    # Also export raw scores for plotting
    for category in ["rna_virus", "phage"]:
        cat_df = df[df["true_category"] == category]
        for lb in sorted(cat_df["length_bin"].unique()):
            sub = cat_df[cat_df["length_bin"] == lb]
            scores_path = os.path.join(output_dir, f"scores_{category}_{lb.replace('-', '_')}.tsv")
            sub[["sequence_id", "length", "viral_score"]].to_csv(scores_path, sep="\t", index=False)

    return results


def analyze_embeddings_pca(
    seq_ids: np.ndarray, embeddings: np.ndarray,
    df: pd.DataFrame, output_dir: str
):
    """1b: PCA on embeddings, project RNA virus sequences."""
    # Build lookup from seq_id to embedding index
    id_to_idx = {sid: i for i, sid in enumerate(seq_ids)}

    # Separate by category
    categories = {}
    for _, row in df.iterrows():
        sid = row["sequence_id"]
        if sid in id_to_idx:
            cat = row["true_category"]
            lb = row["length_bin"]
            if cat not in categories:
                categories[cat] = {"ids": [], "indices": [], "lengths": [], "bins": []}
            categories[cat]["ids"].append(sid)
            categories[cat]["indices"].append(id_to_idx[sid])
            categories[cat]["lengths"].append(row["length"])
            categories[cat]["bins"].append(lb)

    # Fit PCA on phage + chromosome (classifier training domain)
    train_indices = []
    for cat in ["phage", "chromosome"]:
        if cat in categories:
            train_indices.extend(categories[cat]["indices"])

    train_embs = embeddings[train_indices]
    pca = PCA(n_components=10)
    pca.fit(train_embs)
    print(f"PCA explained variance (first 10): {pca.explained_variance_ratio_[:10].sum():.3f}")

    # Project all categories
    pca_results = []
    for cat, data in categories.items():
        cat_embs = embeddings[data["indices"]]
        projected = pca.transform(cat_embs)
        for i, sid in enumerate(data["ids"]):
            pca_results.append({
                "sequence_id": sid,
                "category": cat,
                "length": data["lengths"][i],
                "length_bin": data["bins"][i],
                "pc1": float(projected[i, 0]),
                "pc2": float(projected[i, 1]),
                "pc3": float(projected[i, 2]),
            })

    pca_df = pd.DataFrame(pca_results)
    pca_df.to_csv(os.path.join(output_dir, "pca_projections.tsv"), sep="\t", index=False)

    # Summary: mean PC1/PC2 by category and length bin
    summary = pca_df.groupby(["category", "length_bin"]).agg(
        mean_pc1=("pc1", "mean"),
        mean_pc2=("pc2", "mean"),
        std_pc1=("pc1", "std"),
        std_pc2=("pc2", "std"),
        n=("pc1", "count"),
    ).reset_index()
    print("\nPCA centroids by category and length:")
    print(summary.to_string(index=False))

    return {
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "pca_summary": summary.to_dict(orient="records"),
    }


def analyze_cosine_similarity(
    seq_ids: np.ndarray, embeddings: np.ndarray,
    df: pd.DataFrame, output_dir: str
):
    """1c: Cosine similarity to phage/chromosome centroids."""
    id_to_idx = {sid: i for i, sid in enumerate(seq_ids)}

    # Compute centroids
    centroids = {}
    for cat in ["phage", "chromosome"]:
        cat_df = df[df["true_category"] == cat]
        indices = [id_to_idx[sid] for sid in cat_df["sequence_id"] if sid in id_to_idx]
        centroids[cat] = embeddings[indices].mean(axis=0, keepdims=True)

    # For RNA virus: compute similarity to both centroids
    rna_df = df[df["true_category"] == "rna_virus"].copy()
    rna_indices = [id_to_idx[sid] for sid in rna_df["sequence_id"] if sid in id_to_idx]
    rna_embs = embeddings[rna_indices]

    sim_phage = cosine_similarity(rna_embs, centroids["phage"]).flatten()
    sim_chr = cosine_similarity(rna_embs, centroids["chromosome"]).flatten()
    ratio = sim_phage / (sim_chr + 1e-10)

    rna_df = rna_df.iloc[:len(rna_indices)].copy()
    rna_df["sim_phage"] = sim_phage
    rna_df["sim_chromosome"] = sim_chr
    rna_df["sim_ratio"] = ratio

    rna_df[["sequence_id", "length", "length_bin", "viral_score", "sim_phage", "sim_chromosome", "sim_ratio"]].to_csv(
        os.path.join(output_dir, "cosine_similarity.tsv"), sep="\t", index=False
    )

    # Summary by length bin
    print("\nCosine similarity to centroids (RNA virus):")
    for lb in ["500-1kb", "1-3kb", "3-5kb", "5-10kb", "10-16kb"]:
        sub = rna_df[rna_df["length_bin"] == lb]
        if len(sub) == 0:
            continue
        print(f"  {lb}: sim_phage={sub['sim_phage'].mean():.4f}, "
              f"sim_chr={sub['sim_chromosome'].mean():.4f}, "
              f"ratio={sub['sim_ratio'].mean():.4f}, "
              f"ratio_std={sub['sim_ratio'].std():.4f}")

    return {lb: {
        "sim_phage": float(rna_df[rna_df["length_bin"] == lb]["sim_phage"].mean()),
        "sim_chromosome": float(rna_df[rna_df["length_bin"] == lb]["sim_chromosome"].mean()),
        "sim_ratio": float(rna_df[rna_df["length_bin"] == lb]["sim_ratio"].mean()),
    } for lb in ["500-1kb", "1-3kb", "3-5kb", "5-10kb", "10-16kb"]
        if len(rna_df[rna_df["length_bin"] == lb]) > 0}


def analyze_embedding_norms(
    seq_ids: np.ndarray, embeddings: np.ndarray,
    df: pd.DataFrame, output_dir: str
):
    """1d: Embedding L2 norm and variance by category/length."""
    id_to_idx = {sid: i for i, sid in enumerate(seq_ids)}

    results = {}
    print("\nEmbedding norms by category and length:")
    for cat in ["phage", "chromosome", "rna_virus", "rna_cellular"]:
        cat_df = df[df["true_category"] == cat]
        if len(cat_df) == 0:
            continue
        cat_results = {}
        for lb in sorted(cat_df["length_bin"].unique()):
            sub = cat_df[cat_df["length_bin"] == lb]
            indices = [id_to_idx[sid] for sid in sub["sequence_id"] if sid in id_to_idx]
            if not indices:
                continue
            embs = embeddings[indices]
            norms = np.linalg.norm(embs, axis=1)
            dim_var = np.var(embs, axis=1).mean()  # mean per-sequence variance across dims
            cat_results[lb] = {
                "n": len(indices),
                "mean_norm": float(norms.mean()),
                "std_norm": float(norms.std()),
                "mean_dim_variance": float(dim_var),
            }
            print(f"  {cat}/{lb}: norm={norms.mean():.2f}±{norms.std():.2f}, dim_var={dim_var:.6f}")
        results[cat] = cat_results

    return results


def analyze_false_negatives(df: pd.DataFrame, output_dir: str):
    """1e: Taxonomy of RNA virus false negatives."""
    rna = df[df["true_category"] == "rna_virus"]
    fn = rna[rna["viral_score"] < 0.5].copy()
    tp = rna[rna["viral_score"] >= 0.5].copy()

    print(f"\nRNA virus false negatives: {len(fn)}/{len(rna)} ({100*len(fn)/len(rna):.1f}%)")
    print(f"RNA virus true positives: {len(tp)}/{len(rna)}")

    # Parse taxonomy from sequence IDs (format varies)
    def parse_organism(sid):
        # Try to extract organism name from ID
        parts = sid.split("|")
        if len(parts) > 1:
            return parts[1].strip().split(",")[0].strip()
        return sid.split("_")[0]

    fn["organism"] = fn["sequence_id"].apply(parse_organism)
    tp["organism"] = tp["sequence_id"].apply(parse_organism)

    # Count by organism for false negatives
    fn_orgs = Counter(fn["organism"])
    tp_orgs = Counter(tp["organism"])

    print("\nTop 20 false negative organisms:")
    for org, count in fn_orgs.most_common(20):
        tp_count = tp_orgs.get(org, 0)
        total = count + tp_count
        print(f"  {org}: {count} FN / {total} total ({100*count/total:.0f}% failure rate)")

    # By length bin
    print("\nFalse negative rate by length bin:")
    for lb in ["500-1kb", "1-3kb", "3-5kb", "5-10kb", "10-16kb"]:
        sub_fn = fn[fn["length_bin"] == lb]
        sub_all = rna[rna["length_bin"] == lb]
        if len(sub_all) > 0:
            print(f"  {lb}: {len(sub_fn)}/{len(sub_all)} ({100*len(sub_fn)/len(sub_all):.1f}%)")

    # Export
    fn[["sequence_id", "length", "length_bin", "viral_score"]].to_csv(
        os.path.join(output_dir, "false_negatives.tsv"), sep="\t", index=False
    )

    return {
        "n_false_negatives": len(fn),
        "n_true_positives": len(tp),
        "top_organisms": dict(fn_orgs.most_common(20)),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze RNA virus length-dependent detection failure")
    parser.add_argument("--results", required=True, help="detailed_results.tsv from benchmark")
    parser.add_argument("--cache-dir", required=True, help="Embedding cache directory")
    parser.add_argument("--manifest", required=True, help="Benchmark manifest TSV")
    parser.add_argument("--classifier", help="Classifier joblib path (optional)")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("RNA Virus Length-Dependent Detection Analysis")
    print("=" * 60)

    # Load results
    print("\nLoading detailed results...")
    df = pd.read_csv(args.results, sep="\t")
    print(f"  {len(df)} sequences loaded")

    # Load embeddings
    print("\nLoading cached embeddings...")
    seq_ids, embeddings = load_embeddings(args.cache_dir)
    print(f"  {len(seq_ids)} embeddings loaded, shape: {embeddings.shape}")

    all_results = {}

    # 1a: Score distributions
    print("\n" + "=" * 40)
    print("1a. Score Distributions")
    print("=" * 40)
    all_results["score_distributions"] = analyze_score_distributions(df, args.output)

    # 1b: PCA
    print("\n" + "=" * 40)
    print("1b. PCA Embedding Analysis")
    print("=" * 40)
    all_results["pca"] = analyze_embeddings_pca(seq_ids, embeddings, df, args.output)

    # 1c: Cosine similarity
    print("\n" + "=" * 40)
    print("1c. Cosine Similarity to Centroids")
    print("=" * 40)
    all_results["cosine_similarity"] = analyze_cosine_similarity(seq_ids, embeddings, df, args.output)

    # 1d: Embedding norms
    print("\n" + "=" * 40)
    print("1d. Embedding Norms")
    print("=" * 40)
    all_results["embedding_norms"] = analyze_embedding_norms(seq_ids, embeddings, df, args.output)

    # 1e: False negatives
    print("\n" + "=" * 40)
    print("1e. False Negative Analysis")
    print("=" * 40)
    all_results["false_negatives"] = analyze_false_negatives(df, args.output)

    # Save summary
    summary_path = os.path.join(args.output, "analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
