#!/usr/bin/env python3
"""Analyze ViroSense clustering results on gut virome data.

Applies PCA dimensionality reduction before clustering, then cross-references
with geNomad taxonomy and CheckV quality for validation.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import silhouette_score

BASE = Path(__file__).resolve().parent.parent / "data" / "test"


def load_cached_embeddings(cache_dir: Path, sequence_ids: list[str]) -> np.ndarray:
    """Load cached embeddings in the order of sequence_ids."""
    npz_path = cache_dir / "evo2_7b_blocks_28_mlp_l3_embeddings.npz"
    data = np.load(npz_path, allow_pickle=True)
    cached_ids = list(data["sequence_ids"])
    cached_emb = data["embeddings"]
    id_to_idx = {sid: i for i, sid in enumerate(cached_ids)}
    indices = [id_to_idx[sid] for sid in sequence_ids]
    return cached_emb[indices]


def load_metadata(sequence_ids: list[str]) -> dict[str, dict]:
    """Load geNomad taxonomy, ViroSense scores, CheckV quality, and lengths."""
    meta = {sid: {} for sid in sequence_ids}

    # geNomad taxonomy
    genomad_path = BASE / "genomad_gut_virome" / "gut_virome_test_summary" / "gut_virome_test_virus_summary.tsv"
    with open(genomad_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["seq_name"] in meta:
                meta[row["seq_name"]]["taxonomy"] = row["taxonomy"]
                meta[row["seq_name"]]["n_hallmarks"] = int(row["n_hallmarks"])

    # ViroSense scores
    vs_path = BASE / "gut_virome_results" / "detection_results.tsv"
    with open(vs_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["contig_id"] in meta:
                meta[row["contig_id"]]["viral_score"] = float(row["viral_score"])
                meta[row["contig_id"]]["length"] = int(row["contig_length"])

    # CheckV quality
    checkv_path = BASE / "set5_gut_virome_dataset" / "checkv_quality_summary.csv"
    with open(checkv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sample"] == "SRR5747446" and row["contig_id"] in meta:
                meta[row["contig_id"]]["checkv_quality"] = row["checkv_quality"]

    return meta


def cluster_with_pca(embeddings: np.ndarray, n_components: int, min_cluster_size: int = 3):
    """PCA + StandardScaler + HDBSCAN."""
    # Normalize embeddings to float64 to avoid overflow in matmul
    emb = embeddings.astype(np.float64)
    emb = StandardScaler().fit_transform(emb)

    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(emb)
    explained = pca.explained_variance_ratio_.sum()

    scaled = StandardScaler().fit_transform(reduced)
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=2)
    labels = clusterer.fit_predict(scaled)

    n_clusters = len(set(labels) - {-1})
    n_noise = (labels == -1).sum()
    sil = silhouette_score(scaled, labels) if n_clusters >= 2 else 0.0

    return labels, {
        "n_components": n_components,
        "variance_explained": round(explained, 4),
        "n_clusters": n_clusters,
        "n_noise": int(n_noise),
        "silhouette": round(float(sil), 4),
    }


def main():
    from virosense.io.fasta import read_fasta

    # Load sequences
    fasta_path = BASE / "gut_virome_viral_only.fasta"
    sequences = read_fasta(str(fasta_path))
    sequence_ids = list(sequences.keys())
    print(f"Loaded {len(sequence_ids)} viral contigs")

    # Load cached embeddings
    embeddings = load_cached_embeddings(BASE / "cache", sequence_ids)
    print(f"Embedding shape: {embeddings.shape}")

    # Load metadata
    meta = load_metadata(sequence_ids)

    # Try different PCA dimensions
    print("\n" + "=" * 70)
    print("PCA DIMENSIONALITY REDUCTION + HDBSCAN")
    print("=" * 70)

    best_labels = None
    best_info = None
    best_sil = -1

    for n_comp in [10, 20, 30, 50]:
        labels, info = cluster_with_pca(embeddings, n_comp)
        print(f"\n  PCA-{n_comp}: {info['variance_explained']:.1%} variance, "
              f"{info['n_clusters']} clusters, {info['n_noise']} noise, "
              f"silhouette={info['silhouette']:.3f}")

        if info["silhouette"] > best_sil and info["n_clusters"] >= 2:
            best_sil = info["silhouette"]
            best_labels = labels
            best_info = info

    if best_labels is None:
        print("\nNo valid clustering found.")
        return

    print(f"\nBest: PCA-{best_info['n_components']} "
          f"({best_info['n_clusters']} clusters, silhouette={best_info['silhouette']:.3f})")

    # Analyze best clustering
    print("\n" + "=" * 70)
    print(f"CLUSTER ANALYSIS (PCA-{best_info['n_components']}, HDBSCAN)")
    print("=" * 70)

    cluster_ids = sorted(set(best_labels))
    for cid in cluster_ids:
        members = [sequence_ids[i] for i in range(len(sequence_ids)) if best_labels[i] == cid]
        label = f"Cluster {cid}" if cid >= 0 else "Noise"

        print(f"\n--- {label} ({len(members)} sequences) ---")

        # Taxonomy breakdown
        tax_counts = Counter()
        checkv_counts = Counter()
        lengths = []
        scores = []
        for sid in members:
            m = meta[sid]
            tax = m.get("taxonomy", "not in geNomad")
            # Simplify taxonomy
            if tax != "not in geNomad":
                parts = tax.split(";")
                # Get family level or lowest non-empty
                family = next((p for p in reversed(parts) if p), "Unclassified")
            else:
                family = "Novel (not in geNomad)"
            tax_counts[family] += 1
            checkv_counts[m.get("checkv_quality", "N/A")] += 1
            lengths.append(m.get("length", 0))
            scores.append(m.get("viral_score", 0))

        print(f"  Length: {min(lengths)}-{max(lengths)} bp (median {sorted(lengths)[len(lengths)//2]})")
        print(f"  Viral score: {min(scores):.3f}-{max(scores):.3f} (mean {np.mean(scores):.3f})")
        print(f"  Taxonomy:")
        for tax, count in tax_counts.most_common():
            print(f"    {count}x {tax}")
        print(f"  CheckV quality:")
        for q, count in checkv_counts.most_common():
            print(f"    {count}x {q}")

    # Save results
    output_dir = BASE / "gut_virome_clusters_pca"
    output_dir.mkdir(exist_ok=True)

    # Write assignments with metadata
    with open(output_dir / "cluster_assignments_annotated.tsv", "w") as f:
        f.write("sequence_id\tcluster_id\tlength\tviral_score\tcheckv_quality\tgenonad_taxonomy\n")
        for i, sid in enumerate(sequence_ids):
            m = meta[sid]
            f.write(f"{sid}\t{best_labels[i]}\t{m.get('length', '')}\t"
                    f"{m.get('viral_score', '')}\t{m.get('checkv_quality', '')}\t"
                    f"{m.get('taxonomy', '')}\n")

    with open(output_dir / "cluster_metrics.json", "w") as f:
        json.dump(best_info, f, indent=2)

    print(f"\nResults written to {output_dir}")


if __name__ == "__main__":
    main()
