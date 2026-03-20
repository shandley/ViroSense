#!/usr/bin/env python3
"""Analyze functional clustering in Evo2 40B mean-pooled embeddings.

Compares blocks.10 vs blocks.28, and 40B vs 7B, for gene family clustering.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine


def load_embeddings(emb_dir: Path, panel: list[dict], layer_suffix: str):
    """Load mean-pooled embeddings for a specific layer."""
    embs = []
    labels = []
    names = []
    entries = []

    for entry in panel:
        name = entry["name"]
        path = emb_dir / f"{name}_mean_{layer_suffix}.npy"
        if not path.exists():
            continue
        emb = np.load(path)
        embs.append(emb)
        labels.append(entry.get("gene_family", entry.get("category", "?")))
        names.append(name)
        entries.append(entry)

    return np.array(embs, dtype=np.float64), np.array(labels), names, entries


def analyze_clustering(embs, labels, name_list, title=""):
    """Run full clustering analysis."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"  Sequences: {len(embs)}, Dim: {embs.shape[1]}")
    print(f"  Families: {len(set(labels))}")

    # L2 normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embs_norm = embs / norms

    # PCA
    n_comp = min(50, len(embs_norm) - 1)
    pca = PCA(n_components=n_comp)
    embs_pca = pca.fit_transform(embs_norm)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA-{n_comp} variance explained: {var_explained:.1%}")

    # Silhouette
    sil_raw = silhouette_score(embs_norm, labels, metric="cosine")
    sil_pca = silhouette_score(embs_pca, labels, metric="cosine")
    print(f"\n  Silhouette (raw, cosine): {sil_raw:.3f}")
    print(f"  Silhouette (PCA-{n_comp}, cosine): {sil_pca:.3f}")

    # NN accuracy
    for space_name, space in [("raw", embs_norm), ("PCA", embs_pca)]:
        nn = NearestNeighbors(n_neighbors=2, metric="cosine")
        nn.fit(space)
        _, indices = nn.kneighbors(space)
        correct = sum(1 for i in range(len(labels)) if labels[indices[i, 1]] == labels[i])
        print(f"  NN accuracy ({space_name}): {correct}/{len(labels)} ({100*correct/len(labels):.1f}%)")

    # Within/between by family
    families = defaultdict(list)
    for i, label in enumerate(labels):
        families[label].append(i)

    print(f"\n  Within/between distances (PCA space):")
    family_results = {}
    for fam in sorted(families.keys(), key=lambda x: -len(families[x])):
        idxs = families[fam]
        if len(idxs) < 2:
            continue
        fam_embs = embs_pca[idxs]
        other_idxs = [i for i in range(len(embs_pca)) if i not in idxs]
        other_embs = embs_pca[other_idxs]

        within = [cosine(fam_embs[i], fam_embs[j])
                   for i in range(len(fam_embs)) for j in range(i+1, len(fam_embs))]
        np.random.seed(42)
        sample = np.random.choice(len(other_embs), min(100, len(other_embs)), replace=False)
        between = [cosine(fam_embs[i], other_embs[j])
                   for i in range(min(15, len(fam_embs))) for j in sample]

        w = np.mean(within)
        b = np.mean(between)
        ratio = b / w if w > 0 else float("inf")
        family_results[fam] = {"within": w, "between": b, "ratio": ratio, "n": len(idxs)}
        print(f"    {fam:25s} (N={len(idxs):2d}): within={w:.4f}, between={b:.4f}, ratio={ratio:.2f}x")

    return {
        "silhouette_raw": sil_raw,
        "silhouette_pca": sil_pca,
        "nn_accuracy_pca": correct / len(labels),
        "families": family_results,
    }


def main():
    with open("results/comprehensive/panel.json") as f:
        panel = json.load(f)

    comp_b = [e for e in panel if e.get("component") == "B"]
    print(f"Component B panel: {len(comp_b)} sequences")

    results = {}

    # ── 40B blocks.10 ──
    emb_dir_40b = Path("results/codon_periodicity_panel/embeddings_40b")
    embs_b10, labels_b10, names_b10, _ = load_embeddings(emb_dir_40b, comp_b, "blocks_10")
    if len(embs_b10) > 0:
        results["40B_blocks10"] = analyze_clustering(
            embs_b10, labels_b10, names_b10, "Evo2 40B — blocks.10 (per-position optimal)")

    # ── 40B blocks.28 ──
    embs_b28, labels_b28, names_b28, _ = load_embeddings(emb_dir_40b, comp_b, "blocks_28")
    if len(embs_b28) > 0:
        results["40B_blocks28"] = analyze_clustering(
            embs_b28, labels_b28, names_b28, "Evo2 40B — blocks.28 (classifier layer)")

    # ── 7B decoder.layers.10 (from HTCF) ──
    emb_dir_7b = Path("results/codon_periodicity_panel/embeddings")
    embs_7b = []
    labels_7b = []
    names_7b = []
    for entry in comp_b:
        name = entry["name"]
        path = emb_dir_7b / f"{name}_mean.npy"
        if not path.exists():
            continue
        emb = np.load(path)
        embs_7b.append(emb)
        labels_7b.append(entry.get("gene_family", entry.get("category", "?")))
        names_7b.append(name)

    if embs_7b:
        embs_7b = np.array(embs_7b, dtype=np.float64)
        labels_7b = np.array(labels_7b)
        results["7B_layer10"] = analyze_clustering(
            embs_7b, labels_7b, names_7b, "Evo2 7B — decoder.layers.10 (HTCF)")

    # ── Comparison summary ──
    print(f"\n{'='*70}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model + Layer':<35s} {'Silhouette':>10s} {'NN Acc':>8s} {'Dim':>5s}")
    for key, r in results.items():
        dim = embs_b10.shape[1] if "40B" in key else (embs_7b.shape[1] if embs_7b is not None else 0)
        print(f"  {key:<33s} {r['silhouette_pca']:>10.3f} {r['nn_accuracy_pca']:>7.1%} {dim:>5d}")

    # Save results
    out_path = Path("results/codon_periodicity_panel/functional_clustering_comparison.json")
    # Convert to serializable
    serializable = {}
    for key, r in results.items():
        sr = {k: v for k, v in r.items() if k != "families"}
        sr["families"] = {fam: {k: float(v) for k, v in fv.items()}
                          for fam, fv in r.get("families", {}).items()}
        serializable[key] = sr
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
