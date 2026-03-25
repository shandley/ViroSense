#!/usr/bin/env python3
"""
Compare boundary resolution methods for exon-intron detection.

Methods:
1. Smoothed threshold (current): 100bp uniform filter + threshold at 0
2. Change-point detection (PELT): statistical change-point on raw inversion
3. Derivative edge detection: peaks in |d/dx(inversion)|
4. Linear probe (2-feature lite): logistic regression on (cos1, cos3) features
   trained on HBB, tested on all others

For the full linear probe (4096-D embeddings), see boundary_resolution_full.py
which requires HTCF extraction.
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np

DATA_DIR = Path("results/experiments/exon_intron")
OUT_DIR = Path("results/experiments/boundary_resolution")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_gene(gene_name: str) -> dict | None:
    """Load per-position cosines and annotations for a gene."""
    mp = DATA_DIR / "metrics" / f"{gene_name}_perpos.json"
    if not mp.exists():
        return None

    with open(mp) as f:
        data = json.load(f)

    cos1 = np.array(data["cos1"])
    cos3 = np.array(data["cos3"])
    seq_len = len(cos1)

    # Load annotations
    for ann_file in [DATA_DIR / "annotations_all.json", DATA_DIR / "annotations_fixed.json"]:
        if ann_file.exists():
            with open(ann_file) as f:
                all_ann = json.load(f)
            if gene_name in all_ann:
                ann = all_ann[gene_name]
                regions = ann.get("cds", []) or ann.get("exons", [])
                truth = np.zeros(seq_len)
                for r in regions:
                    s, e = max(0, r["start"]), min(r["end"], seq_len)
                    truth[s:e] = 1
                if truth.mean() < 0.01 or truth.mean() > 0.99:
                    return None
                return {"cos1": cos1, "cos3": cos3, "truth": truth,
                        "seq_len": seq_len, "name": gene_name}
    return None


def method_smoothed_threshold(cos1, cos3, window=100):
    """Current method: smooth + threshold."""
    kernel = np.ones(window) / window
    cos1_s = np.convolve(cos1, kernel, mode="same")
    cos3_s = np.convolve(cos3, kernel, mode="same")
    inversion = cos3_s - cos1_s
    return (inversion > 0).astype(int)


def method_changepoint(cos1, cos3, pen=10):
    """Change-point detection using PELT on raw inversion signal."""
    import ruptures

    # Light smoothing (20bp) to reduce noise without blurring boundaries
    kernel = np.ones(20) / 20
    cos1_s = np.convolve(cos1, kernel, mode="same")
    cos3_s = np.convolve(cos3, kernel, mode="same")
    inversion = cos3_s - cos1_s

    # PELT change-point detection
    signal = inversion.reshape(-1, 1)
    algo = ruptures.Pelt(model="rbf", min_size=50).fit(signal)
    breakpoints = algo.predict(pen=pen)

    # Convert breakpoints to coding/non-coding segments
    predicted = np.zeros(len(cos1))
    prev = 0
    for bp in breakpoints:
        bp = min(bp, len(cos1))
        segment_mean = inversion[prev:bp].mean()
        if segment_mean > 0:
            predicted[prev:bp] = 1
        prev = bp

    return predicted.astype(int)


def method_derivative(cos1, cos3, smooth=30, edge_threshold=0.002):
    """Derivative-based edge detection."""
    kernel = np.ones(smooth) / smooth
    cos1_s = np.convolve(cos1, kernel, mode="same")
    cos3_s = np.convolve(cos3, kernel, mode="same")
    inversion = cos3_s - cos1_s

    # Compute derivative (gradient)
    gradient = np.gradient(inversion)
    abs_grad = np.abs(gradient)

    # Find edges: positions where gradient exceeds threshold
    # Between edges, classify based on mean inversion
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(abs_grad, height=edge_threshold, distance=50)

    # Classify segments between edges
    predicted = np.zeros(len(cos1))
    boundaries = [0] + list(peaks) + [len(cos1)]
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if inversion[s:e].mean() > 0:
            predicted[s:e] = 1

    return predicted.astype(int)


def method_linear_probe(cos1, cos3, truth, window=15):
    """Linear probe using local cosine statistics as features.

    Features per position: cos1, cos3, inversion, plus local mean/std
    in a small window. Trained on one gene, tested on others.
    """
    from sklearn.linear_model import LogisticRegression

    # Build feature matrix
    inversion = cos3 - cos1
    n = len(cos1)

    # Local statistics in a small window
    features = np.zeros((n, 6))
    half_w = window // 2
    for i in range(n):
        s = max(0, i - half_w)
        e = min(n, i + half_w + 1)
        features[i, 0] = cos1[i]
        features[i, 1] = cos3[i]
        features[i, 2] = inversion[i]
        features[i, 3] = np.mean(inversion[s:e])
        features[i, 4] = np.std(inversion[s:e])
        features[i, 5] = np.mean(cos3[s:e]) - np.mean(cos1[s:e])

    return features


def compute_metrics(predicted, truth):
    """Compute precision, recall, F1, boundary precision."""
    tp = float(((predicted == 1) & (truth == 1)).sum())
    fp = float(((predicted == 1) & (truth == 0)).sum())
    fn = float(((predicted == 0) & (truth == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Boundary precision: how close are predicted boundaries to true boundaries?
    pred_edges = np.where(np.diff(predicted))[0]
    true_edges = np.where(np.diff(truth))[0]

    if len(pred_edges) > 0 and len(true_edges) > 0:
        # For each predicted edge, find nearest true edge
        boundary_distances = []
        for pe in pred_edges:
            dist = np.min(np.abs(pe - true_edges))
            boundary_distances.append(dist)
        mean_boundary_dist = np.mean(boundary_distances)
        median_boundary_dist = np.median(boundary_distances)
    else:
        mean_boundary_dist = float("nan")
        median_boundary_dist = float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_boundary_distance": mean_boundary_dist,
        "median_boundary_distance": median_boundary_dist,
        "n_predicted_edges": len(pred_edges),
        "n_true_edges": len(true_edges),
    }


def main():
    # Load all genes
    gene_files = sorted(DATA_DIR.glob("metrics/*_perpos.json"))
    genes = []
    for gf in gene_files:
        gene_name = gf.stem.replace("_perpos", "")
        g = load_gene(gene_name)
        if g is not None:
            genes.append(g)

    print(f"Loaded {len(genes)} genes with annotations")

    # ── Method 1: Smoothed threshold (current, various windows) ──
    print(f"\n{'='*80}")
    print("Method 1: Smoothed threshold")
    for window in [25, 50, 75, 100, 150]:
        results = []
        for g in genes:
            pred = method_smoothed_threshold(g["cos1"], g["cos3"], window=window)
            m = compute_metrics(pred, g["truth"])
            results.append(m)
        mean_f1 = np.mean([r["f1"] for r in results])
        mean_prec = np.mean([r["precision"] for r in results])
        mean_rec = np.mean([r["recall"] for r in results])
        mean_bd = np.nanmean([r["median_boundary_distance"] for r in results])
        print(f"  Window={window:>4d}bp: F1={mean_f1:.3f} Prec={mean_prec:.3f} "
              f"Rec={mean_rec:.3f} Boundary={mean_bd:.0f}bp")

    # ── Method 2: Change-point detection ──
    print(f"\n{'='*80}")
    print("Method 2: Change-point detection (PELT)")
    try:
        import ruptures
        for pen in [5, 10, 20, 50]:
            results = []
            for g in genes:
                try:
                    pred = method_changepoint(g["cos1"], g["cos3"], pen=pen)
                    m = compute_metrics(pred, g["truth"])
                    results.append(m)
                except Exception as e:
                    pass
            if results:
                mean_f1 = np.mean([r["f1"] for r in results])
                mean_prec = np.mean([r["precision"] for r in results])
                mean_rec = np.mean([r["recall"] for r in results])
                mean_bd = np.nanmean([r["median_boundary_distance"] for r in results])
                print(f"  Penalty={pen:>3d}: F1={mean_f1:.3f} Prec={mean_prec:.3f} "
                      f"Rec={mean_rec:.3f} Boundary={mean_bd:.0f}bp ({len(results)} genes)")
    except ImportError:
        print("  ruptures not installed — install with: uv add ruptures")

    # ── Method 3: Derivative edge detection ──
    print(f"\n{'='*80}")
    print("Method 3: Derivative edge detection")
    for smooth in [20, 30, 50]:
        for thresh in [0.001, 0.002, 0.003]:
            results = []
            for g in genes:
                try:
                    pred = method_derivative(g["cos1"], g["cos3"],
                                             smooth=smooth, edge_threshold=thresh)
                    m = compute_metrics(pred, g["truth"])
                    results.append(m)
                except Exception:
                    pass
            if results:
                mean_f1 = np.mean([r["f1"] for r in results])
                mean_prec = np.mean([r["precision"] for r in results])
                mean_rec = np.mean([r["recall"] for r in results])
                mean_bd = np.nanmean([r["median_boundary_distance"] for r in results])
                print(f"  Smooth={smooth:>2d}bp, thresh={thresh:.3f}: F1={mean_f1:.3f} "
                      f"Prec={mean_prec:.3f} Rec={mean_rec:.3f} Boundary={mean_bd:.0f}bp")

    # ── Method 4: Linear probe (leave-one-out) ──
    print(f"\n{'='*80}")
    print("Method 4: Linear probe (2-feature lite, leave-one-out)")
    from sklearn.linear_model import LogisticRegression

    probe_results = []
    for i, test_gene in enumerate(genes):
        # Train on all others
        train_X = []
        train_y = []
        for j, train_gene in enumerate(genes):
            if i == j:
                continue
            feats = method_linear_probe(train_gene["cos1"], train_gene["cos3"],
                                        train_gene["truth"])
            train_X.append(feats)
            train_y.append(train_gene["truth"])

        train_X = np.vstack(train_X)
        train_y = np.concatenate(train_y)

        # Train
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(train_X, train_y)

        # Test
        test_feats = method_linear_probe(test_gene["cos1"], test_gene["cos3"],
                                          test_gene["truth"])
        pred = clf.predict(test_feats)
        m = compute_metrics(pred, test_gene["truth"])
        m["gene"] = test_gene["name"]
        probe_results.append(m)

    mean_f1 = np.mean([r["f1"] for r in probe_results])
    mean_prec = np.mean([r["precision"] for r in probe_results])
    mean_rec = np.mean([r["recall"] for r in probe_results])
    mean_bd = np.nanmean([r["median_boundary_distance"] for r in probe_results])
    print(f"  Leave-one-out: F1={mean_f1:.3f} Prec={mean_prec:.3f} "
          f"Rec={mean_rec:.3f} Boundary={mean_bd:.0f}bp")

    # Per-gene details for probe
    print(f"\n  Per-gene F1:")
    for r in sorted(probe_results, key=lambda x: -x["f1"])[:10]:
        print(f"    {r['gene']:<28s} F1={r['f1']:.3f} Prec={r['precision']:.3f} "
              f"Rec={r['recall']:.3f} Boundary={r['median_boundary_distance']:.0f}bp")

    # ── Summary comparison ──
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Method':<35s} {'F1':>6s} {'Prec':>6s} {'Recall':>6s} {'Boundary':>10s}")
    print(f"{'-'*65}")

    # Best smoothed threshold
    best_smooth = {}
    for window in [25, 50, 75, 100]:
        results = [compute_metrics(
            method_smoothed_threshold(g["cos1"], g["cos3"], window=window),
            g["truth"]) for g in genes]
        f1 = np.mean([r["f1"] for r in results])
        if not best_smooth or f1 > best_smooth["f1"]:
            best_smooth = {"f1": f1, "window": window,
                           "prec": np.mean([r["precision"] for r in results]),
                           "rec": np.mean([r["recall"] for r in results]),
                           "bd": np.nanmean([r["median_boundary_distance"] for r in results])}
    print(f"{'Smoothed threshold ('+str(best_smooth['window'])+'bp)':<35s} "
          f"{best_smooth['f1']:>6.3f} {best_smooth['prec']:>6.3f} "
          f"{best_smooth['rec']:>6.3f} {best_smooth['bd']:>9.0f}bp")

    print(f"{'Linear probe (2-feature LOO)':<35s} "
          f"{mean_f1:>6.3f} {mean_prec:>6.3f} {mean_rec:>6.3f} {mean_bd:>9.0f}bp")

    # Save results
    all_results = {
        "smoothed_threshold": best_smooth,
        "linear_probe": {
            "f1": mean_f1, "precision": mean_prec, "recall": mean_rec,
            "boundary_distance": mean_bd, "per_gene": probe_results,
        },
    }
    with open(OUT_DIR / "boundary_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved to {OUT_DIR}/boundary_comparison.json")


if __name__ == "__main__":
    main()
