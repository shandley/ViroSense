#!/usr/bin/env python3
"""
Recompute RNA dark matter classification with full periodicity features.

Extracts per-position embeddings for 203 sequences, computes all 6 features
(cos3, cos1, lag3, norm_ratio, coding_fraction, norm_cv), and runs classification.

Usage:
    # Cloud NIM (40B, ~1.7 hours):
    NVIDIA_API_KEY=... uv run python scripts/rna_dark_matter_recompute.py

    # HTCF self-hosted (7B, ~17 min):
    uv run python scripts/rna_dark_matter_recompute.py --nim-url http://localhost:8000
"""

import argparse
import asyncio
import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("results/rna_dark_matter_v2")


def compute_full_periodicity_features(emb: np.ndarray) -> dict:
    """Compute all 6 periodicity features from per-position embeddings."""
    n = len(emb)
    norms = np.linalg.norm(emb, axis=1)

    # Lag-3 autocorrelation of norms
    mean_n = norms.mean()
    var_n = norms.var()
    lag3 = float(np.mean((norms[:n-3] - mean_n) * (norms[3:] - mean_n)) / var_n) if var_n > 0 else 0.0
    lag1 = float(np.mean((norms[:n-1] - mean_n) * (norms[1:] - mean_n)) / var_n) if var_n > 0 else 0.0

    # Cosine similarities at offset-1 and offset-3
    cos1_vals = []
    cos3_vals = []
    for j in range(n - 3):
        nj = norms[j]
        nj1 = norms[j + 1]
        nj3 = norms[j + 3]
        if nj > 0 and nj1 > 0:
            cos1_vals.append(float(np.dot(emb[j], emb[j+1]) / (nj * nj1)))
        if nj > 0 and nj3 > 0:
            cos3_vals.append(float(np.dot(emb[j], emb[j+3]) / (nj * nj3)))

    cos1 = float(np.mean(cos1_vals)) if cos1_vals else 0.0
    cos3 = float(np.mean(cos3_vals)) if cos3_vals else 0.0

    # Norm statistics
    norm_mean = float(norms.mean())
    norm_std = float(norms.std())
    norm_cv = norm_std / norm_mean if norm_mean > 0 else 0.0

    return {
        "cos3": round(cos3, 4),
        "cos1": round(cos1, 4),
        "lag3": round(lag3, 4),
        "lag1": round(lag1, 4),
        "norm_mean": round(norm_mean, 2),
        "norm_cv": round(norm_cv, 4),
        "inversion": cos3 > cos1,
        "inversion_gap": round(cos3 - cos1, 4),
    }


async def extract_features(nim_url: str | None = None):
    """Extract per-position embeddings and compute features for all 203 sequences."""
    import httpx

    api_key = os.environ.get("NVIDIA_API_KEY", "")

    if nim_url:
        url = f"{nim_url.rstrip('/')}/biology/arc/evo2/forward"
        layer = "decoder.layers.10"
        concurrency = 1
    else:
        url = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward"
        layer = "blocks.10"
        concurrency = 3
        if not api_key:
            print("ERROR: Set NVIDIA_API_KEY or --nim-url")
            sys.exit(1)

    # Load sample list and sequences
    df = pd.read_csv("results/poc_rna_dark_matter/batch_results.csv")
    df = df.dropna(subset=["category"])

    fasta_path = Path("results/poc_rna_dark_matter/samples.fasta")
    if not fasta_path.exists():
        print(f"ERROR: {fasta_path} not found")
        sys.exit(1)

    # Parse FASTA
    from Bio import SeqIO
    seqs = {}
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        seqs[record.id] = str(record.seq).upper()

    print(f"Sequences: {len(seqs)}, Manifest: {len(df)}")

    # Check cache
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUT_DIR / "features.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"Loaded {len(cached)} cached features")
    else:
        cached = {}

    # Find sequences to extract
    to_extract = []
    for _, row in df.iterrows():
        seq_id = row["seq_id"]
        if seq_id in cached:
            continue
        if seq_id in seqs:
            to_extract.append((seq_id, seqs[seq_id], row["category"]))
        else:
            # Try matching by prefix
            for fasta_id, seq in seqs.items():
                if seq_id in fasta_id or fasta_id in seq_id:
                    to_extract.append((seq_id, seq, row["category"]))
                    break

    print(f"Need extraction: {len(to_extract)}")

    if to_extract:
        sem = asyncio.Semaphore(concurrency)
        completed = 0

        async def extract_one(client, seq_id, seq):
            nonlocal completed
            async with sem:
                headers = {"Content-Type": "application/json"}
                if not nim_url:
                    headers["Authorization"] = f"Bearer {api_key}"
                payload = {"sequence": seq[:16000], "output_layers": [layer]}

                for attempt in range(5):
                    try:
                        resp = await client.post(url, json=payload, headers=headers,
                                                 timeout=300, follow_redirects=True)
                        if resp.status_code == 429:
                            await asyncio.sleep(2 ** attempt * 10)
                            continue
                        resp.raise_for_status()
                        data = resp.json()
                        raw = base64.b64decode(data["data"])
                        npz = np.load(io.BytesIO(raw))
                        emb = npz[f"{layer}.output"]
                        if emb.ndim == 3:
                            if emb.shape[0] == 1: emb = emb.squeeze(0)
                            elif emb.shape[1] == 1: emb = emb.squeeze(1)

                        features = compute_full_periodicity_features(emb)
                        completed += 1
                        if completed % 20 == 0:
                            print(f"  [{completed}/{len(to_extract)}]")
                        return seq_id, features
                    except Exception as e:
                        if attempt < 4:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        print(f"  {seq_id}: FAILED ({str(e)[:50]})")
                        return seq_id, None
                return seq_id, None

        async with httpx.AsyncClient() as client:
            tasks = [extract_one(client, sid, seq) for sid, seq, _ in to_extract]
            for coro in asyncio.as_completed(tasks):
                seq_id, features = await coro
                if features:
                    cached[seq_id] = features

        # Save cache
        with open(cache_path, "w") as f:
            json.dump(cached, f, indent=2)
        print(f"Saved {len(cached)} features to {cache_path}")

    return cached


def classify_and_report(features: dict):
    """Run RNA dark matter classification with full features."""
    from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold

    df = pd.read_csv("results/poc_rna_dark_matter/batch_results.csv")
    df = df.dropna(subset=["category"])

    # Merge features
    rows = []
    for _, row in df.iterrows():
        seq_id = row["seq_id"]
        if seq_id in features:
            feat = features[seq_id]
            rows.append({
                "seq_id": seq_id,
                "category": row["category"],
                "cos3": feat["cos3"],
                "cos1": feat["cos1"],
                "lag3": feat["lag3"],
                "lag1": feat["lag1"],
                "norm_cv": feat["norm_cv"],
                "norm_mean": feat["norm_mean"],
                "inversion_gap": feat["inversion_gap"],
            })

    df_feat = pd.DataFrame(rows)
    print(f"\nClassification dataset: {len(df_feat)} sequences")
    print(f"Categories: {df_feat['category'].value_counts().to_dict()}")

    # Binary: RNA virus vs rest
    y = (df_feat["category"] == "rna_virus").astype(int).values

    # Feature combinations to test
    feature_sets = {
        "full_6": ["cos3", "cos1", "lag3", "norm_cv", "norm_mean", "inversion_gap"],
        "cos3_lag3_cos1": ["cos3", "lag3", "cos1"],
        "cos3_only": ["cos3"],
    }

    results = {}
    for name, feat_cols in feature_sets.items():
        X = df_feat[feat_cols].values

        clf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=5)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
        acc = accuracy_score(y, y_pred)

        # Feature importance
        clf.fit(X, y)
        imp = dict(zip(feat_cols, clf.feature_importances_))

        print(f"\n{name} ({len(feat_cols)} features):")
        print(f"  Accuracy: {acc:.1%}, AUC: {roc_auc:.3f}")
        for feat, importance in sorted(imp.items(), key=lambda x: -x[1]):
            print(f"    {feat:20s}: {importance:.1%}")

        results[name] = {
            "accuracy": acc, "auc": roc_auc,
            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "importances": {k: round(v, 3) for k, v in imp.items()},
        }

    # Save
    with open(OUT_DIR / "classification_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/classification_results.json")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nim-url", type=str, default=None)
    args = parser.parse_args()

    print("=== Step 1: Extract features ===")
    features = asyncio.run(extract_features(nim_url=args.nim_url))

    print("\n=== Step 2: Classify ===")
    classify_and_report(features)


if __name__ == "__main__":
    main()
