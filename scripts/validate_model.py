#!/usr/bin/env python3
"""Validate a trained ViroSense reference model.

Loads the trained classifier and cached embeddings, then runs:
1. Full training-set evaluation (sanity check — should be near-perfect)
2. Stratified 5-fold cross-validation (unbiased performance estimate)
3. Confidence distribution analysis
4. Hard example identification (most uncertain predictions)

Usage:
    python scripts/validate_model.py \
        --model data/reference/model/classifier.joblib \
        --cache-dir data/reference/cache \
        --labels data/reference/labels.tsv \
        --output data/reference/validation/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


def main():
    parser = argparse.ArgumentParser(
        description="Validate a trained ViroSense classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to trained classifier (classifier.joblib)",
    )
    parser.add_argument(
        "--cache-dir", required=True,
        help="Embedding cache directory (contains NPZ files)",
    )
    parser.add_argument(
        "--labels", required=True,
        help="Labels TSV (sequence_id, label)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for validation report",
    )
    parser.add_argument(
        "--layer", default="blocks.28.mlp.l3",
        help="Evo2 layer used for embeddings (default: blocks.28.mlp.l3)",
    )
    parser.add_argument(
        "--model-name", default="evo2_7b",
        help="Evo2 model name (default: evo2_7b)",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    args = parser.parse_args()

    from virosense.models.detector import ClassifierConfig, ViralClassifier

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load classifier ---
    print("Loading classifier...")
    classifier = ViralClassifier.load(args.model)
    class_names = classifier.metadata.get("class_names", ["cellular", "viral"])
    print(f"  Classes: {class_names}")
    print(f"  Input dim: {classifier.metadata.get('input_dim', '?')}")
    print(f"  Trained on: {classifier.metadata.get('n_train', '?')} samples")

    # --- 2. Load cached embeddings ---
    print("\nLoading cached embeddings...")
    cache_dir = Path(args.cache_dir)
    layer_safe = args.layer.replace(".", "_")
    cache_path = cache_dir / f"{args.model_name}_{layer_safe}_embeddings.npz"

    if not cache_path.exists():
        print(f"ERROR: Cache file not found: {cache_path}")
        sys.exit(1)

    data = np.load(cache_path, allow_pickle=True)
    cached_ids = list(data["sequence_ids"])
    cached_embeddings = data["embeddings"]
    print(f"  Loaded {len(cached_ids)} cached embeddings ({cached_embeddings.shape[1]}d)")

    # --- 3. Load labels and align ---
    print("\nLoading labels...")
    labels_df = pd.read_csv(args.labels, sep="\t")
    label_map = dict(zip(labels_df["sequence_id"], labels_df["label"]))

    id_to_idx = {sid: i for i, sid in enumerate(cached_ids)}

    matched_ids = [sid for sid in cached_ids if sid in label_map]
    if len(matched_ids) < len(cached_ids):
        print(f"  Warning: {len(cached_ids) - len(matched_ids)} cached IDs not in labels")
    if len(matched_ids) == 0:
        print("ERROR: No cached sequences matched the labels file")
        sys.exit(1)

    indices = [id_to_idx[sid] for sid in matched_ids]
    X = cached_embeddings[indices]
    y = np.array([label_map[sid] for sid in matched_ids])

    n_per_class = {name: int((y == i).sum()) for i, name in enumerate(class_names)}
    print(f"  Matched: {len(matched_ids)} sequences")
    for name, count in n_per_class.items():
        print(f"    {name}: {count}")

    # --- 4. Full training-set evaluation ---
    print("\n=== Training Set Evaluation (sanity check) ===")
    predictions = classifier.predict(X)
    probas = classifier.predict_proba(X)

    train_acc = accuracy_score(y, predictions)
    train_f1 = f1_score(y, predictions, average="weighted")
    print(f"  Accuracy: {train_acc:.4f}")
    print(f"  F1 (weighted): {train_f1:.4f}")

    if train_acc < 0.90:
        print("  WARNING: Training accuracy below 90% — model may be undertrained")

    # Confusion matrix
    cm = confusion_matrix(y, predictions)
    print(f"\n  Confusion matrix:")
    header = "  " + " " * 15 + "  ".join(f"{name:>10}" for name in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = "  " + f"{name:>15}" + "  ".join(f"{cm[i, j]:>10d}" for j in range(len(class_names)))
        print(row)

    # Per-class report
    print(f"\n  Per-class metrics:")
    report = classification_report(y, predictions, target_names=class_names, output_dict=True)
    for name in class_names:
        r = report[name]
        print(f"    {name}: precision={r['precision']:.3f} recall={r['recall']:.3f} f1={r['f1-score']:.3f}")

    # --- 5. Stratified K-fold cross-validation ---
    print(f"\n=== {args.cv_folds}-Fold Stratified Cross-Validation ===")
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)

    fold_metrics = []
    all_cv_predictions = np.zeros(len(y), dtype=int)
    all_cv_probas = np.zeros((len(y), len(class_names)))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train a fresh classifier for this fold
        config = ClassifierConfig(
            input_dim=X.shape[1],
            num_classes=len(class_names),
        )
        fold_clf = ViralClassifier(config)
        fold_clf.fit(X_train, y_train, class_names=class_names)

        fold_pred = fold_clf.predict(X_val)
        fold_proba = fold_clf.predict_proba(X_val)

        all_cv_predictions[val_idx] = fold_pred
        all_cv_probas[val_idx] = fold_proba

        fold_acc = accuracy_score(y_val, fold_pred)
        fold_f1 = f1_score(y_val, fold_pred, average="weighted")
        fold_prec = precision_score(y_val, fold_pred, average="weighted")
        fold_rec = recall_score(y_val, fold_pred, average="weighted")

        fold_auc = None
        if len(class_names) == 2:
            try:
                fold_auc = roc_auc_score(y_val, fold_proba[:, 1])
            except ValueError:
                pass

        fold_metrics.append({
            "fold": fold + 1,
            "accuracy": fold_acc,
            "f1": fold_f1,
            "precision": fold_prec,
            "recall": fold_rec,
            "auc": fold_auc,
        })

        auc_str = f" auc={fold_auc:.4f}" if fold_auc is not None else ""
        print(f"  Fold {fold + 1}: accuracy={fold_acc:.4f} f1={fold_f1:.4f}{auc_str}")

    # Aggregate CV metrics
    cv_acc = accuracy_score(y, all_cv_predictions)
    cv_f1 = f1_score(y, all_cv_predictions, average="weighted")
    cv_prec = precision_score(y, all_cv_predictions, average="weighted")
    cv_rec = recall_score(y, all_cv_predictions, average="weighted")

    cv_auc = None
    if len(class_names) == 2:
        try:
            cv_auc = roc_auc_score(y, all_cv_probas[:, 1])
        except ValueError:
            pass

    print(f"\n  Overall CV: accuracy={cv_acc:.4f} f1={cv_f1:.4f} precision={cv_prec:.4f} recall={cv_rec:.4f}")
    if cv_auc is not None:
        print(f"  Overall CV AUC: {cv_auc:.4f}")

    # CV confusion matrix
    cv_cm = confusion_matrix(y, all_cv_predictions)
    print(f"\n  CV Confusion matrix:")
    print(header)
    for i, name in enumerate(class_names):
        row = "  " + f"{name:>15}" + "  ".join(f"{cv_cm[i, j]:>10d}" for j in range(len(class_names)))
        print(row)

    cv_report = classification_report(y, all_cv_predictions, target_names=class_names, output_dict=True)
    print(f"\n  CV Per-class metrics:")
    for name in class_names:
        r = cv_report[name]
        print(f"    {name}: precision={r['precision']:.3f} recall={r['recall']:.3f} f1={r['f1-score']:.3f}")

    # --- 6. Confidence distribution ---
    print(f"\n=== Confidence Analysis ===")
    # Use the actual trained model's probabilities (not CV)
    viral_class_idx = 1 if len(class_names) == 2 else class_names.index("viral") if "viral" in class_names else -1
    max_probas = np.max(probas, axis=1)

    print(f"  Confidence distribution (trained model):")
    for cutoff in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        count = (max_probas >= cutoff).sum()
        print(f"    >= {cutoff:.2f}: {count}/{len(max_probas)} ({100 * count / len(max_probas):.1f}%)")

    # Identify low-confidence predictions
    low_confidence_mask = max_probas < 0.7
    n_low = low_confidence_mask.sum()
    print(f"\n  Low confidence (< 0.70): {n_low} sequences ({100 * n_low / len(max_probas):.1f}%)")

    # --- 7. Hard examples ---
    print(f"\n=== Hard Examples (most uncertain predictions) ===")
    uncertainty = 1.0 - max_probas
    hard_indices = np.argsort(uncertainty)[::-1][:20]

    hard_examples = []
    for idx in hard_indices:
        sid = matched_ids[idx]
        true_label = class_names[y[idx]]
        pred_label = class_names[predictions[idx]]
        confidence = float(max_probas[idx])
        correct = true_label == pred_label

        hard_examples.append({
            "sequence_id": sid,
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": round(confidence, 4),
            "correct": correct,
        })

    print(f"  Top 20 most uncertain:")
    for ex in hard_examples:
        mark = "OK" if ex["correct"] else "WRONG"
        print(f"    [{mark}] {ex['sequence_id']}: true={ex['true_label']} "
              f"pred={ex['predicted_label']} conf={ex['confidence']:.3f}")

    # --- 8. Misclassification analysis ---
    misclassified_mask = predictions != y
    n_misclassified = misclassified_mask.sum()
    print(f"\n=== Misclassification Analysis (trained model) ===")
    print(f"  Total misclassified: {n_misclassified}/{len(y)} ({100 * n_misclassified / len(y):.1f}%)")

    if n_misclassified > 0:
        mis_indices = np.where(misclassified_mask)[0]
        misclassified = []
        for idx in mis_indices:
            sid = matched_ids[idx]
            true_label = class_names[y[idx]]
            pred_label = class_names[predictions[idx]]
            confidence = float(max_probas[idx])
            misclassified.append({
                "sequence_id": sid,
                "true_label": true_label,
                "predicted_label": pred_label,
                "confidence": round(confidence, 4),
            })
        misclassified.sort(key=lambda x: x["confidence"], reverse=True)

        # Misclassification by direction
        for i, from_name in enumerate(class_names):
            for j, to_name in enumerate(class_names):
                if i == j:
                    continue
                count = sum(1 for m in misclassified
                            if m["true_label"] == from_name and m["predicted_label"] == to_name)
                if count > 0:
                    print(f"  {from_name} -> {to_name}: {count}")

        # Write full misclassification report
        mis_path = output_dir / "misclassified.tsv"
        pd.DataFrame(misclassified).to_csv(mis_path, sep="\t", index=False)
        print(f"  Full report: {mis_path}")

    # --- 9. Write validation summary ---
    summary = {
        "model_path": str(args.model),
        "n_sequences": len(matched_ids),
        "class_names": class_names,
        "class_counts": n_per_class,
        "training_set": {
            "accuracy": round(float(train_acc), 4),
            "f1": round(float(train_f1), 4),
            "confusion_matrix": cm.tolist(),
            "misclassified": n_misclassified,
        },
        "cross_validation": {
            "n_folds": args.cv_folds,
            "accuracy": round(float(cv_acc), 4),
            "f1": round(float(cv_f1), 4),
            "precision": round(float(cv_prec), 4),
            "recall": round(float(cv_rec), 4),
            "auc": round(float(cv_auc), 4) if cv_auc is not None else None,
            "confusion_matrix": cv_cm.tolist(),
            "per_fold": fold_metrics,
        },
        "confidence": {
            "mean": round(float(max_probas.mean()), 4),
            "median": round(float(np.median(max_probas)), 4),
            "low_confidence_count": int(n_low),
            "low_confidence_fraction": round(float(n_low / len(max_probas)), 4),
        },
        "hard_examples": hard_examples,
    }

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    summary_path = output_dir / "validation_report.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, cls=_NumpyEncoder)
    print(f"\n  Validation report: {summary_path}")

    # Write hard examples TSV
    hard_path = output_dir / "hard_examples.tsv"
    pd.DataFrame(hard_examples).to_csv(hard_path, sep="\t", index=False)
    print(f"  Hard examples: {hard_path}")

    # --- 10. Pass/fail summary ---
    print(f"\n{'=' * 50}")
    print(f"VALIDATION SUMMARY")
    print(f"{'=' * 50}")

    checks = []

    # Check 1: Training accuracy should be high
    if train_acc >= 0.95:
        checks.append(("Training accuracy >= 95%", "PASS", f"{train_acc:.1%}"))
    elif train_acc >= 0.90:
        checks.append(("Training accuracy >= 95%", "WARN", f"{train_acc:.1%}"))
    else:
        checks.append(("Training accuracy >= 95%", "FAIL", f"{train_acc:.1%}"))

    # Check 2: CV accuracy should be reasonable
    if cv_acc >= 0.90:
        checks.append(("CV accuracy >= 90%", "PASS", f"{cv_acc:.1%}"))
    elif cv_acc >= 0.80:
        checks.append(("CV accuracy >= 90%", "WARN", f"{cv_acc:.1%}"))
    else:
        checks.append(("CV accuracy >= 90%", "FAIL", f"{cv_acc:.1%}"))

    # Check 3: CV AUC should be high for binary
    if cv_auc is not None:
        if cv_auc >= 0.95:
            checks.append(("CV AUC >= 0.95", "PASS", f"{cv_auc:.4f}"))
        elif cv_auc >= 0.90:
            checks.append(("CV AUC >= 0.95", "WARN", f"{cv_auc:.4f}"))
        else:
            checks.append(("CV AUC >= 0.95", "FAIL", f"{cv_auc:.4f}"))

    # Check 4: Low confidence should be rare
    low_frac = n_low / len(max_probas)
    if low_frac <= 0.05:
        checks.append(("Low-confidence <= 5%", "PASS", f"{low_frac:.1%}"))
    elif low_frac <= 0.10:
        checks.append(("Low-confidence <= 5%", "WARN", f"{low_frac:.1%}"))
    else:
        checks.append(("Low-confidence <= 5%", "FAIL", f"{low_frac:.1%}"))

    for check_name, status, value in checks:
        print(f"  [{status}] {check_name}: {value}")

    n_fail = sum(1 for _, s, _ in checks if s == "FAIL")
    n_warn = sum(1 for _, s, _ in checks if s == "WARN")

    if n_fail > 0:
        print(f"\n  Result: {n_fail} FAILED checks. Model needs investigation.")
        sys.exit(1)
    elif n_warn > 0:
        print(f"\n  Result: All checks passed ({n_warn} warnings).")
    else:
        print(f"\n  Result: All checks passed.")


if __name__ == "__main__":
    main()
