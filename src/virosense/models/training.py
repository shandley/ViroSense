"""Training loop for classifier heads on Evo2 embeddings."""

from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from virosense.io.results import write_json
from virosense.models.detector import ClassifierConfig, ViralClassifier


def train_classifier(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    epochs: int = 50,
    lr: float = 1e-3,
    val_split: float = 0.2,
    task: str = "viral_vs_cellular",
    class_names: list[str] | None = None,
    layer: str | None = None,
    model: str | None = None,
) -> dict:
    """Train a classification head on frozen embeddings.

    Args:
        embeddings: (N, embed_dim) training embeddings.
        labels: (N,) integer labels.
        output_dir: Directory to save model and metrics.
        epochs: Number of training epochs (passed as max_iter to MLPClassifier).
        lr: Learning rate.
        val_split: Fraction for validation set.
        task: Classification task name.
        class_names: Human-readable class names.
        layer: Evo2 layer used for embeddings.
        model: Evo2 model name.

    Returns:
        Dict with training metrics (accuracy, f1, auc, etc.).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_classes = len(np.unique(labels))
    if class_names is None:
        class_names = [str(c) for c in sorted(np.unique(labels))]

    logger.info(
        f"Training {task} classifier: {len(labels)} samples, "
        f"{n_classes} classes, val_split={val_split}"
    )

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=val_split, random_state=42, stratify=labels
    )

    logger.info(f"Train: {len(y_train)}, Val: {len(y_val)}")

    # Build and train classifier
    config = ClassifierConfig(
        input_dim=embeddings.shape[1],
        num_classes=n_classes,
    )
    classifier = ViralClassifier(config)
    classifier.model.max_iter = epochs
    classifier.model.learning_rate_init = lr
    classifier.fit(
        X_train, y_train,
        class_names=class_names,
        layer=layer,
        model=model,
    )

    # Evaluate on validation set
    metrics = evaluate_classifier(classifier, X_val, y_val, n_classes)
    metrics["task"] = task
    metrics["n_train"] = len(y_train)
    metrics["n_val"] = len(y_val)
    metrics["n_classes"] = n_classes
    metrics["class_names"] = class_names
    metrics["epochs"] = epochs
    metrics["lr"] = lr

    logger.info(
        f"Val metrics: accuracy={metrics['accuracy']:.3f}, "
        f"f1={metrics['f1']:.3f}, auc={metrics.get('auc', 'N/A')}"
    )

    # Save model and metrics
    model_path = output_dir / "classifier.joblib"
    classifier.save(model_path)
    write_json(metrics, output_dir, "metrics.json")

    return metrics


def evaluate_classifier(
    classifier: ViralClassifier,
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
) -> dict:
    """Evaluate a trained classifier and return metrics dict."""
    predictions = classifier.predict(embeddings)
    probas = classifier.predict_proba(embeddings)

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, average="weighted")),
        "precision": float(precision_score(labels, predictions, average="weighted")),
        "recall": float(recall_score(labels, predictions, average="weighted")),
    }

    # AUC only for binary classification
    if n_classes == 2:
        try:
            metrics["auc"] = float(roc_auc_score(labels, probas[:, 1]))
        except ValueError:
            metrics["auc"] = None
    else:
        try:
            metrics["auc"] = float(
                roc_auc_score(labels, probas, multi_class="ovr", average="weighted")
            )
        except ValueError:
            metrics["auc"] = None

    return metrics
