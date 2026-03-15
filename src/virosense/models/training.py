"""Training loop for classifier heads on Evo2 embeddings."""

from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from virosense.io.results import write_json
from virosense.models.detector import ClassifierConfig, ViralClassifier

# Minimum holdout samples required for calibration (per sub-split)
_MIN_CALIBRATION_SAMPLES = 20


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
    normalize_l2: bool = False,
) -> dict:
    """Train a classification head on frozen embeddings with Platt scaling.

    Splits data into train/calibrate/test (3-way). The classifier is trained
    on the training set, then Platt scaling (sigmoid calibration) is applied
    using the calibration set. Metrics are reported on the held-out test set,
    which is seen by neither training nor calibration.

    Args:
        embeddings: (N, embed_dim) training embeddings.
        labels: (N,) integer labels.
        output_dir: Directory to save model and metrics.
        epochs: Number of training epochs (passed as max_iter to MLPClassifier).
        lr: Learning rate.
        val_split: Fraction held out for calibration + testing (split 50/50).
        task: Classification task name.
        class_names: Human-readable class names.
        layer: Evo2 layer used for embeddings.
        model: Evo2 model name.

    Returns:
        Dict with training metrics (accuracy, f1, auc, brier_score, ece, etc.).
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

    # 3-way split: train / calibrate / test
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        embeddings, labels, test_size=val_split, random_state=42, stratify=labels
    )

    can_calibrate = len(y_holdout) >= 2 * _MIN_CALIBRATION_SAMPLES
    if can_calibrate:
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_holdout, y_holdout, test_size=0.5, random_state=42, stratify=y_holdout
        )
        logger.info(
            f"Train: {len(y_train)}, Calibrate: {len(y_cal)}, Test: {len(y_test)}"
        )
    else:
        X_test, y_test = X_holdout, y_holdout
        logger.info(f"Train: {len(y_train)}, Val: {len(y_test)}")
        logger.warning(
            f"Too few holdout samples ({len(y_holdout)}) for calibration, "
            f"skipping Platt scaling"
        )

    # Build and train classifier
    config = ClassifierConfig(
        input_dim=embeddings.shape[1],
        num_classes=n_classes,
        normalize_l2=normalize_l2,
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

    # Evaluate uncalibrated model on test set
    metrics = evaluate_classifier(classifier, X_test, y_test, n_classes)

    # Calibration metrics (uncalibrated)
    probas_uncal = classifier.predict_proba(X_test)
    if n_classes == 2:
        metrics["brier_score_uncalibrated"] = float(
            brier_score_loss(y_test, probas_uncal[:, 1])
        )
        metrics["ece_uncalibrated"] = float(
            _expected_calibration_error(y_test, probas_uncal[:, 1])
        )
    else:
        metrics["log_loss_uncalibrated"] = float(log_loss(y_test, probas_uncal))

    # Probability calibration
    # 2-class: Platt scaling (sigmoid); 3+-class: isotonic regression
    if can_calibrate:
        cal_method = "sigmoid" if n_classes == 2 else "isotonic"
        calibrated_model = CalibratedClassifierCV(
            estimator=FrozenEstimator(classifier.model), method=cal_method
        )
        # Normalize calibration data to match training preprocessing —
        # CalibratedClassifierCV passes data directly to the FrozenEstimator,
        # bypassing the classifier's _normalize() wrapper.
        X_cal_input = classifier._normalize(X_cal)
        calibrated_model.fit(X_cal_input, y_cal)
        classifier.model = calibrated_model
        classifier.metadata["calibrated"] = True
        classifier.metadata["calibration_method"] = cal_method

        # Calibrated metrics on test set
        if n_classes == 2:
            probas_cal = classifier.predict_proba(X_test)
            metrics["brier_score"] = float(
                brier_score_loss(y_test, probas_cal[:, 1])
            )
            metrics["ece"] = float(
                _expected_calibration_error(y_test, probas_cal[:, 1])
            )
            logger.info(
                f"Platt scaling: Brier {metrics['brier_score_uncalibrated']:.4f} "
                f"-> {metrics['brier_score']:.4f}, "
                f"ECE {metrics['ece_uncalibrated']:.4f} "
                f"-> {metrics['ece']:.4f}"
            )
        else:
            probas_cal = classifier.predict_proba(X_test)
            metrics["log_loss"] = float(log_loss(y_test, probas_cal))
            logger.info(
                f"Isotonic calibration: log_loss "
                f"{metrics['log_loss_uncalibrated']:.4f} "
                f"-> {metrics['log_loss']:.4f}"
            )
    else:
        classifier.metadata["calibrated"] = False
        if n_classes == 2:
            metrics["brier_score"] = metrics["brier_score_uncalibrated"]
            metrics["ece"] = metrics["ece_uncalibrated"]
        else:
            metrics["log_loss"] = metrics["log_loss_uncalibrated"]

    metrics["task"] = task
    metrics["n_train"] = len(y_train)
    metrics["n_cal"] = len(y_cal) if can_calibrate else 0
    metrics["n_test"] = len(y_test)
    metrics["n_classes"] = n_classes
    metrics["class_names"] = class_names
    metrics["epochs"] = epochs
    metrics["lr"] = lr

    logger.info(
        f"Test metrics: accuracy={metrics['accuracy']:.3f}, "
        f"f1={metrics['f1']:.3f}, auc={metrics.get('auc', 'N/A')}"
    )

    # Store test arrays for report generation (not serialized to JSON)
    metrics["_y_test"] = y_test
    metrics["_probas_test"] = classifier.predict_proba(X_test)

    # Save model and metrics (filter out non-serializable arrays)
    model_path = output_dir / "classifier.joblib"
    classifier.save(model_path)
    json_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
    write_json(json_metrics, output_dir, "metrics.json")

    return metrics


def _expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Partitions predictions into equal-width probability bins and measures
    the weighted average gap between predicted confidence and actual accuracy.

    Args:
        y_true: (N,) binary labels (0 or 1).
        y_prob: (N,) predicted probabilities for the positive class.
        n_bins: Number of equal-width bins.

    Returns:
        ECE value (0 = perfectly calibrated, 1 = maximally miscalibrated).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        # Include right boundary for the last bin
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue
        bin_accuracy = y_true[mask].mean()
        bin_confidence = y_prob[mask].mean()
        ece += n_in_bin * abs(bin_accuracy - bin_confidence)
    return ece / len(y_true)


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

    # Per-class metrics
    class_names = classifier.metadata.get("class_names", [])
    per_class_f1 = f1_score(labels, predictions, average=None)
    per_class_prec = precision_score(labels, predictions, average=None)
    per_class_rec = recall_score(labels, predictions, average=None)

    if class_names and len(class_names) == len(per_class_f1):
        metrics["per_class"] = {
            name: {
                "f1": round(float(per_class_f1[i]), 4),
                "precision": round(float(per_class_prec[i]), 4),
                "recall": round(float(per_class_rec[i]), 4),
            }
            for i, name in enumerate(class_names)
        }

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    metrics["confusion_matrix"] = cm.tolist()
    if class_names:
        metrics["confusion_matrix_labels"] = class_names

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
