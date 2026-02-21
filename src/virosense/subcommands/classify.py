"""Train or apply a discriminative viral classifier on Evo2 embeddings."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ClassificationResult:
    """Result for a single sequence classification."""

    sequence_id: str
    predicted_class: str
    confidence: float


def run_classify(
    input_file: str,
    labels_file: str,
    output_dir: str,
    backend: str = "nim",
    model: str = "evo2_7b",
    task: str = "viral_vs_cellular",
    epochs: int = 50,
    lr: float = 1e-3,
    val_split: float = 0.2,
    predict_file: str | None = None,
    classifier_model_path: str | None = None,
    layer: str = "blocks.28.mlp.l3",
    cache_dir: str | None = None,
    nim_url: str | None = None,
) -> None:
    """Run discriminative classifier pipeline.

    Training mode:
    1. Read sequences and labels
    2. Extract Evo2 embeddings
    3. Train classification head (frozen embeddings)
    4. Evaluate on validation set
    5. Save model and metrics

    Prediction mode:
    1. Load pre-trained classifier
    2. Extract embeddings for new sequences
    3. Predict and write results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Classification task: {task}")
    logger.info(f"Backend: {backend}, Model: {model}")

    if predict_file and classifier_model_path:
        _run_prediction(
            predict_file=predict_file,
            classifier_model_path=classifier_model_path,
            output_path=output_path,
            backend=backend,
            model=model,
            layer=layer,
            cache_dir=cache_dir,
            nim_url=nim_url,
        )
    else:
        _run_training(
            input_file=input_file,
            labels_file=labels_file,
            output_path=output_path,
            backend=backend,
            model=model,
            task=task,
            epochs=epochs,
            lr=lr,
            val_split=val_split,
            layer=layer,
            cache_dir=cache_dir,
            nim_url=nim_url,
        )


def _run_training(
    input_file: str,
    labels_file: str,
    output_path: Path,
    backend: str,
    model: str,
    task: str,
    epochs: int,
    lr: float,
    val_split: float,
    layer: str,
    cache_dir: str | None,
    nim_url: str | None = None,
) -> None:
    """Train a classifier on labeled sequences."""
    from virosense.backends.base import get_backend
    from virosense.features.evo2_embeddings import extract_embeddings
    from virosense.io.fasta import read_fasta
    from virosense.models.training import train_classifier

    logger.info(f"Training mode: {input_file}, Labels: {labels_file}")
    logger.info(f"Epochs: {epochs}, LR: {lr}, Val split: {val_split}")

    # 1. Read sequences and labels
    sequences = read_fasta(input_file)
    labels_df = pd.read_csv(labels_file, sep="\t", header=0)
    labels_df.columns = labels_df.columns.str.strip()

    if len(labels_df.columns) < 2:
        raise ValueError(
            "Labels file must have at least 2 tab-separated columns: "
            "sequence_id and label"
        )

    id_col, label_col = labels_df.columns[0], labels_df.columns[1]

    # Build label encoding: string labels -> integer labels
    unique_labels = sorted(labels_df[label_col].unique())
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    class_names = [str(c) for c in unique_labels]

    label_map = {
        row[id_col]: label_to_int[row[label_col]]
        for _, row in labels_df.iterrows()
    }

    # Match sequences to labels
    matched_seqs = {}
    matched_labels = []
    for seq_id in sequences:
        if seq_id in label_map:
            matched_seqs[seq_id] = sequences[seq_id]
            matched_labels.append(label_map[seq_id])

    if not matched_seqs:
        raise ValueError(
            "No sequences matched between FASTA and labels file. "
            "Check that sequence IDs match."
        )

    n_skipped = len(sequences) - len(matched_seqs)
    if n_skipped:
        logger.warning(f"Skipped {n_skipped} sequences without labels")

    labels_array = np.array(matched_labels)
    logger.info(
        f"Matched {len(matched_seqs)} labeled sequences: "
        + ", ".join(
            f"{name}: {(labels_array == i).sum()}"
            for i, name in enumerate(class_names)
        )
    )

    # 2. Extract embeddings
    evo2_backend = get_backend(backend, model=model, nim_url=nim_url)
    if not evo2_backend.is_available():
        raise RuntimeError(
            f"Backend {backend!r} is not available. "
            "Check your configuration (e.g., NVIDIA_API_KEY for nim)."
        )

    cache_path = Path(cache_dir) if cache_dir else None
    result = extract_embeddings(
        sequences=matched_seqs,
        backend=evo2_backend,
        layer=layer,
        model=model,
        cache_dir=cache_path,
    )

    # Ensure labels align with embedding order
    ordered_labels = np.array([label_map[sid] for sid in result.sequence_ids])

    # 3. Train classifier
    metrics = train_classifier(
        embeddings=result.embeddings,
        labels=ordered_labels,
        output_dir=output_path,
        epochs=epochs,
        lr=lr,
        val_split=val_split,
        task=task,
        class_names=class_names,
        layer=layer,
        model=model,
    )

    logger.info(
        f"Classifier trained: accuracy={metrics['accuracy']:.3f}, "
        f"f1={metrics['f1']:.3f}, auc={metrics.get('auc', 'N/A')}"
    )


def _run_prediction(
    predict_file: str,
    classifier_model_path: str,
    output_path: Path,
    backend: str,
    model: str,
    layer: str,
    cache_dir: str | None,
    nim_url: str | None = None,
) -> None:
    """Load a trained classifier and predict on new sequences."""
    from virosense.backends.base import get_backend
    from virosense.features.evo2_embeddings import extract_embeddings
    from virosense.io.fasta import read_fasta
    from virosense.io.results import write_json, write_tsv
    from virosense.models.detector import ViralClassifier

    logger.info(f"Prediction mode: {predict_file}")
    logger.info(f"Classifier: {classifier_model_path}")

    # 1. Load classifier
    classifier = ViralClassifier.load(classifier_model_path)
    class_names = classifier.metadata.get("class_names", [])

    # 2. Read and embed new sequences
    sequences = read_fasta(predict_file)
    if not sequences:
        logger.warning("No sequences found in prediction file.")
        return

    evo2_backend = get_backend(backend, model=model, nim_url=nim_url)
    if not evo2_backend.is_available():
        raise RuntimeError(
            f"Backend {backend!r} is not available. "
            "Check your configuration (e.g., NVIDIA_API_KEY for nim)."
        )

    cache_path = Path(cache_dir) if cache_dir else None
    result = extract_embeddings(
        sequences=sequences,
        backend=evo2_backend,
        layer=layer,
        model=model,
        cache_dir=cache_path,
    )

    # 3. Predict (single forward pass — argmax replaces separate predict call)
    probas = classifier.predict_proba(result.embeddings)

    results = []
    for i, seq_id in enumerate(result.sequence_ids):
        pred_idx = int(np.argmax(probas[i]))
        pred_class = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        confidence = float(probas[i, pred_idx])

        results.append(
            ClassificationResult(
                sequence_id=seq_id,
                predicted_class=pred_class,
                confidence=round(confidence, 4),
            )
        )

    # 4. Write results
    write_tsv(results, output_path, "predictions.tsv")

    # Summary
    class_counts = {}
    for r in results:
        class_counts[r.predicted_class] = class_counts.get(r.predicted_class, 0) + 1
    summary = {
        "n_sequences": len(results),
        "class_distribution": class_counts,
        "classifier": classifier_model_path,
        "task": classifier.metadata.get("task", "unknown"),
    }
    write_json(summary, output_path, "prediction_summary.json")

    logger.info(
        f"Predicted {len(results)} sequences: "
        + ", ".join(f"{k}: {v}" for k, v in class_counts.items())
    )
