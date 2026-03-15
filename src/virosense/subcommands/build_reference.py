"""Build a reference classifier for viral detection."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def run_build_reference(
    input_file: str,
    labels_file: str,
    output_dir: str,
    backend: str = "nim",
    model: str = "evo2_7b",
    layer: str = "blocks.28.mlp.l3",
    epochs: int = 200,
    lr: float = 1e-3,
    val_split: float = 0.2,
    install: bool = False,
    batch_size: int = 16,
    cache_dir: str | None = None,
    nim_url: str | None = None,
    max_concurrent: int | None = None,
    normalize_l2: bool = False,
) -> None:
    """Build a reference viral classifier from labeled sequences.

    1. Read labeled sequences and their labels
    2. Extract Evo2 embeddings via selected backend
    3. Train classifier on embeddings + labels
    4. Save model and metrics
    5. Optionally install to default model location
    """
    from virosense.backends.base import get_backend
    from virosense.features.evo2_embeddings import extract_embeddings
    from virosense.io.fasta import read_fasta
    from virosense.models.detector import get_default_model_path
    from virosense.models.training import train_classifier

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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
    raw_label_map = dict(zip(labels_df[id_col], labels_df[label_col]))

    # Build label encoding: supports both integer and string labels
    unique_raw_labels = sorted(labels_df[label_col].unique())
    try:
        # Try integer labels first (backward compatible: 0=cellular, 1=viral)
        _ = [int(l) for l in unique_raw_labels]
        label_to_int = {l: int(l) for l in unique_raw_labels}
        class_names = None  # let train_classifier infer from sorted ints
    except (ValueError, TypeError):
        # String labels (e.g., "chromosome", "plasmid", "viral")
        label_to_int = {label: i for i, label in enumerate(unique_raw_labels)}
        class_names = [str(c) for c in unique_raw_labels]

    # Match sequences to labels
    matched_seqs = {}
    matched_labels = []
    for seq_id in sequences:
        if seq_id in raw_label_map:
            matched_seqs[seq_id] = sequences[seq_id]
            matched_labels.append(label_to_int[raw_label_map[seq_id]])

    if not matched_seqs:
        raise ValueError(
            "No sequences matched between FASTA and labels file. "
            "Check that sequence IDs match."
        )

    n_skipped = len(sequences) - len(matched_seqs)
    if n_skipped:
        logger.warning(f"Skipped {n_skipped} sequences without labels")

    labels_array = np.array(matched_labels)
    unique_labels = np.unique(labels_array)
    label_names = class_names or [str(c) for c in unique_labels]
    logger.info(
        f"Matched {len(matched_seqs)} labeled sequences: "
        + ", ".join(
            f"{label_names[i]}: {(labels_array == c).sum()}"
            for i, c in enumerate(unique_labels)
        )
    )

    # 2. Extract embeddings
    evo2_backend = get_backend(
        backend, model=model, nim_url=nim_url, max_concurrent=max_concurrent,
    )
    model = evo2_backend.model  # Use backend's (possibly auto-corrected) model name
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
        batch_size=batch_size,
        cache_dir=cache_path,
    )

    # Ensure labels align with embedding order
    ordered_labels = np.array(
        [label_to_int[raw_label_map[sid]] for sid in result.sequence_ids]
    )

    # 3. Train classifier
    n_unique = len(unique_labels)
    if class_names is None and n_unique == 2:
        class_names = ["cellular", "viral"]
    task = (
        "viral_vs_cellular" if n_unique == 2
        else f"classification_{n_unique}class"
    )
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
        normalize_l2=normalize_l2,
    )

    logger.info(
        f"Reference model trained: accuracy={metrics['accuracy']:.3f}, "
        f"f1={metrics['f1']:.3f}, auc={metrics.get('auc', 'N/A')}"
    )

    # 4. Generate training report
    from virosense.io.report import generate_training_report

    generate_training_report(
        metrics, output_path,
        y_test=metrics.get("_y_test"),
        probas_test=metrics.get("_probas_test"),
    )

    # 5. Optionally install to default location
    if install:
        model_src = output_path / "classifier.joblib"
        model_dst = get_default_model_path()
        model_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_src, model_dst)
        logger.info(f"Installed reference model to {model_dst}")
