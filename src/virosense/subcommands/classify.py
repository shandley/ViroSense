"""Train or apply a discriminative viral classifier on Evo2 embeddings."""

from pathlib import Path

from loguru import logger


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
        logger.info(f"Prediction mode: {predict_file}")
    else:
        logger.info(f"Training mode: {input_file}, Labels: {labels_file}")
        logger.info(f"Epochs: {epochs}, LR: {lr}, Val split: {val_split}")

    raise NotImplementedError(
        "classify pipeline not yet implemented. See Phase 8 in the plan."
    )
