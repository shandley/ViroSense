"""Viral vs cellular classifier head on frozen Evo2 embeddings."""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger


@dataclass
class ClassifierConfig:
    """Configuration for the classification head."""

    input_dim: int = 4096  # Evo2 embedding dim
    hidden_dims: list[int] = field(default_factory=lambda: [512, 128])
    num_classes: int = 2
    dropout: float = 0.1


@dataclass
class DetectionResult:
    """Result for a single contig classification."""

    contig_id: str
    contig_length: int
    viral_score: float  # 0.0-1.0
    classification: str  # "viral", "cellular", "ambiguous"


class ViralClassifier:
    """Sklearn classifier on frozen Evo2 embeddings.

    Wraps an MLPClassifier with train/predict/save/load. Stores training
    metadata (layer, model, class names) so predictions can be validated
    against the same embedding configuration.
    """

    def __init__(self, config: ClassifierConfig | None = None):
        from sklearn.neural_network import MLPClassifier

        self.config = config or ClassifierConfig()
        self.model = MLPClassifier(
            hidden_layer_sizes=tuple(self.config.hidden_dims),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        self.metadata: dict = {}
        self._is_fitted = False

    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        class_names: list[str] | None = None,
        layer: str | None = None,
        model: str | None = None,
    ) -> "ViralClassifier":
        """Train the classifier on embeddings + labels.

        Args:
            embeddings: (N, embed_dim) training embeddings.
            labels: (N,) integer labels.
            class_names: Human-readable class names (e.g. ["cellular", "viral"]).
            layer: Evo2 layer used for embeddings (stored in metadata).
            model: Evo2 model used (stored in metadata).

        Returns:
            self, for chaining.
        """
        self.model.fit(embeddings, labels)
        self._is_fitted = True
        self.metadata = {
            "n_train": len(labels),
            "n_classes": len(np.unique(labels)),
            "input_dim": embeddings.shape[1],
            "class_names": class_names or [str(c) for c in sorted(np.unique(labels))],
            "layer": layer,
            "model": model,
        }
        logger.info(
            f"Trained classifier: {len(labels)} samples, "
            f"{self.metadata['n_classes']} classes, "
            f"dim={embeddings.shape[1]}"
        )
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        self._check_fitted()
        return self.model.predict(embeddings)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict class probabilities. Shape: (N, n_classes)."""
        self._check_fitted()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*matmul.*", category=RuntimeWarning
            )
            return self.model.predict_proba(embeddings)

    def save(self, path: str | Path) -> Path:
        """Save classifier to disk via joblib.

        Saves both the sklearn model and metadata as a single dict.
        """
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "config": self.config, "metadata": self.metadata},
            path,
        )
        logger.info(f"Saved classifier to {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ViralClassifier":
        """Load a saved classifier from disk."""
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Classifier model not found: {path}")

        data = joblib.load(path)
        obj = cls(config=data["config"])
        obj.model = data["model"]
        obj.metadata = data["metadata"]
        obj._is_fitted = True
        logger.info(
            f"Loaded classifier from {path} "
            f"(trained on {obj.metadata.get('n_train', '?')} samples)"
        )
        return obj

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "Classifier is not trained. Call fit() first or load a "
                "pre-trained model with ViralClassifier.load()."
            )


def get_default_model_path() -> Path:
    """Get the path to the default reference classifier model."""
    from virosense.utils.constants import get_data_dir

    return get_data_dir() / "models" / "reference_classifier.joblib"


def classify_contigs(
    embeddings: np.ndarray,
    sequence_ids: list[str],
    sequence_lengths: list[int],
    classifier: ViralClassifier,
    threshold: float = 0.5,
) -> list[DetectionResult]:
    """Classify contigs as viral or cellular using a trained classifier.

    Args:
        embeddings: (N, embed_dim) embedding matrix.
        sequence_ids: List of contig identifiers.
        sequence_lengths: List of contig lengths in bp.
        classifier: Trained ViralClassifier instance.
        threshold: Score threshold for viral classification.

    Returns:
        List of DetectionResult for each contig.
    """
    probas = classifier.predict_proba(embeddings)

    # Viral class is assumed to be the last class (index 1 for binary)
    viral_idx = probas.shape[1] - 1
    viral_scores = probas[:, viral_idx]

    results = []
    for i, (seq_id, length) in enumerate(zip(sequence_ids, sequence_lengths)):
        score = float(viral_scores[i])
        if score >= threshold:
            classification = "viral"
        elif score <= (1.0 - threshold):
            classification = "cellular"
        else:
            classification = "ambiguous"

        results.append(
            DetectionResult(
                contig_id=seq_id,
                contig_length=length,
                viral_score=round(score, 4),
                classification=classification,
            )
        )

    n_viral = sum(1 for r in results if r.classification == "viral")
    n_cellular = sum(1 for r in results if r.classification == "cellular")
    n_ambiguous = sum(1 for r in results if r.classification == "ambiguous")
    logger.info(
        f"Classification: {n_viral} viral, {n_cellular} cellular, "
        f"{n_ambiguous} ambiguous (threshold={threshold})"
    )
    return results
