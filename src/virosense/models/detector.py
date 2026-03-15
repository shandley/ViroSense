"""Viral vs cellular classifier head on frozen Evo2 embeddings."""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger


@dataclass
class ClassifierConfig:
    """Configuration for the classification head."""

    input_dim: int = 8192  # Evo2 40B embedding dim (4096 for 7B)
    hidden_dims: list[int] = field(default_factory=lambda: [512, 128])
    num_classes: int = 2
    dropout: float = 0.1
    normalize_l2: bool = False  # L2-normalize embeddings before classification


@dataclass
class DetectionResult:
    """Result for a single contig classification."""

    contig_id: str
    contig_length: int
    viral_score: float  # 0.0-1.0 (sum of all viral class probabilities)
    classification: str  # "viral"/"phage"/"rna_virus"/"chromosome"/"plasmid"/"cellular"/"ambiguous"
    chromosome_score: float | None = None
    plasmid_score: float | None = None
    phage_score: float | None = None
    rna_virus_score: float | None = None


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

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2-normalize embeddings if configured.

        Eliminates length-dependent magnitude effects that cause the
        classifier to fail on longer sequences (especially RNA viruses).
        """
        if self.config.normalize_l2:
            from sklearn.preprocessing import normalize

            return normalize(embeddings, norm="l2")
        return embeddings

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
        embeddings = self._normalize(embeddings)
        self.model.fit(embeddings, labels)
        self._is_fitted = True
        self.metadata = {
            "n_train": len(labels),
            "n_classes": len(np.unique(labels)),
            "input_dim": embeddings.shape[1],
            "class_names": class_names or [str(c) for c in sorted(np.unique(labels))],
            "layer": layer,
            "model": model,
            "normalize_l2": self.config.normalize_l2,
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
        return self.model.predict(self._normalize(embeddings))

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict class probabilities. Shape: (N, n_classes)."""
        self._check_fitted()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*matmul.*", category=RuntimeWarning
            )
            return self.model.predict_proba(self._normalize(embeddings))

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
        # Restore normalize_l2 from metadata for backward compatibility
        # (older models won't have this in config)
        if obj.metadata.get("normalize_l2") and not obj.config.normalize_l2:
            obj.config.normalize_l2 = True
        obj._is_fitted = True
        logger.info(
            f"Loaded classifier from {path} "
            f"(trained on {obj.metadata.get('n_train', '?')} samples, "
            f"normalize_l2={obj.config.normalize_l2})"
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


# Known non-viral class names. Any class not in this set is considered viral.
_NON_VIRAL_CLASSES = {"cellular", "chromosome", "plasmid"}


def _get_viral_indices(class_names: list[str]) -> list[int]:
    """Identify which class indices represent viral sequences.

    Any class whose name is NOT in _NON_VIRAL_CLASSES is considered viral.
    This handles 2-class (viral), 3-class (viral), 4-class (phage, rna_virus),
    and future N-class models.

    Args:
        class_names: Ordered class names matching probability indices.

    Returns:
        List of indices into the probability vector that are viral classes.
    """
    return [i for i, name in enumerate(class_names) if name not in _NON_VIRAL_CLASSES]


def _compute_viral_score(probas_row: np.ndarray, class_names: list[str]) -> float:
    """Compute aggregate viral score from probability vector.

    For models with a single viral class, this is just P(viral).
    For models with multiple viral classes (e.g., phage + rna_virus),
    this is the sum of all viral class probabilities.
    """
    viral_indices = _get_viral_indices(class_names)
    if not viral_indices:
        # Fallback: last class is viral (legacy behavior)
        return float(probas_row[-1])
    return float(sum(probas_row[i] for i in viral_indices))


def _classify_from_probas(
    probas_row: np.ndarray,
    threshold: float,
    class_names: list[str],
) -> str:
    """Determine classification label from a single probability vector.

    Supports arbitrary class counts with one or more viral classes.
    Viral score is the sum of all viral class probabilities.

    Args:
        probas_row: (n_classes,) probability vector for one sequence.
        threshold: Score threshold for viral classification.
        class_names: Ordered class names matching probability indices.

    Returns:
        Classification label string.
    """
    viral_score = _compute_viral_score(probas_row, class_names)
    viral_indices = _get_viral_indices(class_names)
    nonviral_indices = [i for i in range(len(probas_row)) if i not in viral_indices]

    if viral_score >= threshold:
        # If multiple viral classes, report the dominant one
        if len(viral_indices) > 1:
            best_viral = max(viral_indices, key=lambda i: probas_row[i])
            return class_names[best_viral]
        return "viral"
    elif viral_score <= (1.0 - threshold):
        if nonviral_indices and class_names:
            best_nonviral = max(nonviral_indices, key=lambda i: probas_row[i])
            return class_names[best_nonviral]
        return "cellular"
    return "ambiguous"


def classify_contigs(
    embeddings: np.ndarray,
    sequence_ids: list[str],
    sequence_lengths: list[int],
    classifier: ViralClassifier,
    threshold: float = 0.5,
) -> list[DetectionResult]:
    """Classify contigs as viral or cellular using a trained classifier.

    Supports both 2-class (viral/cellular) and 3-class
    (chromosome/plasmid/viral) models. The loaded model's metadata
    determines behavior automatically.

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
    class_names = classifier.metadata.get("class_names", [])

    # Compute viral score as sum of all viral class probabilities
    viral_indices = _get_viral_indices(class_names)
    viral_scores = np.sum(probas[:, viral_indices], axis=1) if viral_indices else probas[:, -1]

    # Map class names to DetectionResult field names
    _score_field_map = {
        "chromosome": "chromosome_score",
        "plasmid": "plasmid_score",
        "phage": "phage_score",
        "rna_virus": "rna_virus_score",
    }

    results = []
    for i, (seq_id, length) in enumerate(zip(sequence_ids, sequence_lengths)):
        score = float(viral_scores[i])
        classification = _classify_from_probas(probas[i], threshold, class_names)

        kwargs: dict = {
            "contig_id": seq_id,
            "contig_length": length,
            "viral_score": round(score, 4),
            "classification": classification,
        }

        # Populate per-class scores for multi-class models
        if len(class_names) > 2:
            for j, name in enumerate(class_names):
                field = _score_field_map.get(name)
                if field:
                    kwargs[field] = round(float(probas[i, j]), 4)

        results.append(DetectionResult(**kwargs))

    # Log classification counts
    from collections import Counter

    counts = Counter(r.classification for r in results)
    parts = [f"{counts.get(c, 0)} {c}" for c in sorted(counts)]
    logger.info(
        f"Classification: {', '.join(parts)} (threshold={threshold})"
    )
    return results
