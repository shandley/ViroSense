"""Fast k-mer based sequence classification.

Uses trinucleotide and dinucleotide frequencies for rapid pre-screening.
Achieves ~93% accuracy for viral detection at 1500× the speed of Evo2.

Used by `virosense detect --fast` and as a pre-filter for the two-tier
pipeline (k-mer screening → Evo2 characterization on borderline cases).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from loguru import logger


def compute_kmer_features(sequence: str) -> np.ndarray:
    """Compute trinucleotide + dinucleotide + metadata features for a sequence.

    Returns a 1D feature vector (84 features):
    - 64 trinucleotide frequencies
    - 16 dinucleotide frequencies
    - GC content
    - In-frame trinucleotide entropy
    - Out-of-frame trinucleotide entropy
    - Frame entropy ratio
    """
    seq = sequence.upper()
    seq_len = len(seq)

    # Trinucleotide frequencies (64)
    tris = Counter(seq[i:i + 3] for i in range(seq_len - 2))
    total_3 = sum(tris.values()) or 1
    tri_features = []
    for t1 in "ACGT":
        for t2 in "ACGT":
            for t3 in "ACGT":
                tri_features.append(tris.get(f"{t1}{t2}{t3}", 0) / total_3)

    # Dinucleotide frequencies (16)
    dis = Counter(seq[i:i + 2] for i in range(seq_len - 1))
    total_2 = sum(dis.values()) or 1
    di_features = []
    for d1 in "ACGT":
        for d2 in "ACGT":
            di_features.append(dis.get(f"{d1}{d2}", 0) / total_2)

    # GC content
    gc = (seq.count("G") + seq.count("C")) / max(seq_len, 1)

    # Frame entropy (coding periodicity proxy)
    in_frame = Counter(seq[i:i + 3] for i in range(0, seq_len - 2, 3))
    out_frame = Counter(seq[i:i + 3] for i in range(1, seq_len - 2, 3))

    def _entropy(counter: Counter) -> float:
        total = sum(counter.values())
        if total == 0:
            return 0.0
        return -sum(
            (v / total) * np.log2(v / total)
            for v in counter.values()
            if v > 0
        )

    in_ent = _entropy(in_frame)
    out_ent = _entropy(out_frame)
    frame_ratio = in_ent / max(out_ent, 0.001)

    return np.array(
        tri_features + di_features + [gc, in_ent, out_ent, frame_ratio],
        dtype=np.float32,
    )


def compute_kmer_features_batch(
    sequences: dict[str, str],
) -> tuple[list[str], np.ndarray]:
    """Compute k-mer features for a batch of sequences.

    Returns:
        Tuple of (sequence_ids, feature_matrix) where feature_matrix
        is (N, 84) float32.
    """
    ids = []
    features = []
    for seq_id, seq in sequences.items():
        ids.append(seq_id)
        features.append(compute_kmer_features(seq))
    return ids, np.array(features)


def train_kmer_classifier(
    sequences: dict[str, str],
    labels: dict[str, int],
    output_path: Path | None = None,
) -> "KmerClassifier":
    """Train a k-mer classifier from labeled sequences.

    Args:
        sequences: Dict of sequence_id → DNA sequence.
        labels: Dict of sequence_id → integer label (0=cellular, 1=viral).
        output_path: Optional path to save the trained model.

    Returns:
        Trained KmerClassifier.
    """
    from sklearn.ensemble import RandomForestClassifier

    matched_ids = [sid for sid in sequences if sid in labels]
    ids, X = compute_kmer_features_batch(
        {sid: sequences[sid] for sid in matched_ids}
    )
    y = np.array([labels[sid] for sid in matched_ids])

    logger.info(f"Training k-mer classifier on {len(y)} sequences")

    clf = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    clf.fit(X, y)

    model = KmerClassifier(clf)

    if output_path:
        model.save(output_path)

    return model


class KmerClassifier:
    """Wrapper around a trained k-mer random forest classifier."""

    def __init__(self, model):
        self.model = model

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(features)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(features)

    def save(self, path: Path) -> None:
        """Save model to disk."""
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Saved k-mer classifier to {path}")

    @classmethod
    def load(cls, path: Path) -> "KmerClassifier":
        """Load model from disk."""
        import joblib
        model = joblib.load(path)
        return cls(model)
