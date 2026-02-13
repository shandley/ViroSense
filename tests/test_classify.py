"""Tests for classify module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from virosense.backends.base import EmbeddingRequest, EmbeddingResult
from virosense.subcommands.classify import ClassificationResult


def _make_mock_backend(embed_dim=64):
    """Create a mock backend that returns random embeddings."""
    backend = MagicMock()
    backend.is_available.return_value = True

    def mock_extract(request: EmbeddingRequest) -> EmbeddingResult:
        n = len(request.sequences)
        return EmbeddingResult(
            sequence_ids=list(request.sequences.keys()),
            embeddings=np.random.randn(n, embed_dim).astype(np.float32),
            layer=request.layer,
            model=request.model,
        )

    backend.extract_embeddings.side_effect = mock_extract
    return backend


def _write_fasta(path, seqs):
    """Write sequences to a FASTA file."""
    with open(path, "w") as f:
        for sid, seq in seqs.items():
            f.write(f">{sid}\n{seq}\n")


def _write_labels(path, labels):
    """Write labels TSV file."""
    with open(path, "w") as f:
        f.write("sequence_id\tlabel\n")
        for sid, label in labels.items():
            f.write(f"{sid}\t{label}\n")


class TestTrainingMode:
    def test_train_produces_model_and_metrics(self, tmp_path):
        """Test training mode produces classifier and metrics."""
        from virosense.subcommands.classify import _run_training

        # Create synthetic data
        fasta = tmp_path / "seqs.fasta"
        labels_file = tmp_path / "labels.tsv"
        output = tmp_path / "output"

        seqs = {f"seq_{i}": "ATGC" * 100 for i in range(40)}
        labels = {f"seq_{i}": "classA" if i < 20 else "classB" for i in range(40)}
        _write_fasta(fasta, seqs)
        _write_labels(labels_file, labels)

        mock_backend = _make_mock_backend()

        with patch("virosense.backends.base.get_backend", return_value=mock_backend):
            _run_training(
                input_file=str(fasta),
                labels_file=str(labels_file),
                output_path=output,
                backend="nim",
                model="evo2_7b",
                task="custom_task",
                epochs=50,
                lr=1e-3,
                val_split=0.2,
                layer="blocks.28.mlp.l3",
                cache_dir=None,
            )

        assert (output / "classifier.joblib").exists()
        assert (output / "metrics.json").exists()

    def test_train_multiclass(self, tmp_path):
        """Test training with more than 2 classes."""
        from virosense.subcommands.classify import _run_training

        fasta = tmp_path / "seqs.fasta"
        labels_file = tmp_path / "labels.tsv"
        output = tmp_path / "output"

        seqs = {f"seq_{i}": "ATGC" * 100 for i in range(60)}
        labels = {}
        for i in range(60):
            if i < 20:
                labels[f"seq_{i}"] = "family_A"
            elif i < 40:
                labels[f"seq_{i}"] = "family_B"
            else:
                labels[f"seq_{i}"] = "family_C"

        _write_fasta(fasta, seqs)
        _write_labels(labels_file, labels)

        mock_backend = _make_mock_backend()

        with patch("virosense.backends.base.get_backend", return_value=mock_backend):
            _run_training(
                input_file=str(fasta),
                labels_file=str(labels_file),
                output_path=output,
                backend="nim",
                model="evo2_7b",
                task="family",
                epochs=50,
                lr=1e-3,
                val_split=0.2,
                layer="blocks.28.mlp.l3",
                cache_dir=None,
            )

        assert (output / "classifier.joblib").exists()

    def test_no_matching_sequences_raises(self, tmp_path):
        """Test error when no sequences match labels."""
        from virosense.subcommands.classify import _run_training

        fasta = tmp_path / "seqs.fasta"
        labels_file = tmp_path / "labels.tsv"

        _write_fasta(fasta, {"seq_A": "ATGC"})
        _write_labels(labels_file, {"seq_B": "class1"})

        mock_backend = _make_mock_backend()

        with patch("virosense.backends.base.get_backend", return_value=mock_backend):
            with pytest.raises(ValueError, match="No sequences matched"):
                _run_training(
                    input_file=str(fasta),
                    labels_file=str(labels_file),
                    output_path=tmp_path / "output",
                    backend="nim",
                    model="evo2_7b",
                    task="test",
                    epochs=10,
                    lr=1e-3,
                    val_split=0.2,
                    layer="blocks.28.mlp.l3",
                    cache_dir=None,
                )


class TestPredictionMode:
    def test_predict_produces_results(self, tmp_path):
        """Test prediction mode produces predictions TSV."""
        from virosense.models.detector import ClassifierConfig, ViralClassifier
        from virosense.subcommands.classify import _run_prediction

        # Train a classifier first
        rng = np.random.RandomState(42)
        X = rng.randn(40, 64).astype(np.float32)
        y = np.array([0] * 20 + [1] * 20)

        config = ClassifierConfig(input_dim=64, num_classes=2)
        clf = ViralClassifier(config)
        clf.fit(X, y, class_names=["cellular", "viral"])
        model_path = tmp_path / "model.joblib"
        clf.save(model_path)

        # Create prediction FASTA
        fasta = tmp_path / "predict.fasta"
        _write_fasta(fasta, {f"new_seq_{i}": "ATGC" * 50 for i in range(5)})

        output = tmp_path / "output"
        mock_backend = _make_mock_backend()

        with patch("virosense.backends.base.get_backend", return_value=mock_backend):
            _run_prediction(
                predict_file=str(fasta),
                classifier_model_path=str(model_path),
                output_path=output,
                backend="nim",
                model="evo2_7b",
                layer="blocks.28.mlp.l3",
                cache_dir=None,
            )

        assert (output / "predictions.tsv").exists()
        assert (output / "prediction_summary.json").exists()

        # Check predictions content
        import pandas as pd
        df = pd.read_csv(output / "predictions.tsv", sep="\t")
        assert len(df) == 5
        assert "sequence_id" in df.columns
        assert "predicted_class" in df.columns
        assert "confidence" in df.columns
        assert all(df["confidence"] >= 0)
        assert all(df["confidence"] <= 1)


class TestClassificationResult:
    def test_dataclass_fields(self):
        """Test ClassificationResult dataclass."""
        r = ClassificationResult(
            sequence_id="seq_1",
            predicted_class="viral",
            confidence=0.95,
        )
        assert r.sequence_id == "seq_1"
        assert r.predicted_class == "viral"
        assert r.confidence == 0.95


class TestCLI:
    def test_classify_help(self):
        """Test classify command help works."""
        from click.testing import CliRunner

        from virosense.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["classify", "--help"])
        assert result.exit_code == 0
        assert "--layer" in result.output
        assert "--cache-dir" in result.output
        assert "--predict" in result.output
