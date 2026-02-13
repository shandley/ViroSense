"""Tests for the viral classifier and detection pipeline."""

import numpy as np
import pytest

from virosense.models.detector import (
    ClassifierConfig,
    DetectionResult,
    ViralClassifier,
    classify_contigs,
    get_default_model_path,
)
from virosense.models.training import train_classifier, evaluate_classifier


# --- ViralClassifier tests ---


class TestViralClassifier:
    def _make_training_data(self, n=200, dim=64):
        """Create simple linearly-separable training data."""
        rng = np.random.default_rng(42)
        # Class 0: centered at -1, class 1: centered at +1
        X0 = rng.standard_normal((n // 2, dim)).astype(np.float32) - 1.0
        X1 = rng.standard_normal((n // 2, dim)).astype(np.float32) + 1.0
        X = np.vstack([X0, X1])
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        return X, y

    def test_fit_and_predict(self):
        X, y = self._make_training_data()
        config = ClassifierConfig(input_dim=64, hidden_dims=[32, 16])
        clf = ViralClassifier(config)
        clf.fit(X, y, class_names=["cellular", "viral"])

        predictions = clf.predict(X)
        assert predictions.shape == (200,)
        assert set(predictions).issubset({0, 1})

        # Should get good accuracy on easily separable data
        accuracy = (predictions == y).mean()
        assert accuracy > 0.8

    def test_predict_proba(self):
        X, y = self._make_training_data()
        clf = ViralClassifier(ClassifierConfig(input_dim=64, hidden_dims=[32]))
        clf.fit(X, y)

        probas = clf.predict_proba(X)
        assert probas.shape == (200, 2)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_before_fit_raises(self):
        clf = ViralClassifier()
        with pytest.raises(RuntimeError, match="not trained"):
            clf.predict(np.zeros((1, 4096)))

    def test_save_load_roundtrip(self, tmp_path):
        X, y = self._make_training_data()
        clf = ViralClassifier(ClassifierConfig(input_dim=64, hidden_dims=[32]))
        clf.fit(X, y, class_names=["cellular", "viral"], layer="blocks.28.mlp.l3")

        model_path = tmp_path / "test_model.joblib"
        clf.save(model_path)

        loaded = ViralClassifier.load(model_path)
        assert loaded.metadata["n_train"] == 200
        assert loaded.metadata["class_names"] == ["cellular", "viral"]
        assert loaded.metadata["layer"] == "blocks.28.mlp.l3"

        # Predictions should match
        original_preds = clf.predict(X)
        loaded_preds = loaded.predict(X)
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ViralClassifier.load(tmp_path / "no_such_model.joblib")

    def test_metadata_stored(self):
        X, y = self._make_training_data(n=100, dim=32)
        clf = ViralClassifier(ClassifierConfig(input_dim=32))
        clf.fit(X, y, class_names=["a", "b"], layer="blocks.20.mlp.l3", model="evo2_40b")

        assert clf.metadata["n_train"] == 100
        assert clf.metadata["n_classes"] == 2
        assert clf.metadata["input_dim"] == 32
        assert clf.metadata["layer"] == "blocks.20.mlp.l3"
        assert clf.metadata["model"] == "evo2_40b"


# --- classify_contigs tests ---


class TestClassifyContigs:
    def test_classify_with_trained_model(self):
        rng = np.random.default_rng(42)
        X_train = np.vstack([
            rng.standard_normal((50, 64)).astype(np.float32) - 1.0,
            rng.standard_normal((50, 64)).astype(np.float32) + 1.0,
        ])
        y_train = np.array([0] * 50 + [1] * 50)

        clf = ViralClassifier(ClassifierConfig(input_dim=64, hidden_dims=[32]))
        clf.fit(X_train, y_train, class_names=["cellular", "viral"])

        # Test sequences
        embeddings = np.vstack([
            rng.standard_normal((3, 64)).astype(np.float32) - 2.0,  # clearly cellular
            rng.standard_normal((2, 64)).astype(np.float32) + 2.0,  # clearly viral
        ])
        ids = ["c1", "c2", "c3", "v1", "v2"]
        lengths = [1000, 2000, 3000, 4000, 5000]

        results = classify_contigs(embeddings, ids, lengths, clf, threshold=0.5)

        assert len(results) == 5
        assert all(isinstance(r, DetectionResult) for r in results)
        assert all(0.0 <= r.viral_score <= 1.0 for r in results)
        assert all(r.classification in ("viral", "cellular", "ambiguous") for r in results)

    def test_threshold_affects_classification(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 64)).astype(np.float32)
        y = np.array([0] * 50 + [1] * 50)

        clf = ViralClassifier(ClassifierConfig(input_dim=64, hidden_dims=[32]))
        clf.fit(X, y)

        test_emb = rng.standard_normal((10, 64)).astype(np.float32)
        ids = [f"s{i}" for i in range(10)]
        lengths = [1000] * 10

        results_low = classify_contigs(test_emb, ids, lengths, clf, threshold=0.3)
        results_high = classify_contigs(test_emb, ids, lengths, clf, threshold=0.9)

        n_viral_low = sum(1 for r in results_low if r.classification == "viral")
        n_viral_high = sum(1 for r in results_high if r.classification == "viral")
        assert n_viral_low >= n_viral_high


# --- train_classifier tests ---


class TestTrainClassifier:
    def test_train_produces_model_and_metrics(self, tmp_path):
        rng = np.random.default_rng(42)
        X = np.vstack([
            rng.standard_normal((50, 64)).astype(np.float32) - 1.0,
            rng.standard_normal((50, 64)).astype(np.float32) + 1.0,
        ])
        y = np.array([0] * 50 + [1] * 50)

        metrics = train_classifier(
            embeddings=X,
            labels=y,
            output_dir=tmp_path / "model_output",
            epochs=100,
            lr=1e-3,
            val_split=0.2,
            task="viral_vs_cellular",
            class_names=["cellular", "viral"],
            layer="blocks.28.mlp.l3",
            model="evo2_7b",
        )

        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "auc" in metrics
        assert metrics["accuracy"] > 0.5
        assert (tmp_path / "model_output" / "classifier.joblib").exists()
        assert (tmp_path / "model_output" / "metrics.json").exists()

    def test_evaluate_classifier(self):
        rng = np.random.default_rng(42)
        X = np.vstack([
            rng.standard_normal((50, 64)).astype(np.float32) - 2.0,
            rng.standard_normal((50, 64)).astype(np.float32) + 2.0,
        ])
        y = np.array([0] * 50 + [1] * 50)

        clf = ViralClassifier(ClassifierConfig(input_dim=64, hidden_dims=[32]))
        clf.fit(X, y)

        metrics = evaluate_classifier(clf, X, y, n_classes=2)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "auc" in metrics
        assert metrics["accuracy"] > 0.8


# --- CLI build-reference tests ---


class TestBuildReferenceHelp:
    def test_help(self):
        from click.testing import CliRunner
        from virosense.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["build-reference", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--labels" in result.output
        assert "--install" in result.output


class TestDefaultModelPath:
    def test_returns_path(self):
        path = get_default_model_path()
        assert path.name == "reference_classifier.joblib"
        assert "virosense" in str(path).lower() or ".virosense" in str(path)
