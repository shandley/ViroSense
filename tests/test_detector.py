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
from virosense.models.training import (
    _expected_calibration_error,
    evaluate_classifier,
    train_classifier,
)


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

    def test_normalize_l2_flag(self):
        """L2-normalization config is stored in metadata and applied."""
        X, y = self._make_training_data(n=200, dim=64)
        config = ClassifierConfig(input_dim=64, hidden_dims=[32], normalize_l2=True)
        clf = ViralClassifier(config)
        clf.fit(X, y, class_names=["cellular", "viral"])

        assert clf.config.normalize_l2 is True
        assert clf.metadata["normalize_l2"] is True

        # Predictions should work (normalization applied internally)
        probas = clf.predict_proba(X)
        assert probas.shape == (200, 2)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_normalize_l2_save_load_roundtrip(self, tmp_path):
        """L2-normalization flag is preserved across save/load."""
        X, y = self._make_training_data(n=200, dim=64)
        config = ClassifierConfig(input_dim=64, hidden_dims=[32], normalize_l2=True)
        clf = ViralClassifier(config)
        clf.fit(X, y, class_names=["cellular", "viral"])

        model_path = tmp_path / "l2_model.joblib"
        clf.save(model_path)

        loaded = ViralClassifier.load(model_path)
        assert loaded.config.normalize_l2 is True
        assert loaded.metadata["normalize_l2"] is True

        # Predictions should match
        original_preds = clf.predict(X)
        loaded_preds = loaded.predict(X)
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_normalize_l2_disabled_by_default(self):
        """Default config does not normalize."""
        config = ClassifierConfig(input_dim=64)
        clf = ViralClassifier(config)
        assert clf.config.normalize_l2 is False

        # _normalize should be a no-op when disabled
        X = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(clf._normalize(X), X)

    def test_normalize_l2_changes_embeddings(self):
        """When enabled, _normalize produces unit-norm vectors."""
        config = ClassifierConfig(input_dim=3, normalize_l2=True)
        clf = ViralClassifier(config)

        X = np.array([[3.0, 4.0, 0.0]])
        X_norm = clf._normalize(X)
        np.testing.assert_allclose(np.linalg.norm(X_norm, axis=1), 1.0, atol=1e-6)


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


class TestPlattCalibration:
    def _make_data(self, n=200, dim=64):
        """Create linearly-separable data for calibration tests."""
        rng = np.random.default_rng(42)
        X0 = rng.standard_normal((n // 2, dim)).astype(np.float32) - 1.0
        X1 = rng.standard_normal((n // 2, dim)).astype(np.float32) + 1.0
        X = np.vstack([X0, X1])
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        return X, y

    def test_train_with_calibration(self, tmp_path):
        """Training produces calibrated model with Brier score and ECE."""
        X, y = self._make_data(n=200)
        metrics = train_classifier(
            embeddings=X,
            labels=y,
            output_dir=tmp_path / "cal_output",
            epochs=100,
            lr=1e-3,
            val_split=0.3,
            task="viral_vs_cellular",
            class_names=["cellular", "viral"],
        )

        # Calibration metrics present
        assert "brier_score" in metrics
        assert "ece" in metrics
        assert "brier_score_uncalibrated" in metrics
        assert "ece_uncalibrated" in metrics

        # Scores are valid
        assert 0 <= metrics["brier_score"] <= 1
        assert 0 <= metrics["ece"] <= 1
        assert metrics["n_cal"] > 0
        assert metrics["n_test"] > 0

    def test_calibrated_model_saves_and_loads(self, tmp_path):
        """Calibrated model round-trips through save/load."""
        X, y = self._make_data(n=200)
        train_classifier(
            embeddings=X,
            labels=y,
            output_dir=tmp_path / "cal_model",
            epochs=100,
            val_split=0.3,
            class_names=["cellular", "viral"],
        )

        model_path = tmp_path / "cal_model" / "classifier.joblib"
        loaded = ViralClassifier.load(model_path)
        assert loaded.metadata.get("calibrated") is True

        # Calibrated model still produces valid probabilities
        probas = loaded.predict_proba(X[:10])
        assert probas.shape == (10, 2)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_skips_calibration_with_few_samples(self, tmp_path):
        """Calibration is skipped when holdout set is too small."""
        X, y = self._make_data(n=40)
        metrics = train_classifier(
            embeddings=X,
            labels=y,
            output_dir=tmp_path / "small_output",
            epochs=100,
            val_split=0.2,  # 8 holdout < 20 minimum
            class_names=["cellular", "viral"],
        )

        # Should still produce Brier/ECE (uncalibrated fallback)
        assert "brier_score" in metrics
        assert metrics["n_cal"] == 0

    def test_ece_perfect_calibration(self):
        """ECE is 0 for perfectly calibrated predictions."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        assert _expected_calibration_error(y_true, y_prob) == pytest.approx(0.0)

    def test_ece_worst_calibration(self):
        """ECE is 1 for maximally miscalibrated predictions."""
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert _expected_calibration_error(y_true, y_prob) == pytest.approx(1.0)


class TestThreeClassDetection:
    """Tests for 3-class (chromosome/plasmid/viral) classifier support."""

    def _make_3class_data(self, n=300, dim=64):
        rng = np.random.default_rng(42)
        X0 = rng.standard_normal((n // 3, dim)).astype(np.float32) - 2.0  # chromosome
        X1 = rng.standard_normal((n // 3, dim)).astype(np.float32)        # plasmid
        X2 = rng.standard_normal((n // 3, dim)).astype(np.float32) + 2.0  # viral
        X = np.vstack([X0, X1, X2])
        y = np.array([0] * (n // 3) + [1] * (n // 3) + [2] * (n // 3))
        return X, y

    def test_3class_fit_predict(self):
        X, y = self._make_3class_data()
        clf = ViralClassifier(ClassifierConfig(input_dim=64, hidden_dims=[32, 16], num_classes=3))
        clf.fit(X, y, class_names=["chromosome", "plasmid", "viral"])

        probas = clf.predict_proba(X)
        assert probas.shape == (300, 3)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)
        assert clf.metadata["n_classes"] == 3
        assert clf.metadata["class_names"] == ["chromosome", "plasmid", "viral"]

    def test_3class_classify_contigs_populates_scores(self):
        X, y = self._make_3class_data()
        clf = ViralClassifier(ClassifierConfig(input_dim=64, hidden_dims=[32, 16], num_classes=3))
        clf.fit(X, y, class_names=["chromosome", "plasmid", "viral"])

        test_emb = np.random.default_rng(99).standard_normal((5, 64)).astype(np.float32) + 3.0
        results = classify_contigs(test_emb, [f"v{i}" for i in range(5)], [1000] * 5, clf)

        assert len(results) == 5
        for r in results:
            assert r.chromosome_score is not None
            assert r.plasmid_score is not None
            assert 0 <= r.viral_score <= 1
            assert 0 <= r.chromosome_score <= 1
            assert 0 <= r.plasmid_score <= 1

    def test_3class_nonviral_uses_class_names(self):
        X, y = self._make_3class_data()
        clf = ViralClassifier(ClassifierConfig(input_dim=64, hidden_dims=[32, 16], num_classes=3))
        clf.fit(X, y, class_names=["chromosome", "plasmid", "viral"])

        # Clearly non-viral (chromosome-like)
        chr_emb = np.random.default_rng(7).standard_normal((5, 64)).astype(np.float32) - 3.0
        results = classify_contigs(chr_emb, [f"c{i}" for i in range(5)], [1000] * 5, clf)

        classifications = {r.classification for r in results}
        # 3-class model should NOT produce "cellular" — uses "chromosome"/"plasmid" instead
        assert "cellular" not in classifications

    def test_2class_backward_compatible(self):
        rng = np.random.default_rng(42)
        X = np.vstack([
            rng.standard_normal((50, 64)).astype(np.float32) - 2.0,
            rng.standard_normal((50, 64)).astype(np.float32) + 2.0,
        ])
        y = np.array([0] * 50 + [1] * 50)
        clf = ViralClassifier(ClassifierConfig(input_dim=64, hidden_dims=[32]))
        clf.fit(X, y, class_names=["cellular", "viral"])

        results = classify_contigs(X[:5], [f"s{i}" for i in range(5)], [1000] * 5, clf)
        assert all(r.chromosome_score is None for r in results)
        assert all(r.plasmid_score is None for r in results)
        assert all(r.classification in ("viral", "cellular", "ambiguous") for r in results)

    def test_3class_train_uses_isotonic_calibration(self, tmp_path):
        X, y = self._make_3class_data(n=300)
        metrics = train_classifier(
            embeddings=X,
            labels=y,
            output_dir=tmp_path / "3class_model",
            epochs=100,
            val_split=0.3,
            class_names=["chromosome", "plasmid", "viral"],
        )

        loaded = ViralClassifier.load(tmp_path / "3class_model" / "classifier.joblib")
        assert loaded.metadata.get("calibrated") is True
        assert loaded.metadata.get("calibration_method") == "isotonic"
        assert loaded.metadata["n_classes"] == 3

        # Calibrated probabilities still sum to 1
        probas = loaded.predict_proba(X[:10])
        assert probas.shape == (10, 3)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)

        # Multi-class metrics present
        assert "log_loss" in metrics
        assert "auc" in metrics

    def test_3class_save_load_roundtrip(self, tmp_path):
        X, y = self._make_3class_data()
        clf = ViralClassifier(ClassifierConfig(input_dim=64, hidden_dims=[32], num_classes=3))
        clf.fit(X, y, class_names=["chromosome", "plasmid", "viral"])

        model_path = tmp_path / "3class.joblib"
        clf.save(model_path)
        loaded = ViralClassifier.load(model_path)

        assert loaded.metadata["n_classes"] == 3
        assert loaded.metadata["class_names"] == ["chromosome", "plasmid", "viral"]

        original_preds = clf.predict(X)
        loaded_preds = loaded.predict(X)
        np.testing.assert_array_equal(original_preds, loaded_preds)


class TestFourClassDetection:
    """Tests for 4-class (chromosome/phage/plasmid/rna_virus) classifier."""

    def _make_4class_data(self, n=400, dim=64):
        rng = np.random.default_rng(42)
        X0 = rng.standard_normal((n // 4, dim)).astype(np.float32) - 3.0  # chromosome
        X1 = rng.standard_normal((n // 4, dim)).astype(np.float32) + 3.0  # phage
        X2 = rng.standard_normal((n // 4, dim)).astype(np.float32)        # plasmid
        X3 = rng.standard_normal((n // 4, dim)).astype(np.float32) + 2.0  # rna_virus
        X = np.vstack([X0, X1, X2, X3])
        y = np.array([0] * (n // 4) + [1] * (n // 4) + [2] * (n // 4) + [3] * (n // 4))
        return X, y

    def test_4class_fit_predict(self):
        X, y = self._make_4class_data()
        clf = ViralClassifier(ClassifierConfig(input_dim=64, hidden_dims=[32, 16], num_classes=4))
        clf.fit(X, y, class_names=["chromosome", "phage", "plasmid", "rna_virus"])

        probas = clf.predict_proba(X)
        assert probas.shape == (400, 4)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)
        assert clf.metadata["n_classes"] == 4

    def test_4class_viral_score_is_sum(self):
        """viral_score should be P(phage) + P(rna_virus)."""
        X, y = self._make_4class_data()
        clf = ViralClassifier(ClassifierConfig(input_dim=64, hidden_dims=[32, 16], num_classes=4))
        clf.fit(X, y, class_names=["chromosome", "phage", "plasmid", "rna_virus"])

        # Clearly phage-like embeddings
        phage_emb = np.random.default_rng(99).standard_normal((5, 64)).astype(np.float32) + 4.0
        results = classify_contigs(phage_emb, [f"p{i}" for i in range(5)], [1000] * 5, clf)

        for r in results:
            # viral_score = phage_score + rna_virus_score
            assert r.phage_score is not None
            assert r.rna_virus_score is not None
            expected = round(r.phage_score + r.rna_virus_score, 4)
            assert r.viral_score == pytest.approx(expected, abs=1e-3)

    def test_4class_populates_all_scores(self):
        X, y = self._make_4class_data()
        clf = ViralClassifier(ClassifierConfig(input_dim=64, hidden_dims=[32, 16], num_classes=4))
        clf.fit(X, y, class_names=["chromosome", "phage", "plasmid", "rna_virus"])

        results = classify_contigs(X[:10], [f"s{i}" for i in range(10)], [1000] * 10, clf)
        for r in results:
            assert r.chromosome_score is not None
            assert r.plasmid_score is not None
            assert r.phage_score is not None
            assert r.rna_virus_score is not None

    def test_4class_viral_classification_reports_subclass(self):
        """When classified as viral, should report 'phage' or 'rna_virus', not generic 'viral'."""
        X, y = self._make_4class_data()
        clf = ViralClassifier(ClassifierConfig(input_dim=64, hidden_dims=[32, 16], num_classes=4))
        clf.fit(X, y, class_names=["chromosome", "phage", "plasmid", "rna_virus"])

        # Mix of viral and non-viral
        results = classify_contigs(X, [f"s{i}" for i in range(400)], [1000] * 400, clf)
        viral_results = [r for r in results if r.viral_score >= 0.5]

        if viral_results:
            # With multiple viral classes, classification should be specific
            classifications = {r.classification for r in viral_results}
            assert classifications <= {"phage", "rna_virus", "ambiguous"}
            assert "viral" not in classifications  # should use specific subclass

    def test_4class_nonviral_uses_class_names(self):
        X, y = self._make_4class_data()
        clf = ViralClassifier(ClassifierConfig(input_dim=64, hidden_dims=[32, 16], num_classes=4))
        clf.fit(X, y, class_names=["chromosome", "phage", "plasmid", "rna_virus"])

        chr_emb = np.random.default_rng(7).standard_normal((5, 64)).astype(np.float32) - 4.0
        results = classify_contigs(chr_emb, [f"c{i}" for i in range(5)], [1000] * 5, clf)

        classifications = {r.classification for r in results}
        assert "cellular" not in classifications


class TestViralIndicesHelper:
    """Tests for _get_viral_indices and _compute_viral_score."""

    def test_2class(self):
        from virosense.models.detector import _get_viral_indices
        assert _get_viral_indices(["cellular", "viral"]) == [1]

    def test_3class(self):
        from virosense.models.detector import _get_viral_indices
        assert _get_viral_indices(["chromosome", "plasmid", "viral"]) == [2]

    def test_4class(self):
        from virosense.models.detector import _get_viral_indices
        assert _get_viral_indices(["chromosome", "phage", "plasmid", "rna_virus"]) == [1, 3]

    def test_compute_viral_score_sums(self):
        from virosense.models.detector import _compute_viral_score
        probas = np.array([0.1, 0.4, 0.2, 0.3])  # chr, phage, plasmid, rna_virus
        score = _compute_viral_score(probas, ["chromosome", "phage", "plasmid", "rna_virus"])
        assert score == pytest.approx(0.7)  # 0.4 + 0.3

    def test_compute_viral_score_single_class(self):
        from virosense.models.detector import _compute_viral_score
        probas = np.array([0.3, 0.7])
        score = _compute_viral_score(probas, ["cellular", "viral"])
        assert score == pytest.approx(0.7)


class TestEvaluateClassifierExtended:
    """Tests for confusion matrix and per-class metrics in evaluate_classifier."""

    def test_confusion_matrix_present(self):
        rng = np.random.default_rng(42)
        X = np.vstack([
            rng.standard_normal((50, 64)).astype(np.float32) - 2.0,
            rng.standard_normal((50, 64)).astype(np.float32) + 2.0,
        ])
        y = np.array([0] * 50 + [1] * 50)
        clf = ViralClassifier(ClassifierConfig(input_dim=64, hidden_dims=[32]))
        clf.fit(X, y, class_names=["cellular", "viral"])

        metrics = evaluate_classifier(clf, X, y, n_classes=2)
        assert "confusion_matrix" in metrics
        cm = metrics["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2
        # Row sums should equal class counts
        assert sum(cm[0]) == 50
        assert sum(cm[1]) == 50

    def test_per_class_metrics_present(self):
        rng = np.random.default_rng(42)
        X = np.vstack([
            rng.standard_normal((50, 64)).astype(np.float32) - 2.0,
            rng.standard_normal((50, 64)).astype(np.float32) + 2.0,
        ])
        y = np.array([0] * 50 + [1] * 50)
        clf = ViralClassifier(ClassifierConfig(input_dim=64, hidden_dims=[32]))
        clf.fit(X, y, class_names=["cellular", "viral"])

        metrics = evaluate_classifier(clf, X, y, n_classes=2)
        assert "per_class" in metrics
        assert "cellular" in metrics["per_class"]
        assert "viral" in metrics["per_class"]
        for cls in ("cellular", "viral"):
            assert "f1" in metrics["per_class"][cls]
            assert "precision" in metrics["per_class"][cls]
            assert "recall" in metrics["per_class"][cls]

    def test_confusion_matrix_labels(self):
        rng = np.random.default_rng(42)
        X = np.vstack([
            rng.standard_normal((50, 64)).astype(np.float32) - 2.0,
            rng.standard_normal((50, 64)).astype(np.float32) + 2.0,
        ])
        y = np.array([0] * 50 + [1] * 50)
        clf = ViralClassifier(ClassifierConfig(input_dim=64, hidden_dims=[32]))
        clf.fit(X, y, class_names=["cellular", "viral"])

        metrics = evaluate_classifier(clf, X, y, n_classes=2)
        assert metrics["confusion_matrix_labels"] == ["cellular", "viral"]


class TestThresholdValidation:
    """Test that detect validates threshold bounds."""

    def test_threshold_above_1_raises(self):
        from virosense.subcommands.detect import run_detect
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            run_detect(
                input_file="dummy.fasta",
                output_dir="/tmp/out",
                threshold=1.5,
            )

    def test_threshold_below_0_raises(self):
        from virosense.subcommands.detect import run_detect
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            run_detect(
                input_file="dummy.fasta",
                output_dir="/tmp/out",
                threshold=-0.1,
            )


class TestWriteFasta:
    """Test write_fasta round-trip."""

    def test_write_and_read_back(self, tmp_path):
        from virosense.io.fasta import read_fasta, write_fasta

        seqs = {
            "seq_a": "ATGCATGCATGC",
            "seq_b": "GGGGCCCCTTTTAAAA",
            "seq_c": "A" * 200,
        }
        out_path = tmp_path / "output.fasta"
        write_fasta(seqs, out_path)
        assert out_path.exists()

        loaded = read_fasta(str(out_path))
        assert set(loaded.keys()) == set(seqs.keys())
        for sid, seq in seqs.items():
            assert loaded[sid] == seq

    def test_write_fasta_wrapping(self, tmp_path):
        from virosense.io.fasta import write_fasta

        seqs = {"long_seq": "A" * 200}
        out_path = tmp_path / "wrapped.fasta"
        write_fasta(seqs, out_path, wrap=80)

        lines = out_path.read_text().strip().split("\n")
        assert lines[0] == ">long_seq"
        # All sequence lines should be <= 80 chars
        for line in lines[1:]:
            assert len(line) <= 80

    def test_write_fasta_creates_parent_dirs(self, tmp_path):
        from virosense.io.fasta import write_fasta

        seqs = {"s1": "ATGC"}
        out_path = tmp_path / "nested" / "dir" / "output.fasta"
        write_fasta(seqs, out_path)
        assert out_path.exists()


class TestHTMLReports:
    """Test HTML report generation."""

    def test_detect_report_generated(self, tmp_path):
        from virosense.io.report import generate_detect_report
        from virosense.models.detector import DetectionResult

        results = [
            DetectionResult(
                contig_id=f"c{i}", viral_score=0.1 * i,
                classification="viral" if i >= 5 else "cellular",
                contig_length=1000 + i * 500,
            )
            for i in range(10)
        ]
        sequences = {f"c{i}": "ATGC" * (250 + i * 125) for i in range(10)}
        summary = {
            "n_sequences": 10, "n_viral": 5, "n_cellular": 5, "n_ambiguous": 0,
            "classification_counts": {"viral": 5, "cellular": 5},
            "score_distribution": {
                "mean": 0.45, "median": 0.45, "min": 0.0, "max": 0.9,
                "above_0.9": 1, "between_0.5_0.9": 4, "below_0.5": 5,
            },
            "parameters": {"threshold": 0.5, "min_length": 500,
                           "backend": "nim", "model": "evo2_7b",
                           "layer": "blocks.28.mlp.l3"},
            "classifier": {"input_dim": 64, "n_classes": 2,
                           "class_names": ["cellular", "viral"],
                           "calibrated": True},
        }

        generate_detect_report(results, sequences, summary, tmp_path)
        report = tmp_path / "detection_report.html"
        assert report.exists()
        html = report.read_text()
        assert "plotly" in html.lower() or "Plotly" in html
        assert "Score Distribution" in html or "score" in html.lower()

    def test_training_report_generated(self, tmp_path):
        from virosense.io.report import generate_training_report

        metrics = {
            "accuracy": 0.95, "f1": 0.94, "precision": 0.93, "recall": 0.96,
            "auc": 0.98,
            "n_train": 80, "n_cal": 10, "n_test": 10,
            "n_classes": 2, "class_names": ["cellular", "viral"],
            "task": "viral_vs_cellular", "epochs": 100, "lr": 0.001,
            "confusion_matrix": [[48, 2], [3, 47]],
            "confusion_matrix_labels": ["cellular", "viral"],
            "per_class": {
                "cellular": {"f1": 0.95, "precision": 0.94, "recall": 0.96},
                "viral": {"f1": 0.94, "precision": 0.96, "recall": 0.94},
            },
            "brier_score": 0.05, "ece": 0.03,
            "brier_score_uncalibrated": 0.08, "ece_uncalibrated": 0.06,
        }
        y_test = np.array([0] * 10 + [1] * 10)
        probas_test = np.column_stack([1 - np.linspace(0.1, 0.9, 20),
                                       np.linspace(0.1, 0.9, 20)])

        generate_training_report(metrics, tmp_path, y_test=y_test,
                                 probas_test=probas_test)
        report = tmp_path / "training_report.html"
        assert report.exists()
        html = report.read_text()
        assert "plotly" in html.lower() or "Plotly" in html
        assert "Confusion Matrix" in html or "confusion" in html.lower()


class TestDefaultModelPath:
    def test_returns_path(self):
        path = get_default_model_path()
        assert path.name == "reference_classifier.joblib"
        assert "virosense" in str(path).lower() or ".virosense" in str(path)
