"""End-to-end integration tests for all ViroSense pipelines.

Each test exercises a full subcommand pipeline with a mock backend,
verifying that all components wire together correctly and produce
expected output files with valid content.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from virosense.backends.base import EmbeddingRequest, EmbeddingResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_mock_backend(embed_dim=64):
    """Create a mock backend returning deterministic random embeddings."""
    backend = MagicMock()
    backend.is_available.return_value = True
    backend.max_context_length.return_value = 16_000

    rng = np.random.RandomState(42)

    def mock_extract(request: EmbeddingRequest) -> EmbeddingResult:
        n = len(request.sequences)
        return EmbeddingResult(
            sequence_ids=list(request.sequences.keys()),
            embeddings=rng.randn(n, embed_dim).astype(np.float32),
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


def _write_gff3(path, orfs):
    """Write ORFs to GFF3 format.

    orfs: list of (contig_id, start, end, strand, orf_id)
    """
    with open(path, "w") as f:
        f.write("##gff-version 3\n")
        for contig_id, start, end, strand, orf_id in orfs:
            f.write(
                f"{contig_id}\tprodigal\tCDS\t{start}\t{end}\t.\t{strand}\t0\t"
                f"ID={orf_id}\n"
            )


# ---------------------------------------------------------------------------
# detect pipeline
# ---------------------------------------------------------------------------


class TestDetectPipeline:
    """End-to-end tests for the detect subcommand."""

    def test_detect_full_pipeline(self, tmp_path):
        """Full detect pipeline: FASTA -> embeddings -> classify -> TSV."""
        from virosense.subcommands.detect import run_detect

        fasta = tmp_path / "contigs.fasta"
        output = tmp_path / "results"

        # Create contigs of varying lengths (some below min_length)
        seqs = {}
        for i in range(10):
            length = 300 + i * 100  # 300, 400, ..., 1200
            seqs[f"contig_{i}"] = "ATGC" * (length // 4)
        _write_fasta(fasta, seqs)

        mock_backend = _make_mock_backend()

        # Need a trained classifier for detect to use
        from virosense.models.detector import ClassifierConfig, ViralClassifier

        rng = np.random.RandomState(42)
        X = rng.randn(40, 64).astype(np.float32)
        y = np.array([0] * 20 + [1] * 20)
        config = ClassifierConfig(input_dim=64, num_classes=2)
        clf = ViralClassifier(config)
        clf.fit(X, y, class_names=["cellular", "viral"])

        model_path = tmp_path / "model" / "classifier.joblib"
        model_path.parent.mkdir()
        clf.save(model_path)

        with (
            patch("virosense.backends.base.get_backend", return_value=mock_backend),
            patch(
                "virosense.subcommands.detect._load_classifier",
                return_value=clf,
            ),
        ):
            run_detect(
                input_file=str(fasta),
                output_dir=str(output),
                backend="nim",
                model="evo2_7b",
                threshold=0.5,
                min_length=500,
                batch_size=16,
                threads=4,
                layer="blocks.28.mlp.l3",
                cache_dir=None,
            )

        # Verify output
        result_tsv = output / "detection_results.tsv"
        assert result_tsv.exists()

        df = pd.read_csv(result_tsv, sep="\t")
        # Only contigs >= 500 bp should be included (contig_2 through contig_9)
        assert len(df) == 8
        assert "contig_id" in df.columns
        assert "classification" in df.columns
        assert "viral_score" in df.columns
        assert all(df["viral_score"].between(0, 1))

    def test_detect_all_short_sequences(self, tmp_path):
        """Detect with all sequences below min_length produces no output."""
        from virosense.subcommands.detect import run_detect

        fasta = tmp_path / "short.fasta"
        output = tmp_path / "results"
        _write_fasta(fasta, {f"s{i}": "ATGC" * 10 for i in range(5)})

        mock_backend = _make_mock_backend()
        with patch("virosense.backends.base.get_backend", return_value=mock_backend):
            run_detect(
                input_file=str(fasta),
                output_dir=str(output),
                backend="nim",
                model="evo2_7b",
                threshold=0.5,
                min_length=500,
                batch_size=16,
                threads=4,
                layer="blocks.28.mlp.l3",
                cache_dir=None,
            )

        # No output file since no sequences passed the filter
        assert not (output / "detection_results.tsv").exists()


# ---------------------------------------------------------------------------
# build-reference pipeline
# ---------------------------------------------------------------------------


class TestBuildReferencePipeline:
    """End-to-end tests for the build-reference subcommand."""

    def test_build_reference_full(self, tmp_path):
        """Full build-reference pipeline: data -> train -> model + metrics."""
        from virosense.subcommands.build_reference import run_build_reference

        fasta = tmp_path / "seqs.fasta"
        labels_file = tmp_path / "labels.tsv"
        output = tmp_path / "model"

        seqs = {f"seq_{i}": "ATGC" * 200 for i in range(40)}
        labels = {f"seq_{i}": "0" if i < 20 else "1" for i in range(40)}
        _write_fasta(fasta, seqs)
        _write_labels(labels_file, labels)

        mock_backend = _make_mock_backend()

        with patch("virosense.backends.base.get_backend", return_value=mock_backend):
            run_build_reference(
                input_file=str(fasta),
                labels_file=str(labels_file),
                output_dir=str(output),
                backend="nim",
                model="evo2_7b",
                layer="blocks.28.mlp.l3",
                epochs=50,
                lr=1e-3,
                val_split=0.2,
                install=False,
                batch_size=16,
                cache_dir=None,
            )

        assert (output / "classifier.joblib").exists()
        assert (output / "metrics.json").exists()

        # Verify metrics content
        with open(output / "metrics.json") as f:
            metrics = json.load(f)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_build_reference_with_install(self, tmp_path):
        """Build-reference with --install copies model to default location."""
        from virosense.subcommands.build_reference import run_build_reference

        fasta = tmp_path / "seqs.fasta"
        labels_file = tmp_path / "labels.tsv"
        output = tmp_path / "model"

        seqs = {f"seq_{i}": "ATGC" * 200 for i in range(40)}
        labels = {f"seq_{i}": "0" if i < 20 else "1" for i in range(40)}
        _write_fasta(fasta, seqs)
        _write_labels(labels_file, labels)

        mock_backend = _make_mock_backend()
        fake_default_path = tmp_path / "installed" / "classifier.joblib"

        with (
            patch("virosense.backends.base.get_backend", return_value=mock_backend),
            patch(
                "virosense.models.detector.get_default_model_path",
                return_value=fake_default_path,
            ),
        ):
            run_build_reference(
                input_file=str(fasta),
                labels_file=str(labels_file),
                output_dir=str(output),
                backend="nim",
                model="evo2_7b",
                layer="blocks.28.mlp.l3",
                epochs=50,
                lr=1e-3,
                val_split=0.2,
                install=True,
                batch_size=16,
                cache_dir=None,
            )

        assert fake_default_path.exists()


# ---------------------------------------------------------------------------
# classify pipeline
# ---------------------------------------------------------------------------


class TestClassifyPipeline:
    """End-to-end tests for the classify subcommand."""

    def test_train_then_predict(self, tmp_path):
        """Full classify round-trip: train a model, then predict with it."""
        from virosense.subcommands.classify import _run_prediction, _run_training

        # --- Training ---
        train_fasta = tmp_path / "train.fasta"
        labels_file = tmp_path / "labels.tsv"
        train_output = tmp_path / "trained"

        seqs = {f"seq_{i}": "ATGC" * 200 for i in range(40)}
        labels = {f"seq_{i}": "alpha" if i < 20 else "beta" for i in range(40)}
        _write_fasta(train_fasta, seqs)
        _write_labels(labels_file, labels)

        mock_backend = _make_mock_backend()

        with patch("virosense.backends.base.get_backend", return_value=mock_backend):
            _run_training(
                input_file=str(train_fasta),
                labels_file=str(labels_file),
                output_path=train_output,
                backend="nim",
                model="evo2_7b",
                task="family",
                epochs=50,
                lr=1e-3,
                val_split=0.2,
                layer="blocks.28.mlp.l3",
                cache_dir=None,
            )

        model_path = train_output / "classifier.joblib"
        assert model_path.exists()

        # --- Prediction ---
        pred_fasta = tmp_path / "predict.fasta"
        pred_output = tmp_path / "predictions"
        _write_fasta(pred_fasta, {f"new_{i}": "ATGC" * 150 for i in range(8)})

        with patch("virosense.backends.base.get_backend", return_value=mock_backend):
            _run_prediction(
                predict_file=str(pred_fasta),
                classifier_model_path=str(model_path),
                output_path=pred_output,
                backend="nim",
                model="evo2_7b",
                layer="blocks.28.mlp.l3",
                cache_dir=None,
            )

        # Verify predictions
        pred_tsv = pred_output / "predictions.tsv"
        assert pred_tsv.exists()
        df = pd.read_csv(pred_tsv, sep="\t")
        assert len(df) == 8
        assert set(df["predicted_class"].unique()).issubset({"alpha", "beta"})
        assert all(df["confidence"].between(0, 1))

        # Verify summary
        summary_path = pred_output / "prediction_summary.json"
        assert summary_path.exists()
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["n_sequences"] == 8
        assert "class_distribution" in summary

    def test_classify_via_cli_dispatcher(self, tmp_path):
        """Test run_classify dispatches to training correctly."""
        from virosense.subcommands.classify import run_classify

        fasta = tmp_path / "seqs.fasta"
        labels_file = tmp_path / "labels.tsv"
        output = tmp_path / "output"

        seqs = {f"seq_{i}": "ATGC" * 200 for i in range(30)}
        labels = {f"seq_{i}": "classA" if i < 15 else "classB" for i in range(30)}
        _write_fasta(fasta, seqs)
        _write_labels(labels_file, labels)

        mock_backend = _make_mock_backend()

        with patch("virosense.backends.base.get_backend", return_value=mock_backend):
            run_classify(
                input_file=str(fasta),
                labels_file=str(labels_file),
                output_dir=str(output),
                backend="nim",
                model="evo2_7b",
                task="custom",
                epochs=50,
                lr=1e-3,
                val_split=0.2,
                predict_file=None,
                classifier_model_path=None,
                layer="blocks.28.mlp.l3",
                cache_dir=None,
            )

        assert (output / "classifier.joblib").exists()
        assert (output / "metrics.json").exists()


# ---------------------------------------------------------------------------
# cluster pipeline
# ---------------------------------------------------------------------------


class TestClusterPipeline:
    """End-to-end tests for the cluster subcommand."""

    def test_cluster_hdbscan_dna_only(self, tmp_path):
        """Full cluster pipeline with HDBSCAN, DNA-only mode."""
        from virosense.subcommands.cluster import run_cluster

        fasta = tmp_path / "viral.fasta"
        output = tmp_path / "clusters"
        _write_fasta(fasta, {f"vseq_{i}": "ATGC" * 300 for i in range(30)})

        mock_backend = _make_mock_backend()

        with patch("virosense.backends.base.get_backend", return_value=mock_backend):
            run_cluster(
                input_file=str(fasta),
                output_dir=str(output),
                backend="nim",
                model="evo2_7b",
                mode="dna",
                algorithm="hdbscan",
                min_cluster_size=3,
                n_clusters=None,
                threads=4,
                vhold_embeddings=None,
                layer="blocks.28.mlp.l3",
                cache_dir=None,
            )

        # Verify outputs
        assert (output / "cluster_assignments.tsv").exists()
        assert (output / "cluster_metrics.json").exists()

        df = pd.read_csv(output / "cluster_assignments.tsv", sep="\t")
        assert len(df) == 30
        assert "sequence_id" in df.columns
        assert "cluster_id" in df.columns

        with open(output / "cluster_metrics.json") as f:
            metrics = json.load(f)
        assert "n_clusters" in metrics
        assert "n_noise" in metrics
        assert metrics["algorithm"] == "hdbscan"
        assert metrics["mode"] == "dna"
        assert metrics["n_sequences"] == 30

    def test_cluster_kmeans_with_k(self, tmp_path):
        """Cluster pipeline with KMeans and specified k."""
        from virosense.subcommands.cluster import run_cluster

        fasta = tmp_path / "viral.fasta"
        output = tmp_path / "clusters"
        _write_fasta(fasta, {f"vseq_{i}": "ATGC" * 300 for i in range(30)})

        mock_backend = _make_mock_backend()

        with patch("virosense.backends.base.get_backend", return_value=mock_backend):
            run_cluster(
                input_file=str(fasta),
                output_dir=str(output),
                backend="nim",
                model="evo2_7b",
                mode="dna",
                algorithm="kmeans",
                min_cluster_size=5,
                n_clusters=3,
                threads=4,
                vhold_embeddings=None,
                layer="blocks.28.mlp.l3",
                cache_dir=None,
            )

        df = pd.read_csv(output / "cluster_assignments.tsv", sep="\t")
        assert len(df) == 30
        # KMeans with k=3 should produce exactly 3 clusters
        assert df["cluster_id"].nunique() == 3

        with open(output / "cluster_metrics.json") as f:
            metrics = json.load(f)
        assert metrics["n_clusters"] == 3
        assert metrics["n_noise"] == 0

    def test_cluster_multimodal_with_vhold(self, tmp_path):
        """Cluster with multi-modal fusion (DNA + protein embeddings)."""
        from virosense.subcommands.cluster import run_cluster

        fasta = tmp_path / "viral.fasta"
        output = tmp_path / "clusters"
        n_seqs = 20
        seq_ids = [f"vseq_{i}" for i in range(n_seqs)]
        _write_fasta(fasta, {sid: "ATGC" * 300 for sid in seq_ids})

        # Create mock vHold protein embeddings
        vhold_path = tmp_path / "protein_embeddings.npz"
        prot_dim = 32
        rng = np.random.RandomState(99)
        np.savez(
            vhold_path,
            sequence_ids=np.array(seq_ids),
            embeddings=rng.randn(n_seqs, prot_dim).astype(np.float32),
        )

        mock_backend = _make_mock_backend()

        with patch("virosense.backends.base.get_backend", return_value=mock_backend):
            run_cluster(
                input_file=str(fasta),
                output_dir=str(output),
                backend="nim",
                model="evo2_7b",
                mode="multi",
                algorithm="kmeans",
                min_cluster_size=5,
                n_clusters=3,
                threads=4,
                vhold_embeddings=str(vhold_path),
                layer="blocks.28.mlp.l3",
                cache_dir=None,
            )

        df = pd.read_csv(output / "cluster_assignments.tsv", sep="\t")
        assert len(df) == n_seqs

        with open(output / "cluster_metrics.json") as f:
            metrics = json.load(f)
        assert metrics["mode"] == "multi"

    def test_cluster_empty_fasta(self, tmp_path):
        """Cluster with empty FASTA produces no output."""
        from virosense.subcommands.cluster import run_cluster

        fasta = tmp_path / "empty.fasta"
        fasta.write_text("")
        output = tmp_path / "clusters"

        mock_backend = _make_mock_backend()

        with patch("virosense.backends.base.get_backend", return_value=mock_backend):
            run_cluster(
                input_file=str(fasta),
                output_dir=str(output),
                backend="nim",
                model="evo2_7b",
                mode="dna",
                algorithm="hdbscan",
                min_cluster_size=5,
                n_clusters=None,
                threads=4,
                vhold_embeddings=None,
                layer="blocks.28.mlp.l3",
                cache_dir=None,
            )

        assert not (output / "cluster_assignments.tsv").exists()


# ---------------------------------------------------------------------------
# context pipeline
# ---------------------------------------------------------------------------


class TestContextPipeline:
    """End-to-end tests for the context subcommand."""

    def test_context_full_pipeline(self, tmp_path):
        """Full context pipeline: contigs + ORFs -> window embeddings -> TSV."""
        from virosense.subcommands.context import run_context

        fasta = tmp_path / "contigs.fasta"
        gff = tmp_path / "orfs.gff3"
        output = tmp_path / "context"

        # Create contigs
        contigs = {
            "contig_1": "A" * 10_000,
            "contig_2": "C" * 8_000,
        }
        _write_fasta(fasta, contigs)

        # Create ORFs
        _write_gff3(gff, [
            ("contig_1", 1000, 2000, "+", "orf_1"),
            ("contig_1", 5000, 6000, "-", "orf_2"),
            ("contig_2", 3000, 4000, "+", "orf_3"),
        ])

        mock_backend = _make_mock_backend()

        with patch("virosense.backends.base.get_backend", return_value=mock_backend):
            run_context(
                input_file=str(fasta),
                orfs_file=str(gff),
                output_dir=str(output),
                backend="nim",
                model="evo2_7b",
                window_size=2000,
                vhold_output=None,
                threads=4,
                layer="blocks.28.mlp.l3",
                cache_dir=None,
            )

        # Verify outputs
        assert (output / "context_annotations.tsv").exists()
        assert (output / "context_summary.json").exists()

        df = pd.read_csv(output / "context_annotations.tsv", sep="\t")
        assert len(df) == 3
        assert set(df["orf_id"]) == {"orf_1", "orf_2", "orf_3"}
        assert all(df["window_embedding_norm"] > 0)

        with open(output / "context_summary.json") as f:
            summary = json.load(f)
        assert summary["n_orfs"] == 3
        assert summary["n_contigs"] == 2
        assert summary["window_size"] == 2000

    def test_context_with_vhold_merge(self, tmp_path):
        """Context pipeline with vHold annotations merged."""
        from virosense.subcommands.context import run_context

        fasta = tmp_path / "contigs.fasta"
        gff = tmp_path / "orfs.gff3"
        output = tmp_path / "context"

        _write_fasta(fasta, {"contig_1": "A" * 10_000})
        _write_gff3(gff, [
            ("contig_1", 1000, 2000, "+", "orf_1"),
            ("contig_1", 5000, 6000, "-", "orf_2"),
        ])

        # Create vHold annotations TSV
        vhold_tsv = tmp_path / "vhold_output.tsv"
        vhold_tsv.write_text(
            "orf_id\tannotation\tscore\n"
            "orf_1\tcapsid protein\t0.95\n"
        )

        mock_backend = _make_mock_backend()

        # Mock load_vhold_annotations to return parsed data
        vhold_data = {
            "orf_1": {"annotation": "capsid protein", "score": 0.95},
        }

        with (
            patch("virosense.backends.base.get_backend", return_value=mock_backend),
            patch(
                "virosense.features.prostt5_bridge.load_vhold_annotations",
                return_value=vhold_data,
            ),
        ):
            run_context(
                input_file=str(fasta),
                orfs_file=str(gff),
                output_dir=str(output),
                backend="nim",
                model="evo2_7b",
                window_size=2000,
                vhold_output=str(vhold_tsv),
                threads=4,
                layer="blocks.28.mlp.l3",
                cache_dir=None,
            )

        df = pd.read_csv(output / "context_annotations.tsv", sep="\t")
        assert len(df) == 2

        # orf_1 should have vHold annotation
        orf1 = df[df["orf_id"] == "orf_1"].iloc[0]
        assert orf1["vhold_annotation"] == "capsid protein"
        assert orf1["vhold_score"] == 0.95

        with open(output / "context_summary.json") as f:
            summary = json.load(f)
        assert summary["n_with_vhold"] == 1

    def test_context_missing_contigs_for_orfs(self, tmp_path):
        """ORFs with missing contigs are skipped gracefully."""
        from virosense.subcommands.context import run_context

        fasta = tmp_path / "contigs.fasta"
        gff = tmp_path / "orfs.gff3"
        output = tmp_path / "context"

        _write_fasta(fasta, {"contig_1": "A" * 10_000})
        _write_gff3(gff, [
            ("contig_1", 1000, 2000, "+", "orf_1"),
            ("contig_missing", 500, 1000, "+", "orf_2"),
        ])

        mock_backend = _make_mock_backend()

        with patch("virosense.backends.base.get_backend", return_value=mock_backend):
            run_context(
                input_file=str(fasta),
                orfs_file=str(gff),
                output_dir=str(output),
                backend="nim",
                model="evo2_7b",
                window_size=2000,
                vhold_output=None,
                threads=4,
                layer="blocks.28.mlp.l3",
                cache_dir=None,
            )

        df = pd.read_csv(output / "context_annotations.tsv", sep="\t")
        assert len(df) == 1
        assert df.iloc[0]["orf_id"] == "orf_1"

    def test_context_no_orfs(self, tmp_path):
        """Context with empty ORFs file returns early."""
        from virosense.subcommands.context import run_context

        fasta = tmp_path / "contigs.fasta"
        gff = tmp_path / "orfs.gff3"
        output = tmp_path / "context"

        _write_fasta(fasta, {"contig_1": "A" * 10_000})
        gff.write_text("##gff-version 3\n")  # no ORFs

        mock_backend = _make_mock_backend()

        with patch("virosense.backends.base.get_backend", return_value=mock_backend):
            run_context(
                input_file=str(fasta),
                orfs_file=str(gff),
                output_dir=str(output),
                backend="nim",
                model="evo2_7b",
                window_size=2000,
                vhold_output=None,
                threads=4,
                layer="blocks.28.mlp.l3",
                cache_dir=None,
            )

        # No output since no ORFs
        assert not (output / "context_annotations.tsv").exists()


# ---------------------------------------------------------------------------
# Embedding caching integration
# ---------------------------------------------------------------------------


class TestEmbeddingCaching:
    """Test embedding extraction with caching across pipeline stages."""

    def test_cache_produces_same_results(self, tmp_path):
        """Embeddings with and without cache produce aligned results."""
        from virosense.features.evo2_embeddings import extract_embeddings

        seqs = {f"s{i}": "ATGC" * 100 for i in range(10)}
        mock_backend = _make_mock_backend()

        # Extract without cache
        result_no_cache = extract_embeddings(
            sequences=seqs,
            backend=mock_backend,
            layer="blocks.28.mlp.l3",
            model="evo2_7b",
            cache_dir=None,
        )

        # Extract with cache (first run)
        cache_dir = tmp_path / "cache"
        result_cached = extract_embeddings(
            sequences=seqs,
            backend=mock_backend,
            layer="blocks.28.mlp.l3",
            model="evo2_7b",
            cache_dir=cache_dir,
            checkpoint_every=5,
        )

        assert result_cached.sequence_ids == list(seqs.keys())
        assert result_cached.embeddings.shape == result_no_cache.embeddings.shape

        # Extract with cache (second run — should load from disk)
        mock_backend_2 = _make_mock_backend()
        result_from_cache = extract_embeddings(
            sequences=seqs,
            backend=mock_backend_2,
            layer="blocks.28.mlp.l3",
            model="evo2_7b",
            cache_dir=cache_dir,
            checkpoint_every=5,
        )

        # Backend should NOT be called on second run
        mock_backend_2.extract_embeddings.assert_not_called()
        assert result_from_cache.sequence_ids == list(seqs.keys())
        np.testing.assert_array_equal(
            result_from_cache.embeddings, result_cached.embeddings
        )

    def test_cache_resume_partial(self, tmp_path):
        """Cache resume works when only some sequences are cached."""
        from virosense.features.evo2_embeddings import extract_embeddings

        all_seqs = {f"s{i}": "ATGC" * 100 for i in range(10)}
        cache_dir = tmp_path / "cache"

        mock_backend = _make_mock_backend()

        # Extract first 5 (simulate partial run)
        first_half = {k: v for k, v in list(all_seqs.items())[:5]}
        extract_embeddings(
            sequences=first_half,
            backend=mock_backend,
            layer="blocks.28.mlp.l3",
            model="evo2_7b",
            cache_dir=cache_dir,
            checkpoint_every=5,
        )

        first_call_count = mock_backend.extract_embeddings.call_count

        # Now extract all 10 — should only call backend for the missing 5
        result = extract_embeddings(
            sequences=all_seqs,
            backend=mock_backend,
            layer="blocks.28.mlp.l3",
            model="evo2_7b",
            cache_dir=cache_dir,
            checkpoint_every=5,
        )

        assert result.sequence_ids == list(all_seqs.keys())
        assert result.embeddings.shape[0] == 10
        # Should have made one additional backend call (for the 5 new sequences)
        assert mock_backend.extract_embeddings.call_count == first_call_count + 1


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLIIntegration:
    """Test CLI commands wire up correctly."""

    def test_all_commands_have_help(self):
        """All subcommands produce valid help text."""
        from click.testing import CliRunner

        from virosense.cli import main

        runner = CliRunner()

        for cmd in ["detect", "context", "cluster", "classify", "build-reference"]:
            result = runner.invoke(main, [cmd, "--help"])
            assert result.exit_code == 0, f"{cmd} --help failed: {result.output}"
            assert "--backend" in result.output
            assert "--layer" in result.output or cmd == "detect"

    def test_all_commands_have_cache_dir(self):
        """All subcommands support --cache-dir."""
        from click.testing import CliRunner

        from virosense.cli import main

        runner = CliRunner()

        for cmd in ["detect", "context", "cluster", "classify", "build-reference"]:
            result = runner.invoke(main, [cmd, "--help"])
            assert "--cache-dir" in result.output, f"{cmd} missing --cache-dir"

    def test_version(self):
        """CLI version flag works."""
        from click.testing import CliRunner

        from virosense.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "virosense" in result.output
