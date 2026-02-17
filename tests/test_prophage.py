"""Tests for prophage detection module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from virosense.backends.base import EmbeddingRequest, EmbeddingResult
from virosense.models.prophage import (
    ProphageRegion,
    WindowResult,
    generate_windows,
    merge_prophage_regions,
    score_windows,
)


# ---------------------------------------------------------------------------
# Window generation tests
# ---------------------------------------------------------------------------


class TestGenerateWindows:
    """Tests for the sliding window generator."""

    def test_basic_windows(self):
        """10 kb chromosome, 5 kb window, 2 kb step."""
        chroms = {"chr1": "A" * 10000}
        seqs, meta = generate_windows(chroms, window_size=5000, step_size=2000)
        # Starts: 0, 2000, 4000, plus trailing window at 5000
        assert len(seqs) >= 3
        assert all(len(s) == 5000 for s in seqs.values())
        # All metadata has correct chromosome
        assert all(m["chromosome_id"] == "chr1" for m in meta)

    def test_short_chromosome(self):
        """Chromosome shorter than window → single window."""
        chroms = {"chr1": "A" * 3000}
        seqs, meta = generate_windows(chroms, window_size=5000, step_size=2000)
        assert len(seqs) == 1
        wid = list(seqs.keys())[0]
        assert seqs[wid] == "A" * 3000
        assert meta[0]["start"] == 0
        assert meta[0]["end"] == 3000

    def test_exact_fit(self):
        """Chromosome exactly equal to window size."""
        chroms = {"chr1": "A" * 5000}
        seqs, meta = generate_windows(chroms, window_size=5000, step_size=2000)
        assert len(seqs) == 1
        assert len(list(seqs.values())[0]) == 5000

    def test_ids_encode_coordinates(self):
        """Window IDs encode chr:start:end."""
        chroms = {"mycontig": "A" * 10000}
        seqs, meta = generate_windows(chroms, window_size=5000, step_size=2000)
        for wid in seqs:
            parts = wid.split(":")
            assert len(parts) == 3
            assert parts[0] == "mycontig"
            start, end = int(parts[1]), int(parts[2])
            assert end - start == 5000
            assert len(seqs[wid]) == 5000

    def test_multiple_chromosomes(self):
        """Windows generated for each chromosome independently."""
        chroms = {
            "chr1": "A" * 10000,
            "chr2": "C" * 8000,
        }
        seqs, meta = generate_windows(chroms, window_size=5000, step_size=2000)
        chr1_windows = [m for m in meta if m["chromosome_id"] == "chr1"]
        chr2_windows = [m for m in meta if m["chromosome_id"] == "chr2"]
        assert len(chr1_windows) >= 3
        assert len(chr2_windows) >= 2
        assert len(seqs) == len(meta)

    def test_trailing_window_covers_end(self):
        """Last window covers the end of the chromosome."""
        chroms = {"chr1": "A" * 11000}
        seqs, meta = generate_windows(chroms, window_size=5000, step_size=3000)
        # Check that a window ends at 11000
        ends = [m["end"] for m in meta]
        assert 11000 in ends

    def test_window_sequences_correct(self):
        """Window sequences are the correct substring."""
        seq = "AAAA" * 500 + "CCCC" * 500 + "GGGG" * 500  # 6000 bp
        chroms = {"chr1": seq}
        seqs, meta = generate_windows(chroms, window_size=2000, step_size=2000)
        for m in meta:
            wid = m["window_id"]
            assert seqs[wid] == seq[m["start"]:m["end"]]


# ---------------------------------------------------------------------------
# Region merging tests
# ---------------------------------------------------------------------------


class TestMergeRegions:
    """Tests for prophage region merging logic."""

    def _make_windows(self, scores, window_size=5000, step_size=2000, chrom="chr1"):
        """Create WindowResults from a list of viral scores."""
        results = []
        for i, score in enumerate(scores):
            start = i * step_size
            end = start + window_size
            results.append(WindowResult(
                window_id=f"{chrom}:{start}:{end}",
                chromosome_id=chrom,
                start=start,
                end=end,
                viral_score=score,
                classification="viral" if score >= 0.5 else "cellular",
            ))
        return results

    def test_consecutive_viral(self):
        """5 consecutive viral windows → 1 region."""
        windows = self._make_windows([0.9, 0.8, 0.85, 0.7, 0.95])
        regions = merge_prophage_regions(windows, threshold=0.5, min_region_length=0)
        assert len(regions) == 1
        assert regions[0].n_windows == 5
        assert regions[0].start == 0
        assert regions[0].end == 13000  # 4*2000 + 5000

    def test_no_viral(self):
        """All cellular windows → no regions."""
        windows = self._make_windows([0.1, 0.2, 0.15, 0.3, 0.1])
        regions = merge_prophage_regions(windows, threshold=0.5, min_region_length=0)
        assert len(regions) == 0

    def test_merge_with_gap(self):
        """Two viral regions separated by 1 cellular window within merge_gap."""
        # Windows: V, V, C, V, V (step=2000)
        # Gap between end of window 1 (end=7000) and start of window 3 (start=6000)
        # is within merge_gap=3000
        windows = self._make_windows([0.9, 0.8, 0.1, 0.85, 0.9])
        regions = merge_prophage_regions(
            windows, threshold=0.5, min_region_length=0, merge_gap=3000
        )
        assert len(regions) == 1

    def test_gap_too_large(self):
        """Two viral regions separated by gap > merge_gap → 2 regions."""
        # V, C, C, C, V — gap is large
        windows = self._make_windows(
            [0.9, 0.1, 0.1, 0.1, 0.9],
            step_size=5000,  # non-overlapping windows
        )
        regions = merge_prophage_regions(
            windows, threshold=0.5, min_region_length=0, merge_gap=3000
        )
        assert len(regions) == 2

    def test_min_length_filter(self):
        """Short region below min_region_length → filtered out."""
        # Single viral window = 5000 bp region
        windows = self._make_windows([0.1, 0.9, 0.1])
        regions = merge_prophage_regions(
            windows, threshold=0.5, min_region_length=10000
        )
        assert len(regions) == 0

    def test_min_length_keeps_long(self):
        """Region above min_region_length → kept."""
        windows = self._make_windows([0.9, 0.8, 0.85, 0.9])
        regions = merge_prophage_regions(
            windows, threshold=0.5, min_region_length=5000
        )
        assert len(regions) == 1
        assert regions[0].length >= 5000

    def test_multiple_chromosomes(self):
        """Regions on different chromosomes stay separate."""
        w1 = self._make_windows([0.9, 0.8], chrom="chr1")
        w2 = self._make_windows([0.85, 0.9], chrom="chr2")
        regions = merge_prophage_regions(
            w1 + w2, threshold=0.5, min_region_length=0
        )
        assert len(regions) == 2
        chroms = {r.chromosome_id for r in regions}
        assert chroms == {"chr1", "chr2"}

    def test_region_scores(self):
        """Mean and max scores computed correctly."""
        windows = self._make_windows([0.7, 0.9, 0.8])
        regions = merge_prophage_regions(
            windows, threshold=0.5, min_region_length=0
        )
        assert len(regions) == 1
        assert regions[0].max_score == 0.9
        assert abs(regions[0].mean_score - 0.8) < 0.01

    def test_region_ids_unique(self):
        """Each region gets a unique ID."""
        windows = self._make_windows(
            [0.9, 0.1, 0.1, 0.1, 0.9],
            step_size=5000,
        )
        regions = merge_prophage_regions(
            windows, threshold=0.5, min_region_length=0, merge_gap=0
        )
        ids = [r.region_id for r in regions]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Window scoring tests
# ---------------------------------------------------------------------------


class TestScoreWindows:
    """Tests for window scoring with a classifier."""

    def test_score_windows_basic(self):
        """Score windows with a mock classifier."""
        from virosense.models.detector import ClassifierConfig, ViralClassifier

        # Train a tiny classifier
        rng = np.random.RandomState(42)
        X = rng.randn(40, 16).astype(np.float32)
        y = np.array([0] * 20 + [1] * 20)
        clf = ViralClassifier(ClassifierConfig(input_dim=16, num_classes=2))
        clf.fit(X, y, class_names=["cellular", "viral"])

        # Create window metadata
        meta = [
            {"window_id": "chr1:0:5000", "chromosome_id": "chr1", "start": 0, "end": 5000},
            {"window_id": "chr1:2000:7000", "chromosome_id": "chr1", "start": 2000, "end": 7000},
        ]
        embeddings = rng.randn(2, 16).astype(np.float32)
        seq_ids = ["chr1:0:5000", "chr1:2000:7000"]

        results = score_windows(embeddings, seq_ids, meta, clf, threshold=0.5)
        assert len(results) == 2
        assert all(isinstance(r, WindowResult) for r in results)
        assert all(0.0 <= r.viral_score <= 1.0 for r in results)
        assert all(r.classification in ("viral", "cellular", "ambiguous") for r in results)
        assert results[0].chromosome_id == "chr1"
        assert results[0].start == 0
        assert results[0].end == 5000


# ---------------------------------------------------------------------------
# BED output tests
# ---------------------------------------------------------------------------


class TestBEDOutput:
    """Tests for BED file output."""

    def test_write_bed(self, tmp_path):
        """BED file has correct format."""
        from virosense.io.results import write_bed

        regions = [
            ProphageRegion(
                region_id="prophage_0",
                chromosome_id="chr1",
                start=10000,
                end=45000,
                length=35000,
                n_windows=15,
                mean_score=0.85,
                max_score=0.95,
            ),
            ProphageRegion(
                region_id="prophage_1",
                chromosome_id="chr2",
                start=5000,
                end=20000,
                length=15000,
                n_windows=7,
                mean_score=0.72,
                max_score=0.88,
            ),
        ]

        bed_path = write_bed(regions, tmp_path, "test.bed")
        assert bed_path.exists()

        lines = bed_path.read_text().strip().split("\n")
        assert len(lines) == 2

        # Check first line
        fields = lines[0].split("\t")
        assert len(fields) == 6
        assert fields[0] == "chr1"
        assert fields[1] == "10000"
        assert fields[2] == "45000"
        assert fields[3] == "prophage_0"
        assert fields[4] == "850"  # int(0.85 * 1000)
        assert fields[5] == "."


# ---------------------------------------------------------------------------
# Integration test
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


class TestProphagePipeline:
    """End-to-end tests for the prophage subcommand."""

    def test_prophage_full_pipeline(self, tmp_path):
        """Full prophage pipeline: FASTA -> windows -> score -> merge -> output."""
        from virosense.subcommands.prophage import run_prophage

        fasta = tmp_path / "chromosome.fasta"
        output = tmp_path / "results"

        # Create a 50 kb bacterial chromosome
        with open(fasta, "w") as f:
            f.write(">chromosome_1\n")
            f.write("ATGC" * 12500 + "\n")  # 50,000 bp

        mock_backend = _make_mock_backend()

        # Train a classifier
        from virosense.models.detector import ClassifierConfig, ViralClassifier

        rng = np.random.RandomState(42)
        X = rng.randn(40, 64).astype(np.float32)
        y = np.array([0] * 20 + [1] * 20)
        clf = ViralClassifier(ClassifierConfig(input_dim=64, num_classes=2))
        clf.fit(X, y, class_names=["cellular", "viral"])

        with (
            patch("virosense.backends.base.get_backend", return_value=mock_backend),
            patch(
                "virosense.subcommands.detect._load_classifier",
                return_value=clf,
            ),
        ):
            run_prophage(
                input_file=str(fasta),
                output_dir=str(output),
                backend="nim",
                model="evo2_7b",
                threshold=0.5,
                window_size=5000,
                step_size=2000,
                min_region_length=5000,
                merge_gap=3000,
                batch_size=16,
                layer="blocks.28.mlp.l3",
                cache_dir=None,
                classifier_model=None,
            )

        # Verify output files
        assert (output / "prophage_windows.tsv").exists()
        assert (output / "prophage_regions.tsv").exists()
        assert (output / "prophage_regions.bed").exists()
        assert (output / "prophage_summary.json").exists()

        # Check windows TSV
        windows_df = pd.read_csv(output / "prophage_windows.tsv", sep="\t")
        assert len(windows_df) > 0
        assert "window_id" in windows_df.columns
        assert "chromosome_id" in windows_df.columns
        assert "viral_score" in windows_df.columns
        assert "classification" in windows_df.columns
        assert all(windows_df["viral_score"].between(0, 1))

        # Check summary JSON
        with open(output / "prophage_summary.json") as f:
            summary = json.load(f)
        assert summary["n_chromosomes"] == 1
        assert summary["total_bp"] == 50000
        assert summary["n_windows"] == len(windows_df)

    def test_prophage_empty_input(self, tmp_path):
        """Empty FASTA produces no output files."""
        from virosense.subcommands.prophage import run_prophage

        fasta = tmp_path / "empty.fasta"
        fasta.write_text("")
        output = tmp_path / "results"

        mock_backend = _make_mock_backend()

        with patch("virosense.backends.base.get_backend", return_value=mock_backend):
            run_prophage(
                input_file=str(fasta),
                output_dir=str(output),
                backend="nim",
                model="evo2_7b",
            )

        # No output files should be created
        assert not (output / "prophage_windows.tsv").exists()

    def test_prophage_window_clamping(self, tmp_path):
        """Window size clamped to backend max_context_length."""
        from virosense.subcommands.prophage import run_prophage

        fasta = tmp_path / "chromosome.fasta"
        output = tmp_path / "results"

        with open(fasta, "w") as f:
            f.write(">chr1\n")
            f.write("ATGC" * 10000 + "\n")  # 40,000 bp

        mock_backend = _make_mock_backend()
        mock_backend.max_context_length.return_value = 10000  # limit to 10kb

        from virosense.models.detector import ClassifierConfig, ViralClassifier

        rng = np.random.RandomState(42)
        X = rng.randn(40, 64).astype(np.float32)
        y = np.array([0] * 20 + [1] * 20)
        clf = ViralClassifier(ClassifierConfig(input_dim=64, num_classes=2))
        clf.fit(X, y, class_names=["cellular", "viral"])

        with (
            patch("virosense.backends.base.get_backend", return_value=mock_backend),
            patch(
                "virosense.subcommands.detect._load_classifier",
                return_value=clf,
            ),
        ):
            run_prophage(
                input_file=str(fasta),
                output_dir=str(output),
                backend="nim",
                model="evo2_7b",
                window_size=20000,  # exceeds backend limit
                step_size=5000,
            )

        # Should have used clamped window size
        windows_df = pd.read_csv(output / "prophage_windows.tsv", sep="\t")
        # Verify no window exceeds 10000 bp
        for _, row in windows_df.iterrows():
            assert row["end"] - row["start"] <= 10000
