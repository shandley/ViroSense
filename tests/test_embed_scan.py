"""Tests for virosense embed and scan commands."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from virosense.backends.base import EmbeddingRequest, EmbeddingResult


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def test_fasta(tmp_path):
    """Create a small test FASTA with known sequence content."""
    fasta = tmp_path / "test.fasta"
    # Two sequences — use exact lengths we'll match in per_position_dir
    seq1 = "ATGAAAGCTAGACTGAGCGGCACGAAATTCGAGGACCGCGCGTTCGAAGTGCTGGGCGGCGAGCGCAGCCGCCTGCTGGACGCCGACAGCGTGGGCGCCGCCGTGCGCCAGCGCCGCGGCAACCTGCGCCGCGGCCACCACCACGTGGCGGTGGTGGGCGGCGTGCTGACCGTGCCCGGCCCGCCGCACCGCAGCGAGCGCAGCGCGGTGGTGGGCCGCCGCGAATAA"
    seq2 = "ATGTTCAACAGCTGCGACACCAGCGTGACCCGCAGCGACGGCGACTTCCGCAGCATCCGCAGCATCTGCAGCGCCCTGCCGACCGACCCGATCGGCTAA"
    fasta.write_text(f">seq1\n{seq1}\n>seq2\n{seq2}\n")
    return fasta


@pytest.fixture
def mock_backend_factory():
    """Factory for mock backends with configurable embedding dim."""
    def _make(dim=4096):
        backend = MagicMock()
        backend.is_available.return_value = True
        backend.max_context_length.return_value = 16000
        backend.model = "evo2_7b"

        def mock_extract(request):
            n = len(request.sequences)
            return EmbeddingResult(
                sequence_ids=list(request.sequences.keys()),
                embeddings=np.random.randn(n, dim).astype(np.float32),
                layer=request.layer,
                model=request.model,
            )

        backend.extract_embeddings.side_effect = mock_extract
        return backend
    return _make


@pytest.fixture
def per_position_dir(tmp_path, test_fasta):
    """Create mock per-position embedding files matching test_fasta lengths."""
    pp_dir = tmp_path / "per_position"
    pp_dir.mkdir()

    rng = np.random.default_rng(42)

    # Read actual sequence lengths from FASTA
    from Bio import SeqIO
    records = list(SeqIO.parse(str(test_fasta), "fasta"))

    for rec in records:
        seq_len = len(rec.seq)
        emb = rng.standard_normal((seq_len, 4096)).astype(np.float32)
        # Boost norms for most positions (simulate coding signal)
        coding_end = int(seq_len * 0.85)
        emb[:coding_end] *= 2.0
        safe_name = rec.description.replace("/", "_")[:80]
        np.save(pp_dir / f"{safe_name}.npy", emb)

    return pp_dir


# ======================================================================
# Embed command tests
# ======================================================================


class TestEmbed:
    """Tests for virosense embed subcommand."""

    def _run_embed(self, test_fasta, output_dir, mock_backend_factory, **kwargs):
        """Helper to run embed with mocked backend."""
        from virosense.subcommands.embed import run_embed

        backend = mock_backend_factory()

        with patch("virosense.backends.base.get_backend", return_value=backend):
            run_embed(
                input_file=str(test_fasta),
                output_dir=str(output_dir),
                backend="nim",
                model="evo2_7b",
                **kwargs,
            )
        return backend

    def test_embed_reads_fasta(self, test_fasta, tmp_path, mock_backend_factory):
        """Embed reads FASTA and extracts embeddings."""
        output_dir = tmp_path / "cache"
        self._run_embed(test_fasta, output_dir, mock_backend_factory)
        assert output_dir.exists()

    def test_embed_creates_cache(self, test_fasta, tmp_path, mock_backend_factory):
        """Embed creates NPZ cache file."""
        output_dir = tmp_path / "cache"
        self._run_embed(test_fasta, output_dir, mock_backend_factory)

        npz_files = list(output_dir.glob("*.npz"))
        assert len(npz_files) >= 1

    def test_embed_cache_contains_sequences(self, test_fasta, tmp_path, mock_backend_factory):
        """Embed cache contains the correct number of sequences."""
        output_dir = tmp_path / "cache"
        self._run_embed(test_fasta, output_dir, mock_backend_factory)

        npz_files = list(output_dir.glob("*.npz"))
        data = np.load(npz_files[0])
        assert "sequence_ids" in data
        assert "embeddings" in data
        assert len(data["sequence_ids"]) == 2  # 2 sequences in test FASTA

    def test_embed_empty_fasta(self, tmp_path, mock_backend_factory):
        """Embed handles empty FASTA gracefully."""
        from virosense.subcommands.embed import run_embed

        empty_fasta = tmp_path / "empty.fasta"
        empty_fasta.write_text("")
        output_dir = tmp_path / "cache"

        backend = mock_backend_factory()
        with patch("virosense.backends.base.get_backend", return_value=backend):
            run_embed(
                input_file=str(empty_fasta),
                output_dir=str(output_dir),
                backend="nim",
                model="evo2_7b",
            )

        assert output_dir.exists()


# ======================================================================
# Scan command tests
# ======================================================================


class TestScan:
    """Tests for virosense scan subcommand."""

    def test_scan_finds_per_position_files(self, test_fasta, per_position_dir, tmp_path):
        """Scan finds and loads per-position .npy files."""
        from virosense.subcommands.scan import run_scan

        output_dir = tmp_path / "scan_output"

        run_scan(
            input_file=str(test_fasta),
            output_dir=str(output_dir),
            cache_dir=str(per_position_dir.parent),
            coding=True,
            periodicity=False,
            boundaries=False,
        )

        assert (output_dir / "scan_results.tsv").exists()
        assert (output_dir / "scan_results.json").exists()

    def test_scan_coding_detection(self, test_fasta, per_position_dir, tmp_path):
        """Scan computes coding/intergenic norm ratio."""
        from virosense.subcommands.scan import run_scan

        output_dir = tmp_path / "scan_output"

        run_scan(
            input_file=str(test_fasta),
            output_dir=str(output_dir),
            cache_dir=str(per_position_dir.parent),
            coding=True,
            periodicity=False,
            boundaries=False,
        )

        with open(output_dir / "scan_results.json") as f:
            results = json.load(f)

        # Should have results for sequences with enough genes
        assert len(results) >= 1

    def test_scan_periodicity(self, test_fasta, per_position_dir, tmp_path):
        """Scan computes codon periodicity features."""
        from virosense.subcommands.scan import run_scan

        output_dir = tmp_path / "scan_output"

        run_scan(
            input_file=str(test_fasta),
            output_dir=str(output_dir),
            cache_dir=str(per_position_dir.parent),
            coding=False,
            periodicity=True,
            boundaries=False,
        )

        with open(output_dir / "scan_results.json") as f:
            results = json.load(f)

        assert len(results) >= 1

    def test_scan_boundaries(self, test_fasta, per_position_dir, tmp_path):
        """Scan detects gene boundaries."""
        from virosense.subcommands.scan import run_scan

        output_dir = tmp_path / "scan_output"

        run_scan(
            input_file=str(test_fasta),
            output_dir=str(output_dir),
            cache_dir=str(per_position_dir.parent),
            coding=False,
            periodicity=False,
            boundaries=True,
        )

        with open(output_dir / "scan_results.json") as f:
            results = json.load(f)

        assert len(results) >= 1

    def test_scan_all_features(self, test_fasta, per_position_dir, tmp_path):
        """Scan runs all features together."""
        from virosense.subcommands.scan import run_scan

        output_dir = tmp_path / "scan_output"

        run_scan(
            input_file=str(test_fasta),
            output_dir=str(output_dir),
            cache_dir=str(per_position_dir.parent),
            coding=True,
            periodicity=True,
            boundaries=True,
        )

        assert (output_dir / "scan_results.tsv").exists()

        with open(output_dir / "scan_results.json") as f:
            results = json.load(f)

        assert len(results) >= 1
        # Check that result has expected fields
        r = results[0]
        assert "sequence_id" in r
        assert "length" in r
        assert "n_genes" in r

    def test_scan_missing_per_position_raises(self, test_fasta, tmp_path):
        """Scan raises FileNotFoundError when no per-position files exist."""
        from virosense.subcommands.scan import run_scan

        output_dir = tmp_path / "scan_output"
        empty_cache = tmp_path / "empty_cache"
        empty_cache.mkdir()

        with pytest.raises(FileNotFoundError, match="Per-position"):
            run_scan(
                input_file=str(test_fasta),
                output_dir=str(output_dir),
                cache_dir=str(empty_cache),
            )

    def test_scan_tsv_output_format(self, test_fasta, per_position_dir, tmp_path):
        """Scan TSV output has correct column structure."""
        import pandas as pd
        from virosense.subcommands.scan import run_scan

        output_dir = tmp_path / "scan_output"

        run_scan(
            input_file=str(test_fasta),
            output_dir=str(output_dir),
            cache_dir=str(per_position_dir.parent),
            coding=True,
            periodicity=True,
            boundaries=True,
        )

        df = pd.read_csv(output_dir / "scan_results.tsv", sep="\t")
        assert "sequence_id" in df.columns
        assert "length" in df.columns
        assert "n_genes" in df.columns
        assert len(df) >= 1
