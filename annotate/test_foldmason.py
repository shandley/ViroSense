"""Tests for FoldMason structural alignment module."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from foldmason import (
    FoldMasonResult,
    check_foldmason,
    run_foldmason_msa,
    write_alignment_summary,
)


# ============================================================================
# FoldMasonResult
# ============================================================================


class TestFoldMasonResult:
    def test_fields(self, tmp_path):
        result = FoldMasonResult(
            aa_msa=tmp_path / "aa.fa",
            three_di_msa=tmp_path / "3di.fa",
            guide_tree=tmp_path / "tree.nw",
            num_sequences=10,
            alignment_length=200,
        )
        assert result.num_sequences == 10
        assert result.alignment_length == 200


# ============================================================================
# check_foldmason
# ============================================================================


class TestCheckFoldmason:
    def test_available(self):
        with patch("foldmason.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="1.0\n")
            available, version = check_foldmason()
        assert available is True
        assert version == "1.0"

    def test_not_found(self):
        with patch("foldmason.subprocess.run", side_effect=FileNotFoundError):
            available, msg = check_foldmason()
        assert available is False
        assert "not found" in msg

    def test_timeout(self):
        with patch(
            "foldmason.subprocess.run",
            side_effect=__import__("subprocess").TimeoutExpired(["foldmason"], 10),
        ):
            available, msg = check_foldmason()
        assert available is False
        assert "timed out" in msg


# ============================================================================
# run_foldmason_msa
# ============================================================================


class TestRunFoldmasonMsa:
    def _write_fake_output(self, output_prefix: Path) -> None:
        """Write fake FoldMason output files."""
        aa_msa = Path(str(output_prefix) + "_aa.fa")
        three_di_msa = Path(str(output_prefix) + "_3di.fa")
        tree = Path(str(output_prefix) + ".nw")

        aa_msa.write_text(">seq1\nACDEFG--HI\n>seq2\nACDE--KLHI\n")
        three_di_msa.write_text(">seq1\nabcdefg--hi\n>seq2\nabcde--klhi\n")
        tree.write_text("(seq1:0.1,seq2:0.2);")

    def test_successful_alignment(self, tmp_path):
        query_db = tmp_path / "queryDB"
        output_prefix = tmp_path / "output" / "msa"

        def side_effect(cmd, **_kwargs):
            # Write fake output files
            output_prefix_from_cmd = Path(cmd[3])
            output_prefix_from_cmd.parent.mkdir(parents=True, exist_ok=True)
            self._write_fake_output(output_prefix_from_cmd)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("foldmason.check_foldmason", return_value=(True, "1.0")):
            with patch("foldmason.subprocess.run", side_effect=side_effect):
                result = run_foldmason_msa(query_db, output_prefix)

        assert result.num_sequences == 2
        assert result.alignment_length == 10  # "ACDEFG--HI" = 10 chars

    def test_foldmason_not_available(self, tmp_path):
        with patch("foldmason.check_foldmason", return_value=(False, "not found")):
            with pytest.raises(FileNotFoundError, match="FoldMason not available"):
                run_foldmason_msa(tmp_path / "db", tmp_path / "out")

    def test_foldmason_failure(self, tmp_path):
        with patch("foldmason.check_foldmason", return_value=(True, "1.0")):
            with patch("foldmason.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=1, stderr="Segmentation fault"
                )
                with pytest.raises(RuntimeError, match="foldmason structuremsa failed"):
                    run_foldmason_msa(tmp_path / "db", tmp_path / "out")

    def test_missing_output_raises(self, tmp_path):
        with patch("foldmason.check_foldmason", return_value=(True, "1.0")):
            with patch("foldmason.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
                with pytest.raises(RuntimeError, match="did not produce"):
                    run_foldmason_msa(tmp_path / "db", tmp_path / "out")

    def test_command_includes_threads(self, tmp_path):
        query_db = tmp_path / "queryDB"
        output_prefix = tmp_path / "out"

        def side_effect(cmd, **_kwargs):
            output_prefix_from_cmd = Path(cmd[3])
            output_prefix_from_cmd.parent.mkdir(parents=True, exist_ok=True)
            self._write_fake_output(output_prefix_from_cmd)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("foldmason.check_foldmason", return_value=(True, "1.0")):
            with patch("foldmason.subprocess.run", side_effect=side_effect) as mock_run:
                run_foldmason_msa(query_db, output_prefix, threads=16)

        cmd = mock_run.call_args[0][0]
        assert "--threads" in cmd
        assert "16" in cmd

    def test_refine_iters_added(self, tmp_path):
        query_db = tmp_path / "queryDB"
        output_prefix = tmp_path / "out"

        def side_effect(cmd, **_kwargs):
            output_prefix_from_cmd = Path(cmd[3])
            output_prefix_from_cmd.parent.mkdir(parents=True, exist_ok=True)
            self._write_fake_output(output_prefix_from_cmd)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("foldmason.check_foldmason", return_value=(True, "1.0")):
            with patch("foldmason.subprocess.run", side_effect=side_effect) as mock_run:
                run_foldmason_msa(query_db, output_prefix, refine_iters=3)

        cmd = mock_run.call_args[0][0]
        assert "--refine-iters" in cmd
        assert "3" in cmd

    def test_no_refine_by_default(self, tmp_path):
        query_db = tmp_path / "queryDB"
        output_prefix = tmp_path / "out"

        def side_effect(cmd, **_kwargs):
            output_prefix_from_cmd = Path(cmd[3])
            output_prefix_from_cmd.parent.mkdir(parents=True, exist_ok=True)
            self._write_fake_output(output_prefix_from_cmd)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("foldmason.check_foldmason", return_value=(True, "1.0")):
            with patch("foldmason.subprocess.run", side_effect=side_effect) as mock_run:
                run_foldmason_msa(query_db, output_prefix)

        cmd = mock_run.call_args[0][0]
        assert "--refine-iters" not in cmd


# ============================================================================
# write_alignment_summary
# ============================================================================


class TestWriteAlignmentSummary:
    def test_writes_json(self, tmp_path):
        result = FoldMasonResult(
            aa_msa=tmp_path / "aa.fa",
            three_di_msa=tmp_path / "3di.fa",
            guide_tree=tmp_path / "tree.nw",
            num_sequences=5,
            alignment_length=150,
        )

        output_dir = tmp_path / "summary_output"
        summary_path = write_alignment_summary(result, output_dir)

        assert summary_path.exists()
        data = json.loads(summary_path.read_text())
        assert data["num_sequences"] == 5
        assert data["alignment_length"] == 150
        assert "aa_msa" in data
        assert "guide_tree" in data

    def test_creates_output_dir(self, tmp_path):
        result = FoldMasonResult(
            aa_msa=tmp_path / "aa.fa",
            three_di_msa=tmp_path / "3di.fa",
            guide_tree=tmp_path / "tree.nw",
            num_sequences=1,
            alignment_length=50,
        )

        output_dir = tmp_path / "deep" / "nested" / "dir"
        write_alignment_summary(result, output_dir)
        assert output_dir.exists()
