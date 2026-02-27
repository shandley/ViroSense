"""Tests for Foldseek structural search module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add annotate directory to path for direct imports
sys.path.insert(0, str(Path(__file__).parent))

from foldseek import (
    FOLDSEEK_OUTPUT_COLUMNS,
    FOLDSEEK_OUTPUT_FORMAT,
    FoldseekResult,
    SearchBatch,
    check_foldseek,
    create_query_db_from_pdbs,
    easy_search,
    run_foldseek_search,
    search_databases,
    search_pdb_against_db,
)


# ============================================================================
# FoldseekResult / SearchBatch dataclass tests
# ============================================================================


class TestFoldseekResult:
    def test_empty_result(self):
        result = FoldseekResult(
            hits=pd.DataFrame(columns=FOLDSEEK_OUTPUT_COLUMNS),
            n_queries=5,
            n_hits=0,
            database="bfvd",
        )
        assert result.has_hits is False
        assert "0 hits" in result.summary()
        assert "bfvd" in result.summary()

    def test_result_with_hits(self):
        df = pd.DataFrame(
            [["q1", "t1", 0.8, 100, 5, 0, 1, 100, 1, 100, 1e-10, 50,
              0.95, 0.8, 0.7, 100, 200, 1.0, 0.5]],
            columns=FOLDSEEK_OUTPUT_COLUMNS,
        )
        result = FoldseekResult(hits=df, n_queries=1, n_hits=1, database="bfvd")
        assert result.has_hits is True
        assert "1 hits" in result.summary()


class TestSearchBatch:
    def test_empty_batch(self):
        batch = SearchBatch()
        assert batch.all_hits.empty
        assert batch.best_hits.empty

    def test_all_hits_adds_source_db(self):
        df1 = pd.DataFrame(
            [["q1", "t1", 0.8, 100, 5, 0, 1, 100, 1, 100, 1e-10, 50,
              0.95, 0.8, 0.7, 100, 200, 1.0, 0.5]],
            columns=FOLDSEEK_OUTPUT_COLUMNS,
        )
        df2 = pd.DataFrame(
            [["q1", "t2", 0.6, 80, 10, 1, 1, 80, 1, 80, 1e-5, 40,
              0.85, 0.7, 0.6, 100, 150, 0.8, 0.53]],
            columns=FOLDSEEK_OUTPUT_COLUMNS,
        )
        batch = SearchBatch(results={
            "bfvd": FoldseekResult(hits=df1, n_hits=1, database="bfvd"),
            "viro3d": FoldseekResult(hits=df2, n_hits=1, database="viro3d"),
        })
        all_hits = batch.all_hits
        assert len(all_hits) == 2
        assert "source_db" in all_hits.columns
        assert set(all_hits["source_db"]) == {"bfvd", "viro3d"}

    def test_best_hits_picks_lowest_evalue(self):
        df1 = pd.DataFrame(
            [["q1", "t1", 0.8, 100, 5, 0, 1, 100, 1, 100, 1e-10, 50,
              0.95, 0.8, 0.7, 100, 200, 1.0, 0.5]],
            columns=FOLDSEEK_OUTPUT_COLUMNS,
        )
        df2 = pd.DataFrame(
            [["q1", "t2", 0.6, 80, 10, 1, 1, 80, 1, 80, 1e-5, 40,
              0.85, 0.7, 0.6, 100, 150, 0.8, 0.53]],
            columns=FOLDSEEK_OUTPUT_COLUMNS,
        )
        batch = SearchBatch(results={
            "bfvd": FoldseekResult(hits=df1, n_hits=1, database="bfvd"),
            "viro3d": FoldseekResult(hits=df2, n_hits=1, database="viro3d"),
        })
        best = batch.best_hits
        assert len(best) == 1
        assert best.iloc[0]["target"] == "t1"  # lower evalue
        assert best.iloc[0]["source_db"] == "bfvd"

    def test_summary_format(self):
        batch = SearchBatch(results={
            "bfvd": FoldseekResult(
                hits=pd.DataFrame(columns=FOLDSEEK_OUTPUT_COLUMNS),
                n_queries=5, n_hits=3, database="bfvd",
            ),
        })
        assert "bfvd" in batch.summary()


# ============================================================================
# check_foldseek
# ============================================================================


class TestCheckFoldseek:
    def test_available(self):
        with patch("foldseek.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="8.0\n")
            available, version = check_foldseek()
        assert available is True
        assert version == "8.0"

    def test_not_found(self):
        with patch("foldseek.subprocess.run", side_effect=FileNotFoundError):
            available, msg = check_foldseek()
        assert available is False
        assert "not found" in msg

    def test_timeout(self):
        with patch(
            "foldseek.subprocess.run",
            side_effect=__import__("subprocess").TimeoutExpired(["foldseek"], 10),
        ):
            available, msg = check_foldseek()
        assert available is False
        assert "timed out" in msg


# ============================================================================
# create_query_db_from_pdbs
# ============================================================================


class TestCreateQueryDb:
    def test_successful_creation(self, tmp_path):
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        (pdb_dir / "protein1.pdb").write_text("ATOM dummy\n")
        output_db = tmp_path / "queryDB"

        with patch("foldseek.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = create_query_db_from_pdbs(pdb_dir, output_db)

        assert result == output_db
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "foldseek"
        assert cmd[1] == "createdb"
        assert str(pdb_dir) in cmd
        assert str(output_db) in cmd

    def test_failure_raises(self, tmp_path):
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        output_db = tmp_path / "queryDB"

        with patch("foldseek.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="Error: bad input"
            )
            with pytest.raises(RuntimeError, match="foldseek createdb failed"):
                create_query_db_from_pdbs(pdb_dir, output_db)

    def test_creates_parent_dirs(self, tmp_path):
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        output_db = tmp_path / "deep" / "nested" / "queryDB"

        with patch("foldseek.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            create_query_db_from_pdbs(pdb_dir, output_db)

        assert output_db.parent.exists()


# ============================================================================
# run_foldseek_search
# ============================================================================


class TestRunFoldseekSearch:
    def _make_tsv(self, output_path: Path) -> None:
        """Write a fake Foldseek output TSV."""
        row = "\t".join([
            "query1", "target1", "0.8", "100", "5", "0",
            "1", "100", "1", "100", "1e-10", "50",
            "0.95", "0.8", "0.7",
            "100", "200", "1.0", "0.5",
        ])
        output_path.write_text(row + "\n")

    def test_successful_search(self, tmp_path):
        query_db = tmp_path / "queryDB"
        target_db = tmp_path / "targetDB"
        output_path = tmp_path / "results.tsv"

        def side_effect(cmd, **kwargs):
            # Write fake output when convertalis runs
            if "convertalis" in cmd:
                self._make_tsv(output_path)
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("foldseek.subprocess.run", side_effect=side_effect):
            df = run_foldseek_search(query_db, target_db, output_path)

        assert len(df) == 1
        assert df.iloc[0]["query"] == "query1"
        assert df.iloc[0]["prob"] == 0.95

    def test_no_hits(self, tmp_path):
        query_db = tmp_path / "queryDB"
        target_db = tmp_path / "targetDB"
        output_path = tmp_path / "results.tsv"

        with patch("foldseek.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            df = run_foldseek_search(query_db, target_db, output_path)

        assert df.empty
        assert list(df.columns) == FOLDSEEK_OUTPUT_COLUMNS

    def test_search_failure_raises(self, tmp_path):
        query_db = tmp_path / "queryDB"
        target_db = tmp_path / "targetDB"
        output_path = tmp_path / "results.tsv"

        with patch("foldseek.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stderr="Search error"
            )
            with pytest.raises(RuntimeError, match="foldseek search failed"):
                run_foldseek_search(query_db, target_db, output_path)

    def test_uses_custom_tmp_dir(self, tmp_path):
        query_db = tmp_path / "queryDB"
        target_db = tmp_path / "targetDB"
        output_path = tmp_path / "results.tsv"
        custom_tmp = tmp_path / "my_tmp"

        with patch("foldseek.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            run_foldseek_search(
                query_db, target_db, output_path, tmp_dir=custom_tmp
            )

        assert custom_tmp.exists()

    def test_search_parameters_passed(self, tmp_path):
        query_db = tmp_path / "queryDB"
        target_db = tmp_path / "targetDB"
        output_path = tmp_path / "results.tsv"

        with patch("foldseek.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            run_foldseek_search(
                query_db, target_db, output_path,
                threads=16,
                evalue=1e-5,
                sensitivity=7.5,
                max_seqs=500,
            )

        # Check search command
        search_call = mock_run.call_args_list[0]
        cmd = search_call[0][0]
        assert "--threads" in cmd
        assert "16" in cmd
        assert "-e" in cmd
        assert "1e-05" in cmd
        assert "-s" in cmd
        assert "7.5" in cmd
        assert "--max-seqs" in cmd
        assert "500" in cmd


# ============================================================================
# search_pdb_against_db
# ============================================================================


class TestSearchPdbAgainstDb:
    def test_no_pdb_files(self, tmp_path):
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        target_db = tmp_path / "bfvd"
        output_dir = tmp_path / "output"

        result = search_pdb_against_db(pdb_dir, target_db, output_dir)
        assert result.n_queries == 0
        assert result.n_hits == 0
        assert result.hits.empty

    def test_successful_search(self, tmp_path):
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        (pdb_dir / "protein1.pdb").write_text("ATOM dummy\n")
        (pdb_dir / "protein2.pdb").write_text("ATOM dummy\n")
        target_db = tmp_path / "bfvd"
        output_dir = tmp_path / "output"

        def side_effect(cmd, **_kwargs):
            # Write fake output for convertalis
            # cmd: foldseek convertalis queryDB targetDB resultDB output.tsv ...
            if "convertalis" in cmd:
                output_path = Path(cmd[5])  # output path is 6th element
                row = "\t".join([
                    "protein1", "target1", "0.8", "100", "5", "0",
                    "1", "100", "1", "100", "1e-10", "50",
                    "0.95", "0.8", "0.7",
                    "100", "200", "1.0", "0.5",
                ])
                output_path.write_text(row + "\n")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("foldseek.subprocess.run", side_effect=side_effect):
            result = search_pdb_against_db(
                pdb_dir, target_db, output_dir, db_name="bfvd"
            )

        assert result.n_queries == 2
        assert result.n_hits == 1
        assert result.database == "bfvd"

    def test_failure_returns_empty_result(self, tmp_path):
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        (pdb_dir / "protein1.pdb").write_text("ATOM dummy\n")
        target_db = tmp_path / "bfvd"
        output_dir = tmp_path / "output"

        with patch("foldseek.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stderr="createdb failed"
            )
            result = search_pdb_against_db(pdb_dir, target_db, output_dir)

        assert result.n_hits == 0
        assert result.hits.empty


# ============================================================================
# search_databases
# ============================================================================


class TestSearchDatabases:
    def test_missing_database(self, tmp_path):
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        databases = {"bfvd": tmp_path / "nonexistent" / "bfvd"}

        batch = search_databases(pdb_dir, tmp_path / "output", databases)
        assert "bfvd" in batch.results
        assert batch.results["bfvd"].n_hits == 0

    def test_multiple_databases(self, tmp_path):
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        (pdb_dir / "protein1.pdb").write_text("ATOM dummy\n")

        bfvd_dir = tmp_path / "dbs" / "bfvd"
        bfvd_dir.mkdir(parents=True)
        viro3d_dir = tmp_path / "dbs" / "viro3d"
        viro3d_dir.mkdir(parents=True)

        databases = {
            "bfvd": bfvd_dir / "bfvd",
            "viro3d": viro3d_dir / "viro3d",
        }

        with patch("foldseek.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            batch = search_databases(pdb_dir, tmp_path / "output", databases)

        assert "bfvd" in batch.results
        assert "viro3d" in batch.results


# ============================================================================
# easy_search
# ============================================================================


class TestEasySearch:
    def test_successful_easy_search(self, tmp_path):
        pdb_dir = tmp_path / "pdbs"
        pdb_dir.mkdir()
        target_db = tmp_path / "bfvd"
        output_path = tmp_path / "results.tsv"

        def side_effect(cmd, **kwargs):
            # Write fake output
            row = "\t".join([
                "query1", "target1", "0.8", "100", "5", "0",
                "1", "100", "1", "100", "1e-10", "50",
                "0.95", "0.8", "0.7",
                "100", "200", "1.0", "0.5",
            ])
            output_path.write_text(row + "\n")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("foldseek.subprocess.run", side_effect=side_effect):
            df = easy_search(pdb_dir, target_db, output_path)

        assert len(df) == 1
        assert df.iloc[0]["query"] == "query1"

    def test_easy_search_failure(self, tmp_path):
        pdb_dir = tmp_path / "pdbs"
        target_db = tmp_path / "bfvd"
        output_path = tmp_path / "results.tsv"

        with patch("foldseek.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stderr="easy-search failed"
            )
            with pytest.raises(RuntimeError, match="foldseek easy-search failed"):
                easy_search(pdb_dir, target_db, output_path)

    def test_easy_search_uses_format(self, tmp_path):
        pdb_dir = tmp_path / "pdbs"
        target_db = tmp_path / "bfvd"
        output_path = tmp_path / "results.tsv"

        with patch("foldseek.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            easy_search(pdb_dir, target_db, output_path)

        cmd = mock_run.call_args[0][0]
        assert "--format-output" in cmd
        fmt_idx = cmd.index("--format-output")
        assert cmd[fmt_idx + 1] == FOLDSEEK_OUTPUT_FORMAT


# ============================================================================
# Output format columns
# ============================================================================


class TestOutputFormat:
    def test_columns_include_structural_metrics(self):
        """PDB input enables structural metrics not available with 3Di-only."""
        assert "prob" in FOLDSEEK_OUTPUT_COLUMNS
        assert "lddt" in FOLDSEEK_OUTPUT_COLUMNS
        assert "alntmscore" in FOLDSEEK_OUTPUT_COLUMNS

    def test_format_string_matches_columns(self):
        """Output format string and column list must be in sync."""
        format_cols = FOLDSEEK_OUTPUT_FORMAT.split(",")
        assert format_cols == FOLDSEEK_OUTPUT_COLUMNS
