"""Tests for structure acquisition module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from structure import (
    StructureBatch,
    StructureResult,
    _extract_mean_plddt,
    _write_fasta,
    acquire_structures,
    batch_fetch_alphafold,
    check_colabfold,
    fetch_alphafold_structure,
    run_colabfold,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_pdb(tmp_path):
    """Create a minimal PDB file with B-factors (pLDDT scores)."""
    pdb_content = """\
ATOM      1  N   ALA A   1      27.340  24.430   2.614  1.00 89.50           N
ATOM      2  CA  ALA A   1      26.266  25.413   2.842  1.00 92.30           C
ATOM      3  C   ALA A   1      26.913  26.639   3.531  1.00 87.10           C
ATOM      4  O   ALA A   1      27.886  26.463   4.263  1.00 78.40           O
ATOM      5  CB  ALA A   1      25.112  24.880   3.649  1.00 95.20           C
END
"""
    pdb_path = tmp_path / "test.pdb"
    pdb_path.write_text(pdb_content)
    return pdb_path


@pytest.fixture
def alphafold_api_response():
    """Mock AlphaFold API response."""
    return [
        {
            "entryId": "AF-P00520-F1",
            "gene": "ABL1",
            "uniprotAccession": "P00520",
            "uniprotId": "ABL1_MOUSE",
            "uniprotDescription": "Tyrosine-protein kinase ABL1",
            "taxId": 10090,
            "organismScientificName": "Mus musculus",
            "uniprotStart": 1,
            "uniprotEnd": 1123,
            "uniprotSequence": "MLEICLKLVG...",
            "modelCreatedDate": "2022-06-01",
            "latestVersion": 4,
            "allVersions": [1, 2, 3, 4],
            "isReviewed": True,
            "isReferenceProteome": True,
            "cifUrl": "https://alphafold.ebi.ac.uk/files/AF-P00520-F1-model_v4.cif",
            "bcifUrl": "https://alphafold.ebi.ac.uk/files/AF-P00520-F1-model_v4.bcif",
            "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P00520-F1-model_v4.pdb",
            "paeImageUrl": "https://alphafold.ebi.ac.uk/files/AF-P00520-F1-predicted_aligned_error_v4.png",
            "paeDocUrl": "https://alphafold.ebi.ac.uk/files/AF-P00520-F1-predicted_aligned_error_v4.json",
        }
    ]


# ============================================================================
# Tests: StructureResult / StructureBatch
# ============================================================================


class TestStructureResult:
    def test_default_values(self):
        r = StructureResult(protein_id="test")
        assert r.protein_id == "test"
        assert r.pdb_path is None
        assert r.source == ""
        assert r.error is None

    def test_successful_result(self, tmp_path):
        pdb = tmp_path / "test.pdb"
        pdb.write_text("ATOM ...")
        r = StructureResult(
            protein_id="prot1",
            pdb_path=pdb,
            source="alphafold_db",
            uniprot_accession="P12345",
            plddt_mean=85.5,
        )
        assert r.pdb_path.exists()
        assert r.source == "alphafold_db"
        assert r.plddt_mean == 85.5

    def test_failed_result(self):
        r = StructureResult(
            protein_id="prot1",
            source="failed",
            error="Not found",
        )
        assert r.pdb_path is None
        assert r.source == "failed"


class TestStructureBatch:
    def test_empty_batch(self):
        b = StructureBatch()
        assert b.n_total == 0
        assert b.n_with_structure == 0

    def test_batch_counts(self):
        b = StructureBatch(n_alphafold=5, n_colabfold=3, n_failed=2)
        b.results = {f"p{i}": StructureResult(protein_id=f"p{i}") for i in range(10)}
        assert b.n_total == 10
        assert b.n_with_structure == 8
        assert "8/10" in b.summary()

    def test_summary_format(self):
        b = StructureBatch(n_alphafold=2, n_colabfold=1, n_failed=0)
        b.results = {f"p{i}": StructureResult(protein_id=f"p{i}") for i in range(3)}
        s = b.summary()
        assert "AlphaFold DB: 2" in s
        assert "ColabFold: 1" in s
        assert "failed: 0" in s


# ============================================================================
# Tests: AlphaFold DB Lookup
# ============================================================================


class TestFetchAlphafoldStructure:
    def test_successful_download(self, tmp_path, alphafold_api_response, sample_pdb):
        """Successful AlphaFold DB download."""
        pdb_content = sample_pdb.read_bytes()

        with patch("structure.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            # API metadata response
            api_resp = MagicMock()
            api_resp.status_code = 200
            api_resp.json.return_value = alphafold_api_response

            # PDB file download response
            pdb_resp = MagicMock()
            pdb_resp.status_code = 200
            pdb_resp.content = pdb_content

            mock_client.get.side_effect = [api_resp, pdb_resp]

            result = fetch_alphafold_structure("P00520", tmp_path)

        assert result.source == "alphafold_db"
        assert result.pdb_path is not None
        assert result.pdb_path.exists()
        assert result.uniprot_accession == "P00520"

    def test_not_found_returns_failed(self, tmp_path):
        """404 returns failed StructureResult."""
        with patch("structure.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            resp = MagicMock()
            resp.status_code = 404
            mock_client.get.return_value = resp

            result = fetch_alphafold_structure("NOTREAL", tmp_path)

        assert result.source == "failed"
        assert "No AlphaFold prediction" in result.error

    def test_cached_file_not_redownloaded(self, tmp_path, alphafold_api_response):
        """Existing PDB file is not re-downloaded."""
        cached_pdb = tmp_path / "AF-P00520-F1-model_v4.pdb"
        cached_pdb.write_text("ATOM  cached content")

        with patch("structure.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            api_resp = MagicMock()
            api_resp.status_code = 200
            api_resp.json.return_value = alphafold_api_response
            mock_client.get.return_value = api_resp

            result = fetch_alphafold_structure("P00520", tmp_path)

        assert result.source == "alphafold_db"
        # Only one GET call (API metadata), no PDB download
        assert mock_client.get.call_count == 1

    def test_empty_response(self, tmp_path):
        """Empty API response returns failed."""
        with patch("structure.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = []
            resp.raise_for_status = MagicMock()
            mock_client.get.return_value = resp

            result = fetch_alphafold_structure("P99999", tmp_path)

        assert result.source == "failed"
        assert "Empty prediction" in result.error


class TestBatchFetchAlphafold:
    def test_batch_multiple_accessions(self, tmp_path):
        """Batch fetch processes multiple accessions."""
        with patch("structure.fetch_alphafold_structure") as mock_fetch:
            mock_fetch.side_effect = [
                StructureResult(
                    protein_id="P00520",
                    pdb_path=tmp_path / "a.pdb",
                    source="alphafold_db",
                ),
                StructureResult(
                    protein_id="NOTREAL",
                    source="failed",
                    error="Not found",
                ),
            ]

            results = batch_fetch_alphafold(["P00520", "NOTREAL"], tmp_path)

        assert len(results) == 2
        assert results["P00520"].source == "alphafold_db"
        assert results["NOTREAL"].source == "failed"


# ============================================================================
# Tests: ColabFold
# ============================================================================


class TestCheckColabfold:
    def test_available(self):
        with patch("structure.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            available, _msg = check_colabfold()
        assert available is True

    def test_not_found(self):
        with patch("structure.subprocess.run", side_effect=FileNotFoundError):
            available, msg = check_colabfold()
        assert available is False
        assert "not found" in msg


class TestRunColabfold:
    def test_colabfold_not_available(self, tmp_path):
        """Raises FileNotFoundError when colabfold_batch not installed."""
        fasta = tmp_path / "input.fasta"
        fasta.write_text(">prot1\nMKTAYI\n")

        with patch("structure.check_colabfold", return_value=(False, "not found")):
            with pytest.raises(FileNotFoundError, match="ColabFold not available"):
                run_colabfold(fasta, tmp_path / "output")

    def test_colabfold_command_construction(self, tmp_path):
        """Verify correct CLI args are passed to colabfold_batch."""
        fasta = tmp_path / "input.fasta"
        fasta.write_text(">prot1\nMKTAYI\n")
        out_dir = tmp_path / "output"

        with (
            patch("structure.check_colabfold", return_value=(True, "ok")),
            patch("structure.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)

            # Need to create empty output dir for glob to work
            out_dir.mkdir(parents=True)

            from structure import run_colabfold
            run_colabfold(fasta, out_dir, num_models=3, num_recycles=5)

        args = mock_run.call_args[0][0]
        assert args[0] == "colabfold_batch"
        assert "--num-models" in args
        assert "3" in args
        assert "--num-recycle" in args
        assert "5" in args


class TestParseColabfoldOutput:
    def test_parse_relaxed_pdbs(self, tmp_path):
        """Parse ColabFold output with relaxed PDB files."""
        from structure import _parse_colabfold_output

        # Create mock ColabFold output files
        (tmp_path / "prot1_relaxed_rank_001_alphafold2_model_1.pdb").write_text(
            "ATOM      1  N   ALA A   1      0.0  0.0  0.0  1.00 90.00           N\nEND\n"
        )
        (tmp_path / "prot1_relaxed_rank_002_alphafold2_model_2.pdb").write_text(
            "ATOM  rank2\nEND\n"
        )
        (tmp_path / "prot2_unrelaxed_rank_001_alphafold2_model_1.pdb").write_text(
            "ATOM      1  N   ALA A   1      0.0  0.0  0.0  1.00 85.00           N\nEND\n"
        )

        results = _parse_colabfold_output(tmp_path)

        assert len(results) == 2
        assert results["prot1"].source == "colabfold"
        assert "rank_001" in results["prot1"].pdb_path.name
        assert results["prot2"].source == "colabfold"


# ============================================================================
# Tests: Two-Pass Acquisition
# ============================================================================


class TestAcquireStructures:
    def test_alphafold_only(self, tmp_path):
        """Proteins with UniProt accessions use AlphaFold DB only."""
        sequences = {"prot1": "MKTAYI", "prot2": "GKLDFI"}
        uniprot_map = {"prot1": "P00520", "prot2": "Q9Y6K9"}

        with patch("structure.fetch_alphafold_structure") as mock_fetch:
            mock_fetch.side_effect = [
                StructureResult(
                    protein_id="P00520",
                    pdb_path=tmp_path / "a.pdb",
                    source="alphafold_db",
                    plddt_mean=90.0,
                ),
                StructureResult(
                    protein_id="Q9Y6K9",
                    pdb_path=tmp_path / "b.pdb",
                    source="alphafold_db",
                    plddt_mean=85.0,
                ),
            ]

            batch = acquire_structures(
                sequences, tmp_path, uniprot_map=uniprot_map, skip_colabfold=True
            )

        assert batch.n_alphafold == 2
        assert batch.n_colabfold == 0
        assert batch.n_failed == 0

    def test_fallback_to_colabfold(self, tmp_path):
        """Proteins without accessions fall through to ColabFold."""
        sequences = {"prot1": "MKTAYI", "novel": "GKLDFI"}
        uniprot_map = {"prot1": "P00520"}

        with (
            patch("structure.fetch_alphafold_structure") as mock_fetch,
            patch("structure.run_colabfold") as mock_colabfold,
        ):
            mock_fetch.return_value = StructureResult(
                protein_id="P00520",
                pdb_path=tmp_path / "a.pdb",
                source="alphafold_db",
            )
            mock_colabfold.return_value = {
                "novel": StructureResult(
                    protein_id="novel",
                    pdb_path=tmp_path / "b.pdb",
                    source="colabfold",
                )
            }

            batch = acquire_structures(sequences, tmp_path, uniprot_map=uniprot_map)

        assert batch.n_alphafold == 1
        assert batch.n_colabfold == 1

    def test_skip_colabfold(self, tmp_path):
        """skip_colabfold=True marks remaining as failed."""
        sequences = {"novel1": "MKTAYI", "novel2": "GKLDFI"}

        batch = acquire_structures(
            sequences, tmp_path, skip_colabfold=True
        )

        assert batch.n_alphafold == 0
        assert batch.n_colabfold == 0
        assert batch.n_failed == 2
        for r in batch.results.values():
            assert r.source == "failed"
            assert "skipped" in r.error

    def test_no_uniprot_map(self, tmp_path):
        """All proteins go to ColabFold when no uniprot_map provided."""
        sequences = {"prot1": "MKTAYI"}

        with patch("structure.run_colabfold") as mock_colabfold:
            mock_colabfold.return_value = {
                "prot1": StructureResult(
                    protein_id="prot1",
                    pdb_path=tmp_path / "a.pdb",
                    source="colabfold",
                )
            }

            batch = acquire_structures(sequences, tmp_path)

        assert batch.n_alphafold == 0
        assert batch.n_colabfold == 1

    def test_colabfold_failure_graceful(self, tmp_path):
        """ColabFold failure doesn't crash, marks as failed."""
        sequences = {"prot1": "MKTAYI"}

        with patch("structure.run_colabfold", side_effect=RuntimeError("GPU error")):
            batch = acquire_structures(sequences, tmp_path)

        assert batch.n_failed == 1
        assert "GPU error" in batch.results["prot1"].error

    def test_alphafold_miss_goes_to_colabfold(self, tmp_path):
        """AlphaFold DB miss falls through to ColabFold."""
        sequences = {"prot1": "MKTAYI"}
        uniprot_map = {"prot1": "NOTREAL"}

        with (
            patch("structure.fetch_alphafold_structure") as mock_fetch,
            patch("structure.run_colabfold") as mock_colabfold,
        ):
            mock_fetch.return_value = StructureResult(
                protein_id="NOTREAL",
                source="failed",
                error="Not found",
            )
            mock_colabfold.return_value = {
                "prot1": StructureResult(
                    protein_id="prot1",
                    pdb_path=tmp_path / "a.pdb",
                    source="colabfold",
                )
            }

            batch = acquire_structures(sequences, tmp_path, uniprot_map=uniprot_map)

        assert batch.n_alphafold == 0
        assert batch.n_colabfold == 1


# ============================================================================
# Tests: Utilities
# ============================================================================


class TestExtractMeanPlddt:
    def test_valid_pdb(self, sample_pdb):
        plddt = _extract_mean_plddt(sample_pdb)
        # Mean of 89.50, 92.30, 87.10, 78.40, 95.20 = 88.50
        assert plddt is not None
        assert abs(plddt - 88.50) < 0.01

    def test_nonexistent_file(self, tmp_path):
        plddt = _extract_mean_plddt(tmp_path / "nonexistent.pdb")
        assert plddt is None

    def test_empty_pdb(self, tmp_path):
        pdb = tmp_path / "empty.pdb"
        pdb.write_text("REMARK empty file\nEND\n")
        plddt = _extract_mean_plddt(pdb)
        assert plddt is None


class TestWriteFasta:
    def test_writes_correctly(self, tmp_path):
        path = tmp_path / "out.fasta"
        _write_fasta({"p1": "MKTAYI", "p2": "GKLDFI"}, path)
        content = path.read_text()
        assert ">p1\nMKTAYI\n" in content
        assert ">p2\nGKLDFI\n" in content

    def test_empty_dict(self, tmp_path):
        path = tmp_path / "empty.fasta"
        _write_fasta({}, path)
        assert path.read_text() == ""
