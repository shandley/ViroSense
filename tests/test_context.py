"""Tests for context module: ORF parsing and genomic windowing."""

import numpy as np
import pytest

from virosense.io.orfs import ORF, parse_orfs
from virosense.subcommands.context import ContextAnnotation, extract_orf_windows


# --- GFF3 Parsing ---


def test_parse_gff3(tmp_path):
    """Test parsing standard GFF3 format."""
    gff = tmp_path / "orfs.gff3"
    gff.write_text(
        "##gff-version 3\n"
        "contig_1\tprodigal\tCDS\t100\t500\t.\t+\t0\tID=orf_1;partial=00\n"
        "contig_1\tprodigal\tCDS\t600\t1200\t.\t-\t0\tID=orf_2;partial=01\n"
        "contig_2\tprodigal\tCDS\t50\t300\t.\t+\t0\tID=orf_3\n"
    )
    orfs = parse_orfs(gff)
    assert len(orfs) == 3
    assert orfs[0].orf_id == "orf_1"
    assert orfs[0].contig_id == "contig_1"
    assert orfs[0].start == 100
    assert orfs[0].end == 500
    assert orfs[0].strand == "+"
    assert orfs[1].strand == "-"
    assert orfs[2].contig_id == "contig_2"


def test_parse_gff3_no_id(tmp_path):
    """Test GFF3 without ID attribute generates ID from coordinates."""
    gff = tmp_path / "orfs.gff3"
    gff.write_text(
        "contig_1\tprodigal\tCDS\t100\t500\t.\t+\t0\tpartial=00\n"
    )
    orfs = parse_orfs(gff)
    assert len(orfs) == 1
    assert orfs[0].orf_id == "contig_1_100_500"


def test_parse_gff3_skips_non_cds(tmp_path):
    """Test that non-CDS features are skipped."""
    gff = tmp_path / "orfs.gff3"
    gff.write_text(
        "contig_1\tprodigal\tCDS\t100\t500\t.\t+\t0\tID=orf_1\n"
        "contig_1\tprodigal\tregion\t1\t5000\t.\t+\t0\tID=region_1\n"
        "contig_1\tprodigal\tgene\t100\t500\t.\t+\t0\tID=gene_1\n"
    )
    orfs = parse_orfs(gff)
    assert len(orfs) == 2  # CDS and gene, not region


def test_parse_gff3_comments_and_blank_lines(tmp_path):
    """Test that comments and blank lines are handled."""
    gff = tmp_path / "orfs.gff3"
    gff.write_text(
        "##gff-version 3\n"
        "# This is a comment\n"
        "\n"
        "contig_1\tprodigal\tCDS\t100\t500\t.\t+\t0\tID=orf_1\n"
    )
    orfs = parse_orfs(gff)
    assert len(orfs) == 1


# --- Prodigal FASTA Parsing ---


def test_parse_prodigal_fasta(tmp_path):
    """Test parsing prodigal protein FASTA output."""
    faa = tmp_path / "proteins.faa"
    faa.write_text(
        ">contig_1_1 # 3 # 1205 # 1 # ID=1_1;partial=00\n"
        "MKLTSSSS\n"
        "AAAAKKK\n"
        ">contig_1_2 # 1300 # 2100 # -1 # ID=1_2;partial=00\n"
        "MVVVVV\n"
    )
    orfs = parse_orfs(faa)
    assert len(orfs) == 2
    assert orfs[0].orf_id == "contig_1_1"
    assert orfs[0].contig_id == "contig_1"
    assert orfs[0].start == 3
    assert orfs[0].end == 1205
    assert orfs[0].strand == "+"
    assert orfs[0].protein_sequence == "MKLTSSSSAAAAKKK"
    assert orfs[1].strand == "-"
    assert orfs[1].protein_sequence == "MVVVVV"


# --- Protein FASTA Parsing ---


def test_parse_protein_fasta(tmp_path):
    """Test parsing simple protein FASTA (non-prodigal)."""
    faa = tmp_path / "proteins.fasta"
    faa.write_text(
        ">protein_1 some description\n"
        "MKLTSSSS\n"
        ">protein_2\n"
        "MVVVVV\n"
    )
    orfs = parse_orfs(faa)
    assert len(orfs) == 2
    assert orfs[0].orf_id == "protein_1"
    assert orfs[0].protein_sequence == "MKLTSSSS"
    # No coordinate info for plain FASTA
    assert orfs[0].start == 0
    assert orfs[0].end == 0


# --- Format Detection ---


def test_format_detection_gff3(tmp_path):
    """Test auto-detection of GFF3 by extension."""
    gff = tmp_path / "orfs.gff"
    gff.write_text("contig_1\tprodigal\tCDS\t100\t500\t.\t+\t0\tID=orf_1\n")
    orfs = parse_orfs(gff)
    assert len(orfs) == 1


def test_file_not_found():
    """Test FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        parse_orfs("/nonexistent/path.gff3")


# --- Window Extraction ---


def test_extract_orf_windows_basic():
    """Test basic window extraction around ORFs."""
    contigs = {"contig_1": "A" * 10000}
    orfs = [
        ORF(orf_id="orf_1", contig_id="contig_1", start=3000, end=4000, strand="+"),
    ]
    windows = extract_orf_windows(contigs, orfs, window_size=2000)
    assert len(windows) == 1
    assert "orf_1_w2000" in windows
    assert len(windows["orf_1_w2000"]) == 2000


def test_extract_orf_windows_clipping():
    """Test that windows are clipped to contig bounds."""
    contigs = {"contig_1": "A" * 500}
    orfs = [
        ORF(orf_id="orf_1", contig_id="contig_1", start=10, end=100, strand="+"),
    ]
    windows = extract_orf_windows(contigs, orfs, window_size=2000)
    assert len(windows) == 1
    # Window can't exceed contig length
    assert len(windows["orf_1_w2000"]) <= 500


def test_extract_orf_windows_short_contig_skipped():
    """Test that very short windows (< 100 bp) are skipped."""
    contigs = {"contig_1": "A" * 50}
    orfs = [
        ORF(orf_id="orf_1", contig_id="contig_1", start=10, end=40, strand="+"),
    ]
    windows = extract_orf_windows(contigs, orfs, window_size=2000)
    assert len(windows) == 0  # 50 bp < 100 bp minimum


def test_extract_orf_windows_missing_contig():
    """Test that ORFs with missing contigs are skipped."""
    contigs = {"contig_1": "A" * 5000}
    orfs = [
        ORF(orf_id="orf_1", contig_id="contig_1", start=1000, end=2000, strand="+"),
        ORF(orf_id="orf_2", contig_id="contig_99", start=100, end=500, strand="+"),
    ]
    windows = extract_orf_windows(contigs, orfs, window_size=2000)
    assert len(windows) == 1
    assert "orf_1_w2000" in windows


def test_extract_orf_windows_multiple():
    """Test extraction of multiple windows."""
    contigs = {
        "contig_1": "A" * 10000,
        "contig_2": "C" * 8000,
    }
    orfs = [
        ORF(orf_id="orf_1", contig_id="contig_1", start=1000, end=2000, strand="+"),
        ORF(orf_id="orf_2", contig_id="contig_1", start=5000, end=6000, strand="-"),
        ORF(orf_id="orf_3", contig_id="contig_2", start=3000, end=4000, strand="+"),
    ]
    windows = extract_orf_windows(contigs, orfs, window_size=2000)
    assert len(windows) == 3


def test_extract_orf_windows_edge_start():
    """Test window at the start of a contig."""
    contigs = {"contig_1": "A" * 10000}
    orfs = [
        ORF(orf_id="orf_1", contig_id="contig_1", start=50, end=200, strand="+"),
    ]
    windows = extract_orf_windows(contigs, orfs, window_size=2000)
    assert len(windows) == 1
    # Midpoint = 125, half = 1000, so start clips to 0
    assert len(windows["orf_1_w2000"]) >= 100
