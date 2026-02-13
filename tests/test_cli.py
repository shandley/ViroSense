"""Tests for the virosense CLI."""

from click.testing import CliRunner

from virosense.cli import main


def test_main_help():
    """Test that --help works for main command."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "virosense" in result.output
    assert "detect" in result.output
    assert "context" in result.output
    assert "cluster" in result.output
    assert "classify" in result.output


def test_version():
    """Test that --version works."""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_detect_help():
    """Test detect subcommand help."""
    runner = CliRunner()
    result = runner.invoke(main, ["detect", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--backend" in result.output
    assert "--threshold" in result.output


def test_context_help():
    """Test context subcommand help."""
    runner = CliRunner()
    result = runner.invoke(main, ["context", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--orfs" in result.output
    assert "--window" in result.output


def test_cluster_help():
    """Test cluster subcommand help."""
    runner = CliRunner()
    result = runner.invoke(main, ["cluster", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--mode" in result.output
    assert "--algorithm" in result.output


def test_classify_help():
    """Test classify subcommand help."""
    runner = CliRunner()
    result = runner.invoke(main, ["classify", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--labels" in result.output
    assert "--task" in result.output
    assert "--epochs" in result.output
