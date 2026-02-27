"""Tests for functional category classification module."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from categories import (
    ALL_CATEGORIES,
    CATEGORY_DESCRIPTIONS,
    AnnotationEvidence,
    classify_by_go_bp,
    classify_by_go_mf,
    classify_by_keywords,
    classify_by_pfam,
    classify_by_superfamily,
    classify_protein,
    get_classification_source,
)


# ============================================================================
# classify_by_keywords
# ============================================================================


class TestClassifyByKeywords:
    def test_structural(self):
        assert classify_by_keywords("Major capsid protein") == "structural"
        assert classify_by_keywords("Tail fiber protein") == "structural"
        assert classify_by_keywords("Baseplate wedge subunit") == "structural"

    def test_replication(self):
        assert classify_by_keywords("DNA polymerase") == "replication"
        assert classify_by_keywords("RNA-dependent RNA polymerase") == "replication"
        assert classify_by_keywords("DNA helicase") == "replication"

    def test_protease(self):
        assert classify_by_keywords("3C-like proteinase") == "protease"
        assert classify_by_keywords("Main protease Mpro") == "protease"

    def test_nuclease(self):
        assert classify_by_keywords("DNA endonuclease") == "nuclease"
        assert classify_by_keywords("Phage integrase") == "nuclease"

    def test_packaging(self):
        assert classify_by_keywords("Large terminase subunit") == "packaging"
        assert classify_by_keywords("Scaffold protein") == "packaging"

    def test_host_interaction(self):
        assert classify_by_keywords("Interferon antagonist V protein") == "host_interaction"
        assert classify_by_keywords("Anti-CRISPR protein") == "host_interaction"
        assert classify_by_keywords("Ocr anti-restriction") == "host_interaction"

    def test_regulatory(self):
        assert classify_by_keywords("Transcription factor") == "regulatory"
        assert classify_by_keywords("CI repressor") == "regulatory"

    def test_lysis(self):
        assert classify_by_keywords("Endolysin") == "lysis"
        assert classify_by_keywords("Holin class II") == "lysis"

    def test_movement(self):
        assert classify_by_keywords("Movement protein MP") == "movement"

    def test_entry(self):
        assert classify_by_keywords("Viral entry protein") == "entry"
        assert classify_by_keywords("Receptor binding protein") == "entry"

    def test_unknown(self):
        assert classify_by_keywords("Hypothetical protein") == "unknown"
        assert classify_by_keywords("DUF1234 domain-containing") == "unknown"
        assert classify_by_keywords("Something completely novel") == "unknown"

    def test_gene_name_contributes(self):
        assert classify_by_keywords("Unknown protein", gene="endolysin") == "lysis"

    def test_inhibitor_context(self):
        assert classify_by_keywords("Polymerase inhibitor") == "host_interaction"
        assert classify_by_keywords("Host nuclease inhibitor") == "host_interaction"

    def test_inhibitor_context_not_triggered_for_actual_function(self):
        assert classify_by_keywords("DNA polymerase") == "replication"


# ============================================================================
# classify_by_pfam
# ============================================================================


class TestClassifyByPfam:
    def test_structural(self):
        assert classify_by_pfam("Major capsid protein VP1") == "structural"
        assert classify_by_pfam("Tail protein gp37") == "structural"

    def test_replication(self):
        assert classify_by_pfam("DNA polymerase family B") == "replication"
        assert classify_by_pfam("Superfamily 2 helicase") == "replication"

    def test_lysis(self):
        assert classify_by_pfam("T4-type lysozyme") == "lysis"
        assert classify_by_pfam("Peptidoglycan binding domain") == "lysis"

    def test_empty_returns_none(self):
        assert classify_by_pfam("") is None
        assert classify_by_pfam(None) is None  # type: ignore[arg-type]

    def test_no_match_returns_none(self):
        assert classify_by_pfam("Completely unrelated domain XYZ") is None


# ============================================================================
# classify_by_go_bp / classify_by_go_mf
# ============================================================================


class TestClassifyByGO:
    def test_go_bp_replication(self):
        assert classify_by_go_bp("viral genome replication") == "replication"

    def test_go_bp_lysis(self):
        assert classify_by_go_bp("host cell lysis") == "lysis"

    def test_go_bp_entry(self):
        assert classify_by_go_bp("virion attachment to host cell") == "entry"

    def test_go_mf_protease(self):
        assert classify_by_go_mf("cysteine-type peptidase activity") == "protease"

    def test_go_mf_structural(self):
        assert classify_by_go_mf("structural constituent of virion") == "structural"

    def test_empty_returns_none(self):
        assert classify_by_go_bp("") is None
        assert classify_by_go_mf("") is None


# ============================================================================
# classify_by_superfamily
# ============================================================================


class TestClassifyBySuperfamily:
    def test_structural(self):
        assert classify_by_superfamily("Viral glycoprotein domain") == "structural"

    def test_replication(self):
        assert classify_by_superfamily("DNA/RNA polymerases") == "replication"

    def test_no_match(self):
        assert classify_by_superfamily("Unknown fold") is None


# ============================================================================
# classify_protein (hierarchical)
# ============================================================================


class TestClassifyProtein:
    def test_pfam_takes_priority(self):
        """Pfam evidence should override keyword classification."""
        evidence = AnnotationEvidence(pfam="Major capsid protein")
        # Description says "polymerase" but Pfam says "capsid"
        result = classify_protein("polymerase-like protein", evidence=evidence)
        assert result == "structural"

    def test_superfamily_over_keywords(self):
        evidence = AnnotationEvidence(superfamily="DNA/RNA polymerases")
        result = classify_protein("Unknown protein", evidence=evidence)
        assert result == "replication"

    def test_go_bp_used(self):
        evidence = AnnotationEvidence(go_bp="viral genome replication")
        result = classify_protein("Hypothetical protein", evidence=evidence)
        assert result == "replication"

    def test_falls_back_to_keywords(self):
        result = classify_protein("DNA helicase domain protein")
        assert result == "replication"

    def test_no_evidence_uses_keywords(self):
        result = classify_protein("Endolysin", evidence=None)
        assert result == "lysis"

    def test_empty_evidence_uses_keywords(self):
        evidence = AnnotationEvidence()
        result = classify_protein("Tail fiber protein", evidence=evidence)
        assert result == "structural"


# ============================================================================
# get_classification_source
# ============================================================================


class TestGetClassificationSource:
    def test_pfam_source(self):
        evidence = AnnotationEvidence(pfam="Major capsid protein")
        category, source = get_classification_source("x", evidence=evidence)
        assert category == "structural"
        assert source.startswith("pfam:")

    def test_keyword_source(self):
        category, source = get_classification_source("DNA helicase")
        assert category == "replication"
        assert source == "keywords"

    def test_go_bp_source(self):
        evidence = AnnotationEvidence(go_bp="viral genome replication")
        category, source = get_classification_source("Unknown", evidence=evidence)
        assert category == "replication"
        assert source.startswith("go_bp:")


# ============================================================================
# AnnotationEvidence
# ============================================================================


class TestAnnotationEvidence:
    def test_empty_has_no_evidence(self):
        assert AnnotationEvidence().has_evidence() is False

    def test_pfam_has_evidence(self):
        assert AnnotationEvidence(pfam="Something").has_evidence() is True

    def test_go_has_evidence(self):
        assert AnnotationEvidence(go_bp="Something").has_evidence() is True


# ============================================================================
# Constants
# ============================================================================


class TestConstants:
    def test_all_categories_complete(self):
        assert len(ALL_CATEGORIES) == 11
        assert "unknown" in ALL_CATEGORIES
        assert "structural" in ALL_CATEGORIES

    def test_descriptions_cover_all(self):
        for cat in ALL_CATEGORIES:
            assert cat in CATEGORY_DESCRIPTIONS
