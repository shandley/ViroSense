#!/usr/bin/env python3
"""Comprehensive ~500-sequence panel for codon periodicity & functional clustering.

Two hypotheses:
  1. Codon periodicity universality: Per-position Evo2 embeddings encode the
     triplet genetic code across ALL domains of life.
  2. Functional clustering: Mean-pooled embeddings cluster by protein identity
     (gene family) across species.

Three components:
  A. Universality panel (~170 sequences): 1 CDS per species, diverse genes,
     maximum taxonomic coverage.
  B. Functional clustering panel (10 families × 30 orthologs = 300 sequences):
     Same gene from many species → should cluster by gene family.
  C. Negative controls for clustering (~30 sequences): Random CDS from 30
     species, each a DIFFERENT gene not in the 10 families → noise floor.

Usage:
    # Step 1: Download sequences from NCBI
    uv run python scripts/codon_periodicity_panel.py download

    # Step 2: Extract per-position embeddings via NIM API
    uv run python scripts/codon_periodicity_panel.py extract

    # Step 3: Analyze periodicity + clustering
    uv run python scripts/codon_periodicity_panel.py analyze
"""

from __future__ import annotations

import json
import re
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np

# ── Output directory ──
OUT_DIR = Path("results/codon_periodicity_panel")
FASTA_DIR = OUT_DIR / "fasta"
EMB_DIR = OUT_DIR / "embeddings"
FIG_DIR = OUT_DIR / "figures"

# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT A: UNIVERSALITY PANEL (~170 sequences)
# One CDS per species, diverse genes, maximum taxonomic coverage.
# Each uses a DIFFERENT gene to avoid confounding universality with gene identity.
# ═══════════════════════════════════════════════════════════════════════════════

COMPONENT_A: list[dict] = [
    # ───────────────────────────────────────────────────────────────
    # ARCHAEA — Euryarchaeota (5)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_haloferax_rpoB",
        "component": "A", "category": "archaea_euryarchaeota",
        "lineage": "Archaea; Euryarchaeota; Halobacteria",
        "species": "Haloferax volcanii",
        "gene": "RNA polymerase subunit B",
        "accession": "AAC97368",
        "type": "protein_cds",
        "gc_approx": 65,
    },
    {
        "name": "a_methanocaldococcus_ef2",
        "component": "A", "category": "archaea_euryarchaeota",
        "lineage": "Archaea; Euryarchaeota; Methanococci",
        "species": "Methanocaldococcus jannaschii",
        "gene": "elongation factor 2",
        "accession": "NC_000909",
        "cds_start": 213879, "cds_end": 216092,
        "type": "genomic_region",
        "gc_approx": 31,
    },
    {
        "name": "a_methanosarcina_mcrA",
        "component": "A", "category": "archaea_euryarchaeota",
        "lineage": "Archaea; Euryarchaeota; Methanomicrobia",
        "species": "Methanosarcina acetivorans",
        "gene": "methyl-coenzyme M reductase alpha",
        "accession": "AAM07920",
        "type": "protein_cds",
        "gc_approx": 43,
    },
    {
        "name": "a_thermococcus_hsp60",
        "component": "A", "category": "archaea_euryarchaeota",
        "lineage": "Archaea; Euryarchaeota; Thermococci",
        "species": "Thermococcus kodakarensis",
        "gene": "chaperonin GroEL/HSP60",
        "accession": "BAD85109",
        "type": "protein_cds",
        "gc_approx": 52,
    },
    {
        "name": "a_pyrococcus_gapdh",
        "component": "A", "category": "archaea_euryarchaeota",
        "lineage": "Archaea; Euryarchaeota; Thermococci",
        "species": "Pyrococcus furiosus",
        "gene": "glyceraldehyde-3-phosphate dehydrogenase",
        "accession": "AAL81075",
        "type": "protein_cds",
        "gc_approx": 41,
    },
    # ───────────────────────────────────────────────────────────────
    # ARCHAEA — Crenarchaeota (5)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_sulfolobus_orc1",
        "component": "A", "category": "archaea_crenarchaeota",
        "lineage": "Archaea; Crenarchaeota; Thermoprotei",
        "species": "Sulfolobus solfataricus",
        "gene": "cell division control protein 6 / Orc1",
        "accession": "AAK42240",
        "type": "protein_cds",
        "gc_approx": 36,
    },
    {
        "name": "a_pyrobaculum_ef2",
        "component": "A", "category": "archaea_crenarchaeota",
        "lineage": "Archaea; Crenarchaeota; Thermoprotei",
        "species": "Pyrobaculum aerophilum",
        "gene": "elongation factor 2",
        "accession": "AAL63165",
        "type": "protein_cds",
        "gc_approx": 51,
    },
    {
        "name": "a_thermoproteus_ef1a",
        "component": "A", "category": "archaea_crenarchaeota",
        "lineage": "Archaea; Crenarchaeota; Thermoprotei",
        "species": "Thermoproteus tenax",
        "gene": "elongation factor 1-alpha",
        "accession": "CCC81032",
        "type": "protein_cds",
        "gc_approx": 57,
    },
    {
        "name": "a_aeropyrum_atpA",
        "component": "A", "category": "archaea_crenarchaeota",
        "lineage": "Archaea; Crenarchaeota; Thermoprotei; Desulfurococcales",
        "species": "Aeropyrum pernix",
        "gene": "ATP synthase subunit A",
        "accession": "BAA79593",
        "type": "protein_cds",
        "gc_approx": 56,
    },
    {
        "name": "a_metallosphaera_topA",
        "component": "A", "category": "archaea_crenarchaeota",
        "lineage": "Archaea; Crenarchaeota; Thermoprotei; Sulfolobales",
        "species": "Metallosphaera sedula",
        "gene": "reverse gyrase",
        "accession": "ABP94662",
        "type": "protein_cds",
        "gc_approx": 46,
    },
    # ───────────────────────────────────────────────────────────────
    # ARCHAEA — Thaumarchaeota (3)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_nitrosopumilus_amoA",
        "component": "A", "category": "archaea_thaumarchaeota",
        "lineage": "Archaea; Thaumarchaeota",
        "species": "Nitrosopumilus maritimus",
        "gene": "ammonia monooxygenase subunit A",
        "accession": "ABZ10151",
        "type": "protein_cds",
        "gc_approx": 34,
    },
    {
        "name": "a_cenarchaeum_ef2",
        "component": "A", "category": "archaea_thaumarchaeota",
        "lineage": "Archaea; Thaumarchaeota",
        "species": "Cenarchaeum symbiosum",
        "gene": "elongation factor 2",
        "accession": "ABK78556",
        "type": "protein_cds",
        "gc_approx": 57,
    },
    {
        "name": "a_nitrosocosmicus_ureC",
        "component": "A", "category": "archaea_thaumarchaeota",
        "lineage": "Archaea; Thaumarchaeota; Nitrososphaeria",
        "species": "Nitrosocosmicus oleophilus",
        "gene": "urease alpha subunit",
        "accession": "AYF55891",
        "type": "protein_cds",
        "gc_approx": 39,
    },
    # ───────────────────────────────────────────────────────────────
    # ARCHAEA — Asgardarchaeota (3)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_lokiarchaeum_actinLike",
        "component": "A", "category": "archaea_asgard",
        "lineage": "Archaea; Asgardarchaeota; Lokiarchaeia",
        "species": "Candidatus Lokiarchaeota archaeon",
        "gene": "actin-like protein (profilin)",
        "accession": "KKK40403",
        "type": "protein_cds",
        "gc_approx": 32,
    },
    {
        "name": "a_thorarchaeum_tubulin",
        "component": "A", "category": "archaea_asgard",
        "lineage": "Archaea; Asgardarchaeota; Thorarchaeia",
        "species": "Candidatus Thorarchaeota archaeon",
        "gene": "tubulin-like protein",
        "accession": "OLS27519",
        "type": "protein_cds",
        "gc_approx": 38,
    },
    {
        "name": "a_heimdallarchaeum_ef1a",
        "component": "A", "category": "archaea_asgard",
        "lineage": "Archaea; Asgardarchaeota; Heimdallarchaeia",
        "species": "Candidatus Heimdallarchaeota archaeon",
        "gene": "elongation factor 1-alpha",
        "accession": "OLS18549",
        "type": "protein_cds",
        "gc_approx": 36,
    },
    # ───────────────────────────────────────────────────────────────
    # ARCHAEA — DPANN (3)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_nanoarchaeum_rpoB",
        "component": "A", "category": "archaea_dpann",
        "lineage": "Archaea; DPANN; Nanoarchaeota",
        "species": "Nanoarchaeum equitans",
        "gene": "RNA polymerase subunit B",
        "accession": "AAR39199",
        "type": "protein_cds",
        "gc_approx": 32,
    },
    {
        "name": "a_micrarchaeum_rpsL",
        "component": "A", "category": "archaea_dpann",
        "lineage": "Archaea; DPANN; Micrarchaeota",
        "species": "Candidatus Micrarchaeum acidiphilum",
        "gene": "ribosomal protein S12",
        "accession": "WP_018157650",
        "type": "protein_cds",
        "gc_approx": 39,
    },
    {
        "name": "a_iainarchaeum_dnaB",
        "component": "A", "category": "archaea_dpann",
        "lineage": "Archaea; DPANN; Iainarchaeota",
        "species": "Candidatus Iainarchaeum andersonii",
        "gene": "replicative DNA helicase",
        "accession": "KUK97093",
        "type": "protein_cds",
        "gc_approx": 30,
    },
    # ───────────────────────────────────────────────────────────────
    # ARCHAEA — Deep-branching (3): Korarchaeota, Bathyarchaeota, etc.
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_korarchaeum_rpoB",
        "component": "A", "category": "archaea_deep",
        "lineage": "Archaea; Korarchaeota",
        "species": "Candidatus Korarchaeum cryptofilum",
        "gene": "RNA polymerase subunit B",
        "accession": "ACB07166",
        "type": "protein_cds",
        "gc_approx": 49,
    },
    {
        "name": "a_bathyarchaeum_mcrA",
        "component": "A", "category": "archaea_deep",
        "lineage": "Archaea; Bathyarchaeota",
        "species": "Candidatus Bathyarchaeota archaeon",
        "gene": "methyl-coenzyme M reductase alpha-like",
        "accession": "OPX73920",
        "type": "protein_cds",
        "gc_approx": 37,
    },
    {
        "name": "a_geoarchaeum_atpA",
        "component": "A", "category": "archaea_deep",
        "lineage": "Archaea; Euryarchaeota; Thermoplasmatales",
        "species": "Candidatus Aciduliprofundum boonei",
        "gene": "ATP synthase subunit A",
        "accession": "EDY35789",
        "type": "protein_cds",
        "gc_approx": 38,
    },
    # ───────────────────────────────────────────────────────────────
    # ARCHAEA — Halophiles (3)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_halobacterium_bop",
        "component": "A", "category": "archaea_halophile",
        "lineage": "Archaea; Euryarchaeota; Halobacteria",
        "species": "Halobacterium salinarum",
        "gene": "bacteriorhodopsin",
        "accession": "V00474",
        "type": "full_cds",
        "gc_approx": 66,
    },
    {
        "name": "a_haloquadratum_gasBag",
        "component": "A", "category": "archaea_halophile",
        "lineage": "Archaea; Euryarchaeota; Halobacteria",
        "species": "Haloquadratum walsbyi",
        "gene": "gas vesicle protein GvpA",
        "accession": "CAJ51205",
        "type": "protein_cds",
        "gc_approx": 48,
    },
    {
        "name": "a_natronomonas_rhodopsin",
        "component": "A", "category": "archaea_halophile",
        "lineage": "Archaea; Euryarchaeota; Halobacteria",
        "species": "Natronomonas pharaonis",
        "gene": "sensory rhodopsin II",
        "accession": "AAG19397",
        "type": "protein_cds",
        "gc_approx": 64,
    },
    # ───────────────────────────────────────────────────────────────
    # BACTERIA — Gammaproteobacteria (3)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_ecoli_dnaK",
        "component": "A", "category": "bacteria_gammaproteo",
        "lineage": "Bacteria; Pseudomonadota; Gammaproteobacteria",
        "species": "Escherichia coli K-12",
        "gene": "DnaK (HSP70 chaperone)",
        "accession": "NP_414555",
        "type": "protein_cds",
        "gc_approx": 51,
    },
    {
        "name": "a_pseudomonas_oprF",
        "component": "A", "category": "bacteria_gammaproteo",
        "lineage": "Bacteria; Pseudomonadota; Gammaproteobacteria",
        "species": "Pseudomonas aeruginosa",
        "gene": "outer membrane porin OprF",
        "accession": "NP_250468",
        "type": "protein_cds",
        "gc_approx": 66,
    },
    {
        "name": "a_vibrio_chiA",
        "component": "A", "category": "bacteria_gammaproteo",
        "lineage": "Bacteria; Pseudomonadota; Gammaproteobacteria",
        "species": "Vibrio cholerae",
        "gene": "chitinase A",
        "accession": "NP_230570",
        "type": "protein_cds",
        "gc_approx": 47,
    },
    # ───────────────────────────────────────────────────────────────
    # BACTERIA — Alphaproteobacteria (3)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_caulobacter_ctrA",
        "component": "A", "category": "bacteria_alphaproteo",
        "lineage": "Bacteria; Pseudomonadota; Alphaproteobacteria",
        "species": "Caulobacter vibrioides",
        "gene": "cell cycle master regulator CtrA",
        "accession": "NP_421492",
        "type": "protein_cds",
        "gc_approx": 67,
    },
    {
        "name": "a_rhizobium_nifH",
        "component": "A", "category": "bacteria_alphaproteo",
        "lineage": "Bacteria; Pseudomonadota; Alphaproteobacteria",
        "species": "Sinorhizobium meliloti",
        "gene": "nitrogenase iron protein NifH",
        "accession": "NP_386535",
        "type": "protein_cds",
        "gc_approx": 62,
    },
    {
        "name": "a_rickettsia_rpoB",
        "component": "A", "category": "bacteria_alphaproteo",
        "lineage": "Bacteria; Pseudomonadota; Alphaproteobacteria; Rickettsiales",
        "species": "Rickettsia prowazekii",
        "gene": "RNA polymerase beta subunit",
        "accession": "AJF73992",
        "type": "protein_cds",
        "gc_approx": 29,
    },
    # ───────────────────────────────────────────────────────────────
    # BACTERIA — Firmicutes (3)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_bacillus_spo0A",
        "component": "A", "category": "bacteria_firmicutes",
        "lineage": "Bacteria; Bacillota; Bacilli",
        "species": "Bacillus subtilis 168",
        "gene": "sporulation master regulator Spo0A",
        "accession": "NP_390366",
        "type": "protein_cds",
        "gc_approx": 44,
    },
    {
        "name": "a_clostridium_toxA",
        "component": "A", "category": "bacteria_firmicutes",
        "lineage": "Bacteria; Bacillota; Clostridia",
        "species": "Clostridioides difficile",
        "gene": "toxin A (tcdA) N-terminal domain",
        "accession": "NP_384285",
        "type": "protein_cds",
        "gc_approx": 27,
    },
    {
        "name": "a_staphylococcus_mecA",
        "component": "A", "category": "bacteria_firmicutes",
        "lineage": "Bacteria; Bacillota; Bacilli",
        "species": "Staphylococcus aureus MRSA252",
        "gene": "penicillin-binding protein 2a (mecA)",
        "accession": "CAG40059",
        "type": "protein_cds",
        "gc_approx": 33,
    },
    # ───────────────────────────────────────────────────────────────
    # BACTERIA — Actinobacteria (3)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_mycobacterium_katG",
        "component": "A", "category": "bacteria_actinobacteria",
        "lineage": "Bacteria; Actinomycetota; Actinomycetes",
        "species": "Mycobacterium tuberculosis H37Rv",
        "gene": "catalase-peroxidase KatG",
        "accession": "NP_216424",
        "type": "protein_cds",
        "gc_approx": 66,
    },
    {
        "name": "a_streptomyces_rpoBII",
        "component": "A", "category": "bacteria_actinobacteria",
        "lineage": "Bacteria; Actinomycetota; Actinomycetes",
        "species": "Streptomyces coelicolor A3(2)",
        "gene": "sigma factor WhiG",
        "accession": "NP_628151",
        "type": "protein_cds",
        "gc_approx": 72,
    },
    {
        "name": "a_corynebacterium_dtsR",
        "component": "A", "category": "bacteria_actinobacteria",
        "lineage": "Bacteria; Actinomycetota; Actinomycetes",
        "species": "Corynebacterium glutamicum",
        "gene": "acetyl-CoA carboxylase beta subunit",
        "accession": "NP_601972",
        "type": "protein_cds",
        "gc_approx": 54,
    },
    # ───────────────────────────────────────────────────────────────
    # BACTERIA — Bacteroidetes (3)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_bacteroides_susC",
        "component": "A", "category": "bacteria_bacteroidetes",
        "lineage": "Bacteria; Bacteroidota; Bacteroidia",
        "species": "Bacteroides thetaiotaomicron",
        "gene": "SusC outer membrane transporter",
        "accession": "NP_811179",
        "type": "protein_cds",
        "gc_approx": 43,
    },
    {
        "name": "a_flavobacterium_gliding",
        "component": "A", "category": "bacteria_bacteroidetes",
        "lineage": "Bacteria; Bacteroidota; Flavobacteriia",
        "species": "Flavobacterium johnsoniae",
        "gene": "gliding motility protein GldJ",
        "accession": "ABQ04108",
        "type": "protein_cds",
        "gc_approx": 34,
    },
    {
        "name": "a_porphyromonas_rgpA",
        "component": "A", "category": "bacteria_bacteroidetes",
        "lineage": "Bacteria; Bacteroidota; Bacteroidia",
        "species": "Porphyromonas gingivalis",
        "gene": "arginine-specific cysteine proteinase RgpA",
        "accession": "NP_905463",
        "type": "protein_cds",
        "gc_approx": 48,
    },
    # ───────────────────────────────────────────────────────────────
    # BACTERIA — Cyanobacteria (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_synechocystis_psaA",
        "component": "A", "category": "bacteria_cyanobacteria",
        "lineage": "Bacteria; Cyanobacteriota",
        "species": "Synechocystis sp. PCC 6803",
        "gene": "photosystem I reaction center protein PsaA",
        "accession": "NP_442389",
        "type": "protein_cds",
        "gc_approx": 47,
    },
    {
        "name": "a_prochlorococcus_rbcL",
        "component": "A", "category": "bacteria_cyanobacteria",
        "lineage": "Bacteria; Cyanobacteriota",
        "species": "Prochlorococcus marinus MED4",
        "gene": "RuBisCO large subunit",
        "accession": "NP_892753",
        "type": "protein_cds",
        "gc_approx": 31,
    },
    # ───────────────────────────────────────────────────────────────
    # BACTERIA — Spirochaetes (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_treponema_flaA",
        "component": "A", "category": "bacteria_spirochaetes",
        "lineage": "Bacteria; Spirochaetota",
        "species": "Treponema pallidum",
        "gene": "flagellin FlaA",
        "accession": "NP_218608",
        "type": "protein_cds",
        "gc_approx": 53,
    },
    {
        "name": "a_borrelia_ospA",
        "component": "A", "category": "bacteria_spirochaetes",
        "lineage": "Bacteria; Spirochaetota",
        "species": "Borrelia burgdorferi",
        "gene": "outer surface protein A",
        "accession": "NP_045776",
        "type": "protein_cds",
        "gc_approx": 28,
    },
    # ───────────────────────────────────────────────────────────────
    # BACTERIA — Chlamydiae (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_chlamydia_ompA",
        "component": "A", "category": "bacteria_chlamydiae",
        "lineage": "Bacteria; Chlamydiota",
        "species": "Chlamydia trachomatis",
        "gene": "major outer membrane protein OmpA",
        "accession": "NP_219744",
        "type": "protein_cds",
        "gc_approx": 41,
    },
    {
        "name": "a_waddlia_hsp60",
        "component": "A", "category": "bacteria_chlamydiae",
        "lineage": "Bacteria; Chlamydiota",
        "species": "Waddlia chondrophila",
        "gene": "chaperonin GroEL",
        "accession": "AEN80001",
        "type": "protein_cds",
        "gc_approx": 44,
    },
    # ───────────────────────────────────────────────────────────────
    # BACTERIA — Deinococcus-Thermus (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_deinococcus_recA",
        "component": "A", "category": "bacteria_deinococcus",
        "lineage": "Bacteria; Deinococcota",
        "species": "Deinococcus radiodurans",
        "gene": "recombinase RecA",
        "accession": "NP_295162",
        "type": "protein_cds",
        "gc_approx": 67,
    },
    {
        "name": "a_thermus_taq",
        "component": "A", "category": "bacteria_deinococcus",
        "lineage": "Bacteria; Deinococcota",
        "species": "Thermus aquaticus",
        "gene": "DNA polymerase I (Taq polymerase)",
        "accession": "AAA72962",
        "type": "protein_cds",
        "gc_approx": 65,
    },
    # ───────────────────────────────────────────────────────────────
    # BACTERIA — Planctomycetes (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_planctopirus_atpA",
        "component": "A", "category": "bacteria_planctomycetes",
        "lineage": "Bacteria; Planctomycetota",
        "species": "Planctopirus limnophila",
        "gene": "ATP synthase alpha subunit",
        "accession": "ADG67899",
        "type": "protein_cds",
        "gc_approx": 54,
    },
    {
        "name": "a_gemmata_rpoBII",
        "component": "A", "category": "bacteria_planctomycetes",
        "lineage": "Bacteria; Planctomycetota",
        "species": "Gemmata obscuriglobus",
        "gene": "RNA polymerase beta subunit",
        "accession": "SCZ36073",
        "type": "protein_cds",
        "gc_approx": 64,
    },
    # ───────────────────────────────────────────────────────────────
    # BACTERIA — Acidobacteria (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_acidobacterium_gyrB",
        "component": "A", "category": "bacteria_acidobacteria",
        "lineage": "Bacteria; Acidobacteriota",
        "species": "Acidobacterium capsulatum",
        "gene": "DNA gyrase subunit B",
        "accession": "ACO33375",
        "type": "protein_cds",
        "gc_approx": 60,
    },
    {
        "name": "a_terriglobus_ef_tu",
        "component": "A", "category": "bacteria_acidobacteria",
        "lineage": "Bacteria; Acidobacteriota",
        "species": "Terriglobus roseus",
        "gene": "elongation factor Tu",
        "accession": "WP_018694938",
        "type": "protein_cds",
        "gc_approx": 59,
    },
    # ───────────────────────────────────────────────────────────────
    # BACTERIA — Other phyla (3)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_aquifex_hydrogenase",
        "component": "A", "category": "bacteria_other",
        "lineage": "Bacteria; Aquificota",
        "species": "Aquifex aeolicus",
        "gene": "hydrogenase large subunit",
        "accession": "NP_213649",
        "type": "protein_cds",
        "gc_approx": 43,
    },
    {
        "name": "a_thermotoga_xylanase",
        "component": "A", "category": "bacteria_other",
        "lineage": "Bacteria; Thermotogota",
        "species": "Thermotoga maritima",
        "gene": "xylanase A",
        "accession": "NP_228066",
        "type": "protein_cds",
        "gc_approx": 46,
    },
    {
        "name": "a_mycoplasma_p1",
        "component": "A", "category": "bacteria_other",
        "lineage": "Bacteria; Tenericutes; Mollicutes",
        "species": "Mycoplasma pneumoniae",
        "gene": "cytadhesin P1",
        "accession": "NP_109724",
        "type": "protein_cds",
        "gc_approx": 40,
        "note": "Minimal genome, UGA=Trp in Mycoplasma.",
    },
    # ───────────────────────────────────────────────────────────────
    # VERTEBRATA — Mammals (3)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_human_BRCA1",
        "component": "A", "category": "vertebrata_mammal",
        "lineage": "Eukarya; Chordata; Mammalia; Primates",
        "species": "Homo sapiens",
        "gene": "BRCA1 DNA repair (partial CDS)",
        "accession": "NM_007294",
        "cds_start": 233, "cds_end": 1733,
        "type": "mrna_cds",
        "gc_approx": 45,
    },
    {
        "name": "a_mouse_insulin2",
        "component": "A", "category": "vertebrata_mammal",
        "lineage": "Eukarya; Chordata; Mammalia; Rodentia",
        "species": "Mus musculus",
        "gene": "insulin 2",
        "accession": "NM_008387",
        "cds_start": 69, "cds_end": 399,
        "type": "mrna_cds",
        "gc_approx": 52,
    },
    {
        "name": "a_platypus_venom",
        "component": "A", "category": "vertebrata_mammal",
        "lineage": "Eukarya; Chordata; Mammalia; Monotremata",
        "species": "Ornithorhynchus anatinus",
        "gene": "defensin-like peptide 1 (venom)",
        "accession": "NM_001082110",
        "cds_start": 65, "cds_end": 308,
        "type": "mrna_cds",
        "gc_approx": 45,
    },
    # ───────────────────────────────────────────────────────────────
    # VERTEBRATA — Birds (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_chicken_lysozyme",
        "component": "A", "category": "vertebrata_bird",
        "lineage": "Eukarya; Chordata; Aves; Galliformes",
        "species": "Gallus gallus",
        "gene": "lysozyme C",
        "accession": "NM_205281",
        "cds_start": 39, "cds_end": 479,
        "type": "mrna_cds",
        "gc_approx": 48,
    },
    {
        "name": "a_zebrafinch_foxp2",
        "component": "A", "category": "vertebrata_bird",
        "lineage": "Eukarya; Chordata; Aves; Passeriformes",
        "species": "Taeniopygia guttata",
        "gene": "FOXP2 (vocal learning transcription factor)",
        "accession": "NM_001048263",
        "cds_start": 151, "cds_end": 1300,
        "type": "mrna_cds",
        "gc_approx": 52,
    },
    # ───────────────────────────────────────────────────────────────
    # VERTEBRATA — Reptiles (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_anole_rhodopsin",
        "component": "A", "category": "vertebrata_reptile",
        "lineage": "Eukarya; Chordata; Reptilia; Squamata",
        "species": "Anolis carolinensis",
        "gene": "rhodopsin",
        "accession": "NM_001291265",
        "cds_start": 104, "cds_end": 1159,
        "type": "mrna_cds",
        "gc_approx": 52,
    },
    {
        "name": "a_turtle_hemoglobin",
        "component": "A", "category": "vertebrata_reptile",
        "lineage": "Eukarya; Chordata; Reptilia; Testudines",
        "species": "Chrysemys picta",
        "gene": "hemoglobin alpha",
        "accession": "XM_005283127",
        "cds_start": 59, "cds_end": 487,
        "type": "mrna_cds",
        "gc_approx": 50,
    },
    # ───────────────────────────────────────────────────────────────
    # VERTEBRATA — Bony fish (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_zebrafish_shha",
        "component": "A", "category": "vertebrata_fish",
        "lineage": "Eukarya; Chordata; Actinopterygii; Cypriniformes",
        "species": "Danio rerio",
        "gene": "sonic hedgehog a",
        "accession": "NM_131063",
        "cds_start": 74, "cds_end": 1450,
        "type": "mrna_cds",
        "gc_approx": 48,
    },
    {
        "name": "a_coelacanth_hox",
        "component": "A", "category": "vertebrata_fish",
        "lineage": "Eukarya; Chordata; Sarcopterygii; Coelacanthiformes",
        "species": "Latimeria chalumnae",
        "gene": "homeobox HoxA13",
        "accession": "XM_006008505",
        "cds_start": 244, "cds_end": 1244,
        "type": "mrna_cds",
        "gc_approx": 55,
    },
    # ───────────────────────────────────────────────────────────────
    # VERTEBRATA — Amphibians (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_xenopus_myoD",
        "component": "A", "category": "vertebrata_amphibian",
        "lineage": "Eukarya; Chordata; Amphibia; Anura",
        "species": "Xenopus laevis",
        "gene": "myogenic differentiation factor MyoD",
        "accession": "NM_001085670",
        "cds_start": 179, "cds_end": 1099,
        "type": "mrna_cds",
        "gc_approx": 44,
    },
    {
        "name": "a_axolotl_pax7",
        "component": "A", "category": "vertebrata_amphibian",
        "lineage": "Eukarya; Chordata; Amphibia; Urodela",
        "species": "Ambystoma mexicanum",
        "gene": "paired box 7 (regeneration TF)",
        "accession": "XM_044187009",
        "cds_start": 139, "cds_end": 1600,
        "type": "mrna_cds",
        "gc_approx": 46,
    },
    # ───────────────────────────────────────────────────────────────
    # VERTEBRATA — Cartilaginous fish (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_shark_hemoglobin",
        "component": "A", "category": "vertebrata_cartilaginous",
        "lineage": "Eukarya; Chordata; Chondrichthyes; Elasmobranchii",
        "species": "Callorhinchus milii",
        "gene": "hemoglobin alpha",
        "accession": "XM_007896987",
        "cds_start": 48, "cds_end": 476,
        "type": "mrna_cds",
        "gc_approx": 47,
    },
    {
        "name": "a_skate_hoxD",
        "component": "A", "category": "vertebrata_cartilaginous",
        "lineage": "Eukarya; Chordata; Chondrichthyes; Batoidea",
        "species": "Leucoraja erinacea",
        "gene": "homeobox HoxD13",
        "accession": "XM_055644567",
        "cds_start": 141, "cds_end": 1100,
        "type": "mrna_cds",
        "gc_approx": 54,
    },
    # ───────────────────────────────────────────────────────────────
    # VERTEBRATA — Jawless fish (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_lamprey_hemoglobin",
        "component": "A", "category": "vertebrata_jawless",
        "lineage": "Eukarya; Chordata; Hyperoartia",
        "species": "Petromyzon marinus",
        "gene": "hemoglobin (globin-like)",
        "accession": "XM_032948613",
        "cds_start": 159, "cds_end": 612,
        "type": "mrna_cds",
        "gc_approx": 53,
    },
    {
        "name": "a_hagfish_vLR",
        "component": "A", "category": "vertebrata_jawless",
        "lineage": "Eukarya; Chordata; Myxini",
        "species": "Eptatretus burgeri",
        "gene": "variable lymphocyte receptor B",
        "accession": "AB285309",
        "type": "full_cds",
        "gc_approx": 55,
        "note": "Jawless vertebrate adaptive immunity; LRR-based, not Ig-based.",
    },
    # ───────────────────────────────────────────────────────────────
    # INVERTEBRATA — Insects (3)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_drosophila_hox",
        "component": "A", "category": "invertebrata_insect",
        "lineage": "Eukarya; Arthropoda; Insecta; Diptera",
        "species": "Drosophila melanogaster",
        "gene": "Ultrabithorax (Ubx, homeotic)",
        "accession": "NM_001275048",
        "cds_start": 1, "cds_end": 1200,
        "type": "mrna_cds",
        "gc_approx": 51,
    },
    {
        "name": "a_honeybee_vg",
        "component": "A", "category": "invertebrata_insect",
        "lineage": "Eukarya; Arthropoda; Insecta; Hymenoptera",
        "species": "Apis mellifera",
        "gene": "vitellogenin",
        "accession": "NM_001011578",
        "cds_start": 48, "cds_end": 1548,
        "type": "mrna_cds",
        "gc_approx": 35,
    },
    {
        "name": "a_silkworm_fibroin",
        "component": "A", "category": "invertebrata_insect",
        "lineage": "Eukarya; Arthropoda; Insecta; Lepidoptera",
        "species": "Bombyx mori",
        "gene": "fibroin light chain",
        "accession": "NM_001044023",
        "cds_start": 26, "cds_end": 836,
        "type": "mrna_cds",
        "gc_approx": 40,
    },
    # ───────────────────────────────────────────────────────────────
    # INVERTEBRATA — Nematodes (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_celegans_unc54",
        "component": "A", "category": "invertebrata_nematode",
        "lineage": "Eukarya; Nematoda; Chromadorea",
        "species": "Caenorhabditis elegans",
        "gene": "myosin heavy chain UNC-54 (partial CDS)",
        "accession": "NM_058697",
        "cds_start": 1, "cds_end": 1500,
        "type": "mrna_cds",
        "gc_approx": 42,
    },
    {
        "name": "a_brugia_chitinase",
        "component": "A", "category": "invertebrata_nematode",
        "lineage": "Eukarya; Nematoda; Chromadorea",
        "species": "Brugia malayi",
        "gene": "chitinase",
        "accession": "XM_001898283",
        "cds_start": 1, "cds_end": 1200,
        "type": "mrna_cds",
        "gc_approx": 40,
    },
    # ───────────────────────────────────────────────────────────────
    # INVERTEBRATA — Mollusks (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_octopus_rhodopsin",
        "component": "A", "category": "invertebrata_mollusk",
        "lineage": "Eukarya; Mollusca; Cephalopoda",
        "species": "Octopus bimaculoides",
        "gene": "rhodopsin",
        "accession": "XM_014924783",
        "cds_start": 69, "cds_end": 1200,
        "type": "mrna_cds",
        "gc_approx": 42,
    },
    {
        "name": "a_oyster_defensin",
        "component": "A", "category": "invertebrata_mollusk",
        "lineage": "Eukarya; Mollusca; Bivalvia",
        "species": "Crassostrea gigas",
        "gene": "big defensin",
        "accession": "NM_001308925",
        "cds_start": 23, "cds_end": 380,
        "type": "mrna_cds",
        "gc_approx": 44,
    },
    # ───────────────────────────────────────────────────────────────
    # INVERTEBRATA — Cnidarians (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_hydra_wnt3",
        "component": "A", "category": "invertebrata_cnidarian",
        "lineage": "Eukarya; Cnidaria; Hydrozoa",
        "species": "Hydra vulgaris",
        "gene": "Wnt3 signaling ligand",
        "accession": "NM_001309701",
        "cds_start": 1, "cds_end": 1068,
        "type": "mrna_cds",
        "gc_approx": 35,
    },
    {
        "name": "a_nematostella_nkx",
        "component": "A", "category": "invertebrata_cnidarian",
        "lineage": "Eukarya; Cnidaria; Anthozoa",
        "species": "Nematostella vectensis",
        "gene": "NK homeobox 2.5",
        "accession": "XM_001622805",
        "cds_start": 161, "cds_end": 1100,
        "type": "mrna_cds",
        "gc_approx": 42,
    },
    # ───────────────────────────────────────────────────────────────
    # INVERTEBRATA — Echinoderms (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_urchin_bindin",
        "component": "A", "category": "invertebrata_echinoderm",
        "lineage": "Eukarya; Echinodermata; Echinoidea",
        "species": "Strongylocentrotus purpuratus",
        "gene": "bindin (sperm protein)",
        "accession": "NM_214600",
        "cds_start": 1, "cds_end": 1200,
        "type": "mrna_cds",
        "gc_approx": 42,
    },
    {
        "name": "a_starfish_nodal",
        "component": "A", "category": "invertebrata_echinoderm",
        "lineage": "Eukarya; Echinodermata; Asteroidea",
        "species": "Patiria miniata",
        "gene": "Nodal signaling ligand",
        "accession": "NM_001305363",
        "cds_start": 1, "cds_end": 1200,
        "type": "mrna_cds",
        "gc_approx": 44,
    },
    # ───────────────────────────────────────────────────────────────
    # INVERTEBRATA — Annelids (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_leech_hirudin",
        "component": "A", "category": "invertebrata_annelid",
        "lineage": "Eukarya; Annelida; Hirudinea",
        "species": "Hirudo medicinalis",
        "gene": "hirudin (thrombin inhibitor)",
        "accession": "M17396",
        "type": "full_cds",
        "gc_approx": 49,
    },
    {
        "name": "a_polychaete_hemo",
        "component": "A", "category": "invertebrata_annelid",
        "lineage": "Eukarya; Annelida; Polychaeta",
        "species": "Capitella teleta",
        "gene": "globin",
        "accession": "ELT89652",
        "type": "protein_cds",
        "gc_approx": 40,
    },
    # ───────────────────────────────────────────────────────────────
    # INVERTEBRATA — Other (2): tardigrade, tunicate
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_tardigrade_dsup",
        "component": "A", "category": "invertebrata_other",
        "lineage": "Eukarya; Tardigrada; Eutardigrada",
        "species": "Ramazzottius varieornatus",
        "gene": "damage suppressor (Dsup, radiation protection)",
        "accession": "GAV07234",
        "type": "protein_cds",
        "gc_approx": 43,
        "note": "Unique tardigrade gene: protects DNA from radiation damage.",
    },
    {
        "name": "a_ciona_notochord",
        "component": "A", "category": "invertebrata_other",
        "lineage": "Eukarya; Chordata; Tunicata",
        "species": "Ciona intestinalis",
        "gene": "Brachyury (notochord TF)",
        "accession": "NM_001078487",
        "cds_start": 61, "cds_end": 1400,
        "type": "mrna_cds",
        "gc_approx": 39,
    },
    # ───────────────────────────────────────────────────────────────
    # PLANTAE — Dicots (4)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_arabidopsis_phyB",
        "component": "A", "category": "plantae_dicot",
        "lineage": "Eukarya; Streptophyta; Magnoliopsida; Brassicales",
        "species": "Arabidopsis thaliana",
        "gene": "phytochrome B (light receptor)",
        "accession": "NM_127435",
        "cds_start": 81, "cds_end": 1581,
        "type": "mrna_cds",
        "gc_approx": 44,
    },
    {
        "name": "a_tomato_polygalacturonase",
        "component": "A", "category": "plantae_dicot",
        "lineage": "Eukarya; Streptophyta; Magnoliopsida; Solanales",
        "species": "Solanum lycopersicum",
        "gene": "polygalacturonase (fruit ripening)",
        "accession": "NM_001247874",
        "cds_start": 44, "cds_end": 1062,
        "type": "mrna_cds",
        "gc_approx": 42,
    },
    {
        "name": "a_soybean_leghemoglobin",
        "component": "A", "category": "plantae_dicot",
        "lineage": "Eukarya; Streptophyta; Magnoliopsida; Fabales",
        "species": "Glycine max",
        "gene": "leghemoglobin a (nitrogen fixation)",
        "accession": "NM_001248252",
        "cds_start": 26, "cds_end": 475,
        "type": "mrna_cds",
        "gc_approx": 42,
    },
    {
        "name": "a_grape_stilbene",
        "component": "A", "category": "plantae_dicot",
        "lineage": "Eukarya; Streptophyta; Magnoliopsida; Vitales",
        "species": "Vitis vinifera",
        "gene": "stilbene synthase (resveratrol biosynthesis)",
        "accession": "NM_001281006",
        "cds_start": 1, "cds_end": 1179,
        "type": "mrna_cds",
        "gc_approx": 44,
    },
    # ───────────────────────────────────────────────────────────────
    # PLANTAE — Monocots (3)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_rice_waxy",
        "component": "A", "category": "plantae_monocot",
        "lineage": "Eukarya; Streptophyta; Liliopsida; Poales",
        "species": "Oryza sativa",
        "gene": "granule-bound starch synthase (Waxy)",
        "accession": "NM_001062517",
        "cds_start": 145, "cds_end": 1645,
        "type": "mrna_cds",
        "gc_approx": 58,
    },
    {
        "name": "a_maize_adh1",
        "component": "A", "category": "plantae_monocot",
        "lineage": "Eukarya; Streptophyta; Liliopsida; Poales",
        "species": "Zea mays",
        "gene": "alcohol dehydrogenase 1",
        "accession": "NM_001112104",
        "cds_start": 62, "cds_end": 1200,
        "type": "mrna_cds",
        "gc_approx": 58,
    },
    {
        "name": "a_banana_acc_oxidase",
        "component": "A", "category": "plantae_monocot",
        "lineage": "Eukarya; Streptophyta; Liliopsida; Zingiberales",
        "species": "Musa acuminata",
        "gene": "ACC oxidase (ethylene biosynthesis)",
        "accession": "XM_009389131",
        "cds_start": 70, "cds_end": 1070,
        "type": "mrna_cds",
        "gc_approx": 47,
    },
    # ───────────────────────────────────────────────────────────────
    # PLANTAE — Gymnosperms (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_pine_cab",
        "component": "A", "category": "plantae_gymnosperm",
        "lineage": "Eukarya; Streptophyta; Pinopsida",
        "species": "Pinus taeda",
        "gene": "chlorophyll a/b binding protein",
        "accession": "XM_062643474",
        "cds_start": 143, "cds_end": 943,
        "type": "mrna_cds",
        "gc_approx": 42,
    },
    {
        "name": "a_ginkgo_chs",
        "component": "A", "category": "plantae_gymnosperm",
        "lineage": "Eukarya; Streptophyta; Ginkgoopsida",
        "species": "Ginkgo biloba",
        "gene": "chalcone synthase",
        "accession": "AY496936",
        "type": "full_cds",
        "gc_approx": 42,
    },
    # ───────────────────────────────────────────────────────────────
    # PLANTAE — Ferns (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_azolla_nifH_symbiont",
        "component": "A", "category": "plantae_fern",
        "lineage": "Eukarya; Streptophyta; Polypodiopsida",
        "species": "Salvinia cucullata",
        "gene": "neochrome (phytochrome-phototropin chimera)",
        "accession": "XM_028666244",
        "cds_start": 1, "cds_end": 1500,
        "type": "mrna_cds",
        "gc_approx": 42,
    },
    {
        "name": "a_ceratopteris_CrLFY",
        "component": "A", "category": "plantae_fern",
        "lineage": "Eukarya; Streptophyta; Polypodiopsida",
        "species": "Ceratopteris richardii",
        "gene": "FLORICAULA/LEAFY transcription factor",
        "accession": "XM_059020041",
        "cds_start": 135, "cds_end": 1335,
        "type": "mrna_cds",
        "gc_approx": 45,
    },
    # ───────────────────────────────────────────────────────────────
    # PLANTAE — Mosses/liverworts (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_physcomitrium_phot",
        "component": "A", "category": "plantae_bryophyte",
        "lineage": "Eukarya; Streptophyta; Bryophyta",
        "species": "Physcomitrium patens",
        "gene": "phototropin",
        "accession": "XM_024525296",
        "cds_start": 83, "cds_end": 1583,
        "type": "mrna_cds",
        "gc_approx": 45,
    },
    {
        "name": "a_marchantia_tcp",
        "component": "A", "category": "plantae_bryophyte",
        "lineage": "Eukarya; Streptophyta; Marchantiophyta",
        "species": "Marchantia polymorpha",
        "gene": "TCP transcription factor",
        "accession": "XM_043130137",
        "cds_start": 240, "cds_end": 1240,
        "type": "mrna_cds",
        "gc_approx": 44,
    },
    # ───────────────────────────────────────────────────────────────
    # PLANTAE — Green algae (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_chlamy_flagellar",
        "component": "A", "category": "plantae_green_alga",
        "lineage": "Eukarya; Chlorophyta; Chlorophyceae",
        "species": "Chlamydomonas reinhardtii",
        "gene": "dynein heavy chain (flagellar)",
        "accession": "NM_001324575",
        "cds_start": 89, "cds_end": 1589,
        "type": "mrna_cds",
        "gc_approx": 64,
    },
    {
        "name": "a_volvox_regA",
        "component": "A", "category": "plantae_green_alga",
        "lineage": "Eukarya; Chlorophyta; Chlorophyceae",
        "species": "Volvox carteri",
        "gene": "regA (multicellularity regulator)",
        "accession": "XM_002957316",
        "cds_start": 1, "cds_end": 1500,
        "type": "mrna_cds",
        "gc_approx": 56,
    },
    # ───────────────────────────────────────────────────────────────
    # FUNGI — Ascomycetes (3)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_yeast_cdc28",
        "component": "A", "category": "fungi_ascomycete",
        "lineage": "Eukarya; Ascomycota; Saccharomycetes",
        "species": "Saccharomyces cerevisiae",
        "gene": "cyclin-dependent kinase CDC28",
        "accession": "NM_001180915",
        "cds_start": 1, "cds_end": 900,
        "type": "mrna_cds",
        "gc_approx": 40,
    },
    {
        "name": "a_neurospora_frq",
        "component": "A", "category": "fungi_ascomycete",
        "lineage": "Eukarya; Ascomycota; Sordariomycetes",
        "species": "Neurospora crassa",
        "gene": "frequency (circadian clock)",
        "accession": "XM_960972",
        "cds_start": 1, "cds_end": 1500,
        "type": "mrna_cds",
        "gc_approx": 54,
    },
    {
        "name": "a_aspergillus_aflR",
        "component": "A", "category": "fungi_ascomycete",
        "lineage": "Eukarya; Ascomycota; Eurotiomycetes",
        "species": "Aspergillus flavus",
        "gene": "aflatoxin pathway regulator AflR",
        "accession": "XM_002373908",
        "cds_start": 1, "cds_end": 1300,
        "type": "mrna_cds",
        "gc_approx": 50,
    },
    # ───────────────────────────────────────────────────────────────
    # FUNGI — Basidiomycetes (3)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_agaricus_laccase",
        "component": "A", "category": "fungi_basidiomycete",
        "lineage": "Eukarya; Basidiomycota; Agaricomycetes",
        "species": "Agaricus bisporus",
        "gene": "laccase",
        "accession": "XM_006458750",
        "cds_start": 1, "cds_end": 1500,
        "type": "mrna_cds",
        "gc_approx": 50,
    },
    {
        "name": "a_ustilago_effector",
        "component": "A", "category": "fungi_basidiomycete",
        "lineage": "Eukarya; Basidiomycota; Ustilaginomycetes",
        "species": "Ustilago maydis",
        "gene": "secreted effector protein Pep1",
        "accession": "XM_011388685",
        "cds_start": 1, "cds_end": 600,
        "type": "mrna_cds",
        "gc_approx": 54,
    },
    {
        "name": "a_cryptococcus_cap",
        "component": "A", "category": "fungi_basidiomycete",
        "lineage": "Eukarya; Basidiomycota; Tremellomycetes",
        "species": "Cryptococcus neoformans",
        "gene": "capsule-associated protein Cap10",
        "accession": "XM_012194034",
        "cds_start": 1, "cds_end": 1200,
        "type": "mrna_cds",
        "gc_approx": 49,
    },
    # ───────────────────────────────────────────────────────────────
    # FUNGI — Zygomycetes (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_rhizopus_lipase",
        "component": "A", "category": "fungi_zygomycete",
        "lineage": "Eukarya; Mucoromycota; Mucoromycetes",
        "species": "Rhizopus oryzae",
        "gene": "lipase",
        "accession": "XM_002494879",
        "cds_start": 1, "cds_end": 1200,
        "type": "mrna_cds",
        "gc_approx": 38,
    },
    {
        "name": "a_mucor_hsp70",
        "component": "A", "category": "fungi_zygomycete",
        "lineage": "Eukarya; Mucoromycota; Mucoromycetes",
        "species": "Mucor circinelloides",
        "gene": "HSP70-like chaperone",
        "accession": "XM_018350068",
        "cds_start": 1, "cds_end": 1500,
        "type": "mrna_cds",
        "gc_approx": 38,
    },
    # ───────────────────────────────────────────────────────────────
    # FUNGI — Chytrids (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_batrachochytrium_protease",
        "component": "A", "category": "fungi_chytrid",
        "lineage": "Eukarya; Chytridiomycota",
        "species": "Batrachochytrium dendrobatidis",
        "gene": "subtilisin-like serine protease",
        "accession": "XM_006681891",
        "cds_start": 1, "cds_end": 1200,
        "type": "mrna_cds",
        "gc_approx": 42,
    },
    {
        "name": "a_spizellomyces_gyrB",
        "component": "A", "category": "fungi_chytrid",
        "lineage": "Eukarya; Chytridiomycota",
        "species": "Spizellomyces punctatus",
        "gene": "chitin synthase",
        "accession": "XM_016326399",
        "cds_start": 1, "cds_end": 1500,
        "type": "mrna_cds",
        "gc_approx": 33,
    },
    # ───────────────────────────────────────────────────────────────
    # PROTISTS — Apicomplexa (3)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_plasmodium_csp",
        "component": "A", "category": "protist_apicomplexa",
        "lineage": "Eukarya; Apicomplexa; Aconoidasida",
        "species": "Plasmodium falciparum",
        "gene": "circumsporozoite protein (CSP)",
        "accession": "XM_001351086",
        "cds_start": 1, "cds_end": 1200,
        "type": "mrna_cds",
        "gc_approx": 24,
        "note": "Extreme AT-bias (19% GC genome). Hardest test case.",
    },
    {
        "name": "a_toxoplasma_sag1",
        "component": "A", "category": "protist_apicomplexa",
        "lineage": "Eukarya; Apicomplexa; Conoidasida",
        "species": "Toxoplasma gondii",
        "gene": "surface antigen 1 (SAG1/P30)",
        "accession": "XM_002365686",
        "cds_start": 1, "cds_end": 1020,
        "type": "mrna_cds",
        "gc_approx": 52,
    },
    {
        "name": "a_cryptosporidium_cowp",
        "component": "A", "category": "protist_apicomplexa",
        "lineage": "Eukarya; Apicomplexa; Conoidasida",
        "species": "Cryptosporidium parvum",
        "gene": "oocyst wall protein COWP",
        "accession": "XM_001388100",
        "cds_start": 1, "cds_end": 900,
        "type": "mrna_cds",
        "gc_approx": 31,
    },
    # ───────────────────────────────────────────────────────────────
    # PROTISTS — Kinetoplastids (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_trypanosoma_vsg",
        "component": "A", "category": "protist_kinetoplastid",
        "lineage": "Eukarya; Euglenozoa; Kinetoplastea",
        "species": "Trypanosoma brucei",
        "gene": "variant surface glycoprotein (VSG)",
        "accession": "XM_011775800",
        "cds_start": 1, "cds_end": 1500,
        "type": "mrna_cds",
        "gc_approx": 45,
    },
    {
        "name": "a_leishmania_gp63",
        "component": "A", "category": "protist_kinetoplastid",
        "lineage": "Eukarya; Euglenozoa; Kinetoplastea",
        "species": "Leishmania major",
        "gene": "surface metalloprotease GP63",
        "accession": "XM_001684153",
        "cds_start": 1, "cds_end": 1500,
        "type": "mrna_cds",
        "gc_approx": 62,
    },
    # ───────────────────────────────────────────────────────────────
    # PROTISTS — Amoebozoa (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_dictyostelium_dscA",
        "component": "A", "category": "protist_amoebozoa",
        "lineage": "Eukarya; Amoebozoa; Dictyosteliida",
        "species": "Dictyostelium discoideum",
        "gene": "discoidin I alpha (cell adhesion)",
        "accession": "XM_635861",
        "cds_start": 1, "cds_end": 800,
        "type": "mrna_cds",
        "gc_approx": 28,
    },
    {
        "name": "a_acanthamoeba_actophorin",
        "component": "A", "category": "protist_amoebozoa",
        "lineage": "Eukarya; Amoebozoa; Centramoebida",
        "species": "Acanthamoeba castellanii",
        "gene": "actophorin (actin depolymerization)",
        "accession": "XM_004343419",
        "cds_start": 1, "cds_end": 500,
        "type": "mrna_cds",
        "gc_approx": 58,
    },
    # ───────────────────────────────────────────────────────────────
    # PROTISTS — Ciliates (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_tetrahymena_telomerase",
        "component": "A", "category": "protist_ciliate",
        "lineage": "Eukarya; Ciliophora; Oligohymenophorea",
        "species": "Tetrahymena thermophila",
        "gene": "telomerase reverse transcriptase (TERT)",
        "accession": "XM_001028702",
        "cds_start": 1, "cds_end": 1500,
        "type": "mrna_cds",
        "gc_approx": 32,
    },
    {
        "name": "a_paramecium_trichocyst",
        "component": "A", "category": "protist_ciliate",
        "lineage": "Eukarya; Ciliophora; Oligohymenophorea",
        "species": "Paramecium tetraurelia",
        "gene": "trichocyst matrix protein",
        "accession": "XM_001430826",
        "cds_start": 1, "cds_end": 900,
        "type": "mrna_cds",
        "gc_approx": 28,
    },
    # ───────────────────────────────────────────────────────────────
    # PROTISTS — Oomycetes (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_phytophthora_inf1",
        "component": "A", "category": "protist_oomycete",
        "lineage": "Eukarya; Oomycota; Peronosporales",
        "species": "Phytophthora infestans",
        "gene": "elicitin INF1",
        "accession": "XM_002909427",
        "cds_start": 1, "cds_end": 500,
        "type": "mrna_cds",
        "gc_approx": 51,
    },
    {
        "name": "a_saprolegnia_cellulase",
        "component": "A", "category": "protist_oomycete",
        "lineage": "Eukarya; Oomycota; Saprolegniales",
        "species": "Saprolegnia parasitica",
        "gene": "cellulase",
        "accession": "XM_012351297",
        "cds_start": 1, "cds_end": 1200,
        "type": "mrna_cds",
        "gc_approx": 58,
    },
    # ───────────────────────────────────────────────────────────────
    # PROTISTS — Foraminifera/Radiolaria and other SAR (4)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_reticulomyxa_actin",
        "component": "A", "category": "protist_foraminifera",
        "lineage": "Eukarya; Rhizaria; Foraminifera",
        "species": "Reticulomyxa filosa",
        "gene": "actin",
        "accession": "ETO05498",
        "type": "protein_cds",
        "gc_approx": 30,
    },
    {
        "name": "a_bigelowiella_plastid",
        "component": "A", "category": "protist_other_sar",
        "lineage": "Eukarya; Rhizaria; Chlorarachniophyta",
        "species": "Bigelowiella natans",
        "gene": "nucleomorph housekeeping gene",
        "accession": "XM_001712853",
        "cds_start": 1, "cds_end": 900,
        "type": "mrna_cds",
        "gc_approx": 38,
    },
    {
        "name": "a_thalassiosira_silaffin",
        "component": "A", "category": "protist_other_sar",
        "lineage": "Eukarya; Bacillariophyta; Coscinodiscophyceae",
        "species": "Thalassiosira pseudonana",
        "gene": "silaffin (silica biomineralization)",
        "accession": "XM_002296245",
        "cds_start": 1, "cds_end": 1100,
        "type": "mrna_cds",
        "gc_approx": 48,
    },
    {
        "name": "a_emiliania_coccolith",
        "component": "A", "category": "protist_other_sar",
        "lineage": "Eukarya; Haptophyta; Prymnesiophyceae",
        "species": "Emiliania huxleyi",
        "gene": "GPA (coccolith-associated protein)",
        "accession": "XM_005794263",
        "cds_start": 1, "cds_end": 800,
        "type": "mrna_cds",
        "gc_approx": 64,
    },
    # ───────────────────────────────────────────────────────────────
    # ALGAE (5)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_diatom_fucoxanthin",
        "component": "A", "category": "algae_diatom",
        "lineage": "Eukarya; Bacillariophyta; Mediophyceae",
        "species": "Phaeodactylum tricornutum",
        "gene": "fucoxanthin chlorophyll a/c binding protein",
        "accession": "XM_002186173",
        "cds_start": 1, "cds_end": 700,
        "type": "mrna_cds",
        "gc_approx": 49,
    },
    {
        "name": "a_diatom2_silicon_transporter",
        "component": "A", "category": "algae_diatom",
        "lineage": "Eukarya; Bacillariophyta; Coscinodiscophyceae",
        "species": "Thalassiosira pseudonana",
        "gene": "silicon transporter SIT1",
        "accession": "XM_002295810",
        "cds_start": 1, "cds_end": 1000,
        "type": "mrna_cds",
        "gc_approx": 48,
    },
    {
        "name": "a_ectocarpus_mannitol",
        "component": "A", "category": "algae_brown",
        "lineage": "Eukarya; Phaeophyceae; Ectocarpales",
        "species": "Ectocarpus siliculosus",
        "gene": "mannitol-1-phosphate dehydrogenase",
        "accession": "CBN73949",
        "type": "protein_cds",
        "gc_approx": 55,
    },
    {
        "name": "a_gracilaria_phycoerythrin",
        "component": "A", "category": "algae_red",
        "lineage": "Eukarya; Rhodophyta; Florideophyceae",
        "species": "Gracilariopsis chorda",
        "gene": "R-phycoerythrin alpha subunit",
        "accession": "XM_038300001",
        "cds_start": 1, "cds_end": 600,
        "type": "mrna_cds",
        "gc_approx": 46,
    },
    {
        "name": "a_symbiodinium_peridinin",
        "component": "A", "category": "algae_dinoflagellate",
        "lineage": "Eukarya; Dinophyceae; Suessiales",
        "species": "Symbiodinium microadriaticum",
        "gene": "peridinin-chlorophyll a-binding protein",
        "accession": "XM_021312979",
        "cds_start": 1, "cds_end": 1100,
        "type": "mrna_cds",
        "gc_approx": 54,
    },
    # ───────────────────────────────────────────────────────────────
    # ORGANELLAR — Mitochondrial (3)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_human_mt_co1",
        "component": "A", "category": "organellar_mito",
        "lineage": "Eukarya; Chordata; Mammalia (mitochondrial)",
        "species": "Homo sapiens",
        "gene": "cytochrome c oxidase subunit I (mitochondrial)",
        "accession": "NC_012920",
        "cds_start": 5904, "cds_end": 7445,
        "type": "genomic_region",
        "gc_approx": 38,
        "note": "Mitochondrial genetic code (UGA=Trp).",
    },
    {
        "name": "a_drosophila_mt_nd5",
        "component": "A", "category": "organellar_mito",
        "lineage": "Eukarya; Arthropoda; Insecta (mitochondrial)",
        "species": "Drosophila melanogaster",
        "gene": "NADH dehydrogenase 5 (mitochondrial)",
        "accession": "NC_024511",
        "cds_start": 7920, "cds_end": 9620,
        "type": "genomic_region",
        "gc_approx": 22,
        "note": "Extreme AT-bias mitochondrial genome.",
    },
    {
        "name": "a_yeast_mt_cox2",
        "component": "A", "category": "organellar_mito",
        "lineage": "Eukarya; Ascomycota; Saccharomycetes (mitochondrial)",
        "species": "Saccharomyces cerevisiae",
        "gene": "cytochrome c oxidase subunit II (mitochondrial)",
        "accession": "NC_001224",
        "cds_start": 73758, "cds_end": 74513,
        "type": "genomic_region",
        "gc_approx": 17,
        "note": "Extremely AT-rich yeast mitochondrial genome (17% GC).",
    },
    # ───────────────────────────────────────────────────────────────
    # ORGANELLAR — Chloroplast (3)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_arabidopsis_cp_psbA",
        "component": "A", "category": "organellar_chloroplast",
        "lineage": "Eukarya; Streptophyta; Magnoliopsida (chloroplast)",
        "species": "Arabidopsis thaliana",
        "gene": "photosystem II protein D1 (psbA)",
        "accession": "NC_000932",
        "cds_start": 255, "cds_end": 1315,
        "type": "genomic_region",
        "gc_approx": 40,
    },
    {
        "name": "a_rice_cp_rbcL",
        "component": "A", "category": "organellar_chloroplast",
        "lineage": "Eukarya; Streptophyta; Liliopsida (chloroplast)",
        "species": "Oryza sativa",
        "gene": "RuBisCO large subunit (chloroplast rbcL)",
        "accession": "NC_001320",
        "cds_start": 53936, "cds_end": 55375,
        "type": "genomic_region",
        "gc_approx": 44,
    },
    {
        "name": "a_chlamy_cp_psaB",
        "component": "A", "category": "organellar_chloroplast",
        "lineage": "Eukarya; Chlorophyta (chloroplast)",
        "species": "Chlamydomonas reinhardtii",
        "gene": "photosystem I P700 apoprotein A2 (psaB)",
        "accession": "NC_005353",
        "cds_start": 77900, "cds_end": 80100,
        "type": "genomic_region",
        "gc_approx": 34,
    },
    # ───────────────────────────────────────────────────────────────
    # ORGANELLAR — Apicoplast (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_plasmodium_apicoplast_rpoB",
        "component": "A", "category": "organellar_apicoplast",
        "lineage": "Eukarya; Apicomplexa (apicoplast)",
        "species": "Plasmodium falciparum",
        "gene": "RNA polymerase beta (apicoplast)",
        "accession": "NC_002375",
        "cds_start": 20763, "cds_end": 22000,
        "type": "genomic_region",
        "gc_approx": 14,
        "note": "Apicoplast genome is ~14% GC — most extreme.",
    },
    {
        "name": "a_toxoplasma_apicoplast_tufA",
        "component": "A", "category": "organellar_apicoplast",
        "lineage": "Eukarya; Apicomplexa (apicoplast)",
        "species": "Toxoplasma gondii",
        "gene": "elongation factor Tu (apicoplast)",
        "accession": "NC_004823",
        "cds_start": 21150, "cds_end": 22350,
        "type": "genomic_region",
        "gc_approx": 21,
    },
    # ───────────────────────────────────────────────────────────────
    # ORGANELLAR — Mito relatives: Rickettsia/Wolbachia (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_wolbachia_wsp",
        "component": "A", "category": "organellar_relative",
        "lineage": "Bacteria; Alphaproteobacteria; Rickettsiales",
        "species": "Wolbachia endosymbiont of D. melanogaster",
        "gene": "Wolbachia surface protein (Wsp)",
        "accession": "AAS14330",
        "type": "protein_cds",
        "gc_approx": 35,
        "note": "Obligate endosymbiont of arthropods, mito relative.",
    },
    {
        "name": "a_rickettsia_ompB",
        "component": "A", "category": "organellar_relative",
        "lineage": "Bacteria; Alphaproteobacteria; Rickettsiales",
        "species": "Rickettsia prowazekii",
        "gene": "outer membrane protein B",
        "accession": "NP_220768",
        "type": "protein_cds",
        "gc_approx": 29,
    },
    # ───────────────────────────────────────────────────────────────
    # VIRUSES — Giant DNA (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_mimivirus_capsid",
        "component": "A", "category": "virus_giant_dna",
        "lineage": "Virus; Nucleocytoviricota; Megaviricetes",
        "species": "Acanthamoeba polyphaga mimivirus",
        "gene": "major capsid protein",
        "accession": "AAV50707",
        "type": "protein_cds",
        "gc_approx": 28,
    },
    {
        "name": "a_pandoravirus_dnaPolB",
        "component": "A", "category": "virus_giant_dna",
        "lineage": "Virus; Nucleocytoviricota",
        "species": "Pandoravirus salinus",
        "gene": "DNA polymerase B family",
        "accession": "AGI04718",
        "type": "protein_cds",
        "gc_approx": 62,
    },
    # ───────────────────────────────────────────────────────────────
    # VIRUSES — ssDNA (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_phix174_capsidF",
        "component": "A", "category": "virus_ssdna",
        "lineage": "Virus; Microviridae",
        "species": "Enterobacteria phage phiX174",
        "gene": "major capsid protein F",
        "accession": "NC_001422",
        "cds_start": 1001, "cds_end": 2284,
        "type": "genomic_region",
        "gc_approx": 44,
    },
    {
        "name": "a_aav2_capsid",
        "component": "A", "category": "virus_ssdna",
        "lineage": "Virus; Parvoviridae; Dependoparvovirus",
        "species": "Adeno-associated virus 2",
        "gene": "capsid VP1",
        "accession": "NC_001401",
        "cds_start": 2203, "cds_end": 4410,
        "type": "genomic_region",
        "gc_approx": 48,
    },
    # ───────────────────────────────────────────────────────────────
    # VIRUSES — dsDNA phage (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_t4_gp23",
        "component": "A", "category": "virus_dsdna_phage",
        "lineage": "Virus; Caudoviricetes; Myoviridae",
        "species": "Enterobacteria phage T4",
        "gene": "major capsid protein gp23",
        "accession": "NC_000866",
        "cds_start": 20612, "cds_end": 22141,
        "type": "genomic_region",
        "gc_approx": 35,
    },
    {
        "name": "a_lambda_repressor",
        "component": "A", "category": "virus_dsdna_phage",
        "lineage": "Virus; Caudoviricetes; Siphoviridae",
        "species": "Enterobacteria phage lambda",
        "gene": "cI repressor",
        "accession": "NC_001416",
        "cds_start": 37940, "cds_end": 38654,
        "type": "genomic_region",
        "gc_approx": 49,
    },
    # ───────────────────────────────────────────────────────────────
    # VIRUSES — RNA virus cDNA (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_influenza_ha",
        "component": "A", "category": "virus_rna_cdna",
        "lineage": "Virus; Orthomyxoviridae",
        "species": "Influenza A virus (H1N1)",
        "gene": "hemagglutinin (HA)",
        "accession": "NC_026433",
        "cds_start": 33, "cds_end": 1730,
        "type": "genomic_region",
        "gc_approx": 42,
    },
    {
        "name": "a_sars2_spike",
        "component": "A", "category": "virus_rna_cdna",
        "lineage": "Virus; Coronaviridae; Betacoronavirus",
        "species": "SARS-CoV-2",
        "gene": "spike glycoprotein (partial CDS)",
        "accession": "NC_045512",
        "cds_start": 21563, "cds_end": 23063,
        "type": "genomic_region",
        "gc_approx": 37,
    },
    # ───────────────────────────────────────────────────────────────
    # VIRUSES — Retrovirus (2)
    # ───────────────────────────────────────────────────────────────
    {
        "name": "a_hiv1_gag",
        "component": "A", "category": "virus_retrovirus",
        "lineage": "Virus; Retroviridae; Lentivirus",
        "species": "HIV-1 HXB2",
        "gene": "gag polyprotein (partial)",
        "accession": "NC_001802",
        "cds_start": 790, "cds_end": 2290,
        "type": "genomic_region",
        "gc_approx": 40,
    },
    {
        "name": "a_htlv1_tax",
        "component": "A", "category": "virus_retrovirus",
        "lineage": "Virus; Retroviridae; Deltaretrovirus",
        "species": "HTLV-1",
        "gene": "Tax transactivator",
        "accession": "NC_001436",
        "cds_start": 7296, "cds_end": 8360,
        "type": "genomic_region",
        "gc_approx": 48,
    },
    # ───────────────────────────────────────────────────────────────
    # NON-CODING CONTROLS (~35 sequences)
    # Should NOT show offset-3 cosine inversion or codon periodicity
    # ───────────────────────────────────────────────────────────────
    # rRNAs (5)
    {
        "name": "a_nc_human_18s",
        "component": "A", "category": "noncoding_rRNA",
        "lineage": "Eukarya; Chordata; Mammalia",
        "species": "Homo sapiens",
        "gene": "18S ribosomal RNA",
        "accession": "NR_003286",
        "cds_start": 1, "cds_end": 1200,
        "type": "mrna_cds",
        "gc_approx": 54,
        "noncoding": True,
    },
    {
        "name": "a_nc_ecoli_23s",
        "component": "A", "category": "noncoding_rRNA",
        "lineage": "Bacteria; Gammaproteobacteria",
        "species": "Escherichia coli",
        "gene": "23S ribosomal RNA",
        "accession": "NR_103073",
        "cds_start": 1, "cds_end": 1200,
        "type": "mrna_cds",
        "gc_approx": 53,
        "noncoding": True,
    },
    {
        "name": "a_nc_archaea_16s",
        "component": "A", "category": "noncoding_rRNA",
        "lineage": "Archaea; Euryarchaeota; Methanobacteria",
        "species": "Methanobacterium thermoautotrophicum",
        "gene": "16S ribosomal RNA",
        "accession": "NR_040233",
        "cds_start": 1, "cds_end": 1200,
        "type": "mrna_cds",
        "gc_approx": 55,
        "noncoding": True,
    },
    {
        "name": "a_nc_yeast_5_8s",
        "component": "A", "category": "noncoding_rRNA",
        "lineage": "Eukarya; Ascomycota; Saccharomycetes",
        "species": "Saccharomyces cerevisiae",
        "gene": "5.8S + ITS rRNA region",
        "accession": "NR_132175",
        "type": "full_cds",
        "gc_approx": 42,
        "noncoding": True,
    },
    {
        "name": "a_nc_plastid_16s",
        "component": "A", "category": "noncoding_rRNA",
        "lineage": "Eukarya; Streptophyta (chloroplast)",
        "species": "Arabidopsis thaliana",
        "gene": "16S ribosomal RNA (chloroplast)",
        "accession": "NC_000932",
        "cds_start": 101020, "cds_end": 102510,
        "type": "genomic_region",
        "gc_approx": 53,
        "noncoding": True,
    },
    # tRNA clusters (5)
    {
        "name": "a_nc_ecoli_tRNA_cluster",
        "component": "A", "category": "noncoding_tRNA",
        "lineage": "Bacteria; Gammaproteobacteria",
        "species": "Escherichia coli K-12",
        "gene": "tRNA cluster (thrU-tyrU-glyT-thrT)",
        "accession": "U00096",
        "cds_start": 172690, "cds_end": 173490,
        "type": "genomic_region",
        "gc_approx": 56,
        "noncoding": True,
    },
    {
        "name": "a_nc_human_tRNA_leu",
        "component": "A", "category": "noncoding_tRNA",
        "lineage": "Eukarya; Chordata; Mammalia (mitochondrial)",
        "species": "Homo sapiens",
        "gene": "mt-tRNA cluster (between ND1 and ND2)",
        "accession": "NC_012920",
        "cds_start": 3230, "cds_end": 4262,
        "type": "genomic_region",
        "gc_approx": 38,
        "noncoding": True,
    },
    {
        "name": "a_nc_yeast_tRNA_cluster",
        "component": "A", "category": "noncoding_tRNA",
        "lineage": "Eukarya; Ascomycota; Saccharomycetes",
        "species": "Saccharomyces cerevisiae",
        "gene": "tRNA gene cluster (chr III)",
        "accession": "NC_001135",
        "cds_start": 92500, "cds_end": 93300,
        "type": "genomic_region",
        "gc_approx": 38,
        "noncoding": True,
    },
    {
        "name": "a_nc_bacillus_tRNA",
        "component": "A", "category": "noncoding_tRNA",
        "lineage": "Bacteria; Bacillota; Bacilli",
        "species": "Bacillus subtilis",
        "gene": "tRNA cluster",
        "accession": "NC_000964",
        "cds_start": 62000, "cds_end": 62800,
        "type": "genomic_region",
        "gc_approx": 44,
        "noncoding": True,
    },
    {
        "name": "a_nc_rice_cp_tRNA",
        "component": "A", "category": "noncoding_tRNA",
        "lineage": "Eukarya; Streptophyta (chloroplast)",
        "species": "Oryza sativa",
        "gene": "chloroplast tRNA cluster",
        "accession": "NC_001320",
        "cds_start": 6900, "cds_end": 7700,
        "type": "genomic_region",
        "gc_approx": 42,
        "noncoding": True,
    },
    # lncRNAs (5)
    {
        "name": "a_nc_human_malat1",
        "component": "A", "category": "noncoding_lncRNA",
        "lineage": "Eukarya; Chordata; Mammalia",
        "species": "Homo sapiens",
        "gene": "MALAT1 lncRNA",
        "accession": "NR_002819",
        "cds_start": 1, "cds_end": 1200,
        "type": "mrna_cds",
        "gc_approx": 42,
        "noncoding": True,
    },
    {
        "name": "a_nc_human_xist",
        "component": "A", "category": "noncoding_lncRNA",
        "lineage": "Eukarya; Chordata; Mammalia",
        "species": "Homo sapiens",
        "gene": "XIST lncRNA (X-inactivation, partial)",
        "accession": "NR_001564",
        "cds_start": 1, "cds_end": 1500,
        "type": "mrna_cds",
        "gc_approx": 38,
        "noncoding": True,
    },
    {
        "name": "a_nc_mouse_hotair",
        "component": "A", "category": "noncoding_lncRNA",
        "lineage": "Eukarya; Chordata; Mammalia",
        "species": "Mus musculus",
        "gene": "Hotair lncRNA",
        "accession": "NR_047528",
        "cds_start": 1, "cds_end": 1200,
        "type": "mrna_cds",
        "gc_approx": 40,
        "noncoding": True,
    },
    {
        "name": "a_nc_drosophila_roX1",
        "component": "A", "category": "noncoding_lncRNA",
        "lineage": "Eukarya; Arthropoda; Insecta",
        "species": "Drosophila melanogaster",
        "gene": "roX1 lncRNA (dosage compensation)",
        "accession": "NR_133500",
        "cds_start": 1, "cds_end": 1500,
        "type": "mrna_cds",
        "gc_approx": 40,
        "noncoding": True,
    },
    {
        "name": "a_nc_arabidopsis_coolair",
        "component": "A", "category": "noncoding_lncRNA",
        "lineage": "Eukarya; Streptophyta; Magnoliopsida",
        "species": "Arabidopsis thaliana",
        "gene": "COOLAIR antisense lncRNA (FLC region)",
        "accession": "NC_003076",
        "cds_start": 3172800, "cds_end": 3174000,
        "type": "genomic_region",
        "gc_approx": 36,
        "noncoding": True,
    },
    # Introns (5)
    {
        "name": "a_nc_human_brca1_intron",
        "component": "A", "category": "noncoding_intron",
        "lineage": "Eukarya; Chordata; Mammalia",
        "species": "Homo sapiens",
        "gene": "BRCA1 intron 2",
        "accession": "NG_005905",
        "cds_start": 20000, "cds_end": 21200,
        "type": "genomic_region",
        "gc_approx": 38,
        "noncoding": True,
    },
    {
        "name": "a_nc_drosophila_intron",
        "component": "A", "category": "noncoding_intron",
        "lineage": "Eukarya; Arthropoda; Insecta",
        "species": "Drosophila melanogaster",
        "gene": "white gene intron 3",
        "accession": "NT_033779",
        "cds_start": 18080000, "cds_end": 18081200,
        "type": "genomic_region",
        "gc_approx": 38,
        "noncoding": True,
    },
    {
        "name": "a_nc_maize_adh1_intron",
        "component": "A", "category": "noncoding_intron",
        "lineage": "Eukarya; Streptophyta; Liliopsida",
        "species": "Zea mays",
        "gene": "adh1 intron 1 (often used as enhancer)",
        "accession": "M32984",
        "type": "full_cds",
        "gc_approx": 44,
        "noncoding": True,
    },
    {
        "name": "a_nc_celegans_unc54_intron",
        "component": "A", "category": "noncoding_intron",
        "lineage": "Eukarya; Nematoda",
        "species": "Caenorhabditis elegans",
        "gene": "unc-54 intron region",
        "accession": "NC_003283",
        "cds_start": 7890000, "cds_end": 7891200,
        "type": "genomic_region",
        "gc_approx": 36,
        "noncoding": True,
    },
    {
        "name": "a_nc_yeast_intron",
        "component": "A", "category": "noncoding_intron",
        "lineage": "Eukarya; Ascomycota; Saccharomycetes",
        "species": "Saccharomyces cerevisiae",
        "gene": "ribosomal protein L30 intron (one of few yeast introns)",
        "accession": "NC_001139",
        "cds_start": 397800, "cds_end": 398700,
        "type": "genomic_region",
        "gc_approx": 37,
        "noncoding": True,
    },
    # Intergenic regions (5)
    {
        "name": "a_nc_ecoli_intergenic",
        "component": "A", "category": "noncoding_intergenic",
        "lineage": "Bacteria; Gammaproteobacteria",
        "species": "Escherichia coli K-12",
        "gene": "intergenic region (lacY-lacA)",
        "accession": "U00096",
        "cds_start": 362463, "cds_end": 363463,
        "type": "genomic_region",
        "gc_approx": 50,
        "noncoding": True,
    },
    {
        "name": "a_nc_bacillus_intergenic",
        "component": "A", "category": "noncoding_intergenic",
        "lineage": "Bacteria; Bacillota; Bacilli",
        "species": "Bacillus subtilis 168",
        "gene": "intergenic region",
        "accession": "NC_000964",
        "cds_start": 1450000, "cds_end": 1451000,
        "type": "genomic_region",
        "gc_approx": 44,
        "noncoding": True,
    },
    {
        "name": "a_nc_human_intergenic_chr1",
        "component": "A", "category": "noncoding_intergenic",
        "lineage": "Eukarya; Chordata; Mammalia",
        "species": "Homo sapiens",
        "gene": "gene desert chr1 (no nearby genes)",
        "accession": "NC_000001",
        "cds_start": 150000000, "cds_end": 150001200,
        "type": "genomic_region",
        "gc_approx": 40,
        "noncoding": True,
    },
    {
        "name": "a_nc_archaea_intergenic",
        "component": "A", "category": "noncoding_intergenic",
        "lineage": "Archaea; Euryarchaeota; Methanococci",
        "species": "Methanocaldococcus jannaschii",
        "gene": "intergenic region",
        "accession": "NC_000909",
        "cds_start": 500000, "cds_end": 501000,
        "type": "genomic_region",
        "gc_approx": 31,
        "noncoding": True,
    },
    {
        "name": "a_nc_yeast_intergenic",
        "component": "A", "category": "noncoding_intergenic",
        "lineage": "Eukarya; Ascomycota; Saccharomycetes",
        "species": "Saccharomyces cerevisiae",
        "gene": "chr IV large intergenic region",
        "accession": "NC_001136",
        "cds_start": 460000, "cds_end": 461000,
        "type": "genomic_region",
        "gc_approx": 38,
        "noncoding": True,
    },
    # Repetitive elements (5)
    {
        "name": "a_nc_human_alu",
        "component": "A", "category": "noncoding_repeat",
        "lineage": "Eukarya; Chordata; Mammalia",
        "species": "Homo sapiens",
        "gene": "Alu SINE consensus",
        "accession": "HARDCODED",
        "type": "hardcoded",
        "gc_approx": 56,
        "noncoding": True,
        "sequence": (
            "GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGGCGGGCGG"
            "ATCACGAGGTCAGGAGATCGAGACCATCCTGGCTAACACGGTGAAACCCCGTCTCTACTA"
            "AAAATACAAAAAATTAGCCGGGCGTGGTGGCGGGCGCCTGTAGTCCCAGCTACTCGGGAG"
            "GCTGAGGCAGGAGAATGGCGTGAACCCGGGAGGCGGAGCTTGCAGTGAGCCGAGATTGC"
            "GCCACTGCACTCCAGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAAAAAAAAAAAAA"
        ),
    },
    {
        "name": "a_nc_human_line1",
        "component": "A", "category": "noncoding_repeat",
        "lineage": "Eukarya; Chordata; Mammalia",
        "species": "Homo sapiens",
        "gene": "L1 LINE 5' UTR (promoter region, non-coding part)",
        "accession": "NC_000001",
        "cds_start": 100012000, "cds_end": 100013200,
        "type": "genomic_region",
        "gc_approx": 42,
        "noncoding": True,
    },
    {
        "name": "a_nc_drosophila_roo",
        "component": "A", "category": "noncoding_repeat",
        "lineage": "Eukarya; Arthropoda; Insecta",
        "species": "Drosophila melanogaster",
        "gene": "roo LTR retrotransposon (LTR region only)",
        "accession": "AJ010092",
        "cds_start": 1, "cds_end": 500,
        "type": "genomic_region",
        "gc_approx": 36,
        "noncoding": True,
    },
    {
        "name": "a_nc_yeast_telomere",
        "component": "A", "category": "noncoding_repeat",
        "lineage": "Eukarya; Ascomycota; Saccharomycetes",
        "species": "Saccharomyces cerevisiae",
        "gene": "telomeric/subtelomeric repeat",
        "accession": "HARDCODED",
        "type": "hardcoded",
        "gc_approx": 35,
        "noncoding": True,
        "sequence": (
            "TGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTG"
            "TGTGTGGGTGTGTGTGGTGTGTGTGGTGTGTGGGTGTGTGTGTGGTGTGTGGTGTGTGG"
            "TGTGTGTGGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGGTGTGTGTGG"
            "TGTGTGGTGTGTGTGGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGGTGTGTG"
            "TGTGTGGTGTGTGGGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTG"
        ),
    },
    {
        "name": "a_nc_human_satellite",
        "component": "A", "category": "noncoding_repeat",
        "lineage": "Eukarya; Chordata; Mammalia",
        "species": "Homo sapiens",
        "gene": "alpha-satellite centromeric repeat",
        "accession": "HARDCODED",
        "type": "hardcoded",
        "gc_approx": 38,
        "noncoding": True,
        # 171bp alpha-satellite monomer repeated ~3.5x
        "sequence": (
            "AATCTCAAGTGGATATTTGGAGCGCTTTGAGGCCTATGGTGGAAAAGGAAATATCTTCAC"
            "ATAAAAACTAGACAGAAGCATTCTCAGAAACTTCTTTGTGATGTTTGCATTCAACTCACA"
            "GAGTTGAACGATCCTTTACAACAAAAAGAATCTCAAGTGGATATTTGGAGCGCTTTGAGG"
            "CCTATGGTGGAAAAGGAAATATCTTCACATAAAAACTAGACAGAAGCATTCTCAGAAACT"
            "TCTTTGTGATGTTTGCATTCAACTCACAGAGTTGAACGATCCTTTACAACAAAAAGAAT"
            "CTCAAGTGGATATTTGGAGCGCTTTGAGGCCTATGGTGGAAAAGGAAATATCTTCACAT"
            "AAAAACTAGACAGAAGCATTCTCAGAAACTTCTTTGTGATGTTTGCATTCAACTCACAG"
            "AGTTGAACGATCCTTTACAACAAAAAGAATCTCAAGTGGATATTTGGAGCGCTTTGAGG"
            "CCTATGGTGGAAAAGGAAATATCTTCACATAAAAACTAGAC"
        ),
    },
]

# Count Component A
_n_comp_a = len(COMPONENT_A)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT B: FUNCTIONAL CLUSTERING PANEL (10 families × 30 orthologs = 300)
# Same gene from many species → should cluster by gene family, not by taxonomy.
# ═══════════════════════════════════════════════════════════════════════════════
#
# Helper to create entries compactly. Each family is a list of
# (name_suffix, species, lineage, accession, type, gc_approx) tuples.
# The gene family name and component/category are added automatically.


def _make_family(family_name: str, gene: str, members: list[tuple]) -> list[dict]:
    """Build Component B entries for one gene family."""
    entries = []
    for suffix, species, lineage, acc, acc_type, gc in members:
        entry: dict = {
            "name": f"b_{family_name}_{suffix}",
            "component": "B",
            "category": f"family_{family_name}",
            "lineage": lineage,
            "species": species,
            "gene": gene,
            "accession": acc,
            "type": acc_type,
            "gc_approx": gc,
        }
        entries.append(entry)
    return entries


# ───────────────────────────────────────────────────────────────
# Family 1: ACTIN (eukaryotic cytoskeletal + prokaryotic MreB)
# ───────────────────────────────────────────────────────────────
_ACTIN = _make_family("actin", "actin / MreB", [
    # Bacteria (MreB homologs)
    ("ecoli_mreB", "Escherichia coli K-12", "Bacteria; Gammaproteobacteria", "NP_414687", "protein_cds", 51),
    ("bacillus_mreB", "Bacillus subtilis", "Bacteria; Bacillota", "NP_390456", "protein_cds", 44),
    ("caulobacter_mreB", "Caulobacter vibrioides", "Bacteria; Alphaproteobacteria", "NP_420630", "protein_cds", 67),
    # Archaea
    ("thermoplasma_ta0359", "Thermoplasma acidophilum", "Archaea; Euryarchaeota", "NP_393757", "protein_cds", 46),
    ("lokiarch_actin", "Ca. Lokiarchaeota archaeon", "Archaea; Asgardarchaeota", "KKK40388", "protein_cds", 32),
    # Fungi
    ("yeast_act1", "Saccharomyces cerevisiae", "Eukarya; Ascomycota", "NP_116614", "protein_cds", 40),
    ("neurospora_act", "Neurospora crassa", "Eukarya; Ascomycota", "XP_962212", "protein_cds", 56),
    ("ustilago_act", "Ustilago maydis", "Eukarya; Basidiomycota", "XP_011388763", "protein_cds", 57),
    # Plants
    ("arabidopsis_act2", "Arabidopsis thaliana", "Eukarya; Streptophyta", "NP_188508", "protein_cds", 44),
    ("rice_act1", "Oryza sativa", "Eukarya; Streptophyta", "XP_015620018", "protein_cds", 58),
    ("moss_act", "Physcomitrium patens", "Eukarya; Streptophyta", "XP_024384499", "protein_cds", 48),
    # Vertebrates
    ("human_actb", "Homo sapiens", "Eukarya; Chordata; Mammalia", "NP_001092", "protein_cds", 56),
    ("mouse_actb", "Mus musculus", "Eukarya; Chordata; Mammalia", "NP_031419", "protein_cds", 55),
    ("chicken_actb", "Gallus gallus", "Eukarya; Chordata; Aves", "NP_990849", "protein_cds", 52),
    ("xenopus_actb", "Xenopus laevis", "Eukarya; Chordata; Amphibia", "NP_001082423", "protein_cds", 51),
    ("zebrafish_actb", "Danio rerio", "Eukarya; Chordata; Actinopterygii", "NP_853637", "protein_cds", 52),
    # Invertebrates
    ("drosophila_act5c", "Drosophila melanogaster", "Eukarya; Arthropoda; Insecta", "NP_524822", "protein_cds", 54),
    ("celegans_act1", "Caenorhabditis elegans", "Eukarya; Nematoda", "NP_498092", "protein_cds", 48),
    ("oyster_act", "Crassostrea gigas", "Eukarya; Mollusca", "NP_001292206", "protein_cds", 44),
    ("urchin_act", "Strongylocentrotus purpuratus", "Eukarya; Echinodermata", "NP_999702", "protein_cds", 44),
    # Protists
    ("plasmodium_actI", "Plasmodium falciparum", "Eukarya; Apicomplexa", "XP_001351786", "protein_cds", 24),
    ("toxoplasma_act", "Toxoplasma gondii", "Eukarya; Apicomplexa", "XP_002365003", "protein_cds", 52),
    ("trypanosoma_act", "Trypanosoma brucei", "Eukarya; Euglenozoa", "XP_011775780", "protein_cds", 55),
    ("dictyostelium_act", "Dictyostelium discoideum", "Eukarya; Amoebozoa", "XP_642978", "protein_cds", 28),
    ("tetrahymena_act", "Tetrahymena thermophila", "Eukarya; Ciliophora", "XP_001012807", "protein_cds", 32),
    # Algae
    ("chlamy_act", "Chlamydomonas reinhardtii", "Eukarya; Chlorophyta", "XP_001696476", "protein_cds", 64),
    ("phaeodactylum_act", "Phaeodactylum tricornutum", "Eukarya; Bacillariophyta", "XP_002181263", "protein_cds", 49),
    # Diverse extras
    ("hydra_act", "Hydra vulgaris", "Eukarya; Cnidaria", "XP_012556244", "protein_cds", 37),
    ("honeybee_act", "Apis mellifera", "Eukarya; Arthropoda", "XP_006563113", "protein_cds", 36),
    ("ciona_act", "Ciona intestinalis", "Eukarya; Chordata; Tunicata", "XP_002125261", "protein_cds", 42),
])

# ───────────────────────────────────────────────────────────────
# Family 2: GAPDH (glycolysis, universal)
# ───────────────────────────────────────────────────────────────
_GAPDH = _make_family("gapdh", "glyceraldehyde-3-phosphate dehydrogenase", [
    # Bacteria
    ("ecoli", "Escherichia coli K-12", "Bacteria; Gammaproteobacteria", "NP_416293", "protein_cds", 51),
    ("bacillus", "Bacillus subtilis", "Bacteria; Bacillota", "NP_391534", "protein_cds", 44),
    ("pseudomonas", "Pseudomonas aeruginosa", "Bacteria; Gammaproteobacteria", "NP_249222", "protein_cds", 66),
    ("mycobacterium", "Mycobacterium tuberculosis", "Bacteria; Actinomycetota", "NP_215288", "protein_cds", 66),
    ("synechocystis", "Synechocystis sp. PCC 6803", "Bacteria; Cyanobacteriota", "NP_440657", "protein_cds", 47),
    # Archaea
    ("pyrococcus", "Pyrococcus furiosus", "Archaea; Euryarchaeota", "AAL81075", "protein_cds", 41),
    ("sulfolobus", "Sulfolobus solfataricus", "Archaea; Crenarchaeota", "NP_344361", "protein_cds", 36),
    ("methanosarcina", "Methanosarcina acetivorans", "Archaea; Euryarchaeota", "NP_618055", "protein_cds", 43),
    # Fungi
    ("yeast_tdh3", "Saccharomyces cerevisiae", "Eukarya; Ascomycota", "NP_012483", "protein_cds", 40),
    ("aspergillus", "Aspergillus nidulans", "Eukarya; Ascomycota", "XP_660879", "protein_cds", 54),
    # Plants
    ("arabidopsis", "Arabidopsis thaliana", "Eukarya; Streptophyta", "NP_187062", "protein_cds", 44),
    ("rice", "Oryza sativa", "Eukarya; Streptophyta", "XP_015635459", "protein_cds", 60),
    ("maize", "Zea mays", "Eukarya; Streptophyta", "NP_001104869", "protein_cds", 60),
    ("tomato", "Solanum lycopersicum", "Eukarya; Streptophyta", "NP_001234340", "protein_cds", 42),
    # Vertebrates
    ("human", "Homo sapiens", "Eukarya; Chordata; Mammalia", "NP_002037", "protein_cds", 56),
    ("mouse", "Mus musculus", "Eukarya; Chordata; Mammalia", "NP_001276675", "protein_cds", 51),
    ("chicken", "Gallus gallus", "Eukarya; Chordata; Aves", "NP_989636", "protein_cds", 54),
    ("zebrafish", "Danio rerio", "Eukarya; Chordata; Actinopterygii", "NP_998086", "protein_cds", 50),
    ("xenopus", "Xenopus laevis", "Eukarya; Chordata; Amphibia", "NP_001079511", "protein_cds", 48),
    # Invertebrates
    ("drosophila", "Drosophila melanogaster", "Eukarya; Arthropoda; Insecta", "NP_524375", "protein_cds", 48),
    ("celegans", "Caenorhabditis elegans", "Eukarya; Nematoda", "NP_492055", "protein_cds", 46),
    ("oyster", "Crassostrea gigas", "Eukarya; Mollusca", "XP_011416992", "protein_cds", 42),
    # Protists
    ("plasmodium", "Plasmodium falciparum", "Eukarya; Apicomplexa", "XP_001349727", "protein_cds", 26),
    ("trypanosoma", "Trypanosoma brucei", "Eukarya; Euglenozoa", "XP_011777416", "protein_cds", 52),
    ("leishmania", "Leishmania major", "Eukarya; Euglenozoa", "XP_001684270", "protein_cds", 62),
    ("dictyostelium", "Dictyostelium discoideum", "Eukarya; Amoebozoa", "XP_643571", "protein_cds", 28),
    ("tetrahymena", "Tetrahymena thermophila", "Eukarya; Ciliophora", "XP_001027003", "protein_cds", 32),
    # Algae
    ("chlamy", "Chlamydomonas reinhardtii", "Eukarya; Chlorophyta", "XP_001693456", "protein_cds", 64),
    ("diatom", "Thalassiosira pseudonana", "Eukarya; Bacillariophyta", "XP_002296129", "protein_cds", 48),
    ("ectocarpus", "Ectocarpus siliculosus", "Eukarya; Phaeophyceae", "CBN74866", "protein_cds", 55),
])

# ───────────────────────────────────────────────────────────────
# Family 3: EF-Tu / EF-1alpha (translation elongation, universal)
# ───────────────────────────────────────────────────────────────
_EFTU = _make_family("eftu", "elongation factor Tu / EF-1alpha", [
    # Bacteria
    ("ecoli", "Escherichia coli K-12", "Bacteria; Gammaproteobacteria", "NP_418402", "protein_cds", 51),
    ("bacillus", "Bacillus subtilis", "Bacteria; Bacillota", "NP_391368", "protein_cds", 44),
    ("mycobacterium", "Mycobacterium tuberculosis", "Bacteria; Actinomycetota", "NP_217399", "protein_cds", 66),
    ("thermus", "Thermus thermophilus", "Bacteria; Deinococcota", "WP_011172576", "protein_cds", 68),
    ("synechocystis", "Synechocystis sp. PCC 6803", "Bacteria; Cyanobacteriota", "NP_441742", "protein_cds", 47),
    ("borrelia", "Borrelia burgdorferi", "Bacteria; Spirochaetota", "NP_212432", "protein_cds", 28),
    # Archaea
    ("mjannaschii", "Methanocaldococcus jannaschii", "Archaea; Euryarchaeota", "NP_247637", "protein_cds", 31),
    ("sulfolobus", "Sulfolobus solfataricus", "Archaea; Crenarchaeota", "NP_343795", "protein_cds", 36),
    ("pyrococcus", "Pyrococcus furiosus", "Archaea; Euryarchaeota", "NP_578530", "protein_cds", 41),
    # Fungi
    ("yeast_tef1", "Saccharomyces cerevisiae", "Eukarya; Ascomycota", "NP_009676", "protein_cds", 40),
    ("aspergillus", "Aspergillus nidulans", "Eukarya; Ascomycota", "XP_680782", "protein_cds", 54),
    # Plants
    ("arabidopsis", "Arabidopsis thaliana", "Eukarya; Streptophyta", "NP_189216", "protein_cds", 48),
    ("rice", "Oryza sativa", "Eukarya; Streptophyta", "XP_015639116", "protein_cds", 60),
    ("maize", "Zea mays", "Eukarya; Streptophyta", "NP_001105236", "protein_cds", 58),
    # Vertebrates
    ("human", "Homo sapiens", "Eukarya; Chordata; Mammalia", "NP_001393", "protein_cds", 52),
    ("mouse", "Mus musculus", "Eukarya; Chordata; Mammalia", "NP_034236", "protein_cds", 50),
    ("chicken", "Gallus gallus", "Eukarya; Chordata; Aves", "NP_001026198", "protein_cds", 52),
    ("zebrafish", "Danio rerio", "Eukarya; Chordata; Actinopterygii", "NP_571528", "protein_cds", 48),
    ("xenopus", "Xenopus laevis", "Eukarya; Chordata; Amphibia", "NP_001079052", "protein_cds", 48),
    # Invertebrates
    ("drosophila", "Drosophila melanogaster", "Eukarya; Arthropoda; Insecta", "NP_477375", "protein_cds", 52),
    ("celegans", "Caenorhabditis elegans", "Eukarya; Nematoda", "NP_498320", "protein_cds", 46),
    ("urchin", "Strongylocentrotus purpuratus", "Eukarya; Echinodermata", "XP_011679263", "protein_cds", 44),
    # Protists
    ("plasmodium", "Plasmodium falciparum", "Eukarya; Apicomplexa", "XP_001352029", "protein_cds", 26),
    ("trypanosoma", "Trypanosoma brucei", "Eukarya; Euglenozoa", "XP_011775068", "protein_cds", 52),
    ("dictyostelium", "Dictyostelium discoideum", "Eukarya; Amoebozoa", "XP_644142", "protein_cds", 28),
    ("tetrahymena", "Tetrahymena thermophila", "Eukarya; Ciliophora", "XP_001026888", "protein_cds", 32),
    # Algae
    ("chlamy", "Chlamydomonas reinhardtii", "Eukarya; Chlorophyta", "XP_001703416", "protein_cds", 64),
    ("diatom", "Thalassiosira pseudonana", "Eukarya; Bacillariophyta", "XP_002293858", "protein_cds", 48),
    # Diverse
    ("hydra", "Hydra vulgaris", "Eukarya; Cnidaria", "XP_002162116", "protein_cds", 40),
    ("ciona", "Ciona intestinalis", "Eukarya; Chordata; Tunicata", "XP_002130618", "protein_cds", 42),
])

# ───────────────────────────────────────────────────────────────
# Family 4: HSP70 / DnaK (chaperone, universal)
# ───────────────────────────────────────────────────────────────
_HSP70 = _make_family("hsp70", "HSP70 / DnaK chaperone", [
    # Bacteria
    ("ecoli", "Escherichia coli K-12", "Bacteria; Gammaproteobacteria", "NP_414555", "protein_cds", 51),
    ("bacillus", "Bacillus subtilis", "Bacteria; Bacillota", "NP_389107", "protein_cds", 44),
    ("mycobacterium", "Mycobacterium tuberculosis", "Bacteria; Actinomycetota", "NP_217272", "protein_cds", 66),
    ("thermus", "Thermus thermophilus", "Bacteria; Deinococcota", "WP_011173021", "protein_cds", 68),
    ("synechocystis", "Synechocystis sp. PCC 6803", "Bacteria; Cyanobacteriota", "NP_441675", "protein_cds", 47),
    # Archaea
    ("mjannaschii", "Methanocaldococcus jannaschii", "Archaea; Euryarchaeota", "NP_247340", "protein_cds", 31),
    ("sulfolobus", "Sulfolobus solfataricus", "Archaea; Crenarchaeota", "NP_342740", "protein_cds", 36),
    ("pyrococcus", "Pyrococcus furiosus", "Archaea; Euryarchaeota", "NP_579043", "protein_cds", 41),
    # Fungi
    ("yeast_ssa1", "Saccharomyces cerevisiae", "Eukarya; Ascomycota", "NP_009396", "protein_cds", 40),
    ("aspergillus", "Aspergillus nidulans", "Eukarya; Ascomycota", "XP_658449", "protein_cds", 54),
    # Plants
    ("arabidopsis", "Arabidopsis thaliana", "Eukarya; Streptophyta", "NP_187751", "protein_cds", 44),
    ("rice", "Oryza sativa", "Eukarya; Streptophyta", "XP_015617756", "protein_cds", 60),
    ("soybean", "Glycine max", "Eukarya; Streptophyta", "XP_003525291", "protein_cds", 44),
    # Vertebrates
    ("human_hspa1a", "Homo sapiens", "Eukarya; Chordata; Mammalia", "NP_005336", "protein_cds", 56),
    ("mouse", "Mus musculus", "Eukarya; Chordata; Mammalia", "NP_034609", "protein_cds", 54),
    ("chicken", "Gallus gallus", "Eukarya; Chordata; Aves", "NP_001006686", "protein_cds", 54),
    ("zebrafish", "Danio rerio", "Eukarya; Chordata; Actinopterygii", "NP_571383", "protein_cds", 48),
    ("xenopus", "Xenopus laevis", "Eukarya; Chordata; Amphibia", "NP_001080289", "protein_cds", 48),
    # Invertebrates
    ("drosophila", "Drosophila melanogaster", "Eukarya; Arthropoda; Insecta", "NP_524756", "protein_cds", 48),
    ("celegans", "Caenorhabditis elegans", "Eukarya; Nematoda", "NP_503068", "protein_cds", 44),
    ("beetle", "Tribolium castaneum", "Eukarya; Arthropoda; Insecta", "NP_001036876", "protein_cds", 44),
    # Protists
    ("plasmodium", "Plasmodium falciparum", "Eukarya; Apicomplexa", "XP_001351528", "protein_cds", 26),
    ("leishmania", "Leishmania major", "Eukarya; Euglenozoa", "XP_001684442", "protein_cds", 62),
    ("trypanosoma", "Trypanosoma brucei", "Eukarya; Euglenozoa", "XP_011771838", "protein_cds", 52),
    ("dictyostelium", "Dictyostelium discoideum", "Eukarya; Amoebozoa", "XP_637001", "protein_cds", 28),
    ("tetrahymena", "Tetrahymena thermophila", "Eukarya; Ciliophora", "XP_001011284", "protein_cds", 32),
    # Algae & other
    ("chlamy", "Chlamydomonas reinhardtii", "Eukarya; Chlorophyta", "XP_001693095", "protein_cds", 64),
    ("diatom", "Thalassiosira pseudonana", "Eukarya; Bacillariophyta", "XP_002289997", "protein_cds", 48),
    ("hydra", "Hydra vulgaris", "Eukarya; Cnidaria", "XP_002160498", "protein_cds", 38),
    ("ciona", "Ciona intestinalis", "Eukarya; Chordata; Tunicata", "XP_002119977", "protein_cds", 42),
])

# ───────────────────────────────────────────────────────────────
# Family 5: Cytochrome c oxidase I (COI) — electron transport
# ───────────────────────────────────────────────────────────────
_COI = _make_family("coi", "cytochrome c oxidase subunit I", [
    # Bacteria (aerobic respiration)
    ("ecoli", "Escherichia coli K-12", "Bacteria; Gammaproteobacteria", "NP_415261", "protein_cds", 51),
    ("bacillus", "Bacillus subtilis", "Bacteria; Bacillota", "NP_391096", "protein_cds", 44),
    ("pseudomonas", "Pseudomonas aeruginosa", "Bacteria; Gammaproteobacteria", "NP_251282", "protein_cds", 66),
    ("thermus", "Thermus thermophilus", "Bacteria; Deinococcota", "WP_011174113", "protein_cds", 68),
    ("rhodobacter", "Rhodobacter sphaeroides", "Bacteria; Alphaproteobacteria", "YP_353153", "protein_cds", 64),
    # Mitochondrial (diverse eukaryotes)
    ("human_mt", "Homo sapiens", "Eukarya; Chordata; Mammalia (mt)", "YP_003024028", "protein_cds", 38),
    ("mouse_mt", "Mus musculus", "Eukarya; Chordata; Mammalia (mt)", "YP_003024034", "protein_cds", 37),
    ("chicken_mt", "Gallus gallus", "Eukarya; Chordata; Aves (mt)", "YP_003548963", "protein_cds", 44),
    ("zebrafish_mt", "Danio rerio", "Eukarya; Chordata; Actinopterygii (mt)", "YP_002221250", "protein_cds", 40),
    ("xenopus_mt", "Xenopus laevis", "Eukarya; Chordata; Amphibia (mt)", "YP_009117088", "protein_cds", 42),
    ("drosophila_mt", "Drosophila melanogaster", "Eukarya; Arthropoda; Insecta (mt)", "YP_009047318", "protein_cds", 22),
    ("celegans_mt", "Caenorhabditis elegans", "Eukarya; Nematoda (mt)", "NP_006951", "protein_cds", 28),
    ("octopus_mt", "Octopus vulgaris", "Eukarya; Mollusca (mt)", "YP_009058054", "protein_cds", 36),
    ("urchin_mt", "Strongylocentrotus purpuratus", "Eukarya; Echinodermata (mt)", "NP_007675", "protein_cds", 38),
    ("shark_mt", "Callorhinchus milii", "Eukarya; Chondrichthyes (mt)", "YP_008475450", "protein_cds", 40),
    ("lamprey_mt", "Petromyzon marinus", "Eukarya; Hyperoartia (mt)", "YP_007890410", "protein_cds", 42),
    ("honeybee_mt", "Apis mellifera", "Eukarya; Arthropoda (mt)", "NP_008206", "protein_cds", 16),
    ("yeast_mt", "Saccharomyces cerevisiae", "Eukarya; Ascomycota (mt)", "NP_009308", "protein_cds", 17),
    ("arabidopsis_mt", "Arabidopsis thaliana", "Eukarya; Streptophyta (mt)", "NP_085516", "protein_cds", 44),
    # Diverse euks with mt COI
    ("coral_mt", "Acropora tenuis", "Eukarya; Cnidaria (mt)", "YP_025113067", "protein_cds", 38),
    ("crayfish_mt", "Procambarus clarkii", "Eukarya; Arthropoda; Crustacea (mt)", "YP_654554", "protein_cds", 32),
    ("leech_mt", "Hirudo medicinalis", "Eukarya; Annelida (mt)", "YP_009113408", "protein_cds", 30),
    ("snail_mt", "Biomphalaria glabrata", "Eukarya; Mollusca; Gastropoda (mt)", "YP_009164040", "protein_cds", 30),
    ("starfish_mt", "Asterias rubens", "Eukarya; Echinodermata (mt)", "YP_009088862", "protein_cds", 36),
    ("flatworm_mt", "Schistosoma mansoni", "Eukarya; Platyhelminthes (mt)", "YP_001039704", "protein_cds", 28),
    ("beetle_mt", "Tribolium castaneum", "Eukarya; Arthropoda; Coleoptera (mt)", "NP_203591", "protein_cds", 24),
    ("mosquito_mt", "Anopheles gambiae", "Eukarya; Arthropoda; Diptera (mt)", "YP_009187866", "protein_cds", 20),
    ("tuna_mt", "Thunnus thynnus", "Eukarya; Chordata; Actinopterygii (mt)", "YP_009105148", "protein_cds", 42),
    ("bat_mt", "Myotis lucifugus", "Eukarya; Chordata; Mammalia (mt)", "YP_003058108", "protein_cds", 40),
    ("turtle_mt", "Chelonia mydas", "Eukarya; Chordata; Reptilia (mt)", "YP_004285973", "protein_cds", 42),
])

# ───────────────────────────────────────────────────────────────
# Family 6: Alpha-tubulin (eukaryotic cytoskeletal)
# ───────────────────────────────────────────────────────────────
_TUBULIN = _make_family("tubulin", "alpha-tubulin", [
    # Archaea (FtsZ-like / tubulin homolog)
    ("nitrosopumilus_ftsz", "Nitrosopumilus maritimus", "Archaea; Thaumarchaeota", "ABZ10394", "protein_cds", 34),
    # Fungi
    ("yeast_tub1", "Saccharomyces cerevisiae", "Eukarya; Ascomycota", "NP_012862", "protein_cds", 40),
    ("aspergillus", "Aspergillus nidulans", "Eukarya; Ascomycota", "XP_664216", "protein_cds", 54),
    ("ustilago", "Ustilago maydis", "Eukarya; Basidiomycota", "XP_011390076", "protein_cds", 57),
    # Plants
    ("arabidopsis", "Arabidopsis thaliana", "Eukarya; Streptophyta", "NP_196241", "protein_cds", 44),
    ("rice", "Oryza sativa", "Eukarya; Streptophyta", "XP_015612428", "protein_cds", 60),
    ("maize", "Zea mays", "Eukarya; Streptophyta", "NP_001104919", "protein_cds", 60),
    ("moss", "Physcomitrium patens", "Eukarya; Streptophyta", "XP_024381064", "protein_cds", 48),
    # Vertebrates
    ("human_tuba1a", "Homo sapiens", "Eukarya; Chordata; Mammalia", "NP_006000", "protein_cds", 54),
    ("mouse", "Mus musculus", "Eukarya; Chordata; Mammalia", "NP_035783", "protein_cds", 52),
    ("chicken", "Gallus gallus", "Eukarya; Chordata; Aves", "NP_990622", "protein_cds", 52),
    ("zebrafish", "Danio rerio", "Eukarya; Chordata; Actinopterygii", "NP_998091", "protein_cds", 50),
    ("xenopus", "Xenopus laevis", "Eukarya; Chordata; Amphibia", "NP_001081437", "protein_cds", 48),
    # Invertebrates
    ("drosophila", "Drosophila melanogaster", "Eukarya; Arthropoda; Insecta", "NP_524197", "protein_cds", 52),
    ("celegans", "Caenorhabditis elegans", "Eukarya; Nematoda", "NP_497706", "protein_cds", 46),
    ("octopus", "Octopus bimaculoides", "Eukarya; Mollusca", "XP_014769898", "protein_cds", 42),
    ("urchin", "Strongylocentrotus purpuratus", "Eukarya; Echinodermata", "XP_011676789", "protein_cds", 44),
    ("honeybee", "Apis mellifera", "Eukarya; Arthropoda", "XP_006563899", "protein_cds", 36),
    # Protists
    ("plasmodium", "Plasmodium falciparum", "Eukarya; Apicomplexa", "XP_001351614", "protein_cds", 26),
    ("toxoplasma", "Toxoplasma gondii", "Eukarya; Apicomplexa", "XP_002364651", "protein_cds", 52),
    ("trypanosoma", "Trypanosoma brucei", "Eukarya; Euglenozoa", "XP_011775778", "protein_cds", 55),
    ("leishmania", "Leishmania major", "Eukarya; Euglenozoa", "XP_001686111", "protein_cds", 62),
    ("dictyostelium", "Dictyostelium discoideum", "Eukarya; Amoebozoa", "XP_639476", "protein_cds", 28),
    ("tetrahymena", "Tetrahymena thermophila", "Eukarya; Ciliophora", "XP_001013619", "protein_cds", 32),
    ("paramecium", "Paramecium tetraurelia", "Eukarya; Ciliophora", "XP_001447204", "protein_cds", 28),
    # Algae
    ("chlamy", "Chlamydomonas reinhardtii", "Eukarya; Chlorophyta", "XP_001694268", "protein_cds", 64),
    ("diatom", "Phaeodactylum tricornutum", "Eukarya; Bacillariophyta", "XP_002180012", "protein_cds", 49),
    # Diverse extras
    ("hydra", "Hydra vulgaris", "Eukarya; Cnidaria", "XP_002155428", "protein_cds", 38),
    ("ciona", "Ciona intestinalis", "Eukarya; Chordata; Tunicata", "XP_002127870", "protein_cds", 42),
    ("nematostella", "Nematostella vectensis", "Eukarya; Cnidaria; Anthozoa", "XP_001637988", "protein_cds", 42),
])

# ───────────────────────────────────────────────────────────────
# Family 7: Histone H3 (chromatin, eukaryotic + some archaea)
# ───────────────────────────────────────────────────────────────
_HISTONE_H3 = _make_family("histoneH3", "histone H3 / archaeal histone", [
    # Archaea (archaeal histones)
    ("mjannaschii", "Methanocaldococcus jannaschii", "Archaea; Euryarchaeota", "NP_247642", "protein_cds", 31),
    ("pyrococcus", "Pyrococcus furiosus", "Archaea; Euryarchaeota", "NP_578264", "protein_cds", 41),
    ("methanosarcina", "Methanosarcina acetivorans", "Archaea; Euryarchaeota", "NP_616476", "protein_cds", 43),
    # Fungi
    ("yeast_hht1", "Saccharomyces cerevisiae", "Eukarya; Ascomycota", "NP_014368", "protein_cds", 40),
    ("neurospora", "Neurospora crassa", "Eukarya; Ascomycota", "XP_964832", "protein_cds", 56),
    ("ustilago", "Ustilago maydis", "Eukarya; Basidiomycota", "XP_011392188", "protein_cds", 57),
    # Plants
    ("arabidopsis", "Arabidopsis thaliana", "Eukarya; Streptophyta", "NP_001078178", "protein_cds", 44),
    ("rice", "Oryza sativa", "Eukarya; Streptophyta", "XP_015626792", "protein_cds", 62),
    ("maize", "Zea mays", "Eukarya; Streptophyta", "NP_001105565", "protein_cds", 60),
    # Vertebrates
    ("human_h3f3a", "Homo sapiens", "Eukarya; Chordata; Mammalia", "NP_002098", "protein_cds", 60),
    ("mouse", "Mus musculus", "Eukarya; Chordata; Mammalia", "NP_032232", "protein_cds", 58),
    ("chicken", "Gallus gallus", "Eukarya; Chordata; Aves", "NP_001264694", "protein_cds", 56),
    ("zebrafish", "Danio rerio", "Eukarya; Chordata; Actinopterygii", "NP_998025", "protein_cds", 52),
    ("xenopus", "Xenopus laevis", "Eukarya; Chordata; Amphibia", "NP_001081987", "protein_cds", 46),
    # Invertebrates
    ("drosophila", "Drosophila melanogaster", "Eukarya; Arthropoda; Insecta", "NP_723972", "protein_cds", 48),
    ("celegans", "Caenorhabditis elegans", "Eukarya; Nematoda", "NP_505805", "protein_cds", 44),
    ("urchin", "Strongylocentrotus purpuratus", "Eukarya; Echinodermata", "NP_999630", "protein_cds", 44),
    ("honeybee", "Apis mellifera", "Eukarya; Arthropoda", "XP_006571174", "protein_cds", 36),
    ("hydra", "Hydra vulgaris", "Eukarya; Cnidaria", "XP_002163764", "protein_cds", 36),
    # Protists
    ("plasmodium", "Plasmodium falciparum", "Eukarya; Apicomplexa", "XP_001351921", "protein_cds", 26),
    ("toxoplasma", "Toxoplasma gondii", "Eukarya; Apicomplexa", "XP_002368073", "protein_cds", 52),
    ("trypanosoma", "Trypanosoma brucei", "Eukarya; Euglenozoa", "XP_011774308", "protein_cds", 50),
    ("dictyostelium", "Dictyostelium discoideum", "Eukarya; Amoebozoa", "XP_636756", "protein_cds", 28),
    ("tetrahymena", "Tetrahymena thermophila", "Eukarya; Ciliophora", "XP_001029820", "protein_cds", 32),
    # Algae
    ("chlamy", "Chlamydomonas reinhardtii", "Eukarya; Chlorophyta", "XP_001698003", "protein_cds", 64),
    ("diatom", "Thalassiosira pseudonana", "Eukarya; Bacillariophyta", "XP_002295163", "protein_cds", 48),
    # Diverse extras
    ("ciona", "Ciona intestinalis", "Eukarya; Chordata; Tunicata", "XP_002119832", "protein_cds", 42),
    ("nematostella", "Nematostella vectensis", "Eukarya; Cnidaria; Anthozoa", "XP_001625098", "protein_cds", 42),
    ("octopus", "Octopus bimaculoides", "Eukarya; Mollusca", "XP_014775511", "protein_cds", 42),
    ("moth", "Bombyx mori", "Eukarya; Arthropoda; Lepidoptera", "NP_001091789", "protein_cds", 42),
])

# ───────────────────────────────────────────────────────────────
# Family 8: RNA polymerase beta (rpoB) — transcription, bacteria/archaea
# ───────────────────────────────────────────────────────────────
_RPOB = _make_family("rpoB", "RNA polymerase beta subunit (rpoB)", [
    # Bacteria — diverse phyla
    ("ecoli", "Escherichia coli K-12", "Bacteria; Gammaproteobacteria", "NP_418414", "protein_cds", 51),
    ("bacillus", "Bacillus subtilis", "Bacteria; Bacillota", "NP_388871", "protein_cds", 44),
    ("mycobacterium", "Mycobacterium tuberculosis", "Bacteria; Actinomycetota", "NP_215115", "protein_cds", 66),
    ("pseudomonas", "Pseudomonas aeruginosa", "Bacteria; Gammaproteobacteria", "NP_253680", "protein_cds", 66),
    ("staphylococcus", "Staphylococcus aureus", "Bacteria; Bacillota", "NP_371087", "protein_cds", 33),
    ("synechocystis", "Synechocystis sp. PCC 6803", "Bacteria; Cyanobacteriota", "NP_441741", "protein_cds", 47),
    ("thermus", "Thermus thermophilus", "Bacteria; Deinococcota", "WP_011173372", "protein_cds", 68),
    ("deinococcus", "Deinococcus radiodurans", "Bacteria; Deinococcota", "NP_294283", "protein_cds", 67),
    ("borrelia", "Borrelia burgdorferi", "Bacteria; Spirochaetota", "NP_212562", "protein_cds", 28),
    ("treponema", "Treponema pallidum", "Bacteria; Spirochaetota", "NP_218523", "protein_cds", 53),
    ("chlamydia", "Chlamydia trachomatis", "Bacteria; Chlamydiota", "NP_219833", "protein_cds", 41),
    ("helicobacter", "Helicobacter pylori", "Bacteria; Campylobacterota", "NP_207305", "protein_cds", 39),
    ("rickettsia", "Rickettsia prowazekii", "Bacteria; Alphaproteobacteria", "AJF73992", "protein_cds", 29),
    ("bacteroides", "Bacteroides fragilis", "Bacteria; Bacteroidota", "YP_213064", "protein_cds", 43),
    ("clostridium", "Clostridioides difficile", "Bacteria; Bacillota", "NP_384283", "protein_cds", 29),
    ("aquifex", "Aquifex aeolicus", "Bacteria; Aquificota", "NP_213873", "protein_cds", 43),
    ("thermotoga", "Thermotoga maritima", "Bacteria; Thermotogota", "NP_228693", "protein_cds", 46),
    ("mycoplasma", "Mycoplasma genitalium", "Bacteria; Tenericutes", "NP_072839", "protein_cds", 32),
    # Archaea
    ("mjannaschii", "Methanocaldococcus jannaschii", "Archaea; Euryarchaeota", "NP_248306", "protein_cds", 31),
    ("sulfolobus", "Sulfolobus solfataricus", "Archaea; Crenarchaeota", "NP_342729", "protein_cds", 36),
    ("haloferax", "Haloferax volcanii", "Archaea; Euryarchaeota", "ADE02665", "protein_cds", 65),
    ("pyrococcus", "Pyrococcus furiosus", "Archaea; Euryarchaeota", "NP_578937", "protein_cds", 41),
    ("nanoarchaeum", "Nanoarchaeum equitans", "Archaea; DPANN", "AAR39199", "protein_cds", 32),
    ("korarchaeum", "Ca. Korarchaeum cryptofilum", "Archaea; Korarchaeota", "ACB07166", "protein_cds", 49),
    # Chloroplast rpoB
    ("arabidopsis_cp", "Arabidopsis thaliana (cp)", "Eukarya; Streptophyta (cp)", "NP_051066", "protein_cds", 37),
    ("rice_cp", "Oryza sativa (cp)", "Eukarya; Streptophyta (cp)", "NP_039420", "protein_cds", 39),
    # Organellar relatives
    ("wolbachia", "Wolbachia endosymbiont", "Bacteria; Alphaproteobacteria", "WP_010962494", "protein_cds", 35),
    ("caulobacter", "Caulobacter vibrioides", "Bacteria; Alphaproteobacteria", "NP_422159", "protein_cds", 67),
    ("rhizobium", "Sinorhizobium meliloti", "Bacteria; Alphaproteobacteria", "NP_385893", "protein_cds", 62),
    ("planctomyces", "Planctopirus limnophila", "Bacteria; Planctomycetota", "ADG68145", "protein_cds", 54),
])

# ───────────────────────────────────────────────────────────────
# Family 9: ATP synthase alpha (atpA) — energy metabolism, universal
# ───────────────────────────────────────────────────────────────
_ATPA = _make_family("atpA", "ATP synthase subunit alpha (atpA)", [
    # Bacteria
    ("ecoli", "Escherichia coli K-12", "Bacteria; Gammaproteobacteria", "NP_418192", "protein_cds", 51),
    ("bacillus", "Bacillus subtilis", "Bacteria; Bacillota", "NP_391058", "protein_cds", 44),
    ("mycobacterium", "Mycobacterium tuberculosis", "Bacteria; Actinomycetota", "NP_217315", "protein_cds", 66),
    ("synechocystis", "Synechocystis sp. PCC 6803", "Bacteria; Cyanobacteriota", "NP_443114", "protein_cds", 47),
    ("thermus", "Thermus thermophilus", "Bacteria; Deinococcota", "WP_011173131", "protein_cds", 68),
    # Archaea
    ("mjannaschii", "Methanocaldococcus jannaschii", "Archaea; Euryarchaeota", "NP_247573", "protein_cds", 31),
    ("sulfolobus", "Sulfolobus solfataricus", "Archaea; Crenarchaeota", "NP_341824", "protein_cds", 36),
    ("pyrococcus", "Pyrococcus furiosus", "Archaea; Euryarchaeota", "NP_578173", "protein_cds", 41),
    ("haloferax", "Haloferax volcanii", "Archaea; Euryarchaeota", "ADE02103", "protein_cds", 65),
    # Fungi
    ("yeast", "Saccharomyces cerevisiae", "Eukarya; Ascomycota", "NP_015333", "protein_cds", 40),
    ("aspergillus", "Aspergillus nidulans", "Eukarya; Ascomycota", "XP_661321", "protein_cds", 54),
    # Plants
    ("arabidopsis", "Arabidopsis thaliana", "Eukarya; Streptophyta", "NP_189610", "protein_cds", 44),
    ("rice", "Oryza sativa", "Eukarya; Streptophyta", "XP_015623068", "protein_cds", 60),
    # Vertebrates
    ("human", "Homo sapiens", "Eukarya; Chordata; Mammalia", "NP_001001937", "protein_cds", 52),
    ("mouse", "Mus musculus", "Eukarya; Chordata; Mammalia", "NP_031531", "protein_cds", 50),
    ("chicken", "Gallus gallus", "Eukarya; Chordata; Aves", "NP_001025967", "protein_cds", 52),
    ("zebrafish", "Danio rerio", "Eukarya; Chordata; Actinopterygii", "NP_001007379", "protein_cds", 48),
    ("xenopus", "Xenopus laevis", "Eukarya; Chordata; Amphibia", "NP_001079226", "protein_cds", 48),
    # Invertebrates
    ("drosophila", "Drosophila melanogaster", "Eukarya; Arthropoda; Insecta", "NP_477141", "protein_cds", 48),
    ("celegans", "Caenorhabditis elegans", "Eukarya; Nematoda", "NP_503274", "protein_cds", 44),
    # Protists
    ("plasmodium", "Plasmodium falciparum", "Eukarya; Apicomplexa", "XP_001349185", "protein_cds", 26),
    ("trypanosoma", "Trypanosoma brucei", "Eukarya; Euglenozoa", "XP_011773780", "protein_cds", 52),
    ("tetrahymena", "Tetrahymena thermophila", "Eukarya; Ciliophora", "XP_001032240", "protein_cds", 32),
    # Algae
    ("chlamy", "Chlamydomonas reinhardtii", "Eukarya; Chlorophyta", "XP_001700186", "protein_cds", 64),
    ("diatom", "Thalassiosira pseudonana", "Eukarya; Bacillariophyta", "XP_002289651", "protein_cds", 48),
    # Chloroplast-encoded
    ("arabidopsis_cp", "Arabidopsis thaliana (cp)", "Eukarya; Streptophyta (cp)", "NP_051095", "protein_cds", 37),
    ("rice_cp", "Oryza sativa (cp)", "Eukarya; Streptophyta (cp)", "NP_039393", "protein_cds", 39),
    # Diverse extras
    ("hydra", "Hydra vulgaris", "Eukarya; Cnidaria", "XP_002157356", "protein_cds", 38),
    ("ciona", "Ciona intestinalis", "Eukarya; Chordata; Tunicata", "XP_002122479", "protein_cds", 42),
    ("nematostella", "Nematostella vectensis", "Eukarya; Cnidaria; Anthozoa", "XP_001631758", "protein_cds", 42),
])

# ───────────────────────────────────────────────────────────────
# Family 10: Ribosomal protein S12 (rpsL) — translation, universal
# ───────────────────────────────────────────────────────────────
_RPSL = _make_family("rpsL", "30S ribosomal protein S12 (rpsL)", [
    # Bacteria — broad phylogenetic sampling
    ("ecoli", "Escherichia coli K-12", "Bacteria; Gammaproteobacteria", "NP_417072", "protein_cds", 54),
    ("bacillus", "Bacillus subtilis", "Bacteria; Bacillota", "NP_391322", "protein_cds", 44),
    ("mycobacterium", "Mycobacterium tuberculosis", "Bacteria; Actinomycetota", "NP_215397", "protein_cds", 66),
    ("pseudomonas", "Pseudomonas aeruginosa", "Bacteria; Gammaproteobacteria", "NP_253685", "protein_cds", 66),
    ("synechocystis", "Synechocystis sp. PCC 6803", "Bacteria; Cyanobacteriota", "NP_440709", "protein_cds", 47),
    ("thermus", "Thermus thermophilus", "Bacteria; Deinococcota", "WP_011173443", "protein_cds", 68),
    ("deinococcus", "Deinococcus radiodurans", "Bacteria; Deinococcota", "NP_295291", "protein_cds", 67),
    ("borrelia", "Borrelia burgdorferi", "Bacteria; Spirochaetota", "NP_212559", "protein_cds", 28),
    ("helicobacter", "Helicobacter pylori", "Bacteria; Campylobacterota", "NP_207131", "protein_cds", 39),
    ("chlamydia", "Chlamydia trachomatis", "Bacteria; Chlamydiota", "NP_219831", "protein_cds", 41),
    ("rickettsia", "Rickettsia prowazekii", "Bacteria; Alphaproteobacteria", "NP_220760", "protein_cds", 29),
    ("aquifex", "Aquifex aeolicus", "Bacteria; Aquificota", "NP_214240", "protein_cds", 43),
    ("thermotoga", "Thermotoga maritima", "Bacteria; Thermotogota", "NP_228693", "protein_cds", 46),
    ("mycoplasma", "Mycoplasma genitalium", "Bacteria; Tenericutes", "NP_072697", "protein_cds", 32),
    ("bacteroides", "Bacteroides fragilis", "Bacteria; Bacteroidota", "YP_213148", "protein_cds", 43),
    # Archaea
    ("mjannaschii", "Methanocaldococcus jannaschii", "Archaea; Euryarchaeota", "NP_248291", "protein_cds", 31),
    ("sulfolobus", "Sulfolobus solfataricus", "Archaea; Crenarchaeota", "NP_342660", "protein_cds", 36),
    ("pyrococcus", "Pyrococcus furiosus", "Archaea; Euryarchaeota", "NP_578449", "protein_cds", 41),
    ("haloferax", "Haloferax volcanii", "Archaea; Euryarchaeota", "ADE03949", "protein_cds", 65),
    ("nanoarchaeum", "Nanoarchaeum equitans", "Archaea; DPANN", "AAR39253", "protein_cds", 32),
    # Chloroplast-encoded rps12
    ("arabidopsis_cp", "Arabidopsis thaliana (cp)", "Eukarya; Streptophyta (cp)", "NP_051088", "protein_cds", 37),
    ("rice_cp", "Oryza sativa (cp)", "Eukarya; Streptophyta (cp)", "NP_039385", "protein_cds", 39),
    ("chlamy_cp", "Chlamydomonas reinhardtii (cp)", "Eukarya; Chlorophyta (cp)", "NP_958407", "protein_cds", 34),
    # Mitochondrial-encoded rps12
    ("arabidopsis_mt", "Arabidopsis thaliana (mt)", "Eukarya; Streptophyta (mt)", "NP_085579", "protein_cds", 44),
    # Diverse bacterial extras
    ("caulobacter", "Caulobacter vibrioides", "Bacteria; Alphaproteobacteria", "NP_422151", "protein_cds", 67),
    ("streptomyces", "Streptomyces coelicolor", "Bacteria; Actinomycetota", "NP_628354", "protein_cds", 72),
    ("vibrio", "Vibrio cholerae", "Bacteria; Gammaproteobacteria", "NP_233005", "protein_cds", 47),
    ("staphylococcus", "Staphylococcus aureus", "Bacteria; Bacillota", "NP_373770", "protein_cds", 33),
    ("clostridium", "Clostridioides difficile", "Bacteria; Bacillota", "NP_384285", "protein_cds", 29),
    ("planctomyces", "Planctopirus limnophila", "Bacteria; Planctomycetota", "ADG68063", "protein_cds", 54),
])


# ═══════════════════════════════════════════════════════════════════════════════
# Combine all families into COMPONENT_B
# ═══════════════════════════════════════════════════════════════════════════════
COMPONENT_B: list[dict] = (
    _ACTIN + _GAPDH + _EFTU + _HSP70 + _COI
    + _TUBULIN + _HISTONE_H3 + _RPOB + _ATPA + _RPSL
)

_n_comp_b = len(COMPONENT_B)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT C: NEGATIVE CONTROLS FOR CLUSTERING (~30 sequences)
# Random CDS from 30 different species, each a DIFFERENT gene not in the
# 10 families above. These should NOT cluster with any family.
# ═══════════════════════════════════════════════════════════════════════════════

COMPONENT_C: list[dict] = [
    {
        "name": "c_ecoli_lacZ",
        "component": "C", "category": "negative_control",
        "lineage": "Bacteria; Gammaproteobacteria",
        "species": "Escherichia coli K-12",
        "gene": "beta-galactosidase (lacZ)",
        "accession": "NP_414878",
        "type": "protein_cds",
        "gc_approx": 51,
    },
    {
        "name": "c_human_p53",
        "component": "C", "category": "negative_control",
        "lineage": "Eukarya; Chordata; Mammalia",
        "species": "Homo sapiens",
        "gene": "tumor protein p53",
        "accession": "NP_000537",
        "type": "protein_cds",
        "gc_approx": 48,
    },
    {
        "name": "c_yeast_ura3",
        "component": "C", "category": "negative_control",
        "lineage": "Eukarya; Ascomycota",
        "species": "Saccharomyces cerevisiae",
        "gene": "orotidine-5-phosphate decarboxylase (URA3)",
        "accession": "NP_010893",
        "type": "protein_cds",
        "gc_approx": 40,
    },
    {
        "name": "c_arabidopsis_agamous",
        "component": "C", "category": "negative_control",
        "lineage": "Eukarya; Streptophyta",
        "species": "Arabidopsis thaliana",
        "gene": "MADS-box protein AGAMOUS",
        "accession": "NP_567569",
        "type": "protein_cds",
        "gc_approx": 42,
    },
    {
        "name": "c_drosophila_notch",
        "component": "C", "category": "negative_control",
        "lineage": "Eukarya; Arthropoda; Insecta",
        "species": "Drosophila melanogaster",
        "gene": "Notch receptor (partial)",
        "accession": "NP_476670",
        "type": "protein_cds",
        "gc_approx": 52,
    },
    {
        "name": "c_plasmodium_dhfr",
        "component": "C", "category": "negative_control",
        "lineage": "Eukarya; Apicomplexa",
        "species": "Plasmodium falciparum",
        "gene": "dihydrofolate reductase-thymidylate synthase",
        "accession": "XP_001351095",
        "type": "protein_cds",
        "gc_approx": 24,
    },
    {
        "name": "c_celegans_daf2",
        "component": "C", "category": "negative_control",
        "lineage": "Eukarya; Nematoda",
        "species": "Caenorhabditis elegans",
        "gene": "insulin receptor DAF-2 (partial)",
        "accession": "NP_497825",
        "type": "protein_cds",
        "gc_approx": 42,
    },
    {
        "name": "c_zebrafish_shh",
        "component": "C", "category": "negative_control",
        "lineage": "Eukarya; Chordata; Actinopterygii",
        "species": "Danio rerio",
        "gene": "sonic hedgehog a",
        "accession": "NP_571154",
        "type": "protein_cds",
        "gc_approx": 48,
    },
    {
        "name": "c_bacillus_sporulation",
        "component": "C", "category": "negative_control",
        "lineage": "Bacteria; Bacillota",
        "species": "Bacillus subtilis",
        "gene": "stage V sporulation protein B",
        "accession": "NP_389413",
        "type": "protein_cds",
        "gc_approx": 44,
    },
    {
        "name": "c_mycobacterium_esat6",
        "component": "C", "category": "negative_control",
        "lineage": "Bacteria; Actinomycetota",
        "species": "Mycobacterium tuberculosis",
        "gene": "ESAT-6 (secreted antigen)",
        "accession": "NP_216402",
        "type": "protein_cds",
        "gc_approx": 66,
    },
    {
        "name": "c_mouse_myc",
        "component": "C", "category": "negative_control",
        "lineage": "Eukarya; Chordata; Mammalia",
        "species": "Mus musculus",
        "gene": "c-Myc proto-oncogene",
        "accession": "NP_034979",
        "type": "protein_cds",
        "gc_approx": 52,
    },
    {
        "name": "c_rice_wrky",
        "component": "C", "category": "negative_control",
        "lineage": "Eukarya; Streptophyta",
        "species": "Oryza sativa",
        "gene": "WRKY transcription factor",
        "accession": "XP_015640618",
        "type": "protein_cds",
        "gc_approx": 60,
    },
    {
        "name": "c_synechocystis_phycobili",
        "component": "C", "category": "negative_control",
        "lineage": "Bacteria; Cyanobacteriota",
        "species": "Synechocystis sp. PCC 6803",
        "gene": "phycocyanin alpha subunit",
        "accession": "NP_441268",
        "type": "protein_cds",
        "gc_approx": 47,
    },
    {
        "name": "c_chicken_ovalbumin",
        "component": "C", "category": "negative_control",
        "lineage": "Eukarya; Chordata; Aves",
        "species": "Gallus gallus",
        "gene": "ovalbumin",
        "accession": "NP_990483",
        "type": "protein_cds",
        "gc_approx": 48,
    },
    {
        "name": "c_trypanosoma_ornithine",
        "component": "C", "category": "negative_control",
        "lineage": "Eukarya; Euglenozoa",
        "species": "Trypanosoma brucei",
        "gene": "ornithine decarboxylase",
        "accession": "XP_011771310",
        "type": "protein_cds",
        "gc_approx": 50,
    },
    {
        "name": "c_sulfolobus_topo",
        "component": "C", "category": "negative_control",
        "lineage": "Archaea; Crenarchaeota",
        "species": "Sulfolobus solfataricus",
        "gene": "DNA topoisomerase I",
        "accession": "NP_343817",
        "type": "protein_cds",
        "gc_approx": 36,
    },
    {
        "name": "c_haloferax_dehalogenase",
        "component": "C", "category": "negative_control",
        "lineage": "Archaea; Euryarchaeota",
        "species": "Haloferax volcanii",
        "gene": "haloacid dehalogenase",
        "accession": "ADE03285",
        "type": "protein_cds",
        "gc_approx": 65,
    },
    {
        "name": "c_tetrahymena_serpin",
        "component": "C", "category": "negative_control",
        "lineage": "Eukarya; Ciliophora",
        "species": "Tetrahymena thermophila",
        "gene": "papain-like cysteine protease",
        "accession": "XP_001015267",
        "type": "protein_cds",
        "gc_approx": 32,
    },
    {
        "name": "c_dictyostelium_pks",
        "component": "C", "category": "negative_control",
        "lineage": "Eukarya; Amoebozoa",
        "species": "Dictyostelium discoideum",
        "gene": "polyketide synthase (DIF biosynthesis)",
        "accession": "XP_643897",
        "type": "protein_cds",
        "gc_approx": 28,
    },
    {
        "name": "c_honeybee_royalactin",
        "component": "C", "category": "negative_control",
        "lineage": "Eukarya; Arthropoda",
        "species": "Apis mellifera",
        "gene": "major royal jelly protein 1 (royalactin)",
        "accession": "NP_001011579",
        "type": "protein_cds",
        "gc_approx": 32,
    },
    {
        "name": "c_chlamy_channelrhodopsin",
        "component": "C", "category": "negative_control",
        "lineage": "Eukarya; Chlorophyta",
        "species": "Chlamydomonas reinhardtii",
        "gene": "channelrhodopsin-2 (optogenetics tool origin)",
        "accession": "XP_001694192",
        "type": "protein_cds",
        "gc_approx": 62,
    },
    {
        "name": "c_pseudomonas_phenazine",
        "component": "C", "category": "negative_control",
        "lineage": "Bacteria; Gammaproteobacteria",
        "species": "Pseudomonas aeruginosa",
        "gene": "phenazine biosynthesis protein PhzF",
        "accession": "NP_252975",
        "type": "protein_cds",
        "gc_approx": 66,
    },
    {
        "name": "c_xenopus_noggin",
        "component": "C", "category": "negative_control",
        "lineage": "Eukarya; Chordata; Amphibia",
        "species": "Xenopus laevis",
        "gene": "noggin (BMP antagonist)",
        "accession": "NP_001079052",
        "type": "protein_cds",
        "gc_approx": 46,
    },
    {
        "name": "c_maize_opaque2",
        "component": "C", "category": "negative_control",
        "lineage": "Eukarya; Streptophyta",
        "species": "Zea mays",
        "gene": "Opaque2 transcription factor (zein regulation)",
        "accession": "NP_001105338",
        "type": "protein_cds",
        "gc_approx": 58,
    },
    {
        "name": "c_ustilago_mating",
        "component": "C", "category": "negative_control",
        "lineage": "Eukarya; Basidiomycota",
        "species": "Ustilago maydis",
        "gene": "pheromone receptor Pra1",
        "accession": "XP_011388050",
        "type": "protein_cds",
        "gc_approx": 54,
    },
    {
        "name": "c_urchin_spicule",
        "component": "C", "category": "negative_control",
        "lineage": "Eukarya; Echinodermata",
        "species": "Strongylocentrotus purpuratus",
        "gene": "spicule matrix protein SM30",
        "accession": "NP_999665",
        "type": "protein_cds",
        "gc_approx": 42,
    },
    {
        "name": "c_diatom_frustulin",
        "component": "C", "category": "negative_control",
        "lineage": "Eukarya; Bacillariophyta",
        "species": "Phaeodactylum tricornutum",
        "gene": "frustulin (silica-associated cell wall)",
        "accession": "XP_002181948",
        "type": "protein_cds",
        "gc_approx": 49,
    },
    {
        "name": "c_leishmania_gp63",
        "component": "C", "category": "negative_control",
        "lineage": "Eukarya; Euglenozoa",
        "species": "Leishmania major",
        "gene": "leishmanolysin / GP63 surface protease",
        "accession": "XP_001684153",
        "type": "protein_cds",
        "gc_approx": 62,
    },
    {
        "name": "c_lamprey_vlrB",
        "component": "C", "category": "negative_control",
        "lineage": "Eukarya; Chordata; Hyperoartia",
        "species": "Petromyzon marinus",
        "gene": "variable lymphocyte receptor B",
        "accession": "XP_032804317",
        "type": "protein_cds",
        "gc_approx": 53,
    },
    {
        "name": "c_nematostella_green_fp",
        "component": "C", "category": "negative_control",
        "lineage": "Eukarya; Cnidaria; Anthozoa",
        "species": "Nematostella vectensis",
        "gene": "green fluorescent protein-like",
        "accession": "XP_001625318",
        "type": "protein_cds",
        "gc_approx": 42,
    },
]

_n_comp_c = len(COMPONENT_C)


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PANEL: combine all components
# ═══════════════════════════════════════════════════════════════════════════════

PANEL: list[dict] = COMPONENT_A + COMPONENT_B + COMPONENT_C


# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD, EXTRACT, AND ANALYZE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def download_sequences():
    """Download all sequences from NCBI and save as individual FASTA files."""
    from Bio import Entrez, SeqIO

    Entrez.email = "shandley@wustl.edu"
    FASTA_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []

    for i, entry in enumerate(PANEL):
        name = entry["name"]
        fasta_path = FASTA_DIR / f"{name}.fasta"

        if fasta_path.exists():
            seq = next(SeqIO.parse(str(fasta_path), "fasta"))
            print(f"  [{i+1}/{len(PANEL)}] {name}: already downloaded ({len(seq.seq)} bp)")
            results.append({"name": name, "length": len(seq.seq), "status": "cached"})
            continue

        print(f"  [{i+1}/{len(PANEL)}] {name}: downloading...", end=" ", flush=True)

        try:
            seq_str = _fetch_sequence(entry, Entrez)
            if seq_str is None or len(seq_str) < 50:
                print(f"FAILED (too short: {len(seq_str) if seq_str else 0} bp)")
                failures.append(name)
                continue

            # Validate: only ACGTN
            seq_clean = str(seq_str).upper().replace("U", "T")
            seq_clean = "".join(c for c in seq_clean if c in "ACGTN")

            if len(seq_clean) < 50:
                print(f"FAILED (after cleaning: {len(seq_clean)} bp)")
                failures.append(name)
                continue

            # Truncate very long sequences to 2000 bp
            if len(seq_clean) > 2000:
                seq_clean = seq_clean[:2000]

            # Write FASTA
            gc = sum(1 for c in seq_clean if c in "GC") / len(seq_clean) * 100
            header = (
                f">{name} {entry['species']} {entry['gene']} "
                f"[{len(seq_clean)}bp, GC={gc:.1f}%] "
                f"component={entry['component']} category={entry['category']}"
            )
            with open(fasta_path, "w") as f:
                f.write(f"{header}\n")
                for j in range(0, len(seq_clean), 60):
                    f.write(seq_clean[j : j + 60] + "\n")

            print(f"OK ({len(seq_clean)} bp, GC={gc:.1f}%)")
            results.append({"name": name, "length": len(seq_clean), "gc": gc, "status": "downloaded"})

            # Rate limit: NCBI allows 3 requests/sec with API key, 1/sec without
            time.sleep(0.4)

        except Exception as e:
            print(f"FAILED ({e})")
            failures.append(name)
            continue

    # Write summary
    summary = {
        "total_panel": len(PANEL),
        "component_A": _n_comp_a,
        "component_B": _n_comp_b,
        "component_C": _n_comp_c,
        "downloaded": len(results),
        "failed": len(failures),
        "failures": failures,
        "results": results,
    }
    with open(OUT_DIR / "download_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Panel: {_n_comp_a} (A) + {_n_comp_b} (B) + {_n_comp_c} (C) = {len(PANEL)} total")
    print(f"Downloaded: {len(results)}/{len(PANEL)}")
    if failures:
        print(f"Failed ({len(failures)}): {failures}")


def _fetch_sequence(entry: dict, Entrez) -> str | None:
    """Fetch a sequence from NCBI based on entry type."""
    from Bio import SeqIO
    from Bio.Seq import Seq

    if entry["type"] == "hardcoded":
        return entry["sequence"]

    if entry["type"] == "protein_cds":
        return _fetch_cds_from_protein(entry["accession"], Entrez)

    if entry["type"] in ("mrna_cds", "full_cds"):
        handle = Entrez.efetch(
            db="nucleotide", id=entry["accession"],
            rettype="fasta", retmode="text",
        )
        record = next(SeqIO.parse(StringIO(handle.read()), "fasta"))
        handle.close()

        if entry["type"] == "full_cds":
            return str(record.seq)

        start = entry.get("cds_start", 1) - 1
        end = entry.get("cds_end", len(record.seq))
        return str(record.seq[start:end])

    if entry["type"] == "genomic_region":
        acc = entry.get("nuccore", entry["accession"])
        start = entry["cds_start"]
        end = entry["cds_end"]
        handle = Entrez.efetch(
            db="nucleotide", id=acc,
            rettype="fasta", retmode="text",
            seq_start=start, seq_stop=end,
        )
        record = next(SeqIO.parse(StringIO(handle.read()), "fasta"))
        handle.close()
        return str(record.seq)

    return None


def _fetch_cds_from_protein(protein_acc: str, Entrez) -> str | None:
    """Fetch the CDS nucleotide sequence for a protein accession."""
    from Bio import SeqIO
    from Bio.Seq import Seq

    # Approach 1: coded_by from GenPept
    try:
        handle = Entrez.efetch(
            db="protein", id=protein_acc,
            rettype="gp", retmode="text",
        )
        content = handle.read()
        handle.close()

        match = re.search(r'/coded_by="([^"]+)"', content)
        if match:
            coded_by = match.group(1)
            is_complement = "complement" in coded_by
            coded_by = coded_by.replace("complement(", "").rstrip(")")

            # Handle join() for split CDS
            if "join" in coded_by:
                coded_by = coded_by.replace("join(", "").rstrip(")")
                parts_list = [p.strip() for p in coded_by.split(",")]
                # Take first part for accession, merge coordinates
                segments = []
                nuc_acc = None
                for part in parts_list:
                    if ":" in part:
                        acc_part, coords = part.split(":", 1)
                        if nuc_acc is None:
                            nuc_acc = acc_part
                    else:
                        coords = part
                    if ".." in coords:
                        s, e = coords.split("..")
                        s = int(s.replace("<", "").replace(">", ""))
                        e = int(e.replace("<", "").replace(">", ""))
                        segments.append((s, e))

                if nuc_acc and segments:
                    full_seq = ""
                    for s, e in segments:
                        h = Entrez.efetch(
                            db="nucleotide", id=nuc_acc,
                            rettype="fasta", retmode="text",
                            seq_start=s, seq_stop=e,
                        )
                        rec = next(SeqIO.parse(StringIO(h.read()), "fasta"))
                        h.close()
                        full_seq += str(rec.seq)
                        time.sleep(0.35)

                    if is_complement:
                        full_seq = str(Seq(full_seq).reverse_complement())
                    return full_seq

            else:
                if ":" in coded_by:
                    nuc_acc, coords = coded_by.split(":", 1)
                else:
                    nuc_acc = coded_by
                    coords = None

                if coords and ".." in coords:
                    parts = coords.split("..")
                    start = int(parts[0].replace("<", "").replace(">", ""))
                    end = int(parts[1].replace("<", "").replace(">", ""))
                else:
                    start = 1
                    end = None

                handle = Entrez.efetch(
                    db="nucleotide", id=nuc_acc,
                    rettype="fasta", retmode="text",
                    seq_start=start, seq_stop=end,
                )
                record = next(SeqIO.parse(StringIO(handle.read()), "fasta"))
                handle.close()

                seq = str(record.seq)
                if is_complement:
                    seq = str(Seq(seq).reverse_complement())
                return seq

    except Exception as e:
        print(f"(coded_by failed: {e}, trying fallback) ", end="")

    # Approach 2: elink protein → nuccore, then extract CDS
    try:
        handle = Entrez.elink(dbfrom="protein", db="nuccore", id=protein_acc)
        links = Entrez.read(handle)
        handle.close()

        nuc_ids = [
            link["Id"]
            for linkset in links
            for link in linkset.get("LinkSetDb", [{}])[0].get("Link", [])
        ]

        if nuc_ids:
            handle = Entrez.efetch(
                db="nucleotide", id=nuc_ids[0],
                rettype="gb", retmode="text",
            )
            record = next(SeqIO.parse(StringIO(handle.read()), "genbank"))
            handle.close()

            for feat in record.features:
                if feat.type == "CDS":
                    quals = feat.qualifiers
                    if "protein_id" in quals and protein_acc in quals["protein_id"]:
                        return str(feat.extract(record.seq))
            # Fallback: return first CDS
            for feat in record.features:
                if feat.type == "CDS":
                    return str(feat.extract(record.seq))
    except Exception:
        pass

    return None


def extract_embeddings():
    """Extract per-position embeddings via NIM API for all downloaded sequences."""
    import asyncio
    import base64
    import io
    import os

    import httpx

    EMB_DIR.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("ERROR: Set NVIDIA_API_KEY environment variable")
        sys.exit(1)

    fasta_files = sorted(FASTA_DIR.glob("*.fasta"))
    if not fasta_files:
        print("No FASTA files found. Run 'download' first.")
        return

    from Bio import SeqIO

    url = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    layer = "blocks.20.mlp.l3"

    extracted = 0
    failed = 0

    for fp in fasta_files:
        name = fp.stem
        emb_path = EMB_DIR / f"{name}.npz"

        if emb_path.exists():
            print(f"  {name}: cached")
            extracted += 1
            continue

        record = next(SeqIO.parse(str(fp), "fasta"))
        seq = str(record.seq).upper()

        if len(seq) > 16000:
            seq = seq[:16000]

        print(f"  {name} ({len(seq)} bp): extracting...", end=" ", flush=True)

        try:
            with httpx.Client(timeout=300) as client:
                payload = {"sequence": seq, "output_layers": [layer]}
                resp = client.post(url, json=payload, headers=headers)

                if resp.status_code == 302:
                    redirect_url = resp.headers["Location"]
                    resp = client.get(redirect_url)

                if resp.status_code == 429:
                    print("RATE LIMITED, waiting 60s...")
                    time.sleep(60)
                    resp = client.post(url, json=payload, headers=headers)
                    if resp.status_code == 302:
                        redirect_url = resp.headers["Location"]
                        resp = client.get(redirect_url)

                resp.raise_for_status()
                data = resp.json()

            raw = base64.b64decode(data["data"].encode("ascii"))
            npz = np.load(io.BytesIO(raw))
            key = list(npz.keys())[0]
            emb = npz[key]

            # Shape: (1, seq_len, hidden_dim) → (seq_len, hidden_dim)
            if emb.ndim == 3:
                emb = emb[0]

            np.savez_compressed(str(emb_path), embeddings=emb)
            print(f"OK (shape={emb.shape})")
            extracted += 1

            time.sleep(2)  # Rate limiting

        except Exception as e:
            print(f"FAILED ({e})")
            failed += 1
            continue

    print(f"\nExtracted: {extracted}, Failed: {failed}")


def analyze():
    """Analyze periodicity and clustering from extracted embeddings."""
    import pandas as pd
    from collections import defaultdict

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load all embeddings
    emb_files = sorted(EMB_DIR.glob("*.npz"))
    if not emb_files:
        print("No embeddings found. Run 'extract' first.")
        return

    print(f"Loading {len(emb_files)} embeddings...")

    # Build panel lookup
    panel_lookup = {e["name"]: e for e in PANEL}

    # ── Analysis 1: Codon periodicity (Component A) ──
    print("\n=== Component A: Codon Periodicity Universality ===")

    periodicity_results = []
    for fp in emb_files:
        name = fp.stem
        if name not in panel_lookup:
            continue
        entry = panel_lookup[name]
        if entry["component"] != "A":
            continue

        data = np.load(str(fp))
        emb = data["embeddings"]  # (seq_len, hidden_dim)

        # Compute offset-3 cosine similarity inversion metric
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = emb / norms

        # Adjacent cosine (offset 1)
        cos1 = np.mean(np.sum(normalized[:-1] * normalized[1:], axis=1))
        # Offset 3 cosine
        cos3 = np.mean(np.sum(normalized[:-3] * normalized[3:], axis=1))

        # Inversion: coding shows cos3 > cos1
        inversion = float(cos3 - cos1)

        # Lag-3 autocorrelation on norms
        norm_vec = np.linalg.norm(emb, axis=1)
        if len(norm_vec) > 6:
            norm_centered = norm_vec - np.mean(norm_vec)
            var = np.var(norm_vec)
            if var > 0:
                lag3_autocorr = float(np.mean(norm_centered[:-3] * norm_centered[3:]) / var)
            else:
                lag3_autocorr = 0.0
        else:
            lag3_autocorr = 0.0

        is_noncoding = entry.get("noncoding", False)

        periodicity_results.append({
            "name": name,
            "category": entry["category"],
            "species": entry["species"],
            "gene": entry["gene"],
            "gc_approx": entry["gc_approx"],
            "noncoding": is_noncoding,
            "cos1": float(cos1),
            "cos3": float(cos3),
            "inversion": inversion,
            "lag3_autocorr": lag3_autocorr,
            "seq_len": emb.shape[0],
        })

    if periodicity_results:
        df_period = pd.DataFrame(periodicity_results)
        df_period.to_csv(OUT_DIR / "periodicity_results.tsv", sep="\t", index=False)

        # Summary stats
        coding = df_period[~df_period["noncoding"]]
        noncoding = df_period[df_period["noncoding"]]

        print(f"  Coding sequences: {len(coding)}")
        print(f"    Mean inversion (cos3 - cos1): {coding['inversion'].mean():.4f}")
        print(f"    Mean lag-3 autocorr: {coding['lag3_autocorr'].mean():.4f}")
        print(f"    Inversion positive: {(coding['inversion'] > 0).sum()}/{len(coding)} "
              f"({(coding['inversion'] > 0).mean()*100:.1f}%)")
        print(f"  Non-coding controls: {len(noncoding)}")
        if len(noncoding) > 0:
            print(f"    Mean inversion: {noncoding['inversion'].mean():.4f}")
            print(f"    Inversion positive: {(noncoding['inversion'] > 0).sum()}/{len(noncoding)} "
                  f"({(noncoding['inversion'] > 0).mean()*100:.1f}%)")

    # ── Analysis 2: Functional clustering (Component B + C) ──
    print("\n=== Component B+C: Functional Clustering ===")

    cluster_names = []
    cluster_embeddings = []
    cluster_families = []
    cluster_lineages = []

    for fp in emb_files:
        name = fp.stem
        if name not in panel_lookup:
            continue
        entry = panel_lookup[name]
        if entry["component"] not in ("B", "C"):
            continue

        data = np.load(str(fp))
        emb = data["embeddings"]  # (seq_len, hidden_dim)

        # Mean-pool → single vector
        mean_emb = emb.mean(axis=0)  # (hidden_dim,)

        cluster_names.append(name)
        cluster_embeddings.append(mean_emb)
        cluster_families.append(entry["category"])
        cluster_lineages.append(entry["lineage"])

    if cluster_embeddings:
        X = np.array(cluster_embeddings)
        print(f"  Total sequences for clustering: {X.shape[0]} × {X.shape[1]}")

        # L2-normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X_norm = X / norms

        # Cosine similarity matrix
        cos_sim = X_norm @ X_norm.T
        np.save(str(OUT_DIR / "cosine_similarity_matrix.npy"), cos_sim)

        # PCA for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(20, X.shape[0]))
        X_pca = pca.fit_transform(X_norm)
        print(f"  PCA variance explained (first 5): "
              f"{pca.explained_variance_ratio_[:5].round(3).tolist()}")

        # Cluster using HDBSCAN if available, else KMeans
        try:
            from sklearn.cluster import HDBSCAN as HDBSCANCluster
            clusterer = HDBSCANCluster(min_cluster_size=5)
            labels = clusterer.fit_predict(X_pca[:, :10])
        except ImportError:
            from sklearn.cluster import KMeans
            clusterer = KMeans(n_clusters=11, random_state=42, n_init=10)
            labels = clusterer.fit_predict(X_pca[:, :10])

        # Evaluate: how well do predicted clusters match gene families?
        from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

        true_labels = cluster_families
        ari = adjusted_rand_score(true_labels, labels)
        ami = adjusted_mutual_info_score(true_labels, labels)

        print(f"  ARI (family vs cluster): {ari:.3f}")
        print(f"  AMI (family vs cluster): {ami:.3f}")

        # Within-family vs between-family cosine similarity
        family_set = list(set(cluster_families))
        within_sims = []
        between_sims = []
        for i in range(len(cluster_names)):
            for j in range(i + 1, len(cluster_names)):
                sim = cos_sim[i, j]
                if cluster_families[i] == cluster_families[j]:
                    within_sims.append(sim)
                else:
                    between_sims.append(sim)

        if within_sims and between_sims:
            print(f"  Within-family cosine: {np.mean(within_sims):.3f} ± {np.std(within_sims):.3f}")
            print(f"  Between-family cosine: {np.mean(between_sims):.3f} ± {np.std(between_sims):.3f}")
            print(f"  Separation: {np.mean(within_sims) - np.mean(between_sims):.3f}")

        # Save results
        df_cluster = pd.DataFrame({
            "name": cluster_names,
            "family": cluster_families,
            "lineage": cluster_lineages,
            "cluster_label": labels,
            **{f"pc{i+1}": X_pca[:, i] for i in range(min(5, X_pca.shape[1]))},
        })
        df_cluster.to_csv(OUT_DIR / "clustering_results.tsv", sep="\t", index=False)

        # Summary JSON
        summary = {
            "n_sequences": X.shape[0],
            "embedding_dim": X.shape[1],
            "n_families": len(family_set),
            "ari": float(ari),
            "ami": float(ami),
            "within_family_cosine_mean": float(np.mean(within_sims)),
            "between_family_cosine_mean": float(np.mean(between_sims)),
            "pca_variance_explained": pca.explained_variance_ratio_[:10].tolist(),
        }
        with open(OUT_DIR / "analysis_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\nResults written to {OUT_DIR}/")


def print_panel_summary():
    """Print a summary of the panel composition."""
    from collections import Counter

    print(f"{'='*70}")
    print(f"Codon Periodicity & Functional Clustering Panel")
    print(f"{'='*70}")
    print(f"\nComponent A (Universality): {_n_comp_a} sequences")
    cats_a = Counter(e["category"] for e in COMPONENT_A)
    for cat, n in sorted(cats_a.items()):
        noncoding = sum(1 for e in COMPONENT_A if e["category"] == cat and e.get("noncoding"))
        coding = n - noncoding
        tag = f" [{noncoding} non-coding]" if noncoding else ""
        print(f"  {cat}: {n}{tag}")

    print(f"\nComponent B (Clustering): {_n_comp_b} sequences")
    cats_b = Counter(e["category"] for e in COMPONENT_B)
    for cat, n in sorted(cats_b.items()):
        print(f"  {cat}: {n}")

    print(f"\nComponent C (Negative Controls): {_n_comp_c} sequences")

    print(f"\nTOTAL: {len(PANEL)} sequences")

    # GC range
    gcs = [e["gc_approx"] for e in PANEL if "gc_approx" in e]
    print(f"GC content range: {min(gcs)}% - {max(gcs)}%")

    # Unique species
    species = set(e["species"] for e in PANEL)
    print(f"Unique species: {len(species)}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_panel_summary()
        print(f"\nUsage: {sys.argv[0]} [download|extract|analyze|summary]")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "download":
        download_sequences()
    elif cmd == "extract":
        extract_embeddings()
    elif cmd == "analyze":
        analyze()
    elif cmd == "summary":
        print_panel_summary()
    else:
        print(f"Unknown command: {cmd}")
        print(f"Usage: {sys.argv[0]} [download|extract|analyze|summary]")
        sys.exit(1)
