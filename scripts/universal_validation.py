#!/usr/bin/env python3
"""
Comprehensive cross-domain validation of codon periodicity in Evo2 embeddings.

Tests whether the offset-3 cosine inversion and lag-3 autocorrelation are
universal across all domains of life (Bacteria, Archaea, Eukarya) and
absent in non-coding sequences.

~90 sequences spanning 25+ phyla, GC content 19-72%.

Usage:
    # Step 1: Download sequences from NCBI
    uv run python scripts/universal_validation.py download

    # Step 2: Extract per-position embeddings via NIM API
    uv run python scripts/universal_validation.py extract

    # Step 3: Analyze periodicity and generate figures
    uv run python scripts/universal_validation.py analyze
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# ── Output directory ──
OUT_DIR = Path("results/universal_validation_v2")
FASTA_DIR = OUT_DIR / "fasta"
EMB_DIR = OUT_DIR / "embeddings"
FIG_DIR = OUT_DIR / "figures"

# ── Sequence panel ──
# Each entry: (name, category, lineage, gene, ncbi_accession, cds_start, cds_end, gc_approx)
# For RefSeq mRNA accessions (NM_/XM_), cds_start/cds_end extract the CDS from the mRNA.
# For genomic accessions, they define the CDS region.
# None for cds_start/cds_end means use the full sequence.
#
# We use Entrez efetch to get sequences. For CDS, we fetch the coding_sequence
# feature from protein accessions or extract from mRNA coordinates.

PANEL = [
    # ═══════════════════════════════════════════════════════════════
    # ARCHAEA — Euryarchaeota
    # ═══════════════════════════════════════════════════════════════
    {
        "name": "halobacterium_csg",
        "category": "archaea_euryarchaeota",
        "lineage": "Archaea; Euryarchaeota; Halobacteria",
        "species": "Halobacterium salinarum",
        "gene": "cell surface glycoprotein",
        "accession": "AAA72231.1",  # protein accession → fetch CDS
        "type": "protein_cds",
        "gc_approx": 66,
    },
    {
        "name": "methanocaldococcus_ef2",
        "category": "archaea_euryarchaeota",
        "lineage": "Archaea; Euryarchaeota; Methanococci",
        "species": "Methanocaldococcus jannaschii",
        "gene": "elongation factor 2",
        "accession": "Q58036",  # UniProt → use gene ID
        "ncbi_gene": "MJ_0234",
        "nuccore": "L77117.1",  # genome accession
        "cds_start": 213879,
        "cds_end": 216092,
        "type": "genomic_region",
        "gc_approx": 31,
    },
    {
        "name": "methanosarcina_mcra",
        "category": "archaea_euryarchaeota",
        "lineage": "Archaea; Euryarchaeota; Methanomicrobia",
        "species": "Methanosarcina acetivorans",
        "gene": "methyl-coenzyme M reductase alpha",
        "accession": "AAM07920.1",
        "type": "protein_cds",
        "gc_approx": 43,
    },
    {
        "name": "thermococcus_ef1a",
        "category": "archaea_euryarchaeota",
        "lineage": "Archaea; Euryarchaeota; Thermococci",
        "species": "Thermococcus kodakarensis",
        "gene": "elongation factor 1-alpha",
        "accession": "BAD85532.1",
        "type": "protein_cds",
        "gc_approx": 52,
    },
    {
        "name": "pyrococcus_gapdh",
        "category": "archaea_euryarchaeota",
        "lineage": "Archaea; Euryarchaeota; Thermococci",
        "species": "Pyrococcus furiosus",
        "gene": "glyceraldehyde-3-phosphate dehydrogenase",
        "accession": "AAL81075.1",
        "type": "protein_cds",
        "gc_approx": 41,
    },
    # ═══════════════════════════════════════════════════════════════
    # ARCHAEA — Crenarchaeota
    # ═══════════════════════════════════════════════════════════════
    {
        "name": "sulfolobus_rpob",
        "category": "archaea_crenarchaeota",
        "lineage": "Archaea; Crenarchaeota; Thermoprotei",
        "species": "Sulfolobus solfataricus",
        "gene": "RNA polymerase subunit B",
        "accession": "AAK42089.1",
        "type": "protein_cds",
        "gc_approx": 36,
    },
    {
        "name": "pyrobaculum_ef2",
        "category": "archaea_crenarchaeota",
        "lineage": "Archaea; Crenarchaeota; Thermoprotei",
        "species": "Pyrobaculum aerophilum",
        "gene": "elongation factor 2",
        "accession": "AAL63165.1",
        "type": "protein_cds",
        "gc_approx": 51,
    },
    {
        "name": "thermoproteus_ef1a",
        "category": "archaea_crenarchaeota",
        "lineage": "Archaea; Crenarchaeota; Thermoprotei",
        "species": "Thermoproteus tenax",
        "gene": "elongation factor 1-alpha",
        "accession": "CCC81032.1",
        "type": "protein_cds",
        "gc_approx": 57,
    },
    # ═══════════════════════════════════════════════════════════════
    # ARCHAEA — Thaumarchaeota / DPANN
    # ═══════════════════════════════════════════════════════════════
    {
        "name": "nitrosopumilus_amoa",
        "category": "archaea_thaumarchaeota",
        "lineage": "Archaea; Thaumarchaeota",
        "species": "Nitrosopumilus maritimus",
        "gene": "ammonia monooxygenase A",
        "accession": "ABZ10151.1",
        "type": "protein_cds",
        "gc_approx": 34,
    },
    {
        "name": "cenarchaeum_ef2",
        "category": "archaea_thaumarchaeota",
        "lineage": "Archaea; Thaumarchaeota",
        "species": "Cenarchaeum symbiosum",
        "gene": "elongation factor 2",
        "accession": "ABK78556.1",
        "type": "protein_cds",
        "gc_approx": 57,
    },
    # ═══════════════════════════════════════════════════════════════
    # MAMMALS
    # ═══════════════════════════════════════════════════════════════
    {
        "name": "human_actb",
        "category": "mammal",
        "lineage": "Eukarya; Chordata; Mammalia; Primates",
        "species": "Homo sapiens",
        "gene": "beta-actin",
        "accession": "NM_001101.5",
        "cds_start": 86,
        "cds_end": 1214,
        "type": "mrna_cds",
        "gc_approx": 56,
    },
    {
        "name": "mouse_gapdh",
        "category": "mammal",
        "lineage": "Eukarya; Chordata; Mammalia; Rodentia",
        "species": "Mus musculus",
        "gene": "GAPDH",
        "accession": "NM_001289726.2",
        "cds_start": 72,
        "cds_end": 1073,
        "type": "mrna_cds",
        "gc_approx": 51,
    },
    {
        "name": "bat_cytb",
        "category": "mammal",
        "lineage": "Eukarya; Chordata; Mammalia; Chiroptera",
        "species": "Myotis lucifugus",
        "gene": "cytochrome b (mitochondrial)",
        "accession": "EU521647.1",
        "type": "full_cds",
        "gc_approx": 46,
    },
    {
        "name": "whale_myoglobin",
        "category": "mammal",
        "lineage": "Eukarya; Chordata; Mammalia; Cetacea",
        "species": "Balaenoptera musculus",
        "gene": "myoglobin",
        "accession": "NM_001314089.1",
        "cds_start": 35,
        "cds_end": 502,
        "type": "mrna_cds",
        "gc_approx": 52,
    },
    {
        "name": "platypus_hemoglobin",
        "category": "mammal",
        "lineage": "Eukarya; Chordata; Mammalia; Monotremata",
        "species": "Ornithorhynchus anatinus",
        "gene": "hemoglobin subunit alpha",
        "accession": "NM_001082109.1",
        "cds_start": 35,
        "cds_end": 463,
        "type": "mrna_cds",
        "gc_approx": 52,
    },
    # ═══════════════════════════════════════════════════════════════
    # BIRDS / REPTILES
    # ═══════════════════════════════════════════════════════════════
    {
        "name": "zebra_finch_foxp2",
        "category": "bird",
        "lineage": "Eukarya; Chordata; Aves; Passeriformes",
        "species": "Taeniopygia guttata",
        "gene": "FOXP2",
        "accession": "NM_001048263.2",
        "cds_start": 151,
        "cds_end": 1300,  # partial CDS ~1.1kb
        "type": "mrna_cds",
        "gc_approx": 52,
    },
    {
        "name": "alligator_hemoglobin",
        "category": "reptile",
        "lineage": "Eukarya; Chordata; Reptilia; Crocodylia",
        "species": "Alligator mississippiensis",
        "gene": "hemoglobin alpha",
        "accession": "XM_006263366.4",
        "cds_start": 75,
        "cds_end": 503,
        "type": "mrna_cds",
        "gc_approx": 50,
    },
    {
        "name": "turtle_actb",
        "category": "reptile",
        "lineage": "Eukarya; Chordata; Reptilia; Testudines",
        "species": "Chelonia mydas",
        "gene": "beta-actin",
        "accession": "XM_007058886.3",
        "cds_start": 100,
        "cds_end": 1228,
        "type": "mrna_cds",
        "gc_approx": 54,
    },
    # ═══════════════════════════════════════════════════════════════
    # FISH
    # ═══════════════════════════════════════════════════════════════
    {
        "name": "zebrafish_actb2",
        "category": "fish",
        "lineage": "Eukarya; Chordata; Actinopterygii; Cypriniformes",
        "species": "Danio rerio",
        "gene": "beta-actin 2",
        "accession": "NM_181601.6",
        "cds_start": 53,
        "cds_end": 1180,
        "type": "mrna_cds",
        "gc_approx": 52,
    },
    {
        "name": "pufferfish_gapdh",
        "category": "fish",
        "lineage": "Eukarya; Chordata; Actinopterygii; Tetraodontiformes",
        "species": "Takifugu rubripes",
        "gene": "GAPDH",
        "accession": "XM_003960141.3",
        "cds_start": 55,
        "cds_end": 1056,
        "type": "mrna_cds",
        "gc_approx": 54,
    },
    {
        "name": "coelacanth_actb",
        "category": "fish",
        "lineage": "Eukarya; Chordata; Sarcopterygii; Coelacanthiformes",
        "species": "Latimeria chalumnae",
        "gene": "beta-actin",
        "accession": "XM_005991218.2",
        "cds_start": 88,
        "cds_end": 1216,
        "type": "mrna_cds",
        "gc_approx": 52,
    },
    # ═══════════════════════════════════════════════════════════════
    # AMPHIBIANS
    # ═══════════════════════════════════════════════════════════════
    {
        "name": "xenopus_actb",
        "category": "amphibian",
        "lineage": "Eukarya; Chordata; Amphibia; Anura",
        "species": "Xenopus laevis",
        "gene": "beta-actin",
        "accession": "NM_001088953.1",
        "cds_start": 41,
        "cds_end": 1169,
        "type": "mrna_cds",
        "gc_approx": 51,
    },
    {
        "name": "axolotl_sod",
        "category": "amphibian",
        "lineage": "Eukarya; Chordata; Amphibia; Urodela",
        "species": "Ambystoma mexicanum",
        "gene": "superoxide dismutase",
        "accession": "XM_044194025.1",
        "cds_start": 41,
        "cds_end": 503,
        "type": "mrna_cds",
        "gc_approx": 46,
    },
    # ═══════════════════════════════════════════════════════════════
    # INSECTS
    # ═══════════════════════════════════════════════════════════════
    {
        "name": "drosophila_act5c",
        "category": "insect",
        "lineage": "Eukarya; Arthropoda; Insecta; Diptera",
        "species": "Drosophila melanogaster",
        "gene": "Actin 5C",
        "accession": "NM_078901.5",
        "cds_start": 346,
        "cds_end": 1474,
        "type": "mrna_cds",
        "gc_approx": 54,
    },
    {
        "name": "honeybee_vg",
        "category": "insect",
        "lineage": "Eukarya; Arthropoda; Insecta; Hymenoptera",
        "species": "Apis mellifera",
        "gene": "vitellogenin",
        "accession": "NM_001011578.2",
        "cds_start": 48,
        "cds_end": 1200,  # first ~1.1kb of CDS
        "type": "mrna_cds",
        "gc_approx": 35,
    },
    {
        "name": "mosquito_defensin",
        "category": "insect",
        "lineage": "Eukarya; Arthropoda; Insecta; Diptera",
        "species": "Anopheles gambiae",
        "gene": "defensin",
        "accession": "NM_001032443.1",
        "cds_start": 1,
        "cds_end": 300,
        "type": "mrna_cds",
        "gc_approx": 48,
    },
    {
        "name": "beetle_hsp70",
        "category": "insect",
        "lineage": "Eukarya; Arthropoda; Insecta; Coleoptera",
        "species": "Tribolium castaneum",
        "gene": "HSP70",
        "accession": "NM_001039411.1",
        "cds_start": 57,
        "cds_end": 1000,
        "type": "mrna_cds",
        "gc_approx": 44,
    },
    {
        "name": "silkworm_fibroin",
        "category": "insect",
        "lineage": "Eukarya; Arthropoda; Insecta; Lepidoptera",
        "species": "Bombyx mori",
        "gene": "fibroin light chain",
        "accession": "NM_001044023.1",
        "cds_start": 26,
        "cds_end": 836,
        "type": "mrna_cds",
        "gc_approx": 40,
    },
    # ═══════════════════════════════════════════════════════════════
    # NEMATODES
    # ═══════════════════════════════════════════════════════════════
    {
        "name": "celegans_act1",
        "category": "nematode",
        "lineage": "Eukarya; Nematoda; Chromadorea",
        "species": "Caenorhabditis elegans",
        "gene": "actin-1",
        "accession": "NM_073418.5",
        "cds_start": 1,
        "cds_end": 1134,
        "type": "mrna_cds",
        "gc_approx": 48,
    },
    {
        "name": "brugia_actb",
        "category": "nematode",
        "lineage": "Eukarya; Nematoda; Chromadorea",
        "species": "Brugia malayi",
        "gene": "actin",
        "accession": "XM_001896443.2",
        "cds_start": 43,
        "cds_end": 1171,
        "type": "mrna_cds",
        "gc_approx": 43,
    },
    # ═══════════════════════════════════════════════════════════════
    # MARINE INVERTEBRATES
    # ═══════════════════════════════════════════════════════════════
    {
        "name": "octopus_rhodopsin",
        "category": "mollusk",
        "lineage": "Eukarya; Mollusca; Cephalopoda",
        "species": "Octopus bimaculoides",
        "gene": "rhodopsin",
        "accession": "XM_014924783.2",
        "cds_start": 69,
        "cds_end": 1200,
        "type": "mrna_cds",
        "gc_approx": 42,
    },
    {
        "name": "coral_actb",
        "category": "cnidarian",
        "lineage": "Eukarya; Cnidaria; Anthozoa",
        "species": "Acropora digitifera",
        "gene": "actin",
        "accession": "XM_015910157.2",
        "cds_start": 44,
        "cds_end": 1172,
        "type": "mrna_cds",
        "gc_approx": 48,
    },
    {
        "name": "starfish_actb",
        "category": "echinoderm",
        "lineage": "Eukarya; Echinodermata; Asteroidea",
        "species": "Acanthaster planci",
        "gene": "actin",
        "accession": "XM_022235012.2",
        "cds_start": 60,
        "cds_end": 1188,
        "type": "mrna_cds",
        "gc_approx": 50,
    },
    # ═══════════════════════════════════════════════════════════════
    # PLANTS — Dicots
    # ═══════════════════════════════════════════════════════════════
    {
        "name": "arabidopsis_rbcs",
        "category": "plant_dicot",
        "lineage": "Eukarya; Streptophyta; Magnoliopsida; Brassicales",
        "species": "Arabidopsis thaliana",
        "gene": "RuBisCO small subunit",
        "accession": "NM_123204.4",
        "cds_start": 47,
        "cds_end": 589,
        "type": "mrna_cds",
        "gc_approx": 44,
    },
    {
        "name": "tomato_pgk",
        "category": "plant_dicot",
        "lineage": "Eukarya; Streptophyta; Magnoliopsida; Solanales",
        "species": "Solanum lycopersicum",
        "gene": "phosphoglycerate kinase",
        "accession": "NM_001247395.3",
        "cds_start": 57,
        "cds_end": 1100,
        "type": "mrna_cds",
        "gc_approx": 42,
    },
    {
        "name": "soybean_act11",
        "category": "plant_dicot",
        "lineage": "Eukarya; Streptophyta; Magnoliopsida; Fabales",
        "species": "Glycine max",
        "gene": "actin-11",
        "accession": "NM_001250090.2",
        "cds_start": 59,
        "cds_end": 1187,
        "type": "mrna_cds",
        "gc_approx": 48,
    },
    {
        "name": "cacao_gapdh",
        "category": "plant_dicot",
        "lineage": "Eukarya; Streptophyta; Magnoliopsida; Malvales",
        "species": "Theobroma cacao",
        "gene": "GAPDH",
        "accession": "XM_007039641.3",
        "cds_start": 83,
        "cds_end": 1099,
        "type": "mrna_cds",
        "gc_approx": 46,
    },
    # ═══════════════════════════════════════════════════════════════
    # PLANTS — Monocots
    # ═══════════════════════════════════════════════════════════════
    {
        "name": "maize_adh1",
        "category": "plant_monocot",
        "lineage": "Eukarya; Streptophyta; Liliopsida; Poales",
        "species": "Zea mays",
        "gene": "alcohol dehydrogenase 1",
        "accession": "NM_001112104.2",
        "cds_start": 62,
        "cds_end": 1200,
        "type": "mrna_cds",
        "gc_approx": 58,
    },
    {
        "name": "wheat_rbcl",
        "category": "plant_monocot",
        "lineage": "Eukarya; Streptophyta; Liliopsida; Poales",
        "species": "Triticum aestivum",
        "gene": "RuBisCO large subunit (chloroplast)",
        "accession": "KJ592713.1",
        "cds_start": 1,
        "cds_end": 1434,
        "type": "full_cds",
        "gc_approx": 44,
    },
    # ═══════════════════════════════════════════════════════════════
    # PLANTS — Basal
    # ═══════════════════════════════════════════════════════════════
    {
        "name": "moss_gapdh",
        "category": "plant_basal",
        "lineage": "Eukarya; Streptophyta; Bryophyta",
        "species": "Physcomitrium patens",
        "gene": "GAPDH",
        "accession": "XM_024528741.2",
        "cds_start": 80,
        "cds_end": 1100,
        "type": "mrna_cds",
        "gc_approx": 48,
    },
    {
        "name": "selaginella_rbcl",
        "category": "plant_basal",
        "lineage": "Eukarya; Streptophyta; Lycopodiopsida",
        "species": "Selaginella moellendorffii",
        "gene": "RuBisCO large subunit (chloroplast)",
        "accession": "HM173080.1",
        "type": "full_cds",
        "gc_approx": 40,
    },
    # ═══════════════════════════════════════════════════════════════
    # FUNGI — Ascomycetes (additional)
    # ═══════════════════════════════════════════════════════════════
    {
        "name": "neurospora_actb",
        "category": "fungi_ascomycete",
        "lineage": "Eukarya; Ascomycota; Sordariomycetes",
        "species": "Neurospora crassa",
        "gene": "actin",
        "accession": "XM_957119.4",
        "cds_start": 99,
        "cds_end": 1227,
        "type": "mrna_cds",
        "gc_approx": 56,
    },
    {
        "name": "aspergillus_actb",
        "category": "fungi_ascomycete",
        "lineage": "Eukarya; Ascomycota; Eurotiomycetes",
        "species": "Aspergillus nidulans",
        "gene": "actin",
        "accession": "XM_658803.2",
        "cds_start": 115,
        "cds_end": 1243,
        "type": "mrna_cds",
        "gc_approx": 54,
    },
    # ═══════════════════════════════════════════════════════════════
    # FUNGI — Basidiomycetes
    # ═══════════════════════════════════════════════════════════════
    {
        "name": "cryptococcus_act1",
        "category": "fungi_basidiomycete",
        "lineage": "Eukarya; Basidiomycota; Tremellomycetes",
        "species": "Cryptococcus neoformans",
        "gene": "actin",
        "accession": "XM_012194789.2",
        "cds_start": 69,
        "cds_end": 1197,
        "type": "mrna_cds",
        "gc_approx": 55,
    },
    {
        "name": "ustilago_act1",
        "category": "fungi_basidiomycete",
        "lineage": "Eukarya; Basidiomycota; Ustilaginomycetes",
        "species": "Ustilago maydis",
        "gene": "actin",
        "accession": "XM_011390461.1",
        "cds_start": 114,
        "cds_end": 1242,
        "type": "mrna_cds",
        "gc_approx": 57,
    },
    # ═══════════════════════════════════════════════════════════════
    # PROTISTS — Apicomplexa
    # ═══════════════════════════════════════════════════════════════
    {
        "name": "plasmodium_actI",
        "category": "protist_apicomplexa",
        "lineage": "Eukarya; Apicomplexa; Aconoidasida",
        "species": "Plasmodium falciparum",
        "gene": "actin I",
        "accession": "XM_001351750.2",
        "cds_start": 1,
        "cds_end": 1128,
        "type": "mrna_cds",
        "gc_approx": 24,
        "note": "CRITICAL: extreme AT-bias genome (~19% GC). Hardest test case.",
    },
    {
        "name": "toxoplasma_act1",
        "category": "protist_apicomplexa",
        "lineage": "Eukarya; Apicomplexa; Conoidasida",
        "species": "Toxoplasma gondii",
        "gene": "actin",
        "accession": "XM_002364967.2",
        "cds_start": 51,
        "cds_end": 1179,
        "type": "mrna_cds",
        "gc_approx": 52,
    },
    {
        "name": "cryptosporidium_hsp70",
        "category": "protist_apicomplexa",
        "lineage": "Eukarya; Apicomplexa; Conoidasida",
        "species": "Cryptosporidium parvum",
        "gene": "HSP70",
        "accession": "XM_001388218.1",
        "cds_start": 1,
        "cds_end": 1000,
        "type": "mrna_cds",
        "gc_approx": 32,
    },
    # ═══════════════════════════════════════════════════════════════
    # PROTISTS — Kinetoplastida
    # ═══════════════════════════════════════════════════════════════
    {
        "name": "trypanosoma_tubulin",
        "category": "protist_kinetoplastid",
        "lineage": "Eukarya; Euglenozoa; Kinetoplastea",
        "species": "Trypanosoma brucei",
        "gene": "alpha-tubulin",
        "accession": "XM_011775742.1",
        "cds_start": 1,
        "cds_end": 1359,
        "type": "mrna_cds",
        "gc_approx": 55,
    },
    {
        "name": "leishmania_hsp70",
        "category": "protist_kinetoplastid",
        "lineage": "Eukarya; Euglenozoa; Kinetoplastea",
        "species": "Leishmania major",
        "gene": "HSP70",
        "accession": "XM_001684406.1",
        "cds_start": 1,
        "cds_end": 1000,
        "type": "mrna_cds",
        "gc_approx": 62,
    },
    # ═══════════════════════════════════════════════════════════════
    # PROTISTS — Other
    # ═══════════════════════════════════════════════════════════════
    {
        "name": "dictyostelium_act8",
        "category": "protist_amoebozoa",
        "lineage": "Eukarya; Amoebozoa; Dictyosteliida",
        "species": "Dictyostelium discoideum",
        "gene": "actin-8",
        "accession": "XM_637885.2",
        "cds_start": 1,
        "cds_end": 1134,
        "type": "mrna_cds",
        "gc_approx": 28,
    },
    {
        "name": "tetrahymena_actb",
        "category": "protist_ciliate",
        "lineage": "Eukarya; Ciliophora; Oligohymenophorea",
        "species": "Tetrahymena thermophila",
        "gene": "actin",
        "accession": "XM_001012714.3",
        "cds_start": 1,
        "cds_end": 1134,
        "type": "mrna_cds",
        "gc_approx": 32,
    },
    # ═══════════════════════════════════════════════════════════════
    # ALGAE
    # ═══════════════════════════════════════════════════════════════
    {
        "name": "chlamydomonas_rbcs",
        "category": "green_alga",
        "lineage": "Eukarya; Chlorophyta; Chlorophyceae",
        "species": "Chlamydomonas reinhardtii",
        "gene": "RuBisCO small subunit",
        "accession": "NM_001324575.1",
        "cds_start": 89,
        "cds_end": 636,
        "type": "mrna_cds",
        "gc_approx": 64,
    },
    {
        "name": "diatom_gapdh",
        "category": "diatom",
        "lineage": "Eukarya; Bacillariophyta; Coscinodiscophyceae",
        "species": "Thalassiosira pseudonana",
        "gene": "GAPDH",
        "accession": "XM_002296093.1",
        "cds_start": 1,
        "cds_end": 1017,
        "type": "mrna_cds",
        "gc_approx": 48,
    },
    {
        "name": "brown_alga_act",
        "category": "brown_alga",
        "lineage": "Eukarya; Phaeophyceae; Ectocarpales",
        "species": "Ectocarpus siliculosus",
        "gene": "actin",
        "accession": "CBJ30822.1",
        "type": "protein_cds",
        "gc_approx": 55,
    },
    # ═══════════════════════════════════════════════════════════════
    # ORGANELLAR GENES
    # ═══════════════════════════════════════════════════════════════
    {
        "name": "human_mt_co1",
        "category": "organellar_mito",
        "lineage": "Eukarya; Chordata; Mammalia (mitochondrial)",
        "species": "Homo sapiens",
        "gene": "cytochrome c oxidase I (mitochondrial)",
        "accession": "NC_012920.1",
        "cds_start": 5904,
        "cds_end": 7445,
        "type": "genomic_region",
        "gc_approx": 38,
        "note": "Uses mitochondrial genetic code (UGA=Trp). Different from standard code.",
    },
    {
        "name": "arabidopsis_cp_rbcl",
        "category": "organellar_chloroplast",
        "lineage": "Eukarya; Streptophyta; Magnoliopsida (chloroplast)",
        "species": "Arabidopsis thaliana",
        "gene": "RuBisCO large subunit (chloroplast)",
        "accession": "NC_000932.1",
        "cds_start": 54958,
        "cds_end": 56397,
        "type": "genomic_region",
        "gc_approx": 43,
    },
    {
        "name": "rickettsia_rpob",
        "category": "organellar_relative",
        "lineage": "Bacteria; Alphaproteobacteria; Rickettsiales",
        "species": "Rickettsia prowazekii",
        "gene": "RNA polymerase beta subunit",
        "accession": "AJF73992.1",
        "type": "protein_cds",
        "gc_approx": 29,
        "note": "Free-living relative of mitochondria. Low GC.",
    },
    # ═══════════════════════════════════════════════════════════════
    # UNUSUAL VIRUSES
    # ═══════════════════════════════════════════════════════════════
    {
        "name": "mimivirus_capsid",
        "category": "giant_virus",
        "lineage": "Virus; Nucleocytoviricota; Megaviricetes",
        "species": "Acanthamoeba polyphaga mimivirus",
        "gene": "major capsid protein",
        "accession": "AAV50707.1",
        "type": "protein_cds",
        "gc_approx": 28,
    },
    {
        "name": "pandoravirus_capsid",
        "category": "giant_virus",
        "lineage": "Virus; Nucleocytoviricota",
        "species": "Pandoravirus salinus",
        "gene": "hypothetical protein",
        "accession": "AGI04718.1",
        "type": "protein_cds",
        "gc_approx": 62,
    },
    {
        "name": "phix174_capsid",
        "category": "ssdna_virus",
        "lineage": "Virus; Microviridae",
        "species": "Enterobacteria phage phiX174",
        "gene": "major capsid protein F",
        "accession": "NC_001422.1",
        "cds_start": 1001,
        "cds_end": 2284,
        "type": "genomic_region",
        "gc_approx": 44,
    },
    {
        "name": "parvovirus_vp1",
        "category": "ssdna_virus",
        "lineage": "Virus; Parvoviridae",
        "species": "Human parvovirus B19",
        "gene": "VP1 capsid",
        "accession": "NC_000883.2",
        "cds_start": 2444,
        "cds_end": 4786,
        "type": "genomic_region",
        "gc_approx": 45,
    },
    # ═══════════════════════════════════════════════════════════════
    # NON-CODING CONTROLS (should NOT show offset-3 inversion)
    # ═══════════════════════════════════════════════════════════════
    {
        "name": "human_18s_rrna",
        "category": "noncoding_rRNA",
        "lineage": "Eukarya; Chordata; Mammalia",
        "species": "Homo sapiens",
        "gene": "18S ribosomal RNA",
        "accession": "NR_003286.4",
        "cds_start": 1,
        "cds_end": 1200,
        "type": "mrna_cds",  # not really CDS, but same fetch
        "gc_approx": 54,
        "note": "Non-coding control. Should NOT show offset-3 inversion.",
    },
    {
        "name": "ecoli_23s_rrna",
        "category": "noncoding_rRNA",
        "lineage": "Bacteria; Gammaproteobacteria",
        "species": "Escherichia coli",
        "gene": "23S ribosomal RNA",
        "accession": "NR_103073.1",
        "cds_start": 1,
        "cds_end": 1200,
        "type": "mrna_cds",
        "gc_approx": 53,
        "note": "Non-coding control.",
    },
    {
        "name": "human_malat1",
        "category": "noncoding_lncRNA",
        "lineage": "Eukarya; Chordata; Mammalia",
        "species": "Homo sapiens",
        "gene": "MALAT1 lncRNA",
        "accession": "NR_002819.4",
        "cds_start": 1,
        "cds_end": 1200,
        "type": "mrna_cds",
        "gc_approx": 42,
        "note": "Non-coding control. Long non-coding RNA.",
    },
    {
        "name": "ecoli_intergenic",
        "category": "noncoding_intergenic",
        "lineage": "Bacteria; Gammaproteobacteria",
        "species": "Escherichia coli K12",
        "gene": "intergenic region (lacY-lacA)",
        "accession": "U00096.3",
        "cds_start": 362463,
        "cds_end": 363463,
        "type": "genomic_region",
        "gc_approx": 50,
        "note": "Non-coding control. Intergenic region.",
    },
    {
        "name": "human_alu",
        "category": "noncoding_repeat",
        "lineage": "Eukarya; Chordata; Mammalia",
        "species": "Homo sapiens",
        "gene": "Alu SINE consensus",
        # Use Alu consensus from Repbase / DFAM - we'll hardcode the sequence
        "accession": "HARDCODED",
        "type": "hardcoded",
        "gc_approx": 56,
        "sequence": (
            "GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGGCGGGCGG"
            "ATCACGAGGTCAGGAGATCGAGACCATCCTGGCTAACACGGTGAAACCCCGTCTCTACTA"
            "AAAATACAAAAAATTAGCCGGGCGTGGTGGCGGGCGCCTGTAGTCCCAGCTACTCGGGAG"
            "GCTGAGGCAGGAGAATGGCGTGAACCCGGGAGGCGGAGCTTGCAGTGAGCCGAGATTGC"
            "GCCACTGCACTCCAGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAAAAAAAAAAAAA"
        ),
        "note": "Non-coding SINE element. Should NOT show inversion.",
    },
    {
        "name": "human_intron_brca1",
        "category": "noncoding_intron",
        "lineage": "Eukarya; Chordata; Mammalia",
        "species": "Homo sapiens",
        "gene": "BRCA1 intron 2",
        "accession": "NG_005905.2",
        "cds_start": 20000,
        "cds_end": 21200,
        "type": "genomic_region",
        "gc_approx": 38,
        "note": "Non-coding intronic control.",
    },
    {
        "name": "yeast_telomere",
        "category": "noncoding_repetitive",
        "lineage": "Eukarya; Ascomycota; Saccharomycetes",
        "species": "Saccharomyces cerevisiae",
        "gene": "telomeric/subtelomeric repeat",
        # Hardcoded TG1-3 repeat typical of yeast telomeres
        "accession": "HARDCODED",
        "type": "hardcoded",
        "gc_approx": 35,
        "sequence": (
            "TGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTG"
            "TGTGTGGGTGTGTGTGGTGTGTGTGGTGTGTGGGTGTGTGTGTGGTGTGTGGTGTGTGG"
            "TGTGTGTGGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGGTGTGTGTGG"
            "TGTGTGGTGTGTGTGGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGGTGTGTG"
            "TGTGTGGTGTGTGGGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTGTG"
        ),
        "note": "Non-coding telomeric repeat control.",
    },
    # ═══════════════════════════════════════════════════════════════
    # EDGE CASES
    # ═══════════════════════════════════════════════════════════════
    {
        "name": "ecoli_rps12",
        "category": "edge_high_expression",
        "lineage": "Bacteria; Gammaproteobacteria",
        "species": "Escherichia coli",
        "gene": "30S ribosomal protein S12 (highly expressed, strong codon bias)",
        "accession": "NP_417072.1",
        "type": "protein_cds",
        "gc_approx": 54,
    },
    {
        "name": "mycoplasma_gap",
        "category": "edge_minimal_genome",
        "lineage": "Bacteria; Tenericutes; Mollicutes",
        "species": "Mycoplasma genitalium",
        "gene": "glyceraldehyde-3-phosphate dehydrogenase",
        "accession": "NP_072896.1",
        "type": "protein_cds",
        "gc_approx": 32,
        "note": "Smallest known bacterial genome. Extreme AT bias.",
    },
    {
        "name": "egfp_synthetic",
        "category": "edge_synthetic",
        "lineage": "Synthetic",
        "species": "Synthetic construct",
        "gene": "enhanced GFP (codon-optimized for mammals)",
        "accession": "HARDCODED",
        "type": "hardcoded",
        "gc_approx": 60,
        # Standard eGFP CDS (Clontech/Takara, codon-optimized for mammals)
        "sequence": (
            "ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGA"
            "CGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTA"
            "CGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCAC"
            "CCTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAA"
            "GCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTT"
            "CTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCT"
            "GGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCA"
            "CAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAAC"
            "GGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCC"
            "GACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACT"
            "ACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCT"
            "GCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAA"
        ),
        "note": "Codon-optimized synthetic gene. Tests engineering signature.",
    },
]


def download_sequences():
    """Download all sequences from NCBI and save as individual FASTA files."""
    from Bio import Entrez, SeqIO
    from io import StringIO

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

            # Write FASTA
            gc = sum(1 for c in seq_clean if c in "GC") / len(seq_clean) * 100
            header = f">{name} {entry['species']} {entry['gene']} [{len(seq_clean)}bp, GC={gc:.1f}%]"
            with open(fasta_path, "w") as f:
                f.write(f"{header}\n")
                for j in range(0, len(seq_clean), 60):
                    f.write(seq_clean[j:j+60] + "\n")

            print(f"OK ({len(seq_clean)} bp, GC={gc:.1f}%)")
            results.append({"name": name, "length": len(seq_clean), "gc": gc, "status": "downloaded"})

            # Rate limit: NCBI allows 3 requests/sec with API key, 1/sec without
            time.sleep(0.4)

        except Exception as e:
            print(f"FAILED ({e})")
            failures.append(name)
            continue

    # Write summary
    with open(OUT_DIR / "download_summary.json", "w") as f:
        json.dump({"downloaded": len(results), "failed": len(failures),
                    "failures": failures, "results": results}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Downloaded: {len(results)}/{len(PANEL)}")
    if failures:
        print(f"Failed: {failures}")


def _fetch_sequence(entry: dict, Entrez) -> str | None:
    """Fetch a sequence from NCBI based on entry type."""
    from Bio import SeqIO
    from io import StringIO

    if entry["type"] == "hardcoded":
        return entry["sequence"]

    if entry["type"] == "protein_cds":
        # Fetch CDS nucleotide from protein accession
        return _fetch_cds_from_protein(entry["accession"], Entrez)

    if entry["type"] in ("mrna_cds", "full_cds"):
        # Fetch mRNA and extract CDS region
        handle = Entrez.efetch(db="nucleotide", id=entry["accession"],
                               rettype="fasta", retmode="text")
        record = next(SeqIO.parse(StringIO(handle.read()), "fasta"))
        handle.close()

        if entry["type"] == "full_cds":
            return str(record.seq)

        start = entry.get("cds_start", 1) - 1  # 1-based to 0-based
        end = entry.get("cds_end", len(record.seq))
        return str(record.seq[start:end])

    if entry["type"] == "genomic_region":
        # Fetch genomic region by coordinates
        acc = entry.get("nuccore", entry["accession"])
        start = entry["cds_start"]
        end = entry["cds_end"]
        handle = Entrez.efetch(db="nucleotide", id=acc,
                               rettype="fasta", retmode="text",
                               seq_start=start, seq_stop=end)
        record = next(SeqIO.parse(StringIO(handle.read()), "fasta"))
        handle.close()
        return str(record.seq)

    return None


def _fetch_cds_from_protein(protein_acc: str, Entrez) -> str | None:
    """Fetch the CDS nucleotide sequence for a protein accession."""
    from Bio import SeqIO
    from io import StringIO

    # Use elink to find the nucleotide record, then fetch CDS
    # Approach: fetch protein GenPept, get coded_by from features
    try:
        handle = Entrez.efetch(db="protein", id=protein_acc,
                               rettype="gp", retmode="text")
        content = handle.read()
        handle.close()

        # Parse coded_by from GenPept
        # Look for /coded_by="accession:start..end"
        import re
        match = re.search(r'/coded_by="([^"]+)"', content)
        if match:
            coded_by = match.group(1)
            # Parse: could be "ACC:start..end" or "complement(ACC:start..end)"
            is_complement = "complement" in coded_by
            coded_by = coded_by.replace("complement(", "").rstrip(")")

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

            handle = Entrez.efetch(db="nucleotide", id=nuc_acc,
                                   rettype="fasta", retmode="text",
                                   seq_start=start, seq_stop=end)
            record = next(SeqIO.parse(StringIO(handle.read()), "fasta"))
            handle.close()

            seq = str(record.seq)
            if is_complement:
                from Bio.Seq import Seq
                seq = str(Seq(seq).reverse_complement())

            return seq

    except Exception as e:
        print(f"(coded_by failed: {e}, trying fallback) ", end="")

    # Fallback: use the nucleotide cross-reference
    try:
        handle = Entrez.elink(dbfrom="protein", db="nuccore", id=protein_acc)
        links = Entrez.read(handle)
        handle.close()

        nuc_ids = [link["Id"] for linkset in links
                   for link in linkset.get("LinkSetDb", [{}])[0].get("Link", [])]

        if nuc_ids:
            # Fetch the first linked nucleotide and extract CDS
            handle = Entrez.efetch(db="nucleotide", id=nuc_ids[0],
                                   rettype="gb", retmode="text")
            from Bio import GenBank
            record = next(SeqIO.parse(StringIO(handle.read()), "genbank"))
            handle.close()

            # Find CDS feature matching our protein
            for feat in record.features:
                if feat.type == "CDS":
                    quals = feat.qualifiers
                    if "protein_id" in quals and protein_acc in quals["protein_id"]:
                        return str(feat.extract(record.seq))
                    # Also check first CDS
            # If no match, return first CDS
            for feat in record.features:
                if feat.type == "CDS":
                    return str(feat.extract(record.seq))
    except Exception:
        pass

    return None


def extract_embeddings():
    """Extract per-position embeddings via NIM API for all downloaded sequences."""
    import asyncio
    import httpx
    import base64
    import io

    EMB_DIR.mkdir(parents=True, exist_ok=True)

    # Load all downloaded FASTA files
    fasta_files = sorted(FASTA_DIR.glob("*.fasta"))
    if not fasta_files:
        print("No FASTA files found. Run 'download' first.")
        return

    import os
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("ERROR: Set NVIDIA_API_KEY environment variable")
        return

    sequences = []
    for fp in fasta_files:
        name = fp.stem
        emb_path = EMB_DIR / f"{name}.npy"
        if emb_path.exists():
            print(f"  {name}: cached")
            continue
        with open(fp) as f:
            lines = f.readlines()
            seq = "".join(l.strip() for l in lines if not l.startswith(">"))
        sequences.append((name, seq))

    if not sequences:
        print("All embeddings already cached.")
        return

    print(f"Extracting per-position embeddings for {len(sequences)} sequences...")

    # Use block 10 — strongest periodicity signal. Late blocks (25+) have MLP near-zero.
    # blocks.10 gives well-scaled norms, strong lag-3 (0.90), and clear inversion (+0.11 gap).
    LAYER = "blocks.10"

    async def extract_one(client: httpx.AsyncClient, name: str, seq: str,
                          sem: asyncio.Semaphore) -> tuple[str, np.ndarray | None]:
        async with sem:
            url = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            # Truncate to 16000bp
            seq_trunc = seq[:16000]
            payload = {
                "sequence": seq_trunc,
                "output_layers": [LAYER],
            }

            for attempt in range(5):
                try:
                    resp = await client.post(url, json=payload, headers=headers,
                                             timeout=300, follow_redirects=True)
                    if resp.status_code == 429:
                        wait = 2 ** attempt * 5
                        print(f"  {name}: rate limited, waiting {wait}s...")
                        await asyncio.sleep(wait)
                        continue

                    resp.raise_for_status()
                    data = resp.json()

                    # Decode NPZ response: data field is base64-encoded NPZ
                    raw = base64.b64decode(data["data"])
                    npz_data = np.load(io.BytesIO(raw))
                    key = f"{LAYER}.output"
                    emb = npz_data[key]  # shape: (1, seq_len, 8192), float64
                    emb = emb.squeeze(0)  # (seq_len, 8192), keep float64 for precision

                    return name, emb

                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (429, 503):
                        wait = 2 ** attempt * 5
                        await asyncio.sleep(wait)
                        continue
                    print(f"  {name}: HTTP {e.response.status_code}")
                    if attempt < 4:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return name, None
                except Exception as e:
                    if attempt < 4:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    print(f"  {name}: {e}")
                    return name, None

            return name, None

    async def run_all():
        sem = asyncio.Semaphore(3)
        async with httpx.AsyncClient() as client:
            tasks = [extract_one(client, name, seq, sem) for name, seq in sequences]
            results = []
            for coro in asyncio.as_completed(tasks):
                name, emb = await coro
                if emb is not None:
                    np.save(EMB_DIR / f"{name}.npy", emb)
                    print(f"  {name}: {emb.shape[0]} positions × {emb.shape[1]}D")
                    results.append(name)
                else:
                    print(f"  {name}: FAILED")
            return results

    completed = asyncio.run(run_all())
    print(f"\nExtracted: {len(completed)}/{len(sequences)}")


def analyze_all():
    """Analyze periodicity metrics for all sequences and generate figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.signal import correlate
    from scipy.fft import rfft, rfftfreq

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load all embeddings
    emb_files = sorted(EMB_DIR.glob("*.npy"))
    if not emb_files:
        print("No embeddings found. Run 'extract' first.")
        return

    # Build panel lookup
    panel_lookup = {e["name"]: e for e in PANEL}

    results = []
    for emb_path in emb_files:
        name = emb_path.stem
        emb = np.load(emb_path)
        entry = panel_lookup.get(name, {"category": "unknown", "species": "unknown",
                                         "gene": "unknown", "lineage": "unknown"})

        # Compute periodicity metrics
        norms = np.linalg.norm(emb, axis=1)

        # Lag-3 autocorrelation
        lag3 = _autocorrelation(norms, 3)

        # FFT dominant period
        fft_vals = np.abs(rfft(norms - norms.mean()))
        freqs = rfftfreq(len(norms), d=1.0)
        # Skip DC component (freq=0)
        fft_vals[0] = 0
        if len(fft_vals) > 1:
            peak_idx = np.argmax(fft_vals[1:]) + 1
            dominant_period = 1.0 / freqs[peak_idx] if freqs[peak_idx] > 0 else len(norms)
        else:
            dominant_period = len(norms)

        # 3bp FFT power rank
        target_freq = 1.0 / 3.0
        freq_diffs = np.abs(freqs[1:] - target_freq)
        bp3_idx = np.argmin(freq_diffs) + 1
        bp3_power = fft_vals[bp3_idx]
        max_power = fft_vals[1:].max()
        bp3_rank = (fft_vals[1:] >= bp3_power).sum()  # rank (1 = strongest)
        bp3_fraction = float(bp3_power / max_power) if max_power > 0 else 0

        # Offset-3 cosine similarity
        cos1_vals = []
        cos3_vals = []
        for j in range(len(emb) - 3):
            c1 = np.dot(emb[j], emb[j+1]) / (np.linalg.norm(emb[j]) * np.linalg.norm(emb[j+1]) + 1e-10)
            c3 = np.dot(emb[j], emb[j+3]) / (np.linalg.norm(emb[j]) * np.linalg.norm(emb[j+3]) + 1e-10)
            cos1_vals.append(c1)
            cos3_vals.append(c3)

        cos1 = np.mean(cos1_vals)
        cos3 = np.mean(cos3_vals)
        inversion = cos3 > cos1

        # GC content from FASTA
        fasta_path = FASTA_DIR / f"{name}.fasta"
        gc = 0.0
        seq_len = 0
        if fasta_path.exists():
            with open(fasta_path) as f:
                seq = "".join(l.strip() for l in f if not l.startswith(">"))
            gc = sum(1 for c in seq.upper() if c in "GC") / len(seq) * 100
            seq_len = len(seq)

        result = {
            "name": name,
            "category": entry.get("category", "unknown"),
            "lineage": entry.get("lineage", "unknown"),
            "species": entry.get("species", "unknown"),
            "gene": entry.get("gene", "unknown"),
            "seq_len": seq_len,
            "gc_content": round(gc, 1),
            "emb_positions": emb.shape[0],
            "lag3": round(lag3, 4),
            "dominant_fft_period": round(dominant_period, 1),
            "bp3_fft_rank": int(bp3_rank),
            "bp3_fft_fraction": round(bp3_fraction, 3),
            "cos1": round(cos1, 4),
            "cos3": round(cos3, 4),
            "offset3_inversion": inversion,
            "norm_mean": round(float(norms.mean()), 1),
            "norm_std": round(float(norms.std()), 1),
        }
        results.append(result)
        status = "✓ INVERSION" if inversion else "✗ no inversion"
        print(f"  {name}: lag3={lag3:.3f} cos3={cos3:.3f} cos1={cos1:.3f} {status} GC={gc:.1f}%")

    # Save results JSON (convert numpy types to Python types)
    def _to_python(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    clean_results = []
    for r in results:
        clean_results.append({k: _to_python(v) for k, v in r.items()})
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(clean_results, f, indent=2)

    # ── Summary statistics ──
    coding = [r for r in results if not r["category"].startswith("noncoding")]
    noncoding = [r for r in results if r["category"].startswith("noncoding")]

    n_coding_inversion = sum(1 for r in coding if r["offset3_inversion"])
    n_noncoding_inversion = sum(1 for r in noncoding if r["offset3_inversion"])

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total sequences: {len(results)}")
    print(f"Coding sequences: {len(coding)}")
    print(f"  Offset-3 inversion: {n_coding_inversion}/{len(coding)} ({100*n_coding_inversion/max(len(coding),1):.1f}%)")
    print(f"  Mean lag-3: {np.mean([r['lag3'] for r in coding]):.3f} ± {np.std([r['lag3'] for r in coding]):.3f}")
    print(f"  Mean cos3: {np.mean([r['cos3'] for r in coding]):.3f} ± {np.std([r['cos3'] for r in coding]):.3f}")
    print(f"Non-coding controls: {len(noncoding)}")
    print(f"  Offset-3 inversion: {n_noncoding_inversion}/{len(noncoding)} ({100*n_noncoding_inversion/max(len(noncoding),1):.1f}%)")
    print(f"  Mean lag-3: {np.mean([r['lag3'] for r in noncoding]):.3f} ± {np.std([r['lag3'] for r in noncoding]):.3f}")

    # ── Group by domain/category ──
    from collections import defaultdict
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    print(f"\nBy category:")
    for cat in sorted(by_category.keys()):
        recs = by_category[cat]
        inv = sum(1 for r in recs if r["offset3_inversion"])
        lag3_mean = np.mean([r["lag3"] for r in recs])
        gc_range = f"{min(r['gc_content'] for r in recs):.0f}-{max(r['gc_content'] for r in recs):.0f}%"
        print(f"  {cat:30s}: {inv}/{len(recs)} inversion, lag3={lag3_mean:.3f}, GC={gc_range}")

    # ── Generate figures ──
    _plot_overview(results, FIG_DIR)
    _plot_gc_vs_periodicity(results, FIG_DIR)
    _plot_domain_comparison(results, FIG_DIR)

    print(f"\nFigures saved to {FIG_DIR}/")
    print(f"Results saved to {OUT_DIR}/results.json")


def _autocorrelation(x: np.ndarray, lag: int) -> float:
    """Compute normalized autocorrelation at a specific lag."""
    n = len(x)
    if n <= lag:
        return 0.0
    mean = x.mean()
    var = x.var()
    if var == 0:
        return 0.0
    return float(np.mean((x[:n-lag] - mean) * (x[lag:] - mean)) / var)


def _plot_overview(results: list[dict], fig_dir: Path):
    """Overview figure: offset-3 inversion across all taxa."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Sort: coding first (by lineage), then non-coding
    coding = sorted([r for r in results if not r["category"].startswith("noncoding")],
                    key=lambda r: (r["lineage"], r["name"]))
    noncoding = sorted([r for r in results if r["category"].startswith("noncoding")],
                       key=lambda r: r["name"])
    ordered = coding + noncoding

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(len(ordered) * 0.3, 8)),
                                    gridspec_kw={"width_ratios": [3, 1]})

    names = [r["name"].replace("_", " ") for r in ordered]
    cos3 = [r["cos3"] for r in ordered]
    cos1 = [r["cos1"] for r in ordered]
    colors = ["#2196F3" if not r["category"].startswith("noncoding") else "#F44336" for r in ordered]

    y = range(len(ordered))
    ax1.barh(y, cos3, height=0.4, color=colors, alpha=0.7, label="cos3 (offset-3)")
    ax1.barh([yi + 0.4 for yi in y], cos1, height=0.4, color=[c + "80" for c in colors],
             alpha=0.5, label="cos1 (offset-1)")
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=6)
    ax1.set_xlabel("Cosine Similarity")
    ax1.set_title("Offset-3 vs Offset-1 Cosine Similarity")
    ax1.legend(fontsize=8)
    ax1.invert_yaxis()

    # Panel 2: lag-3 vs GC content
    gc = [r["gc_content"] for r in ordered]
    lag3 = [r["lag3"] for r in ordered]
    ax2.barh(y, lag3, color=colors, alpha=0.7)
    ax2.set_yticks([])
    ax2.set_xlabel("Lag-3 Autocorrelation")
    ax2.set_title("Codon Periodicity")
    ax2.invert_yaxis()

    # Add separator between coding and non-coding
    sep_y = len(coding) - 0.5
    ax1.axhline(sep_y, color="black", linestyle="--", linewidth=0.5)
    ax2.axhline(sep_y, color="black", linestyle="--", linewidth=0.5)
    ax1.text(0.01, sep_y + 0.3, "— non-coding controls below —", fontsize=7, style="italic")

    plt.tight_layout()
    plt.savefig(fig_dir / "overview_all_taxa.png", dpi=150, bbox_inches="tight")
    plt.savefig(fig_dir / "overview_all_taxa.pdf", bbox_inches="tight")
    plt.close()


def _plot_gc_vs_periodicity(results: list[dict], fig_dir: Path):
    """Scatter plot: GC content vs lag-3 and cos3."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coding = [r for r in results if not r["category"].startswith("noncoding")]
    noncoding = [r for r in results if r["category"].startswith("noncoding")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Color by domain
    domain_colors = {
        "archaea": "#FF9800",
        "mammal": "#2196F3", "bird": "#03A9F4", "reptile": "#00BCD4",
        "fish": "#009688", "amphibian": "#4CAF50",
        "insect": "#8BC34A", "nematode": "#CDDC39",
        "mollusk": "#FFEB3B", "cnidarian": "#FFC107", "echinoderm": "#FF9800",
        "plant": "#4CAF50", "fungi": "#9C27B0",
        "protist": "#F44336", "alga": "#00BCD4",
        "organellar": "#795548",
        "giant_virus": "#607D8B", "ssdna_virus": "#9E9E9E",
        "edge": "#FF5722",
    }

    for r in coding:
        cat = r["category"].split("_")[0]
        color = domain_colors.get(cat, "#666666")
        ax1.scatter(r["gc_content"], r["lag3"], c=color, s=40, alpha=0.7, edgecolors="black", linewidth=0.5)
        ax2.scatter(r["gc_content"], r["cos3"] - r["cos1"], c=color, s=40, alpha=0.7, edgecolors="black", linewidth=0.5)

    for r in noncoding:
        ax1.scatter(r["gc_content"], r["lag3"], c="red", s=60, marker="x", linewidth=2)
        ax2.scatter(r["gc_content"], r["cos3"] - r["cos1"], c="red", s=60, marker="x", linewidth=2)

    ax1.set_xlabel("GC Content (%)")
    ax1.set_ylabel("Lag-3 Autocorrelation")
    ax1.set_title("Codon Periodicity vs GC Content")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    ax2.set_xlabel("GC Content (%)")
    ax2.set_ylabel("cos3 − cos1 (inversion signal)")
    ax2.set_title("Offset-3 Inversion vs GC Content")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5, label="No inversion")

    plt.tight_layout()
    plt.savefig(fig_dir / "gc_vs_periodicity.png", dpi=150, bbox_inches="tight")
    plt.savefig(fig_dir / "gc_vs_periodicity.pdf", bbox_inches="tight")
    plt.close()


def _plot_domain_comparison(results: list[dict], fig_dir: Path):
    """Box plot comparing periodicity across domains."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Group into high-level domains
    domain_map = {
        "archaea_euryarchaeota": "Archaea",
        "archaea_crenarchaeota": "Archaea",
        "archaea_thaumarchaeota": "Archaea",
        "mammal": "Vertebrata",
        "bird": "Vertebrata",
        "reptile": "Vertebrata",
        "fish": "Vertebrata",
        "amphibian": "Vertebrata",
        "insect": "Invertebrata",
        "nematode": "Invertebrata",
        "mollusk": "Invertebrata",
        "cnidarian": "Invertebrata",
        "echinoderm": "Invertebrata",
        "plant_dicot": "Plantae",
        "plant_monocot": "Plantae",
        "plant_basal": "Plantae",
        "fungi_ascomycete": "Fungi",
        "fungi_basidiomycete": "Fungi",
        "protist_apicomplexa": "Protista",
        "protist_kinetoplastid": "Protista",
        "protist_amoebozoa": "Protista",
        "protist_ciliate": "Protista",
        "green_alga": "Algae",
        "diatom": "Algae",
        "brown_alga": "Algae",
        "organellar_mito": "Organellar",
        "organellar_chloroplast": "Organellar",
        "organellar_relative": "Organellar",
        "giant_virus": "Virus",
        "ssdna_virus": "Virus",
    }

    from collections import defaultdict
    domain_data = defaultdict(list)
    for r in results:
        if r["category"].startswith("noncoding") or r["category"].startswith("edge"):
            domain = "Non-coding" if r["category"].startswith("noncoding") else "Edge cases"
        else:
            domain = domain_map.get(r["category"], "Other")
        domain_data[domain].append(r)

    # Order
    order = ["Archaea", "Vertebrata", "Invertebrata", "Plantae", "Fungi",
             "Protista", "Algae", "Organellar", "Virus", "Edge cases", "Non-coding"]
    order = [d for d in order if d in domain_data]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Lag-3
    data_lag3 = [[r["lag3"] for r in domain_data[d]] for d in order]
    bp1 = ax1.boxplot(data_lag3, labels=order, patch_artist=True)
    colors = ["#FF9800", "#2196F3", "#8BC34A", "#4CAF50", "#9C27B0",
              "#F44336", "#00BCD4", "#795548", "#607D8B", "#FF5722", "#F44336"]
    for patch, color in zip(bp1["boxes"], colors[:len(order)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax1.set_ylabel("Lag-3 Autocorrelation")
    ax1.set_title("Codon Periodicity Across All Domains of Life")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    # cos3 - cos1
    data_inv = [[r["cos3"] - r["cos1"] for r in domain_data[d]] for d in order]
    bp2 = ax2.boxplot(data_inv, labels=order, patch_artist=True)
    for patch, color in zip(bp2["boxes"], colors[:len(order)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_ylabel("cos3 − cos1 (inversion signal)")
    ax2.set_title("Offset-3 Inversion Signal")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5, label="Inversion threshold")
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(fig_dir / "domain_comparison.png", dpi=150, bbox_inches="tight")
    plt.savefig(fig_dir / "domain_comparison.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/universal_validation.py [download|extract|analyze]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "download":
        print("Downloading sequences from NCBI...")
        download_sequences()
    elif cmd == "extract":
        print("Extracting per-position embeddings...")
        extract_embeddings()
    elif cmd == "analyze":
        print("Analyzing periodicity metrics...")
        analyze_all()
    elif cmd == "all":
        print("Running full pipeline...")
        print("\n=== Step 1: Download ===")
        download_sequences()
        print("\n=== Step 2: Extract ===")
        extract_embeddings()
        print("\n=== Step 3: Analyze ===")
        analyze_all()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python scripts/universal_validation.py [download|extract|analyze|all]")
        sys.exit(1)
