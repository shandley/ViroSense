#!/usr/bin/env python3
"""
Build Component B (functional clustering) and Component C (negative controls)
for the comprehensive validation panel.

Queries NCBI for 30 diverse orthologs per gene family, sampling across
maximum taxonomic diversity.

Usage:
    uv run python scripts/build_functional_panel.py

Outputs:
    results/comprehensive/panel_b.json  — 300 sequences (10 families × 30)
    results/comprehensive/panel_c.json  — 30 negative control sequences
"""

import json
import time
from collections import defaultdict
from io import StringIO
from pathlib import Path

from Bio import Entrez, SeqIO

Entrez.email = "shandley@wustl.edu"

OUT_DIR = Path("results/comprehensive")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Gene families for Component B ──
# Each: (family_name, NCBI protein search query, target CDS length range)
GENE_FAMILIES = [
    ("actin", '(actin[Protein Name] NOT actin-like NOT actin-related) AND RefSeq[Filter] AND 350:400[Sequence Length]', 1000, 1200),
    ("GAPDH", 'glyceraldehyde-3-phosphate dehydrogenase[Protein Name] AND RefSeq[Filter] AND 300:400[Sequence Length]', 900, 1200),
    ("EF_Tu", '(elongation factor Tu[Protein Name] OR EF-Tu[Protein Name]) AND RefSeq[Filter] AND 380:420[Sequence Length]', 1100, 1300),
    ("EF_1alpha", 'elongation factor 1-alpha[Protein Name] AND RefSeq[Filter] AND 440:470[Sequence Length]', 1300, 1500),
    ("HSP70_DnaK", '(molecular chaperone DnaK[Protein Name]) AND RefSeq[Filter] AND 600:650[Sequence Length]', 1800, 2000),
    ("COX1", '(cytochrome c oxidase subunit I[Protein Name] OR COX1[Protein Name]) AND RefSeq[Filter] AND 500:520[Sequence Length]', 1500, 1600),
    ("alpha_tubulin", 'tubulin alpha[Protein Name] AND RefSeq[Filter] AND 440:460[Sequence Length]', 1300, 1400),
    ("rpoB", '(DNA-directed RNA polymerase subunit beta[Protein Name]) AND RefSeq[Filter] AND 1300:1400[Sequence Length]', 1800, 2000),
    ("atpA", '(ATP synthase subunit alpha[Protein Name] OR F1 sector alpha[Protein Name]) AND RefSeq[Filter] AND 500:530[Sequence Length]', 1500, 1600),
    ("ribosomal_S12", '(30S ribosomal protein S12[Protein Name] OR rpsL[Protein Name]) AND RefSeq[Filter] AND 120:140[Sequence Length]', 360, 450),
]

# Taxonomic groups to sample from (aim for 30 per family spanning these)
TAX_GROUPS = [
    ("Archaea", "Archaea[Organism]"),
    ("Gammaproteobacteria", "Gammaproteobacteria[Organism]"),
    ("Alphaproteobacteria", "Alphaproteobacteria[Organism]"),
    ("Firmicutes", "Firmicutes[Organism]"),
    ("Actinobacteria", "Actinobacteria[Organism]"),
    ("Bacteroidetes", "Bacteroidota[Organism]"),
    ("Cyanobacteria", "Cyanobacteriota[Organism]"),
    ("Mammalia", "Mammalia[Organism]"),
    ("Aves", "Aves[Organism]"),
    ("Actinopterygii", "Actinopterygii[Organism]"),
    ("Insecta", "Insecta[Organism]"),
    ("Nematoda", "Nematoda[Organism]"),
    ("Viridiplantae", "Viridiplantae[Organism]"),
    ("Fungi", "Fungi[Organism]"),
    ("Apicomplexa", "Apicomplexa[Organism]"),
    ("Other_Eukarya", "Eukaryota[Organism] NOT Mammalia NOT Aves NOT Actinopterygii NOT Insecta NOT Nematoda NOT Viridiplantae NOT Fungi NOT Apicomplexa"),
]


def search_orthologs(family_name: str, base_query: str, n_target: int = 30) -> list[str]:
    """Search for diverse orthologs of a gene family, returning protein accessions."""
    print(f"\n  Searching {family_name}...")

    accessions = []
    seen_species = set()

    for tax_name, tax_filter in TAX_GROUPS:
        # How many do we still need?
        remaining = n_target - len(accessions)
        if remaining <= 0:
            break

        # Allocate ~2 per taxonomic group, more for diverse groups
        n_per_group = max(1, min(3, remaining))

        query = f"({base_query}) AND ({tax_filter})"
        try:
            handle = Entrez.esearch(db="protein", term=query, retmax=20, sort="relevance")
            result = Entrez.read(handle)
            handle.close()
            time.sleep(0.35)

            if not result["IdList"]:
                continue

            # Fetch summaries to get species
            handle = Entrez.esummary(db="protein", id=",".join(result["IdList"][:20]))
            summaries = Entrez.read(handle)
            handle.close()
            time.sleep(0.35)

            added = 0
            for s in summaries:
                if added >= n_per_group:
                    break
                # Get organism
                org = s.get("Organism", "unknown")
                species = " ".join(org.split()[:2])

                if species in seen_species:
                    continue

                acc = s.get("AccessionVersion", s.get("Caption", ""))
                if acc:
                    accessions.append({
                        "accession": acc,
                        "species": org,
                        "tax_group": tax_name,
                    })
                    seen_species.add(species)
                    added += 1

            if added > 0:
                print(f"    {tax_name}: +{added} (total: {len(accessions)})")

        except Exception as e:
            print(f"    {tax_name}: error ({e})")
            time.sleep(1)

    print(f"  {family_name}: found {len(accessions)} orthologs from {len(seen_species)} species")
    return accessions


def build_component_b():
    """Build the functional clustering panel: 10 families × 30 orthologs."""
    panel_b = []

    for family_name, query, min_cds, max_cds in GENE_FAMILIES:
        orthologs = search_orthologs(family_name, query, n_target=30)

        for i, orth in enumerate(orthologs):
            safe_species = orth["species"].replace(" ", "_").replace("/", "_")[:30]
            name = f"b_{family_name}_{safe_species}_{i}"
            name = "".join(c for c in name if c.isalnum() or c in "_-")

            panel_b.append({
                "name": name,
                "component": "B",
                "category": f"family_{family_name}",
                "gene_family": family_name,
                "species": orth["species"],
                "gene": family_name,
                "accession": orth["accession"],
                "type": "protein_cds",
                "tax_group": orth["tax_group"],
                "domain": _tax_to_domain(orth["tax_group"]),
            })

    return panel_b


def build_component_c():
    """Build negative controls: 30 random CDS from diverse species, different genes."""
    # Use genes NOT in the 10 families
    control_genes = [
        ("alanine racemase", "alanine racemase[Protein Name] AND RefSeq[Filter] AND Bacteria[Organism]", 3),
        ("catalase", "catalase[Protein Name] AND RefSeq[Filter] AND Bacteria[Organism]", 3),
        ("DNA gyrase A", "DNA gyrase subunit A[Protein Name] AND RefSeq[Filter] AND Bacteria[Organism]", 3),
        ("enolase", "enolase[Protein Name] AND RefSeq[Filter] AND Eukaryota[Organism]", 3),
        ("ferredoxin", "ferredoxin[Protein Name] AND RefSeq[Filter] AND Archaea[Organism]", 3),
        ("glutamine synthetase", "glutamine synthetase[Protein Name] AND RefSeq[Filter]", 3),
        ("isocitrate dehydrogenase", "isocitrate dehydrogenase[Protein Name] AND RefSeq[Filter]", 3),
        ("malate dehydrogenase", "malate dehydrogenase[Protein Name] AND RefSeq[Filter]", 3),
        ("phosphoglycerate kinase", "phosphoglycerate kinase[Protein Name] AND RefSeq[Filter]", 3),
        ("superoxide dismutase", "superoxide dismutase[Protein Name] AND RefSeq[Filter]", 3),
    ]

    panel_c = []
    seen_species = set()

    for gene_name, query, n_target in control_genes:
        try:
            handle = Entrez.esearch(db="protein", term=query, retmax=10)
            result = Entrez.read(handle)
            handle.close()
            time.sleep(0.35)

            if not result["IdList"]:
                continue

            handle = Entrez.esummary(db="protein", id=",".join(result["IdList"][:10]))
            summaries = Entrez.read(handle)
            handle.close()
            time.sleep(0.35)

            added = 0
            for s in summaries:
                if added >= n_target:
                    break
                org = s.get("Organism", "unknown")
                species = " ".join(org.split()[:2])
                if species in seen_species:
                    continue

                acc = s.get("AccessionVersion", s.get("Caption", ""))
                if acc:
                    safe_species = species.replace(" ", "_")[:20]
                    name = f"c_{gene_name.replace(' ', '_')}_{safe_species}"
                    name = "".join(c for c in name if c.isalnum() or c in "_-")

                    panel_c.append({
                        "name": name,
                        "component": "C",
                        "category": "negative_control",
                        "gene_family": "none",
                        "species": org,
                        "gene": gene_name,
                        "accession": acc,
                        "type": "protein_cds",
                        "domain": "control",
                    })
                    seen_species.add(species)
                    added += 1

        except Exception as e:
            print(f"  {gene_name}: error ({e})")

    return panel_c


def _tax_to_domain(tax_group: str) -> str:
    """Map tax group to high-level domain."""
    if tax_group == "Archaea":
        return "Archaea"
    if tax_group in ("Mammalia", "Aves", "Actinopterygii"):
        return "Vertebrata"
    if tax_group in ("Insecta", "Nematoda"):
        return "Invertebrata"
    if tax_group == "Viridiplantae":
        return "Plantae"
    if tax_group == "Fungi":
        return "Fungi"
    if tax_group == "Apicomplexa":
        return "Protista"
    if tax_group in ("Gammaproteobacteria", "Alphaproteobacteria", "Firmicutes",
                      "Actinobacteria", "Bacteroidetes", "Cyanobacteria"):
        return "Bacteria"
    return "Other"


def main():
    print("Building Component B (functional clustering panel)...")
    panel_b = build_component_b()

    with open(OUT_DIR / "panel_b.json", "w") as f:
        json.dump(panel_b, f, indent=2)
    print(f"\nComponent B: {len(panel_b)} sequences saved to {OUT_DIR / 'panel_b.json'}")

    # Summary by family
    by_family = defaultdict(int)
    for entry in panel_b:
        by_family[entry["gene_family"]] += 1
    for fam, n in sorted(by_family.items()):
        print(f"  {fam}: {n}")

    print("\nBuilding Component C (negative controls)...")
    panel_c = build_component_c()

    with open(OUT_DIR / "panel_c.json", "w") as f:
        json.dump(panel_c, f, indent=2)
    print(f"\nComponent C: {len(panel_c)} sequences saved to {OUT_DIR / 'panel_c.json'}")


if __name__ == "__main__":
    main()
