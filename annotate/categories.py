"""Functional category classification for viral proteins.

Classifies Foldseek hit descriptions into functional categories using:
1. Pfam domain annotations
2. GO biological process and molecular function terms
3. SUPERFAMILY structural classifications
4. Keyword-based fallback matching

11 categories: structural, replication, protease, nuclease, packaging,
regulatory, movement, lysis, host_interaction, entry, unknown.

Ported from vHold (self-contained, no model dependencies).
Designed for integration into ViroSense's annotate module.
"""

from dataclasses import dataclass

# =============================================================================
# KEYWORD-BASED CLASSIFICATION
# =============================================================================

FUNCTIONAL_CATEGORIES: dict[str, list[str]] = {
    "structural": [
        "capsid", "coat", "envelope", "spike", "matrix", "tail", "fiber",
        "portal", "shell", "head", "virion", "glycoprotein",
        "membrane protein", "membrane-associated", "nucleocapsid",
        "nucleoprotein", "tegument", "baseplate", "assembly", "maturation",
        "fusion protein", "fusion glycoprotein", "fusion",
        "attachment", "structural polyprotein", "surface protein",
        "surface antigen", "surface glycoprotein", "core protein",
        "core antigen", "outer capsid", "inner capsid", "hemagglutinin",
        "neuraminidase", "gag polyprotein", "gag-pro-pol", "pre-membrane",
    ],
    "replication": [
        "polymerase", "replicase", "helicase", "primase", "rdrp",
        "reverse transcriptase", "rep protein", "nsp12", "nsp13",
        "phosphoprotein", "polymerase cofactor",
    ],
    "protease": [
        "protease", "proteinase", "peptidase", "maturase", "3cl", "mpro", "nsp5",
    ],
    "nuclease": [
        "nuclease", "endonuclease", "exonuclease", "integrase", "recombinase",
        "ligase", "rnase", "dnase",
    ],
    "packaging": [
        "terminase", "packaging", "scaffold", "portal protein",
    ],
    "host_interaction": [
        "interferon antagonist", "immune evasion", "immune antagonist",
        "interferon inhibitor", "innate immune", "accessory protein",
        "viroporin", "immune modulator", "ion channel protein", "v protein",
        "anti-restriction", "anti-crispr", "anti-defense", "host shutoff",
        "host defense", "host division inhibitor", "host cell division inhibitor",
        "host takeover", "ocr", "dgtpase inhibitor",
    ],
    "regulatory": [
        "transcription", "repressor", "activator", "regulator",
        "anti-repressor", "antirepressor", "translation enhancer",
        "translation effector", "effector protein", "leader protein",
        "nonstructural protein", "non-structural protein",
        "transactivator", "trans-activating", "immediate-early",
        "methyltransferase",
    ],
    "movement": [
        "movement", "cell-to-cell", "transport protein", "movement protein",
        "plasmodesmata", "tubule-forming", " mp",
    ],
    "lysis": [
        "lysin", "holin", "endolysin", "spanin", "lysis",
    ],
    "entry": [
        "viral entry", "cell entry", "receptor binding",
        "receptor-binding", "entry protein",
    ],
}

UNKNOWN_TERMS = [
    "hypothetical", "uncharacterized", "unknown", "duf", "unipr", "deleted",
]

# =============================================================================
# PFAM DOMAIN TO CATEGORY MAPPING
# =============================================================================

PFAM_TO_CATEGORY: dict[str, list[str]] = {
    "structural": [
        "capsid", "coat protein", "matrix protein", "virion", "envelope",
        "glycoprotein", "spike", "nucleocapsid", "nucleoprotein",
        "vp1", "vp2", "vp3", "vp4", "vp5", "vp6", "vp7",
        "major capsid", "minor capsid", "picornavirus capsid",
        "circovirus capsid", "polyomavirus coat", "parvovirus coat",
        "vesiculovirus matrix", "rhabdovirus spike",
        "herpes virus major capsid", "herpesvirus vp23",
        "herpesvirus glycoprotein", "adenovirus hexon", "adenovirus penton",
        "fiber protein", "tail protein", "baseplate", "portal protein",
        "head protein", "tegument", "membrane protein",
        "structural envelope", "assembly protein", "maturation protein",
        "fusion protein", "fusion glycoprotein", "attachment protein",
        "coronavirus m matrix", "surface protein", "surface antigen",
        "surface glycoprotein", "core protein", "hemagglutinin",
        "neuraminidase", "gag polyprotein", "pre-membrane protein",
        "outer capsid protein", "inner capsid protein",
    ],
    "replication": [
        "polymerase", "rdrp", "rna-dependent rna polymerase",
        "dna polymerase", "reverse transcriptase", "helicase",
        "rna helicase", "dna helicase", "primase", "dna primase",
        "replicase", "rep protein", "parvovirus non-structural",
        "ns1", "nsp12", "nsp13", "viral helicase",
        "superfamily 1 rna helicase", "superfamily 2 helicase",
        "phosphoprotein", "polymerase cofactor",
        "thymidine kinase", "thymidylate", "ribonucleotide reductase",
        "dutpase", "uracil dna glycosylase",
    ],
    "protease": [
        "protease", "peptidase", "proteinase", "maturase", "assemblin",
        "3cl protease", "mpro", "nsp5", "serine endopeptidase",
        "cysteine protease", "cysteine proteinase", "cysteine-type peptidase",
        "metalloprotease", "aspartic protease", "otu", "ovarian tumor",
    ],
    "nuclease": [
        "nuclease", "endonuclease", "exonuclease", "rnase", "dnase",
        "integrase", "recombinase", "ligase", "dna ligase", "rna ligase",
        "dutpase", "uracil dna glycosylase",
        "dna/rna non-specific endonuclease", "pd-(d/e)xk", "giy-yig",
        "hnh", "restriction enzyme",
    ],
    "packaging": [
        "terminase", "packaging", "scaffold", "dna packaging",
        "genome packaging", "portal", "ul6",
    ],
    "regulatory": [
        "transcription", "transcriptional regulator", "repressor",
        "activator", "regulator", "anti-repressor", "trans-activator",
        "transactivator", "immediate-early", "protein kinase",
        "kinase domain", "translation enhancer", "translation effector",
        "translation regulation", "methyltransferase", "cap methyltransferase",
    ],
    "host_interaction": [
        "interferon antagonist", "interferon inhibitor", "immune evasion",
        "immune antagonist", "innate immune", "stat inhibitor",
        "nf-kappa-b inhibitor", "accessory protein", "viroporin",
        "immune modulator", "anti-restriction", "anti-crispr",
        "anti-defense", "host defense inhibitor", "host nuclease inhibitor",
    ],
    "movement": [
        "movement protein", "cell-to-cell", "transport protein", "tubule-forming",
    ],
    "lysis": [
        "lysin", "holin", "endolysin", "spanin", "lysis", "lysozyme",
        "muramidase", "amidase", "peptidoglycan",
    ],
}

# =============================================================================
# GO BIOLOGICAL PROCESS TO CATEGORY MAPPING
# =============================================================================

GO_BP_TO_CATEGORY: dict[str, list[str]] = {
    "structural": [
        "viral capsid assembly", "virion assembly",
        "viral particle assembly", "capsid assembly",
    ],
    "replication": [
        "dna replication", "viral genome replication",
        "viral dna genome replication", "viral rna genome replication",
        "rna-templated transcription", "dna-templated transcription",
        "telomere maintenance", "tmp biosynthetic process",
        "thymidine biosynthetic process", "nucleotide biosynthetic process",
        "deoxyribonucleotide biosynthetic process",
    ],
    "protease": [
        "proteolysis", "viral protein processing", "protein processing",
    ],
    "nuclease": [
        "dna repair", "dna recombination", "dna integration",
    ],
    "packaging": [
        "viral dna genome packaging", "dna packaging",
        "genome packaging", "chromosome organization",
    ],
    "regulatory": [
        "regulation of dna-templated transcription",
        "positive regulation of dna-templated transcription",
        "negative regulation of transcription",
        "viral transcription", "protein phosphorylation",
        "regulation of dna replication",
    ],
    "host_interaction": [
        "virus-mediated perturbation of host defense response",
        "modulation by virus of host process", "immune response",
        "immune evasion", "modulation by virus of host immune response",
        "suppression by virus of host immune response",
        "evasion of host immune response",
    ],
    "entry": [
        "fusion of virus membrane with host plasma membrane",
        "virion attachment to host cell", "viral entry", "membrane fusion",
    ],
    "lysis": [
        "lysis", "cell lysis", "host cell lysis",
    ],
}

# =============================================================================
# GO MOLECULAR FUNCTION TO CATEGORY MAPPING
# =============================================================================

GO_MF_TO_CATEGORY: dict[str, list[str]] = {
    "structural": [
        "structural molecule activity", "structural constituent of virion",
    ],
    "replication": [
        "dna-directed 5'-3' rna polymerase activity",
        "rna-dependent rna polymerase activity",
        "dna-directed dna polymerase activity", "dna polymerase activity",
        "dna-directed 5'-3' dna polymerase activity",
        "dna helicase activity", "rna helicase activity",
        "dna primase activity", "dna binding", "nucleotide binding",
        "atp binding", "atp hydrolysis activity",
        "single-stranded dna binding", "kinase activity",
    ],
    "protease": [
        "serine-type endopeptidase activity", "cysteine-type peptidase activity",
        "metalloendopeptidase activity", "aspartic-type endopeptidase activity",
        "peptidase activity",
    ],
    "nuclease": [
        "nuclease activity", "endonuclease activity", "exonuclease activity",
        "dna ligase activity", "rna ligase activity",
        "endodeoxyribonuclease activity",
    ],
    "regulatory": [
        "protein kinase activity", "methyltransferase activity",
        "transcription factor activity",
    ],
}

# =============================================================================
# SUPERFAMILY TO CATEGORY MAPPING
# =============================================================================

SUPERFAMILY_TO_CATEGORY: dict[str, list[str]] = {
    "structural": [
        "viral glycoprotein", "positive stranded ssrna viruses",
        "vsv matrix protein", "ev matrix protein",
        "group i dsdna viruses", "viral glycoprotein ectodomain",
        "ndv fusion glycoprotein", "wssv envelope", "p40 nucleoprotein",
        "nucleoprotein", "immunoglobulin", "flaviviral glycoprotein",
        "herpesvirus glycoprotein b",
    ],
    "replication": [
        "dna/rna polymerases",
        "p-loop containing nucleoside triphosphate hydrolases",
        "ribonuclease h-like",
        "beta and beta-prime subunits of dna dependent rna-polymerase",
        "dna ligase/mrna capping enzyme", "dna clamp",
    ],
    "protease": [
        "cysteine proteinases", "metalloproteases", "zincins",
    ],
    "nuclease": [
        "dnase i-like", "dutpase-like", "pin domain-like",
    ],
    "regulatory": [
        "s-adenosyl-l-methionine-dependent methyltransferases",
        "protein kinase-like", "ankyrin repeat", "ring/u-box",
        "tata-box binding protein-like", "dna-binding domain of mlu1-box",
        "fkbp12-rapamycin-binding domain",
    ],
    "host_interaction": [
        "bcl-2 inhibitors of programmed cell death", "viroporin",
    ],
}

# =============================================================================
# ALL CATEGORIES
# =============================================================================

ALL_CATEGORIES = [
    "structural", "replication", "protease", "nuclease", "packaging",
    "regulatory", "movement", "lysis", "host_interaction", "entry", "unknown",
]

CATEGORY_DESCRIPTIONS = {
    "structural": "Virion structural proteins (capsid, envelope, matrix)",
    "replication": "Genome replication machinery (polymerase, helicase, primase)",
    "protease": "Proteolytic enzymes (protease, peptidase)",
    "nuclease": "Nucleic acid processing (nuclease, integrase, ligase)",
    "packaging": "Genome packaging (terminase, scaffold)",
    "regulatory": "Gene regulation (transcription factors, kinases)",
    "movement": "Cell-to-cell movement (plant viruses)",
    "lysis": "Host cell lysis (bacteriophages)",
    "host_interaction": "Host interaction and immune evasion",
    "entry": "Host cell entry and fusion",
    "unknown": "Unknown or uncharacterized function",
}

# =============================================================================
# ANNOTATION EVIDENCE
# =============================================================================


@dataclass
class AnnotationEvidence:
    """Evidence for functional classification from various sources."""

    pfam: str | None = None
    go_bp: str | None = None
    go_mf: str | None = None
    superfamily: str | None = None

    def has_evidence(self) -> bool:
        return any([self.pfam, self.go_bp, self.go_mf, self.superfamily])


# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================


def _match_terms(text: str, term_list: list[str]) -> bool:
    """Check if any term from the list is found in the text."""
    text_lower = text.lower()
    return any(term.lower() in text_lower for term in term_list)


def classify_by_pfam(pfam_annotation: str) -> str | None:
    """Classify protein by Pfam domain annotation."""
    if not pfam_annotation:
        return None
    for category, terms in PFAM_TO_CATEGORY.items():
        if _match_terms(pfam_annotation, terms):
            return category
    return None


def classify_by_go_bp(go_bp: str) -> str | None:
    """Classify protein by GO biological process."""
    if not go_bp:
        return None
    for category, terms in GO_BP_TO_CATEGORY.items():
        if _match_terms(go_bp, terms):
            return category
    return None


def classify_by_go_mf(go_mf: str) -> str | None:
    """Classify protein by GO molecular function."""
    if not go_mf:
        return None
    for category, terms in GO_MF_TO_CATEGORY.items():
        if _match_terms(go_mf, terms):
            return category
    return None


def classify_by_superfamily(superfamily: str) -> str | None:
    """Classify protein by SUPERFAMILY annotation."""
    if not superfamily:
        return None
    for category, terms in SUPERFAMILY_TO_CATEGORY.items():
        if _match_terms(superfamily, terms):
            return category
    return None


def _is_inhibitor_context(text: str, keyword: str) -> bool:
    """Check if a keyword match occurs in an inhibitor/antagonist context."""
    idx = text.find(keyword)
    if idx < 0:
        return False

    after = text[idx + len(keyword):idx + len(keyword) + 30]
    if any(term in after for term in ["inhibitor", "inhibition"]):
        return True

    before = text[max(0, idx - 15):idx]
    return "host" in before


def classify_by_keywords(description: str, gene: str | None = None) -> str:
    """Classify protein by keyword matching.

    Args:
        description: Protein description/annotation text
        gene: Optional gene name

    Returns:
        Functional category string
    """
    text = (description + " " + (gene or "")).lower()

    for category, keywords in FUNCTIONAL_CATEGORIES.items():
        for keyword in keywords:
            if keyword in text:
                if category != "host_interaction" and _is_inhibitor_context(text, keyword):
                    return "host_interaction"
                return category

    for term in UNKNOWN_TERMS:
        if term in text:
            return "unknown"

    return "unknown"


def classify_protein(
    description: str,
    gene: str | None = None,
    evidence: AnnotationEvidence | None = None,
) -> str:
    """Classify protein into functional category using all available evidence.

    Hierarchical approach:
    1. Pfam domain classification (most reliable)
    2. SUPERFAMILY structural classification
    3. GO biological process
    4. GO molecular function
    5. Keyword-based fallback

    Args:
        description: Protein description/annotation text
        gene: Optional gene name
        evidence: Optional AnnotationEvidence with Pfam/GO/SUPERFAMILY data

    Returns:
        Functional category string
    """
    if evidence and evidence.has_evidence():
        for classify_fn, field in [
            (classify_by_pfam, evidence.pfam),
            (classify_by_superfamily, evidence.superfamily),
            (classify_by_go_bp, evidence.go_bp),
            (classify_by_go_mf, evidence.go_mf),
        ]:
            if field:
                category = classify_fn(field)
                if category:
                    return category

    return classify_by_keywords(description, gene)


def get_classification_source(
    description: str,
    gene: str | None = None,
    evidence: AnnotationEvidence | None = None,
) -> tuple[str, str]:
    """Get classification and its evidence source.

    Returns:
        Tuple of (category, source) where source indicates what matched
    """
    if evidence and evidence.has_evidence():
        for classify_fn, field, source_prefix in [
            (classify_by_pfam, evidence.pfam, "pfam"),
            (classify_by_superfamily, evidence.superfamily, "superfamily"),
            (classify_by_go_bp, evidence.go_bp, "go_bp"),
            (classify_by_go_mf, evidence.go_mf, "go_mf"),
        ]:
            if field:
                category = classify_fn(field)
                if category:
                    return category, f"{source_prefix}:{field}"

    category = classify_by_keywords(description, gene)
    return category, "keywords"
