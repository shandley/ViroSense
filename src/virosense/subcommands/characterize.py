"""Comprehensive sequence characterization from Evo2 embeddings.

Produces a multi-dimensional biological profile ("DNA passport") for each
input sequence, combining mean-pooled and per-position embedding features
into identity, origin, structure, and novelty assessments.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def run_characterize(
    input_file: str,
    output_dir: str,
    backend: str = "nim",
    model: str = "evo2_7b",
    layer: str = "blocks.28.mlp.l3",
    cache_dir: str | None = None,
    reference_panel: str | None = None,
    nim_url: str | None = None,
    max_concurrent: int = 3,
    per_position: bool = False,
) -> None:
    """Characterize sequences with a comprehensive biological profile.

    For each sequence, computes:
    - Identity: similarity to known categories, nearest match, anomaly score
    - Origin: viral/cellular, RNA/DNA, mobile/chromosomal signatures
    - Structure: coding density, codon periodicity (requires per-position)
    - Novelty: anomaly percentile against reference panel

    Args:
        input_file: Input FASTA.
        output_dir: Output directory.
        backend: Evo2 backend.
        model: Evo2 model name.
        layer: Embedding layer.
        cache_dir: Embedding cache directory.
        reference_panel: Path to reference embeddings NPZ with known categories.
                        If None, uses a built-in minimal panel.
        nim_url: Self-hosted NIM URL.
        max_concurrent: Max concurrent NIM requests.
        per_position: Also analyze per-position embeddings (requires embed --per-position).
    """
    from virosense.backends.base import get_backend
    from virosense.features.evo2_embeddings import extract_embeddings
    from virosense.io.fasta import read_fasta

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read sequences
    sequences = read_fasta(input_file)
    if not sequences:
        logger.warning("No sequences found.")
        return

    logger.info(f"Characterizing {len(sequences)} sequences")

    # Extract mean-pooled embeddings
    evo2_backend = get_backend(
        backend, model=model, nim_url=nim_url, max_concurrent=max_concurrent
    )
    resolved_model = evo2_backend.model

    if not evo2_backend.is_available():
        raise RuntimeError(
            f"Backend {backend!r} is not available. "
            "Check your configuration."
        )

    cache_path = Path(cache_dir) if cache_dir else None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
        result = extract_embeddings(
            sequences=sequences,
            backend=evo2_backend,
            layer=layer,
            model=resolved_model,
            cache_dir=cache_path,
        )

    # Load or build reference panel
    ref_centroids, ref_embeddings, ref_labels = _load_reference_panel(
        reference_panel, cache_path, layer, resolved_model
    )

    # Build anomaly detector from reference
    from sklearn.neighbors import NearestNeighbors

    nn = None
    ref_anomaly_scores = None
    if ref_embeddings is not None and len(ref_embeddings) > 20:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
            nn = NearestNeighbors(n_neighbors=min(10, len(ref_embeddings) - 1), metric="cosine")
            nn.fit(ref_embeddings)
            ref_dists, _ = nn.kneighbors(ref_embeddings)
            ref_anomaly_scores = ref_dists[:, -1]

    # Per-position data (optional)
    pp_dir = None
    if per_position and cache_path:
        for candidate in [cache_path / "per_position", cache_path]:
            if candidate.exists() and list(candidate.glob("*.npy")):
                pp_dir = candidate
                break

    # Characterize each sequence
    reports = []
    for i, seq_id in enumerate(result.sequence_ids):
        emb = result.embeddings[i].astype(np.float64)
        report = _characterize_one(
            seq_id=seq_id,
            embedding=emb,
            centroids=ref_centroids,
            nn=nn,
            ref_anomaly_scores=ref_anomaly_scores,
            pp_dir=pp_dir,
            sequence=sequences.get(seq_id),
        )
        report["length"] = len(sequences.get(seq_id, ""))
        reports.append(report)

    # Write outputs
    _write_reports(reports, output_path)
    logger.info(f"Characterized {len(reports)} sequences → {output_path}/")


def _characterize_one(
    seq_id: str,
    embedding: np.ndarray,
    centroids: dict[str, np.ndarray],
    nn=None,
    ref_anomaly_scores=None,
    pp_dir=None,
    sequence: str | None = None,
) -> dict:
    """Compute full biological profile for one sequence."""
    norm = np.linalg.norm(embedding)
    report = {"sequence_id": seq_id}

    # === IDENTITY ===
    sims = {}
    for cat, centroid in centroids.items():
        c_norm = np.linalg.norm(centroid)
        if norm > 0 and c_norm > 0:
            sims[cat] = float(np.dot(embedding, centroid) / (norm * c_norm))
        else:
            sims[cat] = 0.0
    report["category_similarities"] = sims

    nearest = max(sims, key=sims.get) if sims else "unknown"
    report["nearest_category"] = nearest
    report["nearest_similarity"] = sims.get(nearest, 0)

    # Anomaly score
    if nn is not None:
        dists, _ = nn.kneighbors(embedding.reshape(1, -1))
        report["anomaly_score"] = float(dists[0, -1])
        if ref_anomaly_scores is not None:
            report["anomaly_percentile"] = float(
                (ref_anomaly_scores < report["anomaly_score"]).mean()
            )
    else:
        report["anomaly_score"] = None
        report["anomaly_percentile"] = None

    # === ORIGIN ===
    viral_sim = max(sims.get("phage", 0), sims.get("rna_virus", 0))
    nonviral_sim = max(sims.get("chromosome", 0), sims.get("cellular", 0), 0.001)
    report["viral_signature"] = round(viral_sim / nonviral_sim, 3)

    phage_sim = max(sims.get("phage", 0), 0.001)
    rna_sim = sims.get("rna_virus", 0)
    report["rna_origin_score"] = round(rna_sim / phage_sim, 3)

    chr_sim = max(sims.get("chromosome", 0), 0.001)
    report["mobile_element_score"] = round(
        (sims.get("phage", 0) + sims.get("plasmid", 0)) / (2 * chr_sim), 3
    )

    report["embedding_norm"] = float(norm)

    # === STRUCTURE (per-position, optional) ===
    if pp_dir is not None:
        pp_report = _analyze_per_position(seq_id, pp_dir, sequence)
        report.update(pp_report)

    # === INTERPRETATION ===
    report["interpretation"] = _interpret(report)

    return report


def _analyze_per_position(
    seq_id: str,
    pp_dir: Path,
    sequence: str | None,
) -> dict:
    """Analyze per-position embeddings for structural features."""
    from scipy.ndimage import uniform_filter1d

    safe_name = seq_id.replace("/", "_")[:80]
    npy_path = None
    for candidate in pp_dir.glob("*.npy"):
        if candidate.stem[:40] == safe_name[:40]:
            npy_path = candidate
            break

    if npy_path is None:
        return {}

    per_pos = np.load(npy_path)
    norms = np.linalg.norm(per_pos, axis=1)
    seq_len = per_pos.shape[0]

    if seq_len < 50:
        return {}

    pp_report = {}

    # Coding density estimate (norm-based)
    smooth = uniform_filter1d(norms, size=30)
    median_norm = np.median(smooth)
    high_norm = smooth > median_norm
    pp_report["estimated_coding_fraction"] = round(float(high_norm.mean()), 3)

    # Norm statistics
    pp_report["norm_mean"] = round(float(norms.mean()), 1)
    pp_report["norm_std"] = round(float(norms.std()), 1)
    pp_report["norm_cv"] = round(float(norms.std() / max(norms.mean(), 0.001)), 3)

    # Codon periodicity
    nc = norms - norms.mean()
    autocorr = np.correlate(nc, nc, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr /= autocorr[0] if autocorr[0] > 0 else 1

    if len(autocorr) > 3:
        pp_report["lag1_autocorr"] = round(float(autocorr[1]), 4)
        pp_report["lag3_autocorr"] = round(float(autocorr[3]), 4)
        pp_report["codon_periodicity_ratio"] = round(
            float(autocorr[3] / max(abs(autocorr[1]), 0.001)), 2
        )

    # Offset cosine features
    norm_mat = np.linalg.norm(per_pos, axis=1, keepdims=True)
    norm_mat[norm_mat == 0] = 1
    normalized = per_pos / norm_mat

    cos1 = float(np.sum(normalized[:-1] * normalized[1:], axis=1).mean())
    cos3 = float(np.sum(normalized[:-3] * normalized[3:], axis=1).mean()) if seq_len > 3 else 0
    pp_report["cos1"] = round(cos1, 4)
    pp_report["cos3"] = round(cos3, 4)
    pp_report["offset3_inversion"] = cos3 > cos1

    # Compositional uniformity (how many distinct regions?)
    window = 100
    if seq_len > 2 * window:
        windowed = []
        for j in range(0, seq_len - window, window // 2):
            windowed.append(per_pos[j:j + window].mean(axis=0))
        windowed = np.array(windowed)
        w_norms = np.linalg.norm(windowed, axis=1, keepdims=True)
        w_norms[w_norms == 0] = 1
        w_normalized = windowed / w_norms
        adj_dists = 1 - np.sum(w_normalized[:-1] * w_normalized[1:], axis=1)
        pp_report["compositional_uniformity"] = round(float(1 - adj_dists.std()), 3)
        pp_report["max_compositional_shift"] = round(float(adj_dists.max()), 4)

    return pp_report


def _interpret(report: dict) -> dict:
    """Generate human-readable interpretation from scores."""
    interp = {}

    vs = report.get("viral_signature", 0)
    if vs > 1.3:
        interp["viral"] = "likely viral"
    elif vs > 1.0:
        interp["viral"] = "possibly viral"
    elif vs > 0.7:
        interp["viral"] = "ambiguous"
    else:
        interp["viral"] = "likely non-viral"

    ro = report.get("rna_origin_score", 0)
    if ro > 1.1:
        interp["origin"] = "RNA-origin"
    elif ro > 0.9:
        interp["origin"] = "ambiguous origin"
    else:
        interp["origin"] = "DNA-origin"

    me = report.get("mobile_element_score", 0)
    if me > 1.2:
        interp["mobility"] = "mobile element"
    elif me > 0.8:
        interp["mobility"] = "possibly mobile"
    else:
        interp["mobility"] = "chromosomal"

    ap = report.get("anomaly_percentile")
    if ap is not None:
        if ap > 0.99:
            interp["novelty"] = "highly novel (top 1%)"
        elif ap > 0.95:
            interp["novelty"] = "unusual (top 5%)"
        elif ap > 0.80:
            interp["novelty"] = "somewhat unusual"
        else:
            interp["novelty"] = "typical"

    cp = report.get("lag3_autocorr")
    if cp is not None:
        if cp > 0.8:
            interp["coding"] = "strong codon structure"
        elif cp > 0.6:
            interp["coding"] = "moderate codon structure"
        elif cp > 0.3:
            interp["coding"] = "weak codon structure"
        else:
            interp["coding"] = "minimal coding signal"

    return interp


def _load_reference_panel(
    reference_panel: str | None,
    cache_dir: Path | None,
    layer: str,
    model: str,
) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    """Load reference embeddings for computing similarities and anomaly scores."""
    centroids = {}
    ref_embeddings = None
    ref_labels = None

    # Try loading from explicit reference panel
    if reference_panel:
        ref_path = Path(reference_panel)
        if ref_path.exists():
            data = np.load(ref_path, allow_pickle=True)
            if "embeddings" in data and "labels" in data:
                ref_embeddings = data["embeddings"]
                ref_labels = data["labels"]
                for label in np.unique(ref_labels):
                    mask = ref_labels == label
                    centroids[str(label)] = ref_embeddings[mask].mean(axis=0)
                logger.info(f"Loaded reference panel: {len(ref_embeddings)} sequences, {len(centroids)} categories")
                return centroids, ref_embeddings, ref_labels

    # Try loading from cache directory (our benchmark cache format)
    if cache_dir:
        safe_layer = layer.replace(".", "_")
        safe_model = model.replace("-", "_")
        cache_file = cache_dir / f"{safe_model}_{safe_layer}_embeddings.npz"
        if cache_file.exists():
            data = np.load(cache_file)
            if "embeddings" in data and "sequence_ids" in data:
                ref_embeddings = data["embeddings"]
                logger.info(f"Loaded {len(ref_embeddings)} reference embeddings from cache")
                # Without labels, compute a single centroid as the overall mean
                centroids["reference"] = ref_embeddings.mean(axis=0)
                return centroids, ref_embeddings, None

    # Fallback: no reference panel
    logger.warning("No reference panel found. Anomaly scoring will be disabled.")
    return centroids, None, None


def _write_reports(reports: list[dict], output_path: Path) -> None:
    """Write characterization reports to TSV and JSON."""
    # JSON (full detail)
    with open(output_path / "characterization.json", "w") as f:
        json.dump(reports, f, indent=2, default=str)

    # TSV (flat summary)
    flat = []
    for r in reports:
        row = {
            "sequence_id": r["sequence_id"],
            "length": r.get("length", ""),
            "nearest_category": r.get("nearest_category", ""),
            "nearest_similarity": r.get("nearest_similarity", ""),
            "anomaly_percentile": r.get("anomaly_percentile", ""),
            "viral_signature": r.get("viral_signature", ""),
            "rna_origin_score": r.get("rna_origin_score", ""),
            "mobile_element_score": r.get("mobile_element_score", ""),
            "embedding_norm": r.get("embedding_norm", ""),
        }

        # Per-position fields (if available)
        for key in ["estimated_coding_fraction", "lag3_autocorr", "cos3",
                     "norm_cv", "compositional_uniformity"]:
            row[key] = r.get(key, "")

        # Interpretation
        interp = r.get("interpretation", {})
        row["interp_viral"] = interp.get("viral", "")
        row["interp_origin"] = interp.get("origin", "")
        row["interp_mobility"] = interp.get("mobility", "")
        row["interp_novelty"] = interp.get("novelty", "")
        row["interp_coding"] = interp.get("coding", "")

        flat.append(row)

    df = pd.DataFrame(flat)
    df.to_csv(output_path / "characterization.tsv", sep="\t", index=False)

    # Print summary
    n = len(reports)
    logger.info(f"Characterized {n} sequences")

    if n > 0:
        interps = [r.get("interpretation", {}) for r in reports]
        viral_counts = pd.Series([i.get("viral", "") for i in interps]).value_counts()
        origin_counts = pd.Series([i.get("origin", "") for i in interps]).value_counts()
        novelty_counts = pd.Series([i.get("novelty", "") for i in interps]).value_counts()

        logger.info(f"  Viral: {viral_counts.to_dict()}")
        logger.info(f"  Origin: {origin_counts.to_dict()}")
        if any("novelty" in i for i in interps):
            logger.info(f"  Novelty: {novelty_counts.to_dict()}")
