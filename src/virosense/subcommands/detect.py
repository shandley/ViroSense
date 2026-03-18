"""Detect viral sequences in metagenomic contigs."""

from pathlib import Path

import numpy as np
from loguru import logger


def run_detect(
    input_file: str,
    output_dir: str,
    backend: str = "nim",
    model: str = "evo2_7b",
    threshold: float = 0.5,
    min_length: int = 500,
    batch_size: int = 16,
    layer: str = "blocks.28.mlp.l3",
    cache_dir: str | None = None,
    nim_url: str | None = None,
    max_concurrent: int | None = None,
    classifier_model: str | None = None,
    fast: bool = False,
) -> None:
    """Run viral detection pipeline.

    Standard mode:
    1. Read metagenomic contigs from FASTA
    2. Filter by minimum length
    3. Extract Evo2 embeddings via selected backend
    4. Classify contigs as viral/cellular using trained classifier
    5. Write results TSV, summary JSON, filtered FASTA, and HTML report

    Fast mode (--fast):
    1. Read and filter sequences
    2. Classify ALL sequences with k-mer features (~1500× faster, ~93% accuracy)
    3. Identify borderline cases (k-mer probability 0.3-0.7)
    4. Send ONLY borderline cases to Evo2 for full classification
    5. Merge results and write output
    """
    if fast:
        return _run_detect_fast(
            input_file=input_file,
            output_dir=output_dir,
            backend=backend,
            model=model,
            threshold=threshold,
            min_length=min_length,
            batch_size=batch_size,
            layer=layer,
            cache_dir=cache_dir,
            nim_url=nim_url,
            max_concurrent=max_concurrent,
            classifier_model=classifier_model,
        )
    from collections import Counter

    from virosense.backends.base import get_backend
    from virosense.features.evo2_embeddings import extract_embeddings
    from virosense.io.fasta import filter_by_length, read_fasta, write_fasta
    from virosense.io.report import generate_detect_report
    from virosense.io.results import write_json, write_tsv
    from virosense.models.detector import (
        ViralClassifier,
        classify_contigs,
        get_default_model_path,
    )

    # Validate threshold
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"Threshold must be between 0.0 and 1.0, got {threshold}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Detecting viral sequences in {input_file}")
    logger.info(f"Backend: {backend}, Model: {model}")
    logger.info(f"Threshold: {threshold}, Min length: {min_length}")

    # 1. Read and filter sequences
    sequences = read_fasta(input_file)
    sequences = filter_by_length(sequences, min_length)
    if not sequences:
        logger.warning("No sequences remaining after length filter.")
        return

    # 2. Load classifier
    classifier = _load_classifier(classifier_model)

    # 3. Get backend and extract embeddings
    evo2_backend = get_backend(
        backend, api_key=None, model=model, nim_url=nim_url,
        max_concurrent=max_concurrent,
    )
    model = evo2_backend.model  # Use backend's (possibly auto-corrected) model name
    if not evo2_backend.is_available():
        raise RuntimeError(
            f"Backend {backend!r} is not available. "
            "Check your configuration (e.g., NVIDIA_API_KEY for nim)."
        )

    cache_path = Path(cache_dir) if cache_dir else None
    result = extract_embeddings(
        sequences=sequences,
        backend=evo2_backend,
        layer=layer,
        model=model,
        batch_size=batch_size,
        cache_dir=cache_path,
    )

    # 4. Validate embedding dimensions match classifier
    expected_dim = classifier.metadata.get("input_dim")
    actual_dim = result.embeddings.shape[1]
    if expected_dim and actual_dim != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: classifier expects {expected_dim}-D "
            f"but backend produced {actual_dim}-D. "
            f"This typically means the classifier was trained with a different "
            f"Evo2 model (e.g., 40B produces 8192-D, 7B produces 4096-D). "
            f"Retrain the classifier with: virosense build-reference --model {model}"
        )

    # 5. Classify contigs
    sequence_lengths = [len(sequences[sid]) for sid in result.sequence_ids]
    detection_results = classify_contigs(
        embeddings=result.embeddings,
        sequence_ids=result.sequence_ids,
        sequence_lengths=sequence_lengths,
        classifier=classifier,
        threshold=threshold,
    )

    # 6. Write results TSV
    write_tsv(detection_results, output_path, "detection_results.tsv")

    # 7. Write filtered viral contigs FASTA
    _NON_VIRAL = {"cellular", "chromosome", "plasmid", "ambiguous"}
    viral_ids = {r.contig_id for r in detection_results if r.classification not in _NON_VIRAL}
    if viral_ids:
        viral_seqs = {sid: sequences[sid] for sid in result.sequence_ids if sid in viral_ids}
        write_fasta(viral_seqs, output_path / "viral_contigs.fasta")

    # 8. Build and write summary JSON
    scores = [r.viral_score for r in detection_results]
    counts = Counter(r.classification for r in detection_results)

    n_viral = sum(1 for r in detection_results if r.classification not in _NON_VIRAL)
    n_cellular = sum(1 for r in detection_results if r.classification in {"cellular", "chromosome", "plasmid"})
    n_ambiguous = counts.get("ambiguous", 0)

    summary = {
        "n_sequences": len(detection_results),
        "n_viral": n_viral,
        "n_cellular": n_cellular,
        "n_ambiguous": n_ambiguous,
        "classification_counts": dict(counts),
        "score_distribution": {
            "mean": round(float(np.mean(scores)), 4),
            "median": round(float(np.median(scores)), 4),
            "min": round(float(np.min(scores)), 4),
            "max": round(float(np.max(scores)), 4),
            "above_0.9": sum(1 for s in scores if s >= 0.9),
            "between_0.5_0.9": sum(1 for s in scores if 0.5 <= s < 0.9),
            "below_0.5": sum(1 for s in scores if s < 0.5),
        },
        "parameters": {
            "threshold": threshold,
            "min_length": min_length,
            "backend": backend,
            "model": model,
            "layer": layer,
        },
        "classifier": {
            "model_path": str(classifier_model) if classifier_model else "default",
            "input_dim": classifier.metadata.get("input_dim"),
            "n_classes": classifier.metadata.get("n_classes"),
            "class_names": classifier.metadata.get("class_names"),
            "calibrated": classifier.metadata.get("calibrated"),
        },
    }
    write_json(summary, output_path, "detection_summary.json")

    # 9. Generate interactive HTML report
    generate_detect_report(detection_results, sequences, summary, output_path)

    logger.info(
        f"Detection complete: {n_viral} viral, "
        f"{n_cellular} cellular, {n_ambiguous} ambiguous. "
        f"Results in {output_path}"
    )


def _load_classifier(model_path: str | None) -> "ViralClassifier":
    """Load classifier from specified path or default location."""
    from virosense.models.detector import ViralClassifier, get_default_model_path

    if model_path:
        return ViralClassifier.load(model_path)

    default_path = get_default_model_path()
    if default_path.exists():
        return ViralClassifier.load(default_path)

    raise FileNotFoundError(
        f"No classifier model found at {default_path}. "
        "Train a reference model first with:\n"
        "  virosense build-reference -i sequences.fasta --labels labels.tsv -o model_dir/\n"
        "Then copy the model to the default location:\n"
        f"  cp model_dir/classifier.joblib {default_path}"
    )


def _run_detect_fast(
    input_file: str,
    output_dir: str,
    backend: str = "nim",
    model: str = "evo2_7b",
    threshold: float = 0.5,
    min_length: int = 500,
    batch_size: int = 16,
    layer: str = "blocks.28.mlp.l3",
    cache_dir: str | None = None,
    nim_url: str | None = None,
    max_concurrent: int | None = None,
    classifier_model: str | None = None,
    kmer_low: float = 0.3,
    kmer_high: float = 0.7,
) -> None:
    """Two-tier detection: k-mer screening → Evo2 on borderline cases.

    Tier 1: Classify all sequences with trinucleotide features (~1500× faster).
    Tier 2: Send borderline cases (k-mer probability between kmer_low and kmer_high)
            to Evo2 for full embedding-based classification.
    """
    from collections import Counter
    from dataclasses import dataclass

    from virosense.features.kmer_classifier import compute_kmer_features_batch
    from virosense.io.fasta import filter_by_length, read_fasta, write_fasta
    from virosense.io.results import write_json, write_tsv
    from virosense.models.detector import DetectionResult

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fast mode: k-mer screening → Evo2 on borderline cases")
    logger.info(f"K-mer thresholds: confident non-viral < {kmer_low}, confident viral > {kmer_high}")

    # 1. Read and filter sequences
    sequences = read_fasta(input_file)
    sequences = filter_by_length(sequences, min_length)
    if not sequences:
        logger.warning("No sequences remaining after length filter.")
        return

    logger.info(f"Loaded {len(sequences)} sequences")

    # 2. K-mer classification (Tier 1)
    seq_ids, kmer_features = compute_kmer_features_batch(sequences)
    logger.info(f"Computed k-mer features: {kmer_features.shape}")

    # Train a quick k-mer classifier from the reference model's training data
    # For now, use a simple heuristic based on our validated features
    # In production, this would load a pre-trained k-mer model
    #
    # Heuristic: use the reference Evo2 classifier to get labels for cached
    # sequences, then train k-mer model on those. If no cache, fall back to
    # Evo2 for everything (fast mode degrades gracefully to standard mode).
    classifier = _load_classifier(classifier_model)

    # Check if we have cached Evo2 embeddings to bootstrap the k-mer classifier
    kmer_model = _get_or_train_kmer_model(
        sequences, classifier, cache_dir, layer, model
    )

    if kmer_model is None:
        logger.warning(
            "No cached embeddings to train k-mer model. "
            "Falling back to standard Evo2 detection."
        )
        return run_detect(
            input_file=input_file,
            output_dir=output_dir,
            backend=backend,
            model=model,
            threshold=threshold,
            min_length=min_length,
            batch_size=batch_size,
            layer=layer,
            cache_dir=cache_dir,
            nim_url=nim_url,
            max_concurrent=max_concurrent,
            classifier_model=classifier_model,
            fast=False,
        )

    # K-mer predictions
    kmer_proba = kmer_model.predict_proba(kmer_features)
    kmer_viral_prob = kmer_proba[:, 1] if kmer_proba.shape[1] == 2 else kmer_proba.max(axis=1)

    # 3. Triage: confident, borderline, Evo2-needed
    confident_viral = kmer_viral_prob >= kmer_high
    confident_nonviral = kmer_viral_prob <= kmer_low
    borderline = ~confident_viral & ~confident_nonviral

    n_confident = confident_viral.sum() + confident_nonviral.sum()
    n_borderline = borderline.sum()

    logger.info(
        f"K-mer triage: {confident_viral.sum()} confident viral, "
        f"{confident_nonviral.sum()} confident non-viral, "
        f"{n_borderline} borderline → Evo2"
    )

    # 4. Evo2 on borderline cases only (Tier 2)
    results = []
    for i, sid in enumerate(seq_ids):
        if confident_viral[i]:
            results.append(DetectionResult(
                contig_id=sid,
                viral_score=round(float(kmer_viral_prob[i]), 4),
                classification="viral",
                contig_length=len(sequences[sid]),
            ))
        elif confident_nonviral[i]:
            results.append(DetectionResult(
                contig_id=sid,
                viral_score=round(float(kmer_viral_prob[i]), 4),
                classification="cellular",
                contig_length=len(sequences[sid]),
            ))
        # Borderline cases handled below

    if n_borderline > 0:
        logger.info(f"Running Evo2 on {n_borderline} borderline sequences...")

        from virosense.backends.base import get_backend
        from virosense.features.evo2_embeddings import extract_embeddings
        from virosense.models.detector import classify_contigs

        borderline_seqs = {
            seq_ids[i]: sequences[seq_ids[i]]
            for i in range(len(seq_ids)) if borderline[i]
        }

        evo2_backend = get_backend(
            backend, model=model, nim_url=nim_url,
            max_concurrent=max_concurrent,
        )
        resolved_model = evo2_backend.model

        if evo2_backend.is_available():
            cache_path = Path(cache_dir) if cache_dir else None
            emb_result = extract_embeddings(
                sequences=borderline_seqs,
                backend=evo2_backend,
                layer=layer,
                model=resolved_model,
                batch_size=batch_size,
                cache_dir=cache_path,
            )

            borderline_lengths = [
                len(borderline_seqs[sid]) for sid in emb_result.sequence_ids
            ]
            borderline_results = classify_contigs(
                embeddings=emb_result.embeddings,
                sequence_ids=emb_result.sequence_ids,
                sequence_lengths=borderline_lengths,
                classifier=classifier,
                threshold=threshold,
            )
            results.extend(borderline_results)
        else:
            logger.warning(
                "Evo2 backend not available — classifying borderline "
                "cases with k-mer scores only"
            )
            for i in range(len(seq_ids)):
                if borderline[i]:
                    results.append(DetectionResult(
                        contig_id=seq_ids[i],
                        viral_score=round(float(kmer_viral_prob[i]), 4),
                        classification="viral" if kmer_viral_prob[i] >= threshold else "cellular",
                        contig_length=len(sequences[seq_ids[i]]),
                    ))

    # 5. Write results
    write_tsv(results, output_path, "detection_results.tsv")

    _NON_VIRAL = {"cellular", "chromosome", "plasmid", "ambiguous"}
    viral_ids = {r.contig_id for r in results if r.classification not in _NON_VIRAL}
    if viral_ids:
        viral_seqs = {sid: sequences[sid] for sid in seq_ids if sid in viral_ids}
        write_fasta(viral_seqs, output_path / "viral_contigs.fasta")

    counts = Counter(r.classification for r in results)
    n_viral = sum(1 for r in results if r.classification not in _NON_VIRAL)
    n_total = len(results)

    summary = {
        "mode": "fast",
        "n_sequences": n_total,
        "n_viral": n_viral,
        "n_cellular": n_total - n_viral,
        "kmer_confident": int(n_confident),
        "evo2_borderline": int(n_borderline),
        "compute_savings": f"{(1 - n_borderline / max(n_total, 1)):.0%} sequences skipped Evo2",
        "classification_counts": dict(counts),
    }
    write_json(summary, output_path, "detection_summary.json")

    logger.info(
        f"Fast detection complete: {n_viral} viral, {n_total - n_viral} non-viral. "
        f"{n_confident}/{n_total} classified by k-mers, "
        f"{n_borderline} by Evo2. "
        f"Results in {output_path}"
    )


def _get_or_train_kmer_model(sequences, classifier, cache_dir, layer, model):
    """Get or train a k-mer classifier bootstrapped from Evo2 predictions.

    Uses cached Evo2 embeddings (if available) to generate training labels,
    then trains a fast k-mer classifier on those labels.
    """
    from virosense.features.kmer_classifier import (
        KmerClassifier,
        compute_kmer_features_batch,
        train_kmer_classifier,
    )

    # Check for cached embeddings
    if not cache_dir:
        return None

    cache_path = Path(cache_dir)
    npz_files = list(cache_path.glob("*.npz"))
    if not npz_files:
        return None

    # Load cached embeddings and use Evo2 classifier to generate labels
    data = np.load(npz_files[0])
    if "embeddings" not in data or "sequence_ids" not in data:
        return None

    cached_ids = set(str(s) for s in data["sequence_ids"])
    cached_embs = data["embeddings"]
    cached_id_list = [str(s) for s in data["sequence_ids"]]

    # Get Evo2 predictions for cached sequences
    proba = classifier.predict_proba(cached_embs)
    if proba.shape[1] == 2:
        viral_scores = proba[:, 1]
    else:
        # Multi-class: viral class index
        class_names = classifier.metadata.get("class_names", [])
        viral_idx = next(
            (i for i, n in enumerate(class_names) if n in ("viral", "phage")), 0
        )
        viral_scores = proba[:, viral_idx]

    # Build training labels from high-confidence Evo2 predictions
    labels = {}
    for i, sid in enumerate(cached_id_list):
        if sid in sequences:
            if viral_scores[i] > 0.8:
                labels[sid] = 1  # viral
            elif viral_scores[i] < 0.2:
                labels[sid] = 0  # non-viral

    if len(labels) < 50:
        logger.warning(f"Only {len(labels)} high-confidence labels from cache — insufficient for k-mer model")
        return None

    n_viral = sum(v for v in labels.values())
    n_nonviral = len(labels) - n_viral
    logger.info(
        f"Training k-mer model from {len(labels)} cached Evo2 predictions "
        f"({n_viral} viral, {n_nonviral} non-viral)"
    )

    return train_kmer_classifier(sequences, labels)
