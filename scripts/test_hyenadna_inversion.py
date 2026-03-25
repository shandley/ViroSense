#!/usr/bin/env python3
"""
Test whether HyenaDNA shows the offset-3 cosine inversion.

If a completely different architecture (Hyena convolution) trained on DNA
shows the same codon periodicity pattern as Evo2 (StripedHyena), this
confirms the signal is a universal feature of DNA language models.

Uses HyenaDNA-small (256-D, 32k context, 3.6M params) via HuggingFace.
Runs on CPU/MPS — no GPU needed.
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

OUT_DIR = Path("results/experiments/hyenadna_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Test sequences: a mix of coding and non-coding
TEST_SEQUENCES = {
    # Coding sequences (should show inversion: cos3 > cos1)
    "ecoli_lacZ_cds": None,        # Will load from FASTA
    "human_HBB_exon1": None,       # Will load from FASTA
    "drosophila_eve_cds": None,    # Will load from FASTA

    # Non-coding (should NOT show inversion: cos1 > cos3)
    "human_HBB_intron1": None,
}

# Use sequences from our existing exon-intron data
SEQ_DIR = Path("results/experiments/exon_intron/sequences")
ANN_DIR = Path("results/experiments/exon_intron")


def load_test_sequences() -> dict[str, str]:
    """Load a diverse set of coding and non-coding sequences."""
    sequences = {}

    # Load full gene sequences and extract coding/non-coding regions
    for gene_name, ann_file in [
        ("human_HBB", "annotations_all.json"),
        ("drosophila_eve", "annotations_all.json"),
        ("arabidopsis_AG", "annotations_all.json"),
        ("celegans_unc54", "annotations_all.json"),
        ("yeast_ACT1", "annotations_fixed.json"),
    ]:
        fasta_path = SEQ_DIR / f"{gene_name}.fasta"
        if not fasta_path.exists():
            continue

        with open(fasta_path) as f:
            seq = "".join(l.strip() for l in f if not l.startswith(">"))

        # Load annotations to get CDS regions
        with open(ANN_DIR / ann_file) as f:
            all_ann = json.load(f)
        if gene_name not in all_ann:
            continue

        ann = all_ann[gene_name]
        regions = ann.get("cds", []) or ann.get("exons", [])

        # Extract first exon (coding)
        if regions:
            r = regions[0]
            exon_seq = seq[r["start"]:r["end"]]
            if len(exon_seq) >= 200:
                sequences[f"{gene_name}_exon1"] = exon_seq[:2000]

        # Extract intron (non-coding) — region between first two exons
        if len(regions) >= 2:
            intron_start = regions[0]["end"]
            intron_end = regions[1]["start"]
            intron_seq = seq[intron_start:intron_end]
            if len(intron_seq) >= 200:
                sequences[f"{gene_name}_intron1"] = intron_seq[:2000]

    # Also add the E. coli lacZ region (pure coding, no introns)
    lac_path = Path("results/experiments/fig1_data")
    for f in lac_path.glob("*.fasta"):
        with open(f) as fh:
            seq = "".join(l.strip() for l in fh if not l.startswith(">"))
        if len(seq) > 500:
            sequences["ecoli_lac_operon"] = seq[:5000]
            break

    return sequences


def extract_embeddings(model, tokenizer, sequence: str, device: str) -> np.ndarray:
    """Extract per-position embeddings from HyenaDNA."""
    # Tokenize (single-character tokenizer)
    inputs = tokenizer(sequence, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    # Get hidden states from the last layer
    # outputs.hidden_states is a tuple of (n_layers+1) tensors
    last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
    embeddings = last_hidden.squeeze(0).cpu().numpy()  # (seq_len, hidden_dim)

    # Remove special tokens (first and last)
    # HyenaDNA adds [CLS] at start
    if embeddings.shape[0] > len(sequence):
        embeddings = embeddings[1:len(sequence) + 1]

    return embeddings


def compute_cosines(embeddings: np.ndarray, max_offset: int = 15) -> dict:
    """Compute cosine similarity at offsets 1-15."""
    n = embeddings.shape[0]
    norms = np.linalg.norm(embeddings, axis=1)

    cosines = {}
    for offset in range(1, max_offset + 1):
        values = []
        for i in range(n - offset):
            ni, nj = norms[i], norms[i + offset]
            if ni > 0 and nj > 0:
                cos = np.dot(embeddings[i], embeddings[i + offset]) / (ni * nj)
                values.append(cos)
        cosines[f"offset_{offset}"] = np.mean(values) if values else 0

    return cosines


def main():
    print("Loading HyenaDNA-small (32k context)...")
    model_name = "LongSafari/hyenadna-small-32k-seqlen-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32,
    )
    model.eval()

    device = "cpu"  # M4 MPS could also work
    model = model.to(device)
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Hidden dim: {model.config.d_model}")

    # Load test sequences
    sequences = load_test_sequences()
    print(f"\nLoaded {len(sequences)} test sequences:")
    for name, seq in sequences.items():
        print(f"  {name}: {len(seq)} bp")

    # Extract and analyze
    results = []
    print(f"\n{'='*70}")
    print(f"{'Sequence':<30s} {'Length':>6s} {'cos1':>7s} {'cos3':>7s} {'Gap':>7s} {'Inversion?':>11s}")
    print(f"{'-'*70}")

    for name, seq in sequences.items():
        # Truncate to 5000bp for speed
        seq = seq[:5000]

        embeddings = extract_embeddings(model, tokenizer, seq, device)
        cosines = compute_cosines(embeddings)

        cos1 = cosines["offset_1"]
        cos3 = cosines["offset_3"]
        gap = cos3 - cos1
        has_inversion = gap > 0

        is_coding = "exon" in name or "cds" in name or "lac" in name
        correct = has_inversion == is_coding

        results.append({
            "name": name, "length": len(seq),
            "cos1": cos1, "cos3": cos3, "gap": gap,
            "has_inversion": has_inversion,
            "is_coding": is_coding, "correct": correct,
            "cosines": cosines,
        })

        status = "YES" if has_inversion else "no"
        mark = "ok" if correct else "WRONG"
        print(f"{name:<30s} {len(seq):>6d} {cos1:>7.4f} {cos3:>7.4f} {gap:>+7.4f} {status:>7s}  {mark}")

    # Summary
    n_correct = sum(1 for r in results if r["correct"])
    print(f"\n{'='*70}")
    print(f"SUMMARY: {n_correct}/{len(results)} correct")
    print(f"{'='*70}")

    coding = [r for r in results if r["is_coding"]]
    noncoding = [r for r in results if not r["is_coding"]]

    if coding:
        print(f"  Coding sequences (n={len(coding)}):")
        print(f"    Mean cos1: {np.mean([r['cos1'] for r in coding]):.4f}")
        print(f"    Mean cos3: {np.mean([r['cos3'] for r in coding]):.4f}")
        print(f"    Mean gap:  {np.mean([r['gap'] for r in coding]):+.4f}")
        print(f"    Inversion present: {sum(1 for r in coding if r['has_inversion'])}/{len(coding)}")

    if noncoding:
        print(f"  Non-coding sequences (n={len(noncoding)}):")
        print(f"    Mean cos1: {np.mean([r['cos1'] for r in noncoding]):.4f}")
        print(f"    Mean cos3: {np.mean([r['cos3'] for r in noncoding]):.4f}")
        print(f"    Mean gap:  {np.mean([r['gap'] for r in noncoding]):+.4f}")
        print(f"    Inversion present: {sum(1 for r in noncoding if r['has_inversion'])}/{len(noncoding)}")

    # Multi-offset comb filter for coding vs non-coding
    print(f"\n  Multi-offset comb filter (coding):")
    if coding:
        for offset in range(1, 16):
            mean_cos = np.mean([r["cosines"][f"offset_{offset}"] for r in coding])
            bar = "#" * int(mean_cos * 50) if mean_cos > 0 else ""
            mod3 = " *" if offset % 3 == 0 else ""
            print(f"    Offset {offset:>2d}: {mean_cos:>7.4f} {bar}{mod3}")

    # Save results
    with open(OUT_DIR / "hyenadna_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {OUT_DIR}/hyenadna_results.json")


if __name__ == "__main__":
    main()
