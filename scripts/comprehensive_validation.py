#!/usr/bin/env python3
"""
Comprehensive cross-domain validation of Evo2 embeddings.

~500 sequences testing:
1. Codon periodicity universality (per-position analysis)
2. Functional clustering (mean-pooled analysis)

Designed for HTCF execution (self-hosted NIM, 2 TB scratch).

Usage:
    # Step 1: Download sequences from NCBI
    uv run python scripts/comprehensive_validation.py download --output results/comprehensive/

    # Step 2: Extract embeddings (per-position + mean-pooled) via NIM
    uv run python scripts/comprehensive_validation.py extract \
        --input results/comprehensive/sequences/ \
        --output results/comprehensive/embeddings/ \
        --nim-url http://localhost:8000  # HTCF self-hosted
        # or omit --nim-url for cloud NIM

    # Step 3: Analyze
    uv run python scripts/comprehensive_validation.py analyze \
        --input results/comprehensive/ \
        --output results/comprehensive/analysis/

    # Or run all steps:
    uv run python scripts/comprehensive_validation.py all --output results/comprehensive/
"""

import argparse
import asyncio
import base64
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def download_sequences(output_dir: Path, panel: list[dict]):
    """Download all sequences from NCBI."""
    from Bio import Entrez, SeqIO
    from io import StringIO
    import re

    Entrez.email = "shandley@wustl.edu"

    seq_dir = output_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []

    for i, entry in enumerate(panel):
        name = entry["name"]
        fasta_path = seq_dir / f"{name}.fasta"

        if fasta_path.exists():
            with open(fasta_path) as f:
                seq = "".join(l.strip() for l in f if not l.startswith(">"))
            if len(seq) >= 50:
                print(f"  [{i+1}/{len(panel)}] {name}: cached ({len(seq)} bp)")
                results.append({"name": name, "length": len(seq), "status": "cached"})
                continue

        print(f"  [{i+1}/{len(panel)}] {name}: downloading...", end=" ", flush=True)

        try:
            seq_str = _fetch_sequence(entry, Entrez)
            if seq_str is None or len(seq_str) < 50:
                print(f"FAILED (too short: {len(seq_str) if seq_str else 0} bp)")
                failures.append({"name": name, "reason": "too short"})
                time.sleep(0.4)
                continue

            seq_clean = str(seq_str).upper().replace("U", "T")
            seq_clean = "".join(c for c in seq_clean if c in "ACGTN")

            if len(seq_clean) < 50:
                print(f"FAILED (after cleaning: {len(seq_clean)} bp)")
                failures.append({"name": name, "reason": "too short after cleaning"})
                time.sleep(0.4)
                continue

            # Truncate to max 2000 bp CDS
            if len(seq_clean) > 2000:
                seq_clean = seq_clean[:2000]

            gc = sum(1 for c in seq_clean if c in "GC") / len(seq_clean) * 100
            header = (f">{name} {entry.get('species', '')} {entry.get('gene', '')} "
                      f"[{len(seq_clean)}bp, GC={gc:.1f}%]")
            with open(fasta_path, "w") as f:
                f.write(f"{header}\n")
                for j in range(0, len(seq_clean), 60):
                    f.write(seq_clean[j:j+60] + "\n")

            print(f"OK ({len(seq_clean)} bp, GC={gc:.1f}%)")
            results.append({"name": name, "length": len(seq_clean), "gc": gc, "status": "downloaded"})
            time.sleep(0.35)

        except Exception as e:
            print(f"FAILED ({e})")
            failures.append({"name": name, "reason": str(e)})
            time.sleep(0.4)

    # Save summary
    with open(output_dir / "download_summary.json", "w") as f:
        json.dump({"total": len(panel), "downloaded": len(results),
                    "failed": len(failures), "failures": failures}, f, indent=2)

    print(f"\nDownloaded: {len(results)}/{len(panel)}, Failed: {len(failures)}")


def _fetch_sequence(entry: dict, Entrez) -> str | None:
    """Fetch a sequence from NCBI."""
    from Bio import SeqIO
    from io import StringIO
    import re

    fetch_type = entry.get("type", "mrna_cds")
    acc = entry.get("accession", "")

    if fetch_type == "hardcoded":
        return entry.get("sequence", "")

    if fetch_type == "protein_cds":
        return _fetch_cds_from_protein(acc, Entrez)

    if fetch_type in ("mrna_cds", "full_cds", "noncoding"):
        # Strip version number
        acc_base = acc.split(".")[0] if "." in acc else acc
        handle = Entrez.efetch(db="nucleotide", id=acc_base,
                               rettype="fasta", retmode="text")
        record = next(SeqIO.parse(StringIO(handle.read()), "fasta"))
        handle.close()

        if fetch_type == "full_cds" or fetch_type == "noncoding":
            seq = str(record.seq)
        else:
            start = entry.get("cds_start", 1) - 1
            end = entry.get("cds_end", len(record.seq))
            seq = str(record.seq[start:end])
        return seq

    if fetch_type == "genomic_region":
        acc_base = entry.get("nuccore", acc).split(".")[0]
        start = entry.get("cds_start", 1)
        end = entry.get("cds_end")
        handle = Entrez.efetch(db="nucleotide", id=acc_base,
                               rettype="fasta", retmode="text",
                               seq_start=start, seq_stop=end)
        record = next(SeqIO.parse(StringIO(handle.read()), "fasta"))
        handle.close()
        return str(record.seq)

    return None


def _fetch_cds_from_protein(protein_acc: str, Entrez) -> str | None:
    """Fetch CDS nucleotide from a protein accession via coded_by."""
    from Bio import SeqIO
    from io import StringIO
    import re

    try:
        handle = Entrez.efetch(db="protein", id=protein_acc,
                               rettype="gp", retmode="text")
        content = handle.read()
        handle.close()

        match = re.search(r'/coded_by="([^"]+)"', content)
        if match:
            coded_by = match.group(1)
            is_complement = "complement" in coded_by
            coded_by = coded_by.replace("complement(", "").rstrip(")")

            if ":" in coded_by:
                nuc_acc, coords = coded_by.split(":", 1)
            else:
                nuc_acc = coded_by
                coords = None

            if coords and ".." in coords:
                # Handle join() for multi-exon
                if "join" in coords:
                    # For simplicity, take the full range
                    nums = re.findall(r"(\d+)", coords)
                    if len(nums) >= 2:
                        start = int(nums[0])
                        end = int(nums[-1])
                    else:
                        return None
                else:
                    parts = coords.split("..")
                    start = int(parts[0].replace("<", "").replace(">", ""))
                    end = int(parts[1].replace("<", "").replace(">", ""))
            else:
                start = 1
                end = None

            nuc_acc_base = nuc_acc.split(".")[0]
            handle = Entrez.efetch(db="nucleotide", id=nuc_acc_base,
                                   rettype="fasta", retmode="text",
                                   seq_start=start, seq_stop=end)
            record = next(SeqIO.parse(StringIO(handle.read()), "fasta"))
            handle.close()

            seq = str(record.seq)
            if is_complement:
                from Bio.Seq import Seq
                seq = str(Seq(seq).reverse_complement())
            return seq

    except Exception:
        pass
    return None


def extract_embeddings(input_dir: Path, output_dir: Path, nim_url: str | None = None):
    """Extract per-position + mean-pooled embeddings via NIM."""
    import httpx

    output_dir.mkdir(parents=True, exist_ok=True)
    perpos_dir = output_dir / "per_position"
    perpos_dir.mkdir(exist_ok=True)

    import os
    api_key = os.environ.get("NVIDIA_API_KEY", "")

    # Determine endpoint
    if nim_url:
        url = f"{nim_url.rstrip('/')}/biology/arc/evo2/forward"
        layer = "decoder.layers.10"  # self-hosted layer naming
    else:
        url = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward"
        layer = "blocks.10"

    # Load sequences
    fasta_files = sorted(input_dir.glob("*.fasta"))
    sequences = []
    for fp in fasta_files:
        name = fp.stem
        # Check if already extracted
        mean_path = output_dir / f"{name}_mean.npy"
        if mean_path.exists():
            continue
        with open(fp) as f:
            seq = "".join(l.strip() for l in f if not l.startswith(">"))
        sequences.append((name, seq))

    if not sequences:
        print("All embeddings cached.")
        return

    print(f"Extracting embeddings for {len(sequences)} sequences...")
    print(f"  Endpoint: {url}")
    print(f"  Layer: {layer}")

    concurrency = 1 if nim_url else 3  # Self-hosted NIM 7B handles only 1 concurrent

    async def extract_one(client: httpx.AsyncClient, name: str, seq: str,
                          sem: asyncio.Semaphore):
        async with sem:
            headers = {"Content-Type": "application/json"}
            if not nim_url:
                headers["Authorization"] = f"Bearer {api_key}"

            payload = {
                "sequence": seq[:16000],
                "output_layers": [layer],
            }

            for attempt in range(5):
                try:
                    resp = await client.post(url, json=payload, headers=headers,
                                             timeout=300, follow_redirects=True)
                    if resp.status_code == 429:
                        await asyncio.sleep(2 ** attempt * 5)
                        continue
                    resp.raise_for_status()
                    data = resp.json()

                    raw = base64.b64decode(data["data"])
                    npz_data = np.load(io.BytesIO(raw))
                    key = f"{layer}.output"
                    emb = npz_data[key]
                    # Handle both tensor shapes:
                    # Cloud NIM 40B: (1, seq_len, 8192)
                    # Self-hosted NIM 7B: (seq_len, 1, 4096)
                    if emb.ndim == 3:
                        if emb.shape[0] == 1:
                            emb = emb.squeeze(0)  # (1, N, D) → (N, D)
                        elif emb.shape[1] == 1:
                            emb = emb.squeeze(1)  # (N, 1, D) → (N, D)
                        else:
                            emb = emb[0]  # fallback

                    # Save mean-pooled (small file)
                    mean_emb = emb.mean(axis=0).astype(np.float32)
                    np.save(output_dir / f"{name}_mean.npy", mean_emb)

                    # Save per-position (large — only on HTCF or if requested)
                    # For periodicity analysis, compute metrics here and save only the metrics
                    metrics = _compute_periodicity_metrics(emb)
                    with open(output_dir / f"{name}_metrics.json", "w") as f:
                        json.dump(metrics, f)

                    return name, mean_emb.shape[0]

                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (429, 503):
                        await asyncio.sleep(2 ** attempt * 5)
                        continue
                    if attempt < 4:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    print(f"  {name}: HTTP {e.response.status_code}")
                    return name, None
                except Exception as e:
                    if attempt < 4:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    print(f"  {name}: {e}")
                    return name, None

            return name, None

    async def run_all():
        sem = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient() as client:
            tasks = [extract_one(client, n, s, sem) for n, s in sequences]
            completed = 0
            failed = 0
            for coro in asyncio.as_completed(tasks):
                name, dim = await coro
                if dim is not None:
                    completed += 1
                    print(f"  [{completed}/{len(sequences)}] {name}: {dim}D")
                else:
                    failed += 1
                    print(f"  {name}: FAILED")
            return completed, failed

    completed, failed = asyncio.run(run_all())
    print(f"\nExtracted: {completed}/{len(sequences)}, Failed: {failed}")


def _compute_periodicity_metrics(emb: np.ndarray) -> dict:
    """Compute per-position periodicity metrics without saving the full embedding."""
    norms = np.linalg.norm(emb, axis=1)
    n = len(norms)

    # Lag-3 autocorrelation
    mean = norms.mean()
    var = norms.var()
    lag3 = float(np.mean((norms[:n-3] - mean) * (norms[3:] - mean)) / var) if var > 0 else 0.0

    # FFT
    fft_vals = np.abs(np.fft.rfft(norms - mean))
    freqs = np.fft.rfftfreq(n, d=1.0)
    fft_vals[0] = 0
    if len(fft_vals) > 1:
        peak_idx = np.argmax(fft_vals[1:]) + 1
        dominant_period = 1.0 / freqs[peak_idx] if freqs[peak_idx] > 0 else n
    else:
        dominant_period = n

    # 3bp FFT power
    target_freq = 1.0 / 3.0
    bp3_idx = np.argmin(np.abs(freqs[1:] - target_freq)) + 1
    bp3_power = float(fft_vals[bp3_idx])
    max_power = float(fft_vals[1:].max())
    bp3_fraction = bp3_power / max_power if max_power > 0 else 0

    # Cosine similarities
    cos1_vals = []
    cos3_vals = []
    for j in range(len(emb) - 3):
        nj = np.linalg.norm(emb[j])
        nj1 = np.linalg.norm(emb[j+1])
        nj3 = np.linalg.norm(emb[j+3])
        if nj > 0 and nj1 > 0:
            cos1_vals.append(float(np.dot(emb[j], emb[j+1]) / (nj * nj1)))
        if nj > 0 and nj3 > 0:
            cos3_vals.append(float(np.dot(emb[j], emb[j+3]) / (nj * nj3)))

    cos1 = float(np.mean(cos1_vals)) if cos1_vals else 0.0
    cos3 = float(np.mean(cos3_vals)) if cos3_vals else 0.0

    return {
        "lag3": round(lag3, 4),
        "dominant_fft_period": round(dominant_period, 1),
        "bp3_fft_fraction": round(bp3_fraction, 3),
        "cos1": round(cos1, 4),
        "cos3": round(cos3, 4),
        "offset3_inversion": cos3 > cos1,
        "inversion_gap": round(cos3 - cos1, 4),
        "norm_mean": round(float(norms.mean()), 2),
        "norm_std": round(float(norms.std()), 2),
    }


def analyze_all(input_dir: Path, output_dir: Path, panel: list[dict]):
    """Run comprehensive analysis: periodicity + functional clustering."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import cosine

    output_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = input_dir / "embeddings"

    panel_lookup = {e["name"]: e for e in panel}

    # ── Load all data ──
    results = []
    mean_embs = {}

    for entry in panel:
        name = entry["name"]
        metrics_path = emb_dir / f"{name}_metrics.json"
        mean_path = emb_dir / f"{name}_mean.npy"

        if not metrics_path.exists() or not mean_path.exists():
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        mean_emb = np.load(mean_path)
        mean_embs[name] = mean_emb

        # Load sequence for GC
        fasta_path = input_dir / "sequences" / f"{name}.fasta"
        gc = 0.0
        seq_len = 0
        if fasta_path.exists():
            with open(fasta_path) as f:
                seq = "".join(l.strip() for l in f if not l.startswith(">"))
            gc = sum(1 for c in seq.upper() if c in "GC") / max(len(seq), 1) * 100
            seq_len = len(seq)

        results.append({
            "name": name,
            "component": entry.get("component", "?"),
            "category": entry.get("category", "unknown"),
            "species": entry.get("species", ""),
            "gene": entry.get("gene", ""),
            "gene_family": entry.get("gene_family", ""),
            "domain": entry.get("domain", ""),
            "gc_content": round(gc, 1),
            "seq_len": seq_len,
            **metrics,
        })

    print(f"Loaded {len(results)} sequences with embeddings")

    # Save full results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Periodicity analysis ──
    print("\n" + "="*70)
    print("PERIODICITY UNIVERSALITY")
    print("="*70)

    coding = [r for r in results if not r["category"].startswith("noncoding")]
    noncoding = [r for r in results if r["category"].startswith("noncoding")]

    n_coding_inv = sum(1 for r in coding if r["offset3_inversion"])
    n_noncoding_inv = sum(1 for r in noncoding if r["offset3_inversion"])

    print(f"Coding: {n_coding_inv}/{len(coding)} ({100*n_coding_inv/max(len(coding),1):.1f}%) show inversion")
    print(f"Non-coding: {n_noncoding_inv}/{len(noncoding)} ({100*n_noncoding_inv/max(len(noncoding),1):.1f}%) show inversion")

    # By domain
    by_domain = defaultdict(list)
    for r in results:
        by_domain[r.get("domain", "unknown")].append(r)

    print("\nBy domain:")
    for domain in sorted(by_domain.keys()):
        recs = by_domain[domain]
        inv = sum(1 for r in recs if r["offset3_inversion"])
        print(f"  {domain:30s}: {inv}/{len(recs)} inversion")

    # ── Functional clustering analysis ──
    print("\n" + "="*70)
    print("FUNCTIONAL CLUSTERING")
    print("="*70)

    # Only component B (functional clustering panel)
    comp_b = [r for r in results if r.get("component") == "B" and r["name"] in mean_embs]
    if comp_b:
        gene_families = defaultdict(list)
        for r in comp_b:
            gf = r.get("gene_family", r["gene"])
            gene_families[gf].append(r["name"])

        print(f"Component B sequences: {len(comp_b)}")
        print(f"Gene families: {len(gene_families)}")
        for gf, names in sorted(gene_families.items(), key=lambda x: -len(x[1])):
            print(f"  {gf:30s}: N={len(names)}")

        # Build embedding matrix for component B
        b_names = [r["name"] for r in comp_b]
        b_embs = np.array([mean_embs[n] for n in b_names], dtype=np.float64)
        b_labels = np.array([r.get("gene_family", r["gene"]) for r in comp_b])

        # L2 normalize
        b_norms = np.linalg.norm(b_embs, axis=1, keepdims=True)
        b_norms[b_norms == 0] = 1
        b_embs_norm = b_embs / b_norms

        # PCA
        n_components = min(50, len(b_embs_norm) - 1)
        pca = PCA(n_components=n_components)
        b_pca = pca.fit_transform(b_embs_norm)

        # Silhouette
        if len(set(b_labels)) > 1:
            sil = silhouette_score(b_pca, b_labels, metric="cosine")
            print(f"\nSilhouette (gene family, PCA-{n_components}): {sil:.3f}")

        # Nearest-neighbor accuracy
        nn = NearestNeighbors(n_neighbors=2, metric="cosine")
        nn.fit(b_pca)
        _, indices = nn.kneighbors(b_pca)
        correct = sum(1 for i in range(len(b_labels)) if b_labels[indices[i, 1]] == b_labels[i])
        print(f"NN gene family accuracy: {correct}/{len(b_labels)} ({100*correct/len(b_labels):.1f}%)")

        # Within vs between distances
        print("\nWithin/between cosine distances by gene family:")
        for gf in sorted(gene_families.keys(), key=lambda x: -len(gene_families[x])):
            gf_mask = b_labels == gf
            gf_embs = b_pca[gf_mask]
            other_embs = b_pca[~gf_mask]
            if len(gf_embs) < 2:
                continue
            within = [cosine(gf_embs[i], gf_embs[j])
                      for i in range(len(gf_embs)) for j in range(i+1, len(gf_embs))]
            np.random.seed(42)
            sample = np.random.choice(len(other_embs), min(100, len(other_embs)), replace=False)
            between = [cosine(gf_embs[i], other_embs[j])
                       for i in range(len(gf_embs)) for j in sample]
            w = np.mean(within)
            b = np.mean(between)
            print(f"  {gf:30s}: within={w:.4f}, between={b:.4f}, ratio={b/w:.2f}x")

    print(f"\nResults saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Evo2 embedding validation")
    parser.add_argument("command", choices=["download", "extract", "analyze", "all"])
    parser.add_argument("--output", type=Path, default=Path("results/comprehensive"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--nim-url", type=str, default=None)
    parser.add_argument("--panel", type=Path, default=None,
                        help="Path to panel JSON file")
    args = parser.parse_args()

    # Load panel
    if args.panel and args.panel.exists():
        with open(args.panel) as f:
            panel = json.load(f)
    else:
        # Try default location
        default_panel = Path("results/comprehensive/panel.json")
        if default_panel.exists():
            with open(default_panel) as f:
                panel = json.load(f)
        else:
            print("ERROR: No panel file found. Provide --panel or create results/comprehensive/panel.json")
            sys.exit(1)

    print(f"Panel: {len(panel)} sequences")

    if args.command == "download":
        download_sequences(args.output, panel)
    elif args.command == "extract":
        input_dir = args.input or args.output / "sequences"
        extract_embeddings(input_dir, args.output / "embeddings", args.nim_url)
    elif args.command == "analyze":
        input_dir = args.input or args.output
        analyze_all(input_dir, args.output / "analysis", panel)
    elif args.command == "all":
        print("\n=== Step 1: Download ===")
        download_sequences(args.output, panel)
        print("\n=== Step 2: Extract ===")
        extract_embeddings(args.output / "sequences", args.output / "embeddings", args.nim_url)
        print("\n=== Step 3: Analyze ===")
        analyze_all(args.output, args.output / "analysis", panel)


if __name__ == "__main__":
    main()
