#!/usr/bin/env python3
"""
Exon-intron boundary detection from Evo2 per-position embeddings.

Tests whether the offset-3 cosine inversion flips at splice sites.

Genes tested:
- Human HBB (beta-globin): 1.6kb, 3 exons, 2 introns — textbook gene
- Human TP53: ~19kb, 11 exons — tumor suppressor, complex structure
- Human BRCA1 (partial): 16kb region, multiple exons
- Drosophila Adh: ~4kb, 3 exons, 2 introns — model organism
- C. elegans unc-54: ~6kb, multiple exons — nematode
- Arabidopsis AGAMOUS: ~4kb, 7 exons — plant

Usage:
    # Cloud NIM:
    NVIDIA_API_KEY=... uv run python scripts/exon_intron_scan.py

    # HTCF:
    uv run python scripts/exon_intron_scan.py --nim-url http://localhost:8000
"""

import argparse
import asyncio
import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT_DIR = Path("results/exon_intron")

# Genes to test: (name, accession, start, end, description)
# Using RefSeqGene (NG_) accessions which include full genomic context with introns
GENES = [
    {
        "name": "human_HBB",
        "accession": "NG_000007.3",
        "start": 70544,
        "end": 72152,
        "species": "Homo sapiens",
        "gene": "HBB (beta-globin)",
        "note": "3 exons, 2 introns, 1.6kb. Textbook example.",
    },
    {
        "name": "human_TP53_part1",
        "accession": "NG_017013.2",
        "start": 5000,
        "end": 20000,
        "species": "Homo sapiens",
        "gene": "TP53 (tumor suppressor), first 15kb",
        "note": "11 exons, ~19kb total. Taking first 15kb.",
    },
    {
        "name": "human_TP53_part2",
        "accession": "NG_017013.2",
        "start": 18000,
        "end": 32772,
        "species": "Homo sapiens",
        "gene": "TP53 (tumor suppressor), last 15kb",
        "note": "Overlaps with part1 for stitching.",
    },
    {
        "name": "human_BRCA1",
        "accession": "NG_005905.2",
        "start": 15000,
        "end": 31000,
        "species": "Homo sapiens",
        "gene": "BRCA1 (breast cancer 1), 16kb region",
        "note": "Large gene, ~80kb total. Taking exon-rich 16kb region.",
    },
    {
        "name": "drosophila_Adh",
        "accession": "NT_033779.5",
        "start": 15680652,
        "end": 15684700,
        "species": "Drosophila melanogaster",
        "gene": "Adh (alcohol dehydrogenase)",
        "note": "3 exons, 2 introns, ~4kb.",
    },
    {
        "name": "celegans_unc54",
        "accession": "NC_003279.8",
        "start": 12844667,
        "end": 12850900,
        "species": "Caenorhabditis elegans",
        "gene": "unc-54 (myosin heavy chain)",
        "note": "Multiple exons, ~6kb.",
    },
    {
        "name": "arabidopsis_AG",
        "accession": "NC_003074.8",
        "start": 16019399,
        "end": 16024700,
        "species": "Arabidopsis thaliana",
        "gene": "AGAMOUS (floral development)",
        "note": "7 exons, ~5kb.",
    },
    {
        "name": "yeast_ACT1",
        "accession": "NC_001136.10",
        "start": 54695,
        "end": 56115,
        "species": "Saccharomyces cerevisiae",
        "gene": "ACT1 (actin, single intron)",
        "note": "1 intron — yeast has very few introns.",
    },
]


def download_sequences():
    """Download genomic sequences from NCBI."""
    from Bio import Entrez, SeqIO
    from io import StringIO

    Entrez.email = "shandley@wustl.edu"
    seq_dir = OUT_DIR / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    sequences = {}
    for gene in GENES:
        name = gene["name"]
        fasta_path = seq_dir / f"{name}.fasta"

        if fasta_path.exists():
            with open(fasta_path) as f:
                seq = "".join(l.strip() for l in f if not l.startswith(">"))
            print(f"  {name}: cached ({len(seq)} bp)")
            sequences[name] = seq
            continue

        print(f"  {name}: downloading...", end=" ", flush=True)
        try:
            acc = gene["accession"].split(".")[0]
            handle = Entrez.efetch(db="nucleotide", id=acc, rettype="fasta", retmode="text",
                                   seq_start=gene["start"], seq_stop=gene["end"])
            record = next(SeqIO.parse(StringIO(handle.read()), "fasta"))
            handle.close()
            seq = str(record.seq).upper()

            with open(fasta_path, "w") as f:
                f.write(f">{name} {gene['species']} {gene['gene']} [{len(seq)}bp]\n")
                for i in range(0, len(seq), 60):
                    f.write(seq[i:i+60] + "\n")

            print(f"OK ({len(seq)} bp)")
            sequences[name] = seq
            import time
            time.sleep(0.5)
        except Exception as e:
            print(f"FAILED ({e})")

    return sequences


def download_annotations():
    """Download exon annotations for each gene."""
    from Bio import Entrez, SeqIO
    from io import StringIO

    Entrez.email = "shandley@wustl.edu"
    ann_path = OUT_DIR / "annotations.json"

    if ann_path.exists():
        with open(ann_path) as f:
            return json.load(f)

    annotations = {}
    for gene in GENES:
        name = gene["name"]
        acc = gene["accession"].split(".")[0]
        start = gene["start"]
        end = gene["end"]

        print(f"  {name}: getting annotations...", end=" ", flush=True)
        try:
            handle = Entrez.efetch(db="nucleotide", id=acc, rettype="gb", retmode="text",
                                   seq_start=start, seq_stop=end)
            record = next(SeqIO.parse(StringIO(handle.read()), "genbank"))
            handle.close()

            exons = []
            cds_regions = []
            for feat in record.features:
                if feat.type == "exon":
                    exons.append({"start": int(feat.location.start), "end": int(feat.location.end)})
                if feat.type == "CDS":
                    # Extract CDS parts (may be join of multiple exons)
                    for part in feat.location.parts:
                        cds_regions.append({"start": int(part.start), "end": int(part.end)})

            annotations[name] = {"exons": exons, "cds": cds_regions, "length": end - start}
            print(f"{len(exons)} exons, {len(cds_regions)} CDS parts")
            import time
            time.sleep(0.5)
        except Exception as e:
            print(f"FAILED ({e})")
            annotations[name] = {"exons": [], "cds": [], "length": end - start}

    with open(ann_path, "w") as f:
        json.dump(annotations, f, indent=2)

    return annotations


async def extract_embeddings(sequences: dict, nim_url: str | None = None):
    """Extract per-position embeddings for each gene."""
    import httpx

    api_key = os.environ.get("NVIDIA_API_KEY", "")

    if nim_url:
        url = f"{nim_url.rstrip('/')}/biology/arc/evo2/forward"
        layer = "decoder.layers.10"
    else:
        url = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward"
        layer = "blocks.10"

    metrics_dir = OUT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    for name, seq in sequences.items():
        metrics_path = metrics_dir / f"{name}_perpos.json"
        if metrics_path.exists():
            print(f"  {name}: cached")
            continue

        # For sequences > 16kb, use windowed extraction
        if len(seq) > 16000:
            print(f"  {name}: {len(seq)} bp — using windows...")
            cos1_all, cos3_all = await _extract_windowed(seq, url, layer, api_key, nim_url, name)
        else:
            print(f"  {name}: {len(seq)} bp — single extraction...", end=" ", flush=True)
            cos1_all, cos3_all = await _extract_single(seq, url, layer, api_key, nim_url)
            print("OK")

        if cos1_all is not None:
            # Save per-position cosine values
            result = {
                "cos1": cos1_all.tolist(),
                "cos3": cos3_all.tolist(),
                "length": len(seq),
            }
            with open(metrics_path, "w") as f:
                json.dump(result, f)


async def _extract_single(seq, url, layer, api_key, nim_url):
    """Extract per-position embeddings for a single sequence."""
    import httpx

    headers = {"Content-Type": "application/json"}
    if not nim_url:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"sequence": seq, "output_layers": [layer]}

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, headers=headers, timeout=300, follow_redirects=True)
        resp.raise_for_status()
        data = resp.json()

        raw = base64.b64decode(data["data"])
        npz = np.load(io.BytesIO(raw))
        emb = npz[f"{layer}.output"]
        if emb.ndim == 3:
            if emb.shape[0] == 1: emb = emb.squeeze(0)
            elif emb.shape[1] == 1: emb = emb.squeeze(1)

        return _compute_cosines(emb)


async def _extract_windowed(seq, url, layer, api_key, nim_url, name):
    """Extract per-position embeddings using overlapping windows."""
    import httpx

    window = 16000
    step = 12000

    cos1_sum = np.zeros(len(seq))
    cos3_sum = np.zeros(len(seq))
    count = np.zeros(len(seq))

    headers = {"Content-Type": "application/json"}
    if not nim_url:
        headers["Authorization"] = f"Bearer {api_key}"

    n_windows = (len(seq) - window) // step + 2
    done = 0

    async with httpx.AsyncClient() as client:
        for start in range(0, len(seq) - window + 1, step):
            end = min(start + window, len(seq))
            window_seq = seq[start:end]

            payload = {"sequence": window_seq, "output_layers": [layer]}
            for attempt in range(3):
                try:
                    resp = await client.post(url, json=payload, headers=headers,
                                             timeout=300, follow_redirects=True)
                    if resp.status_code == 429:
                        await asyncio.sleep(30)
                        continue
                    resp.raise_for_status()
                    data = resp.json()

                    raw = base64.b64decode(data["data"])
                    npz = np.load(io.BytesIO(raw))
                    emb = npz[f"{layer}.output"]
                    if emb.ndim == 3:
                        if emb.shape[0] == 1: emb = emb.squeeze(0)
                        elif emb.shape[1] == 1: emb = emb.squeeze(1)

                    c1, c3 = _compute_cosines(emb)
                    wlen = end - start
                    cos1_sum[start:end] += c1[:wlen]
                    cos3_sum[start:end] += c3[:wlen]
                    count[start:end] += 1
                    done += 1
                    print(f"    window {done}/{n_windows}: {start}-{end}")
                    break
                except Exception as e:
                    if attempt < 2:
                        await asyncio.sleep(5)
                    else:
                        print(f"    window {start}: FAILED ({e})")

        # Handle last window
        if count[-1] == 0:
            last_start = len(seq) - window
            if last_start >= 0:
                payload = {"sequence": seq[last_start:], "output_layers": [layer]}
                resp = await client.post(url, json=payload, headers=headers,
                                         timeout=300, follow_redirects=True)
                resp.raise_for_status()
                data = resp.json()
                raw = base64.b64decode(data["data"])
                npz = np.load(io.BytesIO(raw))
                emb = npz[f"{layer}.output"]
                if emb.ndim == 3:
                    if emb.shape[0] == 1: emb = emb.squeeze(0)
                    elif emb.shape[1] == 1: emb = emb.squeeze(1)
                c1, c3 = _compute_cosines(emb)
                cos1_sum[last_start:] += c1
                cos3_sum[last_start:] += c3
                count[last_start:] += 1

    mask = count > 0
    cos1_avg = np.zeros(len(seq))
    cos3_avg = np.zeros(len(seq))
    cos1_avg[mask] = cos1_sum[mask] / count[mask]
    cos3_avg[mask] = cos3_sum[mask] / count[mask]

    return cos1_avg, cos3_avg


def _compute_cosines(emb):
    """Compute per-position cos1 and cos3."""
    n = len(emb)
    norms = np.linalg.norm(emb, axis=1)

    cos1 = np.zeros(n)
    cos3 = np.zeros(n)

    for i in range(n - 1):
        ni = norms[i]
        ni1 = norms[i + 1]
        if ni > 0 and ni1 > 0:
            cos1[i] = np.dot(emb[i], emb[i+1]) / (ni * ni1)

    for i in range(n - 3):
        ni = norms[i]
        ni3 = norms[i + 3]
        if ni > 0 and ni3 > 0:
            cos3[i] = np.dot(emb[i], emb[i+3]) / (ni * ni3)

    return cos1, cos3


def plot_results(annotations):
    """Plot per-position cosine profiles with exon annotations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_dir = OUT_DIR / "metrics"
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    for gene_info in GENES:
        name = gene_info["name"]
        metrics_path = metrics_dir / f"{name}_perpos.json"
        if not metrics_path.exists():
            continue

        with open(metrics_path) as f:
            data = json.load(f)

        cos1 = np.array(data["cos1"])
        cos3 = np.array(data["cos3"])

        # Smooth
        window = 50
        kernel = np.ones(window) / window
        cos1_s = np.convolve(cos1, kernel, mode="same")
        cos3_s = np.convolve(cos3, kernel, mode="same")
        inversion = cos3_s - cos1_s

        ann = annotations.get(name, {})
        exons = ann.get("exons", [])
        cds = ann.get("cds", [])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                                        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08})

        x = np.arange(len(cos1_s))

        # Top: cos1 vs cos3
        ax1.plot(x, cos1_s, color="#C62828", linewidth=0.5, alpha=0.7, label="cos(offset-1)")
        ax1.plot(x, cos3_s, color="#1565C0", linewidth=0.5, alpha=0.7, label="cos(offset-3)")
        ax1.set_ylabel("Cosine similarity")
        ax1.legend(loc="upper right", fontsize=7)
        ax1.set_title(f"{gene_info['species']} — {gene_info['gene']}", fontsize=10, fontweight="bold")

        # Bottom: inversion signal
        ax2.fill_between(x, 0, inversion, where=inversion > 0, alpha=0.6, color="#1565C0", label="Exon (coding)")
        ax2.fill_between(x, 0, inversion, where=inversion < 0, alpha=0.6, color="#C62828", label="Intron/intergenic")
        ax2.axhline(0, color="black", linewidth=0.3)
        ax2.set_ylabel("cos3 - cos1")
        ax2.set_xlabel(f"Position (bp)")
        ax2.legend(loc="upper right", fontsize=7)

        # Mark exons
        for exon in exons:
            for ax in [ax1, ax2]:
                ax.axvspan(exon["start"], exon["end"], alpha=0.1, color="#4CAF50", zorder=0)

        # Mark CDS
        for c in cds:
            ax2.plot([c["start"], c["end"]], [ax2.get_ylim()[0]] * 2,
                     color="#4CAF50", linewidth=3, solid_capstyle="butt")

        plt.savefig(fig_dir / f"{name}_exon_intron.png", dpi=200, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  {name}: saved figure")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nim-url", type=str, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Download sequences ===")
    sequences = download_sequences()

    print("\n=== Step 2: Download annotations ===")
    annotations = download_annotations()

    print(f"\n=== Step 3: Extract embeddings ===")
    asyncio.run(extract_embeddings(sequences, nim_url=args.nim_url))

    print(f"\n=== Step 4: Plot ===")
    plot_results(annotations)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
