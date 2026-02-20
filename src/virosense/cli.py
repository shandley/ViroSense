"""Command-line interface for virosense."""

import click

from virosense import __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="virosense")
def main():
    """virosense - Multi-modal viral detection and characterization.

    Combines DNA-level (Evo2) and protein-level (ProstT5/vHold) analysis
    for viral sequence detection, annotation, and classification.
    """
    pass


@main.command()
@click.option("-i", "--input", "input_file", required=True,
              type=click.Path(exists=True),
              help="Input FASTA file with metagenomic contigs")
@click.option("-o", "--output", required=True, type=click.Path(),
              help="Output directory")
@click.option("--backend", type=click.Choice(["local", "nim", "modal"]),
              default="nim", help="Evo2 inference backend (default: nim)")
@click.option("--model", type=click.Choice(["evo2_1b_base", "evo2_7b"]),
              default="evo2_7b", help="Evo2 model (default: evo2_7b)")
@click.option("--threshold", default=0.5, type=float,
              help="Viral classification threshold (default: 0.5)")
@click.option("--min-length", default=500, type=int,
              help="Minimum contig length in bp (default: 500)")
@click.option("--batch-size", default=16, type=int,
              help="Batch size for embedding extraction (default: 16)")
@click.option("-t", "--threads", default=4, type=int,
              help="Number of threads (default: 4)")
@click.option("--layer", default="blocks.28.mlp.l3",
              help="Evo2 layer for embedding extraction")
@click.option("--cache-dir", type=click.Path(), default=None,
              help="Directory to cache embeddings")
@click.option("--nim-url", default=None,
              help="Self-hosted NIM container URL (e.g. http://localhost:8000)")
def detect(input_file, output, backend, model, threshold,
           min_length, batch_size, threads, layer, cache_dir, nim_url):
    """Detect viral sequences in metagenomic contigs.

    Uses Evo2 DNA embeddings to classify contigs as viral or cellular.
    """
    from virosense.subcommands.detect import run_detect
    run_detect(
        input_file=input_file,
        output_dir=output,
        backend=backend,
        model=model,
        threshold=threshold,
        min_length=min_length,
        batch_size=batch_size,
        threads=threads,
        layer=layer,
        cache_dir=cache_dir,
        nim_url=nim_url,
    )


@main.command()
@click.option("-i", "--input", "input_file", required=True,
              type=click.Path(exists=True),
              help="Input viral contigs FASTA")
@click.option("--orfs", required=True, type=click.Path(exists=True),
              help="ORF predictions (GFF3, prodigal, or FASTA)")
@click.option("-o", "--output", required=True, type=click.Path(),
              help="Output directory")
@click.option("--backend", type=click.Choice(["local", "nim", "modal"]),
              default="nim", help="Evo2 inference backend (default: nim)")
@click.option("--model", default="evo2_7b",
              help="Evo2 model (default: evo2_7b)")
@click.option("--window", default=2000, type=int,
              help="Genomic context window size in bp (default: 2000)")
@click.option("--vhold-output", type=click.Path(),
              help="vHold annotation output to merge (optional)")
@click.option("-t", "--threads", default=4, type=int,
              help="Number of threads (default: 4)")
@click.option("--layer", default="blocks.28.mlp.l3",
              help="Evo2 layer for embedding extraction")
@click.option("--cache-dir", type=click.Path(), default=None,
              help="Directory to cache embeddings")
@click.option("--nim-url", default=None,
              help="Self-hosted NIM container URL (e.g. http://localhost:8000)")
def context(input_file, orfs, output, backend, model, window,
            vhold_output, threads, layer, cache_dir, nim_url):
    """Annotate ORFs with genomic context from Evo2 embeddings.

    Combines DNA-level context (Evo2) with protein-level annotation (vHold)
    for enhanced functional prediction.
    """
    from virosense.subcommands.context import run_context
    run_context(
        input_file=input_file,
        orfs_file=orfs,
        output_dir=output,
        backend=backend,
        model=model,
        window_size=window,
        vhold_output=vhold_output,
        threads=threads,
        layer=layer,
        cache_dir=cache_dir,
        nim_url=nim_url,
    )


@main.command()
@click.option("-i", "--input", "input_file", required=True,
              type=click.Path(exists=True),
              help="Input FASTA of unclassified viral sequences")
@click.option("-o", "--output", required=True, type=click.Path(),
              help="Output directory")
@click.option("--backend", type=click.Choice(["local", "nim", "modal"]),
              default="nim", help="Evo2 inference backend (default: nim)")
@click.option("--model", default="evo2_7b",
              help="Evo2 model (default: evo2_7b)")
@click.option("--mode", type=click.Choice(["dna", "protein", "multi"]),
              default="multi",
              help="Embedding modality for clustering (default: multi)")
@click.option("--algorithm", type=click.Choice(["hdbscan", "leiden", "kmeans"]),
              default="hdbscan",
              help="Clustering algorithm (default: hdbscan)")
@click.option("--min-cluster-size", default=5, type=int,
              help="Minimum cluster size (default: 5)")
@click.option("--n-clusters", default=None, type=int,
              help="Number of clusters (kmeans only; auto-estimated if omitted)")
@click.option("-t", "--threads", default=4, type=int,
              help="Number of threads (default: 4)")
@click.option("--vhold-embeddings", type=click.Path(),
              help="Pre-computed ProstT5 embeddings from vHold (optional)")
@click.option("--layer", default="blocks.28.mlp.l3",
              help="Evo2 layer for embedding extraction")
@click.option("--cache-dir", type=click.Path(), default=None,
              help="Directory to cache embeddings")
@click.option("--pca-dims", default=0, type=int,
              help="PCA dimensions before clustering (0=auto 90pct variance, -1=disable)")
@click.option("--nim-url", default=None,
              help="Self-hosted NIM container URL (e.g. http://localhost:8000)")
def cluster(input_file, output, backend, model, mode, algorithm,
            min_cluster_size, n_clusters, threads, vhold_embeddings,
            layer, cache_dir, pca_dims, nim_url):
    """Cluster unclassified viral sequences using multi-modal embeddings.

    Combines DNA (Evo2) and protein (ProstT5) embeddings to organize
    viral dark matter into putative families.
    """
    from virosense.subcommands.cluster import run_cluster
    run_cluster(
        input_file=input_file,
        output_dir=output,
        backend=backend,
        model=model,
        mode=mode,
        algorithm=algorithm,
        min_cluster_size=min_cluster_size,
        n_clusters=n_clusters,
        threads=threads,
        vhold_embeddings=vhold_embeddings,
        layer=layer,
        cache_dir=cache_dir,
        pca_dims=None if pca_dims < 0 else pca_dims,
        nim_url=nim_url,
    )


@main.command()
@click.option("-i", "--input", "input_file", required=True,
              type=click.Path(exists=True),
              help="Input FASTA of DNA sequences")
@click.option("--labels", required=True, type=click.Path(exists=True),
              help="TSV file with sequence_id and label columns")
@click.option("-o", "--output", required=True, type=click.Path(),
              help="Output directory for model and predictions")
@click.option("--backend", type=click.Choice(["local", "nim", "modal"]),
              default="nim", help="Evo2 inference backend (default: nim)")
@click.option("--model", default="evo2_7b",
              help="Evo2 model for embeddings (default: evo2_7b)")
@click.option("--task", type=click.Choice(["viral_vs_cellular", "family", "host"]),
              default="viral_vs_cellular",
              help="Classification task (default: viral_vs_cellular)")
@click.option("--epochs", default=50, type=int,
              help="Training epochs (default: 50)")
@click.option("--lr", default=1e-3, type=float,
              help="Learning rate (default: 1e-3)")
@click.option("--val-split", default=0.2, type=float,
              help="Validation split fraction (default: 0.2)")
@click.option("--predict", type=click.Path(exists=True), default=None,
              help="FASTA file to predict on (skip training)")
@click.option("--classifier-model", type=click.Path(exists=True), default=None,
              help="Pre-trained classifier model for prediction")
@click.option("--layer", default="blocks.28.mlp.l3",
              help="Evo2 layer for embedding extraction")
@click.option("--cache-dir", type=click.Path(), default=None,
              help="Directory to cache embeddings")
@click.option("--nim-url", default=None,
              help="Self-hosted NIM container URL (e.g. http://localhost:8000)")
def classify(input_file, labels, output, backend, model, task,
             epochs, lr, val_split, predict, classifier_model,
             layer, cache_dir, nim_url):
    """Train or apply a discriminative viral classifier on Evo2 embeddings.

    Trains a classification head on frozen Evo2 embeddings to predict
    viral family, host range, or viral vs cellular origin.
    """
    from virosense.subcommands.classify import run_classify
    run_classify(
        input_file=input_file,
        labels_file=labels,
        output_dir=output,
        backend=backend,
        model=model,
        task=task,
        epochs=epochs,
        lr=lr,
        val_split=val_split,
        predict_file=predict,
        classifier_model_path=classifier_model,
        layer=layer,
        cache_dir=cache_dir,
        nim_url=nim_url,
    )


@main.command()
@click.option("-i", "--input", "input_file", required=True,
              type=click.Path(exists=True),
              help="Input FASTA of bacterial chromosomes/contigs")
@click.option("-o", "--output", required=True, type=click.Path(),
              help="Output directory")
@click.option("--backend", type=click.Choice(["local", "nim", "modal"]),
              default="nim", help="Evo2 inference backend (default: nim)")
@click.option("--model", type=click.Choice(["evo2_1b_base", "evo2_7b"]),
              default="evo2_7b", help="Evo2 model (default: evo2_7b)")
@click.option("--threshold", default=0.5, type=float,
              help="Viral score threshold (default: 0.5)")
@click.option("--window-size", default=5000, type=int,
              help="Sliding window size in bp (default: 5000)")
@click.option("--step-size", default=2000, type=int,
              help="Step size between windows in bp (default: 2000)")
@click.option("--min-region-length", default=5000, type=int,
              help="Minimum prophage region length in bp (default: 5000)")
@click.option("--merge-gap", default=3000, type=int,
              help="Max gap to merge adjacent regions in bp (default: 3000)")
@click.option("--batch-size", default=16, type=int,
              help="Batch size for embedding extraction (default: 16)")
@click.option("--layer", default="blocks.28.mlp.l3",
              help="Evo2 layer for embedding extraction")
@click.option("--cache-dir", type=click.Path(), default=None,
              help="Directory to cache embeddings")
@click.option("--classifier-model", type=click.Path(exists=True),
              default=None,
              help="Pre-trained classifier model (default: reference model)")
@click.option("--nim-url", default=None,
              help="Self-hosted NIM container URL (e.g. http://localhost:8000)")
@click.option("--scan-mode", type=click.Choice(["adaptive", "full"]),
              default="adaptive",
              help="Scanning strategy: adaptive (two-pass, default) or full (single-pass)")
@click.option("--coarse-window-size", default=15000, type=int,
              help="Coarse pass window size in bp (default: 15000, adaptive only)")
@click.option("--coarse-step-size", default=10000, type=int,
              help="Coarse pass step size in bp (default: 10000, adaptive only)")
@click.option("--coarse-threshold", default=0.3, type=float,
              help="Score threshold for coarse-pass hits (default: 0.3, adaptive only)")
@click.option("--margin", default=20000, type=int,
              help="Margin in bp around coarse hits for fine pass (default: 20000)")
def prophage(input_file, output, backend, model, threshold,
             window_size, step_size, min_region_length, merge_gap,
             batch_size, layer, cache_dir, classifier_model, nim_url,
             scan_mode, coarse_window_size, coarse_step_size,
             coarse_threshold, margin):
    """Detect prophage regions in bacterial chromosomes.

    Scans input sequences with a sliding window, scores each window
    using a trained viral classifier, and merges consecutive high-scoring
    windows into prophage region calls. Outputs TSV and BED files.

    Default mode is adaptive two-pass scanning: a coarse pass identifies
    candidate regions, then a fine pass scans only those regions. This
    reduces API calls ~5x for typical bacterial chromosomes. Use
    --scan-mode full for single-pass scanning at fine resolution.
    """
    from virosense.subcommands.prophage import run_prophage
    run_prophage(
        input_file=input_file,
        output_dir=output,
        backend=backend,
        model=model,
        threshold=threshold,
        window_size=window_size,
        step_size=step_size,
        min_region_length=min_region_length,
        merge_gap=merge_gap,
        batch_size=batch_size,
        layer=layer,
        cache_dir=cache_dir,
        classifier_model=classifier_model,
        nim_url=nim_url,
        scan_mode=scan_mode,
        coarse_window_size=coarse_window_size,
        coarse_step_size=coarse_step_size,
        coarse_threshold=coarse_threshold,
        margin=margin,
    )


@main.command("build-reference")
@click.option("-i", "--input", "input_file", required=True,
              type=click.Path(exists=True),
              help="Input FASTA with labeled viral + cellular sequences")
@click.option("--labels", required=True, type=click.Path(exists=True),
              help="TSV file: sequence_id<tab>label (0=cellular, 1=viral)")
@click.option("-o", "--output", required=True, type=click.Path(),
              help="Output directory for model and metrics")
@click.option("--backend", type=click.Choice(["local", "nim", "modal"]),
              default="nim", help="Evo2 inference backend (default: nim)")
@click.option("--model", default="evo2_7b",
              help="Evo2 model (default: evo2_7b)")
@click.option("--layer", default="blocks.28.mlp.l3",
              help="Evo2 layer for embedding extraction")
@click.option("--epochs", default=200, type=int,
              help="Training epochs (default: 200)")
@click.option("--lr", default=1e-3, type=float,
              help="Learning rate (default: 1e-3)")
@click.option("--val-split", default=0.2, type=float,
              help="Validation split fraction (default: 0.2)")
@click.option("--install", is_flag=True, default=False,
              help="Copy trained model to default location (~/.virosense/models/)")
@click.option("--batch-size", default=16, type=int,
              help="Batch size for embedding extraction (default: 16)")
@click.option("--cache-dir", type=click.Path(), default=None,
              help="Directory to cache embeddings")
@click.option("--nim-url", default=None,
              help="Self-hosted NIM container URL (e.g. http://localhost:8000)")
def build_reference(input_file, labels, output, backend, model, layer,
                    epochs, lr, val_split, install, batch_size, cache_dir,
                    nim_url):
    """Build a reference classifier for viral detection.

    Takes a FASTA file of labeled viral and cellular sequences plus a
    labels TSV, extracts Evo2 embeddings, and trains a classifier.

    The labels file should be tab-separated with two columns:
    sequence_id and label (0 for cellular, 1 for viral).
    """
    from virosense.subcommands.build_reference import run_build_reference
    run_build_reference(
        input_file=input_file,
        labels_file=labels,
        output_dir=output,
        backend=backend,
        model=model,
        layer=layer,
        epochs=epochs,
        lr=lr,
        val_split=val_split,
        install=install,
        batch_size=batch_size,
        cache_dir=cache_dir,
        nim_url=nim_url,
    )
