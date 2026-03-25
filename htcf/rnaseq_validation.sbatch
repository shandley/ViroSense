#!/bin/bash
#SBATCH --job-name=rnaseq_val
#SBATCH --partition=general
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/sahlab/shandley/virosense/logs/rnaseq_val_%j.out
#SBATCH --error=/scratch/sahlab/shandley/virosense/logs/rnaseq_val_%j.err

# RNA-seq validation of Evo2 false positive predictions
# Downloads P. patens RNA-seq, aligns to genome, checks coverage at FP positions
#
# Uses A100 or any GPU node (no NIM needed — pure bioinformatics)

set -euo pipefail
export PATH=$HOME/.local/bin:$PATH

SCRATCH=/scratch/sahlab/shandley
WORK=$SCRATCH/virosense/rnaseq_validation
mkdir -p $WORK/fastq $WORK/genome $WORK/aligned

# ── Step 1: Install tools if needed ──
if [ ! -f "$SCRATCH/miniforge/bin/conda" ]; then
    echo "=== Installing miniforge ==="
    curl -L -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash /tmp/miniforge.sh -b -p $SCRATCH/miniforge
    rm /tmp/miniforge.sh
fi
export PATH=$SCRATCH/miniforge/bin:$PATH

if [ ! -d "$SCRATCH/miniforge/envs/rnaseq" ]; then
    echo "=== Creating rnaseq environment ==="
    conda create -y -n rnaseq -c bioconda -c conda-forge \
        hisat2 samtools sra-tools bedtools biopython unzip
fi

eval "$(conda shell.bash hook)"
conda activate rnaseq

# ── Step 2: Download P. patens reference genome ──
GENOME=$WORK/genome/Ppatens_genome.fa
if [ ! -f "$GENOME" ]; then
    echo "=== Downloading P. patens genome from NCBI FTP ==="
    # Download full genome FASTA from RefSeq FTP (most reliable method)
    curl -L -o $WORK/genome/genome.fna.gz \
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/002/425/GCF_000002425.5_Phypa_V5/GCF_000002425.5_Phypa_V5_genomic.fna.gz"
    gunzip -c $WORK/genome/genome.fna.gz > $GENOME
    echo "  Downloaded $(grep -c '^>' $GENOME) sequences, $(wc -c < $GENOME) bytes"
fi

# ── Step 3: Build HISAT2 index ──
INDEX=$WORK/genome/ppatens
if [ ! -f "${INDEX}.1.ht2" ]; then
    echo "=== Building HISAT2 index ==="
    hisat2-build -p 16 $GENOME $INDEX
fi

# ── Step 4: Download RNA-seq data ──
# Use a subset of the JGI Plant Gene Atlas — pick 3 representative conditions
# These are from PRJNA681040 (JGI Plant Gene Atlas P. patens)
# Protonema, gametophore, sporophyte — 3 conditions, ~200M reads total
#
# Alternative: use SRA accessions from the Gene Atlas
# For speed, download just 2 representative libraries
# P. patens WT RNA-seq from PRJNA1415520 (3 replicates, ~20M PE reads each)
SRRS=("SRR37012529" "SRR37012531" "SRR37012532")
SRRNAMES=("wt_rep3" "wt_rep2" "wt_rep1")

for i in "${!SRRS[@]}"; do
    SRR=${SRRS[$i]}
    NAME=${SRRNAMES[$i]}
    FQ1=$WORK/fastq/${NAME}_1.fastq.gz
    FQ2=$WORK/fastq/${NAME}_2.fastq.gz

    if [ ! -f "$FQ1" ]; then
        echo "=== Downloading $SRR ($NAME) ==="
        cd $WORK/fastq
        fasterq-dump --split-files --threads 8 $SRR
        gzip ${SRR}_1.fastq ${SRR}_2.fastq 2>/dev/null || true
        mv ${SRR}_1.fastq.gz $FQ1 2>/dev/null || true
        mv ${SRR}_2.fastq.gz $FQ2 2>/dev/null || true
        cd -
        echo "  Downloaded $NAME"
    else
        echo "  Cached: $NAME"
    fi
done

# ── Step 5: Align with HISAT2 ──
for i in "${!SRRS[@]}"; do
    NAME=${SRRNAMES[$i]}
    BAM=$WORK/aligned/${NAME}.sorted.bam

    if [ ! -f "$BAM" ]; then
        echo "=== Aligning $NAME ==="
        FQ1=$WORK/fastq/${NAME}_1.fastq.gz
        FQ2=$WORK/fastq/${NAME}_2.fastq.gz

        if [ -f "$FQ2" ]; then
            hisat2 -p 16 -x $INDEX -1 $FQ1 -2 $FQ2 --rna-strandness RF \
                2>$WORK/aligned/${NAME}_hisat2.log | \
                samtools sort -@ 8 -o $BAM
        else
            hisat2 -p 16 -x $INDEX -U $FQ1 --rna-strandness R \
                2>$WORK/aligned/${NAME}_hisat2.log | \
                samtools sort -@ 8 -o $BAM
        fi
        samtools index $BAM
        echo "  Aligned: $(samtools flagstat $BAM | head -1)"
    else
        echo "  Cached: $NAME alignment"
    fi
done

# ── Step 6: Merge BAMs ──
MERGED=$WORK/aligned/merged.sorted.bam
if [ ! -f "$MERGED" ]; then
    echo "=== Merging alignments ==="
    samtools merge -@ 8 $MERGED $WORK/aligned/*.sorted.bam
    samtools index $MERGED
fi

# ── Step 7: Compute coverage at our scan region ──
echo "=== Computing coverage at scan region (16,665,000-17,165,000) ==="
# Our scan region on chr1
REGION_START=16665000
REGION_END=17165000

# Per-base coverage in our region
samtools depth -r "NC_037253.2:${REGION_START}-${REGION_END}" $MERGED > $WORK/coverage_region.tsv
echo "  Coverage file: $(wc -l < $WORK/coverage_region.tsv) positions"

# Overall alignment stats
echo "=== Alignment summary ==="
samtools flagstat $MERGED

echo "=== Done ==="
echo "Coverage file ready for analysis: $WORK/coverage_region.tsv"
