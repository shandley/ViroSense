#!/bin/bash
#
# RNA-seq validation for a single organism.
# Called by sbatch wrapper with organism-specific parameters.
#
# Usage: bash rnaseq_validate_organism.sh <organism> <genome_url> <region_chrom> <region_start> <region_end> <srr1> <srr2> <srr3>

set -euo pipefail

ORGANISM=$1
GENOME_URL=$2
REGION_CHROM=$3
REGION_START=$4
REGION_END=$5
shift 5
SRRS=("$@")

SCRATCH=/scratch/sahlab/shandley
WORK=$SCRATCH/virosense/rnaseq_validation/$ORGANISM
mkdir -p $WORK/fastq $WORK/genome $WORK/aligned

export PATH=$SCRATCH/miniforge/bin:$PATH
eval "$(conda shell.bash hook)"
conda activate rnaseq

echo "=== RNA-seq validation: $ORGANISM ==="
echo "  Genome: $GENOME_URL"
echo "  Region: $REGION_CHROM:$REGION_START-$REGION_END"
echo "  SRRs: ${SRRS[*]}"

# ── Step 1: Download genome ──
GENOME=$WORK/genome/genome.fa
if [ ! -f "$GENOME" ]; then
    echo "=== Downloading genome ==="
    curl -L -o $WORK/genome/genome.fna.gz "$GENOME_URL"
    gunzip -c $WORK/genome/genome.fna.gz > $GENOME
    echo "  $(grep -c '^>' $GENOME) sequences, $(wc -c < $GENOME) bytes"
fi

# ── Step 2: Build HISAT2 index ──
INDEX=$WORK/genome/index
if [ ! -f "${INDEX}.1.ht2" ]; then
    echo "=== Building HISAT2 index ==="
    hisat2-build -p 16 $GENOME $INDEX
fi

# ── Step 3: Download and align each SRR ──
for SRR in "${SRRS[@]}"; do
    BAM=$WORK/aligned/${SRR}.sorted.bam

    if [ -f "$BAM" ]; then
        echo "  Cached: $SRR alignment"
        continue
    fi

    FQ1=$WORK/fastq/${SRR}_1.fastq.gz
    FQ2=$WORK/fastq/${SRR}_2.fastq.gz

    # Download if needed
    if [ ! -f "$FQ1" ]; then
        echo "=== Downloading $SRR ==="
        cd $WORK/fastq
        fasterq-dump --split-files --threads 8 $SRR
        # Compress (fasterq-dump outputs uncompressed)
        for f in ${SRR}_*.fastq; do
            [ -f "$f" ] && gzip "$f"
        done
        cd -
    fi

    # Align
    echo "=== Aligning $SRR ==="
    if [ -f "$FQ2" ]; then
        hisat2 -p 16 -x $INDEX -1 $FQ1 -2 $FQ2 \
            2>$WORK/aligned/${SRR}_hisat2.log | \
            samtools sort -@ 8 -o $BAM
    else
        hisat2 -p 16 -x $INDEX -U $FQ1 \
            2>$WORK/aligned/${SRR}_hisat2.log | \
            samtools sort -@ 8 -o $BAM
    fi
    samtools index $BAM
    echo "  $(samtools flagstat $BAM | head -1)"

    # Clean up uncompressed fastq to save space
    rm -f $WORK/fastq/${SRR}_*.fastq
done

# ── Step 4: Merge BAMs ──
MERGED=$WORK/aligned/merged.sorted.bam
if [ ! -f "$MERGED" ]; then
    echo "=== Merging alignments ==="
    BAMS=$(ls $WORK/aligned/*.sorted.bam | grep -v merged)
    samtools merge -@ 8 $MERGED $BAMS
    samtools index $MERGED
fi

# ── Step 5: Extract coverage at scan region ──
echo "=== Computing coverage at $REGION_CHROM:$REGION_START-$REGION_END ==="
samtools depth -r "${REGION_CHROM}:${REGION_START}-${REGION_END}" $MERGED > $WORK/coverage_region.tsv
echo "  Coverage: $(wc -l < $WORK/coverage_region.tsv) positions"

# ── Step 6: Summary stats ──
echo "=== Alignment summary ==="
samtools flagstat $MERGED

echo "=== Done: $ORGANISM ==="
echo "Coverage file: $WORK/coverage_region.tsv"
