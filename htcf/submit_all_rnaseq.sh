#!/bin/bash
# Submit RNA-seq validation jobs for all 4 remaining organisms
# Run from HTCF login node

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIRO_HOME="/scratch/sahlab/shandley/virosense/ViroSense"

submit_job() {
    local NAME=$1
    local GENOME_URL=$2
    local CHROM=$3
    local START=$4
    local END=$5
    shift 5
    local SRRS=("$@")

    sbatch \
        --job-name="rv_${NAME}" \
        --partition=general \
        --cpus-per-task=16 \
        --mem=64G \
        --time=12:00:00 \
        --output="/scratch/sahlab/shandley/virosense/logs/rv_${NAME}_%j.out" \
        --error="/scratch/sahlab/shandley/virosense/logs/rv_${NAME}_%j.err" \
        --wrap="bash ${VIRO_HOME}/htcf/rnaseq_validate_organism.sh $NAME $GENOME_URL $CHROM $START $END ${SRRS[*]}"
}

# 1. Dictyostelium discoideum (social amoeba) — chr2
submit_job dictyostelium \
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/004/695/GCF_000004695.1_dicty_2.7/GCF_000004695.1_dicty_2.7_genomic.fna.gz" \
    NC_007088.5 3820124 4320124 \
    SRR31609131 SRR31609130 SRR31609129

# 2. Acropora millepora (staghorn coral) — chr1
submit_job acropora \
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/013/753/865/GCF_013753865.1_Amil_v2.1/GCF_013753865.1_Amil_v2.1_genomic.fna.gz" \
    NC_058066.1 9850000 10350000 \
    SRR1853208 SRR1853209 SRR1853198

# 3. Danaus plexippus (monarch butterfly) — chr2
submit_job danaus \
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/009/731/565/GCF_009731565.1_Dplex_v4/GCF_009731565.1_Dplex_v4_genomic.fna.gz" \
    NC_045808.1 5830000 6330000 \
    SRR12695265 SRR12695261 SRR14766482

# 4. Magallana gigas (Pacific oyster) — chr1
submit_job magallana \
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/963/853/765/GCF_963853765.1_xbMagGiga1.1/GCF_963853765.1_xbMagGiga1.1_genomic.fna.gz" \
    NC_088853.1 41855000 42355000 \
    SRR9089188 SRR9089191 SRR9089190

echo "All 4 jobs submitted."
