#!/bin/bash
#SBATCH --job-name=sto_tissue_h5ad
#SBATCH --array=1-8
#SBATCH --output=sto_tissue_h5ad_%A_%a.out
#SBATCH --error=sto_tissue_h5ad_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4

module purge > /dev/null 2>&1
module add tools/miniconda/python3.8/4.9.2

conda activate stereo38

cd /exports/nieromics-hpc/cnovellarausell/organoid_tx_realigned_files

samples=(
  "C04689E2"
  "C04895D5"
  "C04895F1"
  "C04896A3"
  "C04897C6"
  "C04897F3"
  "D04923A6"
  "D04923E2"
)

sample=${samples[$SLURM_ARRAY_TASK_ID-1]}

echo "Processing sample: $sample"

CHIP_DIR="realigned_${sample}"
GEF_PATH=$(find "$CHIP_DIR" -name "${sample}.tissue.gef" | head -n 1)

if [[ -z "$GEF_PATH" ]]; then
    echo "ERROR: tissue GEF file not found for $sample in $CHIP_DIR" >&2
    exit 1
fi

OUT_PATH="${GEF_PATH%.tissue.gef}_tissue.h5ad"

if [[ -f "$OUT_PATH" ]]; then
    echo "H5AD already exists: $OUT_PATH. Skipping."
    exit 0
fi

CONVERT_SCRIPT="/exports/nieromics-hpc/cnovellarausell/organoid_tx/segmentation/sto_export_h5ad.py"

python "$CONVERT_SCRIPT" \
    --gef "$GEF_PATH" \
    --out "$OUT_PATH"

echo "Done sample: $sample"
