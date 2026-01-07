#!/bin/bash
#SBATCH --job-name=sto_tissue_h5ad
#SBATCH --array=1-8
#SBATCH --output=sto_tissue_h5ad_%A_%a.out
#SBATCH --error=sto_tissue_h5ad_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4

# ==============================================================================
# Pre-processing Script: GEF -> H5AD
# ==============================================================================
# This script converts Stereo-seq GEF files to AnnData H5AD format.
# Requirements:
#   - Linux Environment (HPC)
#   - 'stereopy' installed (e.g., conda activate stereo38)
# ==============================================================================

# 1. Environment Setup
module purge > /dev/null 2>&1
# Adjust to your specific module/conda setup
# module add tools/miniconda/python3.8/4.9.2
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stereo38

# 2. Configuration
# Root directory containing the realigned chip folders
DATA_ROOT="/exports/nieromics-hpc/cnovellarausell/organoid_tx_realigned_files"

# Path to the python conversion script (sto_export_h5ad.py)
# Assuming this script is run from project root or similar, but better to be explicit or relative
# We assume the user submits this from the pipeline/scripts folder or root
SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT=$(readlink -f "$SCRIPT_DIR/../..")
CONVERT_SCRIPT="$PROJECT_ROOT/pipeline/src/sto_export_h5ad.py"

if [[ ! -f "$CONVERT_SCRIPT" ]]; then
    # Fallback to hardcoded path if relative fails (HPC specific)
    CONVERT_SCRIPT="/exports/nieromics-hpc/cnovellarausell/organoid_tx/pipeline/src/sto_export_h5ad.py"
fi

cd "$DATA_ROOT"

# 3. Sample List
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

# 4. Locate Input File
CHIP_DIR="realigned_${sample}"
# Find the .tissue.gef file automatically
GEF_PATH=$(find "$CHIP_DIR" -name "${sample}.tissue.gef" | head -n 1)

if [[ -z "$GEF_PATH" ]]; then
    echo "ERROR: tissue GEF file not found for $sample in $CHIP_DIR" >&2
    exit 1
fi

# 5. Define Output
# Save .h5ad next to the .gef file
OUT_PATH="${GEF_PATH%.tissue.gef}_tissue.h5ad"

if [[ -f "$OUT_PATH" ]]; then
    echo "H5AD already exists: $OUT_PATH. Skipping."
    exit 0
fi

# 6. Run Conversion
python "$CONVERT_SCRIPT" \
    --gef "$GEF_PATH" \
    --out "$OUT_PATH"

echo "Done sample: $sample"

