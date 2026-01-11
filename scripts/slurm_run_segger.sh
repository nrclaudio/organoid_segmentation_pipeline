#!/bin/bash
#SBATCH --job-name=segger_run
#SBATCH --output=logs/segger_%A_%a.out
#SBATCH --error=logs/segger_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=0-20  # 21 Samples found in data/processed/segger_data/segger_inputs

# ==============================================================================
# Run Segger Pipeline on HPC
# ==============================================================================

# 1. Environment Setup
module purge || true

echo "conda"
module add tools/miniconda/python3.8/4.9.2

hostname
echo "Cuda devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Initialize Conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate segger_env

# Dynamically add pip-installed NVIDIA libraries to LD_LIBRARY_PATH.
# This is necessary because CuPy and Torch look for CUDA runtimes in the system path by default.
export LD_LIBRARY_PATH=$(find $(python -c "import site; print(site.getsitepackages()[0])")/nvidia -maxdepth 2 -type d -name "lib" | paste -sd ":" -):$LD_LIBRARY_PATH

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
python -c "import torch; import cupy; print(f'Torch CUDA: {torch.cuda.is_available()}'); print(f'CuPy version: {cupy.__version__}')"


# 2. Setup Paths
# Navigate to the pipeline directory if not already there
# cd $(dirname "$0")/..

# Create logs directory
mkdir -p logs

# 3. Identify Sample
# List all subdirectories in the correct inputs directory
INPUTS_DIR="../data/processed/segger_data/segger_inputs"
SAMPLES=($(ls -d $INPUTS_DIR/*/ | xargs -n 1 basename | sort))
NUM_SAMPLES=${#SAMPLES[@]}

# Get current sample based on Array Task ID
SAMPLE_IDX=$SLURM_ARRAY_TASK_ID

# Validation
if [ "$SAMPLE_IDX" -ge "$NUM_SAMPLES" ]; then
    echo "Error: Array index $SAMPLE_IDX exceeds number of samples ($NUM_SAMPLES)."
    exit 1
fi

SAMPLE=${SAMPLES[$SAMPLE_IDX]}

echo "=================================================="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SAMPLE_IDX"
echo "Processing Sample: $SAMPLE"
echo "=================================================="

# 4. Run Pipeline
# Note: We set --workers 4 to utilize the requested CPUs. 
# If you encounter OOM errors, reduce workers to 0.
python -u src/run_segger_pipeline.py \
    --inputs-dir ../data/processed/segger_data/segger_inputs \
    --datasets-dir ../data/processed/segger_data/segger_datasets \
    --models-dir ../data/processed/segger_data/segger_models \
    --sample "$SAMPLE" \
    --workers 4 \
    --epochs 10

echo "Job finished for $SAMPLE"