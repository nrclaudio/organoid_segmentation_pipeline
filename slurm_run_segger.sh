#!/bin/bash
#SBATCH --job-name=segger_run
#SBATCH --output=logs/segger_%A_%a.out
#SBATCH --error=logs/segger_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=0-23  # ADJUST THIS: 0 to (Number_of_Samples - 1)

# ==============================================================================
# Run Segger Pipeline on HPC
# ==============================================================================

# 1. Environment Setup
module purge || true
# Load necessary modules (Uncomment and adjust if your HPC requires module loading for CUDA)
# module load cuda/11.7 

# Initialize Conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate segger_env  # <--- CHANGE THIS to your actual environment name

# 2. Setup Paths
# Navigate to the script's directory
cd $(dirname "$0")

# Create logs directory
mkdir -p logs

# 3. Identify Sample
# List all subdirectories in segger_inputs/ to get sample names
SAMPLES=($(ls -d segger_inputs/*/ | xargs -n 1 basename | sort))
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
python run_segger_pipeline.py \
    --inputs-dir segger_inputs \
    --datasets-dir segger_datasets \
    --models-dir segger_models \
    --sample "$SAMPLE" \
    --workers 4 \
    --epochs 10

echo "Job finished for $SAMPLE"
