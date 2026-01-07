# Snakemake workflow for parallel Segger training
# Usage: snakemake --cores 4

import glob
import os

# Updated paths for reorganized project
KIDNEY_DIR = "../data/processed/kidneys"
INPUTS_DIR = "../data/processed/segger_data/segger_inputs"
MODELS_DIR = "../data/processed/segger_data/segger_models"

# 1. Discover Kidneys
KIDNEY_ZARRS = glob.glob(f"{KIDNEY_DIR}/*.sdata.zarr")
KIDNEY_IDS = [os.path.basename(k).replace(".sdata.zarr", "") for k in KIDNEY_ZARRS]

if not KIDNEY_IDS:
    print(f"Warning: No kidneys found in '{KIDNEY_DIR}'. Run 'process_all_chips.py' first.")

rule all:
    input:
        expand(f"{MODELS_DIR}/{{kidney}}", kidney=KIDNEY_IDS)

# 2. Prepare Segger Inputs (Parallelizable)
rule prepare_inputs:
    input:
        f"{KIDNEY_DIR}/{{kidney}}.sdata.zarr"
    output:
        f"{INPUTS_DIR}/{{kidney}}/transcripts.parquet",
        f"{INPUTS_DIR}/{{kidney}}/boundaries.parquet",
        f"{INPUTS_DIR}/{{kidney}}/genes.parquet"
    params:
        out_dir = INPUTS_DIR
    shell:
        "python src/prepare_segger_inputs.py --input-file {input} --out-dir {params.out_dir}"

# 3. Train Model (Parallelizable)
rule train_model:
    input:
        f"{INPUTS_DIR}/{{kidney}}/transcripts.parquet"
    output:
        directory(f"{MODELS_DIR}/{{kidney}}")
    shell:
        "python src/run_segger_pipeline.py --sample {wildcards.kidney}"