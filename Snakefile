# Snakemake workflow for parallel Segger training
# Usage: snakemake --cores 4

import glob
import os

# 1. Discover Kidneys
# Expects 'kidneys/' directory to be populated by 'process_all_chips.py'
KIDNEY_ZARRS = glob.glob("kidneys/*.sdata.zarr")
KIDNEY_IDS = [os.path.basename(k).replace(".sdata.zarr", "") for k in KIDNEY_ZARRS]

if not KIDNEY_IDS:
    print("Warning: No kidneys found in 'kidneys/'. Run 'process_all_chips.py' first.")

rule all:
    input:
        expand("segger_models/{kidney}", kidney=KIDNEY_IDS)

# 2. Prepare Segger Inputs (Parallelizable)
rule prepare_inputs:
    input:
        "kidneys/{kidney}.sdata.zarr"
    output:
        "segger_inputs/{kidney}/transcripts.parquet",
        "segger_inputs/{kidney}/boundaries.parquet",
        "segger_inputs/{kidney}/genes.parquet"
    params:
        out_dir = "segger_inputs"
    shell:
        "python prepare_segger_inputs.py --input-file {input} --out-dir {params.out_dir}"

# 3. Train Model (Parallelizable)
rule train_model:
    input:
        "segger_inputs/{kidney}/transcripts.parquet"
    output:
        directory("segger_models/{kidney}")
    shell:
        "python run_segger_pipeline.py --sample {wildcards.kidney}"
