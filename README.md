# Organoid Segmentation Pipeline

This repository contains the pipeline for processing Stereo-seq organoid data, from raw GEF files to segmented Parquet datasets. It is designed to work within the larger project structure and utilizes the [Segger](https://github.com/nrclaudio/stereosegger) repository.

## Project Structure

This pipeline is intended to be located in the `pipeline/` directory of the project root:

```
project_root/
├── data/                  # Centralized data storage
│   ├── raw/               # Raw chip Zarrs
│   └── processed/
│       ├── kidneys/       # Extracted kidney slices
│       └── segger_data/   # Segger inputs, datasets, and models
├── tools/
│   └── segger/            # Segger library source
├── pipeline/              # This repository
│   ├── src/               # Python source code
│   ├── scripts/           # SLURM and shell execution scripts
│   ├── utils/             # Validation and helper scripts
│   └── Snakefile          # Snakemake workflow
└── scripts/               # Downstream analysis scripts
```

## Workflow

### 1. Extract Kidneys
Converts realigned chip data (H5AD + Images) into centered SpatialData crops for each kidney.
```bash
python src/process_all_chips.py
```
*Input: `data/processed/realigned/`*
*Output: `data/processed/kidneys/`*

### 2. Prepare Segger Inputs
Converts kidney Zarrs into the Parquet format required by Segger, including boundary and transcript graphs.
```bash
python src/prepare_segger_inputs.py
```
*Input: `data/processed/kidneys/`*
*Output: `data/processed/segger_data/segger_inputs/`*

### 3. Run Segmentation / Training
Runs the Segger GNN pipeline to train models and predict cell assignments.
```bash
python src/run_segger_pipeline.py
```
*Input: `data/processed/segger_data/segger_inputs/`*
*Output: `data/processed/segger_data/segger_models/`*

---

### Parallel Execution (Snakemake)
To parallelize steps 2 and 3:
```bash
snakemake --cores 4
```

### HPC Execution (SLURM)
Scripts for HPC clusters (e.g., GEF to H5AD conversion) are located in `scripts/`.
```bash
sbatch scripts/slurm_export_h5ad.sh
```

## Requirements

- Python 3.10+

- [SpatialData](https://spatialdata.scverse.org/)

- [Segger](https://github.com/nrclaudio/stereosegger) (must be placed in `tools/segger` relative to the project root)



## Server Setup (GPU Acceleration)

To run the segmentation pipeline with GPU support, you must install the correct version of `cupy` matching your server's CUDA toolkit.



1. **Check CUDA version:** `nvidia-smi` or `nvcc --version`

2. **Install cupy:**

   - For CUDA 12.x: `pip install cupy-cuda12x`

   - For CUDA 11.x: `pip install cupy-cuda11x`
