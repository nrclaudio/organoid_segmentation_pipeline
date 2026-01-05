# Organoid Segmentation Pipeline

This repository contains the pipeline for processing Stereo-seq organoid data. It is designed to work side-by-side with the [Segger](https://github.com/nrclaudio/stereosegger) repository.

## Directory Structure

```
segmentation/              (Project Root)
├── segger/                (The Segger library code)
└── organoid_segmentation_pipeline/  (This pipeline code)
    ├── Dockerfile
    ├── process_all_chips.py
    └── ...
```

## Setup & Usage

### 1. Local (macOS)

Ensure you have the `segger_mac` environment set up (see `../segger/README.md`).

**Running the Pipeline:**
You can run these scripts from inside this directory. They will automatically look for `../segger` if needed.

```bash
cd organoid_segmentation_pipeline

# Step 1: Extract Kidneys
python process_all_chips.py

# Step 2: Convert Data
python prepare_segger_inputs.py

# Step 3: Run Segmentation
python run_segger_pipeline.py
```

### 2. Docker

To build the Docker image, you must run the build command **from the project root** (`segmentation/`) so that it can access both `segger` and the pipeline.

**Build:**
```bash
# From 'segmentation/' root:
docker build -t organoid-pipeline -f organoid_segmentation_pipeline/Dockerfile .
```

**Run:**
```bash
docker run -it -v $(pwd):/data organoid-pipeline
```