import sys
import subprocess
from pathlib import Path
import os
import argparse

# Update paths for reorganized project
SEGMENTATION_DIR = Path(__file__).resolve().parent
DATA_ROOT = SEGMENTATION_DIR.parent / "data" / "processed" / "realigned"
OUTPUT_KIDNEYS = SEGMENTATION_DIR.parent / "data" / "processed" / "kidneys"

# Use current python environment
SDATA_PYTHON = sys.executable

# ... CHIP_ORDER ...

# ... run_step ...

def main():
    parser = argparse.ArgumentParser(description="Process Stereo-seq chips: H5AD -> SpatialData -> Extract Kidneys")
    parser.add_argument("--chip", type=str, help="Specific chip ID to process (e.g., C04689E2). If omitted, processes all found chips.")
    args = parser.parse_args()

    # Realigned directories are now directly inside DATA_ROOT
    realigned_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir() and d.name.startswith("realigned_")])
    
    for chip_dir in realigned_dirs:
        # ... logic ...
            
        # ... logic ...

        # Step 2: Chip SpatialData
        sdata_path = chip_dir / f"{chip_id}.sdata.zarr"
        # ... logic ...
             
        # Step 3: Extract Kidneys
        # Determine chip index
        try:
            chip_idx = CHIP_ORDER.index(chip_id) + 1
        except ValueError:
            print(f"Warning: {chip_id} not in CHIP_ORDER, assigning 99")
            chip_idx = 99
            
        OUTPUT_KIDNEYS.mkdir(parents=True, exist_ok=True)
        
        # Check if kidneys already exist
        kidneys_exist = True
        for i in range(1, 4):
            k_path = OUTPUT_KIDNEYS / f"{chip_idx:02d}_{chip_id}_kidney{i}.sdata.zarr"
            if not k_path.exists():
                kidneys_exist = False
                break
        
        if kidneys_exist:
            print(f"Kidneys already extracted for {chip_id}. Skipping.")
        else:
            print(f"Extracting kidneys for {chip_id} (Index {chip_idx})...")
            # Use local robust extraction script
            cmd = [
                SDATA_PYTHON, str(SEGMENTATION_DIR / "extract_kidneys_robust.py"),
                "--sdata", str(sdata_path),
                "--out-dir", str(OUTPUT_KIDNEYS),
                "--prefix", f"{chip_idx:02d}_{chip_id}",
                "--margin", "50"
            ]
            run_step(cmd)

if __name__ == "__main__":
    main()
