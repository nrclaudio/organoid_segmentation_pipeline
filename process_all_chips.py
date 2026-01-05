import sys
import subprocess
from pathlib import Path
import os
import argparse

SEGMENTATION_DIR = Path(__file__).resolve().parent

# Check for data in current dir or parent
if list(SEGMENTATION_DIR.glob("realigned_*")):
    DATA_ROOT = SEGMENTATION_DIR
else:
    DATA_ROOT = SEGMENTATION_DIR.parent

# Use current python environment
SDATA_PYTHON = sys.executable

# Define chip order for consistent IDs
CHIP_ORDER = [
    "C04895D5", # Assuming this is chip 1 (replacing C04895F1)
    "C04689E2",
    "C04897F3",
    "D04923A6",
    "C04897C6",
    "D04923E2",
    "C04896A3",
]

def run_step(cmd):
    print(f"Running: {' '.join(str(x) for x in cmd)}")
    subprocess.check_call(cmd)

def main():
    parser = argparse.ArgumentParser(description="Process Stereo-seq chips: H5AD -> SpatialData -> Extract Kidneys")
    parser.add_argument("--chip", type=str, help="Specific chip ID to process (e.g., C04689E2). If omitted, processes all found chips.")
    args = parser.parse_args()

    realigned_dirs = sorted(list(DATA_ROOT.glob("realigned_*")))
    
    for chip_dir in realigned_dirs:
        chip_id = chip_dir.name.replace("realigned_", "")
        
        if args.chip and chip_id != args.chip:
            continue

        print(f"\nProcessing {chip_id}...")
        
        if chip_id not in CHIP_ORDER:
             print(f"Note: {chip_id} is not in the hardcoded CHIP_ORDER list.")
        
        # Paths
        # Find outs directory (might be directly under chip_dir or in a subfolder like <chip_id>_A)
        outs_dirs = list(chip_dir.glob("**/outs"))
        if not outs_dirs:
            print(f"Skipping {chip_id} (no outs dir found in {chip_dir})")
            continue
        outs_dir = outs_dirs[0] # Take the first one found
        print(f"Using outs dir: {outs_dir}")
            
        # Look for existing tissue H5AD
        h5ad_path = outs_dir / "feature_expression" / f"{chip_id}_tissue.h5ad"
        
        if not h5ad_path.exists():
             print(f"H5AD not found: {h5ad_path}")
             # Check for raw or cellbin h5ad as fallback?
             fallback = outs_dir / "feature_expression" / f"{chip_id}_cellbin.h5ad"
             if fallback.exists():
                 print(f"Found fallback: {fallback}")
                 h5ad_path = fallback
             else:
                 print(f"Skipping {chip_id} (No H5AD found)")
                 continue

        print(f"Using H5AD: {h5ad_path}")

        # Step 2: Chip SpatialData
        sdata_path = chip_dir / f"{chip_id}.sdata.zarr"
        img_path = outs_dir / "image" / f"{chip_id}_ssDNA_regist.tif"
        labels_path = outs_dir / "image" / f"{chip_id}_ssDNA_mask.tif"
        tissue_mask_path = outs_dir / "image" / f"{chip_id}_ssDNA_tissue_cut.tif"
        
        if not sdata_path.exists():
             cmd = [
                SDATA_PYTHON, str(SEGMENTATION_DIR / "sto_to_sdata.py"),
                "--h5ad", str(h5ad_path),
                "--image", str(img_path),
                "--labels", str(labels_path),
                "--sample-id", chip_id,
                "--out", str(sdata_path)
            ]
             if tissue_mask_path.exists():
                 cmd.extend(["--tissue-mask", str(tissue_mask_path)])
             
             run_step(cmd)
        else:
             print(f"SpatialData already exists: {sdata_path}")
             
        # Step 3: Extract Kidneys
        # Determine chip index
        try:
            chip_idx = CHIP_ORDER.index(chip_id) + 1
        except ValueError:
            print(f"Warning: {chip_id} not in CHIP_ORDER, assigning 99")
            chip_idx = 99
            
        output_kidneys = SEGMENTATION_DIR / "kidneys"
        output_kidneys.mkdir(exist_ok=True)
        
        # Check if kidneys already exist
        kidneys_exist = True
        for i in range(1, 4):
            k_path = output_kidneys / f"{chip_idx:02d}_{chip_id}_kidney{i}.sdata.zarr"
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
                "--out-dir", str(output_kidneys),
                "--prefix", f"{chip_idx:02d}_{chip_id}",
                "--margin", "50"
            ]
            run_step(cmd)

if __name__ == "__main__":
    main()
