import argparse
import sys
import subprocess
from pathlib import Path
import spatialdata as sd
import scanpy as sc
import os
import tifffile
import numpy as np
from scipy.ndimage import label

# Adjust path to find segger if not installed
# Check current dir or sibling dir
SEGGER_REPO = Path(__file__).resolve().parent / "segger"
if not SEGGER_REPO.exists():
    SEGGER_REPO = Path(__file__).resolve().parent.parent / "segger"

SEGGER_SRC = SEGGER_REPO / "src"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kidneys-dir", default="kidneys", help="Directory containing extracted kidney .sdata.zarr files")
    parser.add_argument("--out-dir", default="segger_inputs", help="Output directory for Segger data")
    parser.add_argument("--segger-python", help="Python executable for Segger conversion tool (if different from current)")
    args = parser.parse_args()
    
    kidneys_dir = Path(args.kidneys_dir)
    out_dir = Path(args.out_dir)
    
    segger_py = args.segger_python if args.segger_python else sys.executable
    out_dir.mkdir(exist_ok=True)
    
    # Find all kidney zarrs
    zarrs = sorted(list(kidneys_dir.glob("*_kidney*.sdata.zarr")))
    
    if not zarrs:
        print(f"No kidney zarrs found in {kidneys_dir}")
        return

    print(f"Found {len(zarrs)} kidneys to process.")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SEGGER_SRC) + ":" + env.get("PYTHONPATH", "")
    
    for z_path in zarrs:
        name = z_path.stem.replace(".sdata", "")
        sample_out = out_dir / name
        
        if sample_out.exists():
            print(f"Skipping {name} (output exists)")
            continue
            
        print(f"Processing {name}...")
        
        try:
            sdata = sd.read_zarr(z_path.resolve())
            
            if not sdata.tables:
                print(f"No tables found in {z_path}, skipping.")
                continue
            
            table_name = list(sdata.tables.keys())[0]
            adata = sdata.tables[table_name]
            
            # Save temp H5AD
            temp_h5ad = out_dir / f"{name}_temp.h5ad"
            adata.write_h5ad(temp_h5ad)
            
            # Extract labels if available (ssdna_mask)
            # Used for boundaries.parquet
            temp_labels = None
            if "ssdna_mask" in sdata.labels:
                lbl = sdata.labels["ssdna_mask"]
                # Handle multiscale
                if hasattr(lbl, 'keys') and 'scale0' in lbl:
                    ds = lbl['scale0']
                    var_name = list(ds.data_vars.keys())[0]
                    mask = ds[var_name].values
                else:
                    mask = lbl.values
                
                if mask.ndim == 3: mask = mask[0]
                
                # Label connected components so each nucleus has a unique ID
                mask, _ = label(mask)
                mask = mask.astype(np.int32)
                
                temp_labels = out_dir / f"{name}_labels_temp.tif"
                tifffile.imwrite(temp_labels, mask)
            
            # Construct command - Run as script file to avoid package init (avoids torch dependency)
            script_path = SEGGER_SRC / "segger" / "cli" / "convert_saw_h5ad_to_segger_parquet.py"
            
            cmd = [
                segger_py, str(script_path),
                "--h5ad", str(temp_h5ad),
                "--out_dir", str(sample_out),
                "--bin_pitch", "1.0",
                "--min_count", "1",
                # Use var_names if gene_name column not found
                "--gene_name_source", "gene_name" 
            ]
            
            if temp_labels:
                cmd.extend(["--labels_tif", str(temp_labels)])
                
            subprocess.check_call(cmd, env=env)
            
            # Cleanup
            if temp_h5ad.exists(): temp_h5ad.unlink()
            if temp_labels and temp_labels.exists(): temp_labels.unlink()
            
        except Exception as e:
            print(f"Failed to process {name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
