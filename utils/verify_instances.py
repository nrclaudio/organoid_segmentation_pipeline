import pandas as pd
from pathlib import Path
import sys

def verify_sample(sample_dir):
    sample_dir = Path(sample_dir)
    tx_path = sample_dir / "transcripts.parquet"
    bd_path = sample_dir / "boundaries.parquet"
    
    if not tx_path.exists() or not bd_path.exists():
        print(f"  Missing files in {sample_dir.name}")
        return False

    # Load data
    tx = pd.read_parquet(tx_path, columns=["cell_id", "overlaps_nucleus"])
    bd = pd.read_parquet(bd_path, columns=["boundary_id"])

    # Analysis
    unique_tx_cells = tx[tx["cell_id"] > 0]["cell_id"].nunique()
    unique_bd_ids = bd["boundary_id"].nunique()
    total_nuclear_tx = (tx["overlaps_nucleus"] == 1).sum()
    
    print(f"\n=== Sample: {sample_dir.name} ===")
    print(f"  Unique Boundary Polygons: {unique_bd_ids}")
    print(f"  Unique Cell IDs in Transcripts: {unique_tx_cells}")
    print(f"  Total Nuclear Transcripts: {total_nuclear_tx}")

    if unique_tx_cells == 0:
        print("  ❌ ERROR: No cell IDs found in transcripts!")
    elif unique_tx_cells == 1 and unique_bd_ids > 1:
        print("  ❌ ERROR: Only 1 Cell ID found (mask is still binary)!")
    elif unique_tx_cells != unique_bd_ids:
        print(f"  ⚠️  WARNING: Mismatch! (Some polygons might not have transcripts)")
    else:
        print("  ✅ SUCCESS: Instance IDs are correctly mapped.")
    
    return True

def main():
    inputs_dir = Path("organoid_segmentation_pipeline/segger_inputs")
    if not inputs_dir.exists():
        print(f"Directory not found: {inputs_dir}")
        return

    samples = sorted([d for d in inputs_dir.iterdir() if d.is_dir()])
    for s in samples:
        verify_sample(s)

if __name__ == "__main__":
    main()
