import sys
import argparse
from pathlib import Path
import os
import torch
import logging
import pandas as pd
import warnings
import glob
import re
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# Suppress shapely warnings regarding oriented_envelope
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in oriented_envelope")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in oriented_envelope")

# Ensure segger is importable
# Reorganization: segger is now in tools/segger
# Script in pipeline/src -> parent=pipeline -> parent.parent=root -> tools/segger
SEGGER_REPO = Path(__file__).resolve().parent.parent.parent / "tools" / "segger"

SEGGER_SRC = SEGGER_REPO / "src"
if str(SEGGER_SRC) not in sys.path:
    sys.path.insert(0, str(SEGGER_SRC))

# ... imports ...

# ... functions ...

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs-dir", default="../data/processed/segger_data/segger_inputs")
    parser.add_argument("--datasets-dir", default="../data/processed/segger_data/segger_datasets")
    parser.add_argument("--models-dir", default="../data/processed/segger_data/segger_models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0) # Default to 0 to avoid OOM
    parser.add_argument("--sample", type=str, help="Specific sample name to process (overrides scanning inputs-dir)")
    args = parser.parse_args()
    
    inputs_dir = Path(args.inputs_dir)
    datasets_dir = Path(args.datasets_dir)
    models_dir = Path(args.models_dir)
    
    datasets_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    
    if args.sample:
        sample_dir = inputs_dir / args.sample
        if not sample_dir.exists():
             print(f"Error: Sample {args.sample} not found in {inputs_dir}")
             return
        samples = [sample_dir]
    else:
        samples = sorted([d for d in inputs_dir.iterdir() if d.is_dir()])
    
    if not samples:
        print("No samples found.")
        return

    print(f"Found {len(samples)} samples.")
    
    for sample_dir in samples:
        name = sample_dir.name
        print(f"\n=== Processing {name} ===")
        
        d_out = datasets_dir / name
        m_out = models_dir / name
        
        # 1. Create Dataset
        if (d_out / "train_tiles" / "processed").exists():
            print("  Dataset exists, skipping.")
        else:
            try:
                create_dataset(sample_dir, d_out, args)
            except Exception as e:
                print(f"  Failed to create dataset: {e}")
                import traceback
                traceback.print_exc()
                continue
            
        # 2. Train
        if (m_out / "lightning_logs").exists():
            print("  Model logs exist, skipping training (delete to retrain).")
        else:
            try:
                train_sample(d_out, m_out, sample_dir, args)
            except Exception as e:
                print(f"  Failed to train: {e}")
                import traceback
                traceback.print_exc()
                continue
            
        # 3. Predict
        try:
            predict_sample(d_out, m_out, m_out, sample_dir, args)
        except Exception as e:
            print(f"  Failed to predict: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
