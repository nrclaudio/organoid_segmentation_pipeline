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
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import Trainer

# Suppress shapely warnings regarding oriented_envelope
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in oriented_envelope")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in oriented_envelope")

# Ensure segger is importable
SEGGER_REPO = Path(__file__).resolve().parent.parent.parent / "tools" / "segger"
SEGGER_SRC = SEGGER_REPO / "src"
if str(SEGGER_SRC) not in sys.path:
    sys.path.insert(0, str(SEGGER_SRC))

try:
    from segger.data.io import XeniumSample
    from segger.training.train import LitSegger
    from segger.training.segger_data_module import SeggerDataModule
    from segger.prediction.predict_parquet import segment, load_model
except ImportError as e:
    print(f"Error importing Segger modules: {e}")
    sys.exit(1)

def create_dataset(sample_dir, d_out, args):
    """Create a Segger-compatible dataset from parquets."""
    print(f"  Creating dataset in {d_out}...")
    xs = XeniumSample(verbose=True)
    xs.set_file_paths(
        transcripts_path=sample_dir / "transcripts.parquet",
        boundaries_path=sample_dir / "boundaries.parquet",
    )
    xs.set_metadata()
    
    # Save dataset with standard tile sizes
    xs.save_dataset_for_segger(
        processed_dir=d_out,
        x_size=220,
        y_size=220,
        d_x=200,
        d_y=200,
        margin_x=10,
        margin_y=10,
        compute_labels=True,
        num_workers=args.workers if args.workers > 0 else 1
    )

def train_sample(d_out, m_out, sample_dir, args):
    """Train a Segger model on the sample dataset."""
    print(f"  Training model in {m_out}...")
    dm = SeggerDataModule(
        data_dir=d_out,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    dm.setup()

    # Define GNN metadata
    metadata = (["tx", "bd"], [("tx", "belongs", "bd"), ("tx", "neighbors", "tx")])
    
    # Initialize model
    ls = LitSegger(
        num_tx_tokens=10000, 
        init_emb=8,
        hidden_channels=64,
        out_channels=16,
        heads=4,
        num_mid_layers=1,
        aggr="sum",
        metadata=metadata,
    )

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=args.epochs,
        default_root_dir=m_out,
        logger=CSVLogger(m_out),
        precision="16-mixed" if torch.cuda.is_available() else 32
    )

    trainer.fit(model=ls, datamodule=dm)

def predict_sample(d_out, m_out, results_dir, sample_dir, args):
    """Run prediction using the official Segger library function."""
    print(f"  Running prediction using Segger library...")
    
    # Detect Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Using device for inference: {device}")

    # Load the best/latest checkpoint
    ckpt_dir = m_out / "lightning_logs" / "version_0" / "checkpoints"
    if not ckpt_dir.exists():
        versions = sorted(glob.glob(str(m_out / "lightning_logs" / "version_*")), reverse=True)
        if versions:
            ckpt_dir = Path(versions[0]) / "checkpoints"
        else:
            raise FileNotFoundError(f"No checkpoints found in {m_out}")

    model = load_model(ckpt_dir)
    
    dm = SeggerDataModule(
        data_dir=d_out,
        batch_size=1,
        num_workers=0,
    )
    dm.setup()

    receptive_field = {"k_bd": 4, "dist_bd": 15, "k_tx": 5, "dist_tx": 3}

    segment(
        model,
        dm,
        save_dir=results_dir,
        seg_tag=sample_dir.name,
        transcript_file=sample_dir / "transcripts.parquet",
        receptive_field=receptive_field,
        min_transcripts=5,
        score_cut=0.1,
        cell_id_col="segger_cell_id",
        use_cc=True,
        knn_method="kd_tree",
        verbose=True,
        gpu_ids=["0"] if torch.cuda.is_available() else None,
    )

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
    
    datasets_dir.mkdir(exist_ok=True, parents=True)
    models_dir.mkdir(exist_ok=True, parents=True)
    
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