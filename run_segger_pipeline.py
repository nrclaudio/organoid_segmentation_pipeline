import sys
import argparse
from pathlib import Path
import os
import torch
import logging
import pandas as pd
import warnings

# Suppress shapely warnings regarding oriented_envelope
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in oriented_envelope")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in oriented_envelope")

# Ensure segger is importable
# Check current dir or sibling dir
SEGGER_REPO = Path(__file__).resolve().parent / "segger"
if not SEGGER_REPO.exists():
    SEGGER_REPO = Path(__file__).resolve().parent.parent / "segger"

SEGGER_SRC = SEGGER_REPO / "src"
if str(SEGGER_SRC) not in sys.path:
    sys.path.insert(0, str(SEGGER_SRC))

try:
    from segger.data.parquet.sample import STSampleParquet
    from segger.training.train import LitSegger
    from segger.training.segger_data_module import SeggerDataModule
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import CSVLogger
except ImportError as e:
    print(f"Error importing Segger modules: {e}")
    print("Please ensure you are running this script within the 'segger_mac' environment.")
    sys.exit(1)

def create_dataset(input_dir, output_dir, args):
    print(f"  Creating Dataset from {input_dir}...")
    
    # Ensure at least 1 worker for NDTree partitioning
    workers = args.workers if args.workers > 0 else 1
    
    # Initialize sample
    # Note: weights=None for now (no scRNAseq embeddings used yet)
    sample = STSampleParquet(
        base_dir=input_dir,
        n_workers=workers,
        sample_type="saw_bin1",
        weights=None 
    )

    # Save dataset (Create graph)
    sample.save(
        data_dir=output_dir,
        k_bd=3,
        dist_bd=15.0,
        k_tx=3,
        dist_tx=5.0,
        tx_graph_mode="grid_same_gene",
        grid_connectivity=8,
        within_bin_edges="star",
        bin_pitch=1.0,
        allow_missing_boundaries=True, 
        tile_width=200,
        tile_height=200,
        val_prob=0.1,
        test_prob=0.1,
        neg_sampling_ratio=5.0,
        frac=1.0
    )

def train_sample(dataset_dir, model_dir, raw_input_dir, args):
    print(f"  Training Model on {dataset_dir}...")
    
    dm = SeggerDataModule(
        data_dir=dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    dm.setup()
    
    # Determine number of gene tokens
    num_tx_tokens = 30000 # Default
    genes_file = raw_input_dir / "genes.parquet"
    if genes_file.exists():
        df = pd.read_parquet(genes_file)
        num_tx_tokens = len(df) + 10 # Buffer
        print(f"  Detected {len(df)} genes. Setting num_tx_tokens={num_tx_tokens}")

    # Inspect data for feature dims
    sample_data = dm.train[0]
    
    if "tx" in sample_data.x_dict and sample_data.x_dict["tx"].ndim == 1:
        is_token_based = True
        num_tx_features = num_tx_tokens
    else:
        is_token_based = False
        num_tx_features = sample_data.x_dict["tx"].shape[1]

    # Handle case where 'bd' might be missing if no boundaries
    if "bd" in sample_data.x_dict:
        num_bd_features = sample_data.x_dict["bd"].shape[1]
    else:
        num_bd_features = 0 
        print("  Warning: No boundary features found.")

    model = LitSegger(
        is_token_based=is_token_based,
        num_node_features={"tx": num_tx_features, "bd": num_bd_features},
        init_emb=8,
        hidden_channels=32,
        out_channels=8,
        heads=2,
        num_mid_layers=2,
        aggr="sum",
        learning_rate=1e-3
    )

    acc = "cpu"
    if torch.cuda.is_available():
        acc = "cuda"
    elif torch.backends.mps.is_available():
        acc = "mps"
    
    print(f"  Using accelerator: {acc}")

    trainer = Trainer(
        accelerator=acc,
        devices=1,
        max_epochs=args.epochs,
        default_root_dir=model_dir,
        logger=CSVLogger(model_dir),
    )
    
    trainer.fit(model=model, datamodule=dm)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs-dir", default="segger_inputs")
    parser.add_argument("--datasets-dir", default="segger_datasets")
    parser.add_argument("--models-dir", default="segger_models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0) # Default to 0 to avoid OOM
    args = parser.parse_args()
    
    inputs_dir = Path(args.inputs_dir)
    datasets_dir = Path(args.datasets_dir)
    models_dir = Path(args.models_dir)
    
    datasets_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    
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

if __name__ == "__main__":
    main()