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
# Reorganization: segger is now in tools/stereosegger
# Script in pipeline/src -> parent=pipeline -> parent.parent=root -> tools/stereosegger
SEGGER_REPO = Path(__file__).resolve().parent.parent.parent / "tools" / "stereosegger"

SEGGER_SRC = SEGGER_REPO / "src"
if str(SEGGER_SRC) not in sys.path:
    sys.path.insert(0, str(SEGGER_SRC))

try:
    from stereosegger.data.parquet.sample import STSampleParquet
    from stereosegger.training.train import LitSegger
    from stereosegger.training.segger_data_module import SeggerDataModule
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import CSVLogger
    from stereosegger.data.utils import get_edge_index, create_anndata, coo_to_dense_adj
    # Import the GPU-accelerated predict_batch
    from stereosegger.prediction import predict_batch
except ImportError as e:
    import traceback
    traceback.print_exc()
    print(f"Error importing Segger modules: {e}")
    sys.exit(1)

# ... (keep helper functions like get_similarity_scores_cpu if you want, or delete them later)

def predict_sample(dataset_dir, model_dir, output_dir, raw_input_dir, args):
    print(f"  Predicting on {dataset_dir}...")
    
    # 1. Detect Device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"    Using device for inference: {device}")

    # 2. Load Model
    ckpt_glob = list((model_dir / "lightning_logs").glob("version_*/checkpoints/*.ckpt"))
    if not ckpt_glob:
        print("    No checkpoint found. Skipping prediction.")
        return
    
    ckpt_path = sorted(ckpt_glob, key=lambda p: p.stat().st_mtime)[-1]
    print(f"    Loading checkpoint: {ckpt_path.name}")
    
    model = LitSegger.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()
    
    # 3. Data Module
    dm = SeggerDataModule(
        data_dir=dataset_dir,
        batch_size=1,
        num_workers=0 
    )
    dm.setup()
    
    all_assignments = []
    loaders = [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]
    
    print("    Running inference on tiles (GPU)...")

    for loader in loaders:
        for batch in tqdm(loader, leave=False):
            batch = batch.to(device)
            # CALL LIBRARY FUNCTION (GPU)
            # Note: predict_batch returns a DataFrame
            df = predict_batch(model.model, batch, score_cut=0.5, use_cc=False)
            all_assignments.append(df)
            
    if not all_assignments:
        print("    No assignments generated.")
        return

    full_df = pd.concat(all_assignments, ignore_index=True)
    full_df = full_df.sort_values(["bound", "score"], ascending=[False, False])
    full_df = full_df.drop_duplicates(subset="transcript_id", keep="first")
    
    transcripts_file = raw_input_dir / "transcripts.parquet"
    if transcripts_file.exists():
        orig_df = pd.read_parquet(transcripts_file)
        orig_df["transcript_id"] = orig_df["transcript_id"].astype(str)
        
        merged_df = orig_df.merge(full_df, on="transcript_id", how="left")
        out_file = output_dir / f"{raw_input_dir.name}_segmentation.parquet"
        merged_df.to_parquet(out_file)
        print(f"    Saved segmentation to {out_file}")
        
        merged_df["cell_id"] = merged_df["segger_cell_id"].fillna("UNASSIGNED")
        
        genes_file = raw_input_dir / "genes.parquet"
        if genes_file.exists():
            genes_df = pd.read_parquet(genes_file)
            gene_map = dict(zip(genes_df.gene_id, genes_df.gene_name))
            merged_df["feature_name"] = merged_df["gene_id"].map(gene_map)
            merged_df.rename(columns={"x": "x_location", "y": "y_location"}, inplace=True)
            
            print("    Creating AnnData...")
            adata = create_anndata(merged_df, min_transcripts=3)
            adata_out = output_dir / f"{raw_input_dir.name}_segmentation.h5ad"
            adata.write_h5ad(adata_out)
            print(f"    Saved AnnData to {adata_out}")

def create_dataset(input_dir, output_dir, args):
    print(f"  Creating Dataset from {input_dir}...")
    
    # Ensure at least 1 worker for NDTree partitioning
    workers = args.workers if args.workers > 0 else 1
    
    # Initialize sample
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
    
    # Check if explicitly flagged as token_based in the dataset (handles [idx, count] case)
    if hasattr(sample_data["tx"], "token_based") and sample_data["tx"].token_based:
        is_token_based = True
        num_tx_features = num_tx_tokens
    elif "tx" in sample_data.x_dict and sample_data.x_dict["tx"].ndim == 1:
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

    acc = "auto"
    
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