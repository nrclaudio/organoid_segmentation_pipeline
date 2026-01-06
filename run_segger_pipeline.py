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
    from segger.data.utils import get_edge_index, create_anndata, coo_to_dense_adj
except ImportError as e:
    print(f"Error importing Segger modules: {e}")
    print("Please ensure you are running this script within the 'segger_mac' environment.")
    sys.exit(1)

def get_similarity_scores(model, batch, from_type, to_type, k_to, dist_to):
    # Compute similarity scores on best available device
    device = batch[from_type].x.device
    shape = batch[from_type].x.shape[0], batch[to_type].x.shape[0]
    
    # Compute edge indices (CPU KDTree)
    if from_type == to_type:
        coords_1 = coords_2 = batch[to_type].pos
    else:
        coords_1 = batch[to_type].pos[:, :2]
        coords_2 = batch[from_type].pos[:, :2]
        
    edge_index = get_edge_index(
        coords_1.cpu(),
        coords_2.cpu(),
        k=k_to,
        dist=dist_to,
        method="kd_tree"
    ).to(device)
    
    # Convert to dense adjacency
    edge_index = coo_to_dense_adj(edge_index.T, num_nodes=shape[0], num_nbrs=k_to)
    
    model.eval()
    with torch.no_grad():
        embeddings = model(batch.x_dict, batch.edge_index_dict)
    
    emb_to = embeddings[to_type]
    emb_from = embeddings[from_type]
    
    # Handle padding (-1)
    mask = edge_index != -1
    flat_indices = edge_index[mask]
    
    # Gather 'from' embeddings
    gathered_from = emb_from[flat_indices]
    
    # Repeat 'to' embeddings
    row_indices = torch.arange(shape[0], device=device).unsqueeze(1).expand_as(edge_index)
    flat_rows = row_indices[mask]
    gathered_to = emb_to[flat_rows]
    
    # Compute dot product and sigmoid on device
    similarity = (gathered_to * gathered_from).sum(dim=1)
    similarity = torch.sigmoid(similarity)
    
    # Move back to CPU for SciPy sparse matrix
    values = similarity.cpu().numpy()
    rows = flat_rows.cpu().numpy()
    cols = flat_indices.cpu().numpy()
    
    return coo_matrix((values, (rows, cols)), shape=shape)

def predict_batch(model, batch, score_cut=0.5, use_cc=True):
    # Device-agnostic version of predict_batch
    transcript_id = batch["tx"].id.cpu().numpy().astype(str)
    assignments = pd.DataFrame({"transcript_id": transcript_id})
    assignments["score"] = 0.0
    assignments["segger_cell_id"] = None
    assignments["bound"] = 0
    
    if len(batch["bd"].pos) < 10:
        return assignments

    # tx -> bd (expansion)
    scores = get_similarity_scores(model, batch, "tx", "bd", k_to=4, dist_to=12.0)
    dense_scores = scores.toarray()
    
    if dense_scores.shape[1] == 0:
        return assignments

    max_scores = dense_scores.max(axis=1)
    assignments["score"] = max_scores
    
    mask = max_scores > score_cut
    
    all_ids = np.concatenate(batch["bd"].id.cpu().numpy())
    max_indices = dense_scores.argmax(axis=1)
    
    assignments.loc[mask, "segger_cell_id"] = all_ids[max_indices[mask]]
    assignments.loc[mask, "bound"] = 1
    
    # tx -> tx (floating cells)
    if use_cc:
        unassigned_mask = assignments["segger_cell_id"].isna()
        if unassigned_mask.any():
            scores_tx = get_similarity_scores(model, batch, "tx", "tx", k_to=5, dist_to=5.0)
            dense_tx = scores_tx.toarray()
            
            sub_matrix = dense_tx[unassigned_mask][:, unassigned_mask]
            sub_matrix[sub_matrix < score_cut] = 0
            
            n_comps, labels = connected_components(sub_matrix, connection='weak', directed=False)
            
            def _get_id():
                return "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), 8)) + "-nx"
            
            new_ids = np.array([_get_id() for _ in range(n_comps)])
            assignments.loc[unassigned_mask, "segger_cell_id"] = new_ids[labels]
            
    return assignments

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
    
    print("    Running inference on tiles...")
    for loader in loaders:
        for batch in tqdm(loader, leave=False):
            # Move batch to device
            batch = batch.to(device)
            # Run prediction (Torch math will happen on GPU/MPS, logic on CPU)
            df = predict_batch(model.model, batch, score_cut=0.5, use_cc=True)
            all_assignments.append(df)
            
    if not all_assignments:
        print("    No assignments generated.")
        return

    full_df = pd.concat(all_assignments, ignore_index=True)
    
    # Resolve duplicates (transcripts appearing in multiple tiles due to padding/overlap)
    # Prefer assigned (bound=1) and higher score
    full_df = full_df.sort_values(["bound", "score"], ascending=[False, False])
    full_df = full_df.drop_duplicates(subset="transcript_id", keep="first")
    
    # Merge with original transcripts to get gene names and coordinates
    transcripts_file = raw_input_dir / "transcripts.parquet"
    if transcripts_file.exists():
        orig_df = pd.read_parquet(transcripts_file)
        orig_df["transcript_id"] = orig_df["transcript_id"].astype(str)
        
        merged_df = orig_df.merge(full_df, on="transcript_id", how="left")
        
        # Save results
        out_file = output_dir / f"{raw_input_dir.name}_segmentation.parquet"
        merged_df.to_parquet(out_file)
        print(f"    Saved segmentation to {out_file}")
        
        # Create H5AD (CellBin)
        # We need 'cell_id' column for create_anndata. Segger uses 'segger_cell_id'
        merged_df["cell_id"] = merged_df["segger_cell_id"].fillna("UNASSIGNED")
        
        # Map back to original coordinate system if needed?
        # transcripts.parquet usually has global x/y.
        # We need columns: cell_id, feature_name (gene), x, y
        # orig_df has 'gene_id', need mapping to names?
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
            
        