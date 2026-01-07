import torch
from pathlib import Path

def check_tile(filepath):
    try:
        data = torch.load(filepath)
    except Exception as e:
        return f"Error: {e}"

    edge_type = ("tx", "belongs", "bd")
    if edge_type not in data.edge_types:
        return "missing_edge_type"

    store = data[edge_type]
    has_index = hasattr(store, "edge_label_index")
    
    if not has_index:
        return "missing_labels (Test Tile)"

    labels = store.edge_label
    pos = (labels == 1).sum().item()
    neg = (labels == 0).sum().item()
    
    return {"pos": pos, "neg": neg}

def main():
    # Looking in current directory since you are inside organoid_segmentation_pipeline
    dataset_dir = Path("segger_datasets/01_C04895D5_kidney1")
    train_dir = dataset_dir / "train_tiles" / "processed"
    
    if not train_dir.exists():
        print(f"Directory not found: {train_dir}\n(Wait for the pipeline to create some tiles first)")
        return

    pt_files = sorted(list(train_dir.glob("*.pt")))
    print(f"Found {len(pt_files)} tiles. Checking first 10...")
    
    for pt in pt_files[:10]:
        print(f"{pt.name}: {check_tile(pt)}")

if __name__ == "__main__":
    main()