import argparse
import sys
from pathlib import Path
import numpy as np
import dask.array as da
import spatialdata as sd
from spatialdata.models import Image2DModel, Labels2DModel, TableModel
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import shutil
import pandas as pd
import xarray as xr

def get_centroids_and_areas(mask):
    """
    Finds connected components in the mask.
    Returns a list of dicts: {'label': int, 'centroid': (y, x), 'area': int, 'bbox': (min_y, min_x, max_y, max_x)}
    """
    # Label components
    labeled_mask, num_features = ndimage.label(mask > 0)
    
    components = []
    if num_features == 0:
        return components, labeled_mask

    # Find objects
    slices = ndimage.find_objects(labeled_mask)
    
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        label_id = i + 1
        
        y_slice, x_slice = sl
        sub_mask = (labeled_mask[y_slice, x_slice] == label_id)
        
        area = np.sum(sub_mask)
        cy_local, cx_local = ndimage.center_of_mass(sub_mask)
        cy = cy_local + y_slice.start
        cx = cx_local + x_slice.start
        
        components.append({
            'label': label_id,
            'centroid': (cy, cx),
            'area': area,
            'bbox': (y_slice.start, x_slice.start, y_slice.stop, x_slice.stop),
            'slice': sl
        })
        
    return components, labeled_mask

def split_merged_components(mask, min_distance=200):
    """
    Applies watershed segmentation to split touching components.
    """
    print(f"  Attempting to split merged components (min_distance={min_distance})...")
    
    # Distance transform
    distance = ndimage.distance_transform_edt(mask)
    
    # Smooth distance map slightly to merge local peaks
    distance_smoothed = ndimage.gaussian_filter(distance, sigma=10)
    
    # Find local maxima (peaks)
    # min_distance: Minimum number of pixels separating peaks.
    # For kidneys (~5000px wide), 200-500 is reasonable.
    local_maxi = peak_local_max(distance_smoothed, min_distance=min_distance, labels=mask)
    
    num_peaks = len(local_maxi)
    print(f"  Found {num_peaks} peaks for watershed markers.")
    
    if num_peaks < 2:
        print("  Not enough peaks to split. Returning original mask.")
        return mask

    # Markers
    markers = np.zeros_like(mask, dtype=int)
    for i, (r, c) in enumerate(local_maxi):
        markers[r, c] = i + 1
    
    # Watershed
    labels = watershed(-distance_smoothed, markers, mask=mask)
    return labels

def sort_kidneys(components):
    """
    Sorts components into [Top, Left, Right] based on centroids.
    Expects exactly 3 components.
    """
    if len(components) != 3:
        print(f"Warning: Expected 3 components for sorting, got {len(components)}. Sorting by X coordinate.")
        return sorted(components, key=lambda c: c['centroid'][1])

    top_comp = min(components, key=lambda c: c['centroid'][0])
    remaining = [c for c in components if c != top_comp]
    
    remaining_sorted_x = sorted(remaining, key=lambda c: c['centroid'][1])
    left_comp = remaining_sorted_x[0]
    right_comp = remaining_sorted_x[1]
    
    return [top_comp, left_comp, right_comp]

def crop_and_save(sdata, component, output_path, margin=50):
    min_y, min_x, max_y, max_x = component['bbox']
    min_y = max(0, min_y - margin)
    min_x = max(0, min_x - margin)
    max_y += margin
    max_x += margin
    
    crop_y = slice(min_y, max_y)
    crop_x = slice(min_x, max_x)
    
    # --- 1. Crop Images ---
    new_images = {}
    for name, img in sdata.images.items():
        if hasattr(img, 'keys') and 'scale0' in img:
             ds = img['scale0']
             var_name = list(ds.data_vars.keys())[0]
             da_img = ds[var_name]
             cropped_da = da_img.isel(y=crop_y, x=crop_x)
             new_images[name] = Image2DModel.parse(
                 cropped_da.data,
                 dims=("c", "y", "x"),
                 scale_factors=[2, 2, 2]
             )
        elif isinstance(img, (xr.DataArray, da.Array, np.ndarray)) or hasattr(img, 'shape'):
             cropped_img = img[:, crop_y, crop_x]
             new_images[name] = Image2DModel.parse(
                 cropped_img, 
                 dims=("c", "y", "x"),
                 scale_factors=[2, 2, 2]
             )

    # --- 2. Crop Labels ---
    new_labels = {}
    for name, lbl in sdata.labels.items():
        if hasattr(lbl, 'keys') and 'scale0' in lbl:
            ds = lbl['scale0']
            var_name = list(ds.data_vars.keys())[0]
            da_lbl = ds[var_name]
            cropped_lbl = da_lbl.isel(y=crop_y, x=crop_x)
            new_labels[name] = Labels2DModel.parse(
                cropped_lbl.data,
                dims=("y", "x"),
                scale_factors=[2, 2, 2]
            )
        else:
            cropped_lbl = lbl[crop_y, crop_x]
            new_labels[name] = Labels2DModel.parse(
                cropped_lbl,
                dims=("y", "x"),
                scale_factors=[2, 2, 2]
            )

    # --- 3. Filter Table ---
    new_tables = {}
    for name, table in sdata.tables.items():
        if "spatial" in table.obsm:
            coords = table.obsm["spatial"]
            in_box = (
                (coords[:, 0] >= min_x) & (coords[:, 0] < max_x) &
                (coords[:, 1] >= min_y) & (coords[:, 1] < max_y)
            )
            sub_table = table[in_box].copy()
            sub_table.obsm["spatial"][:, 0] -= min_x
            sub_table.obsm["spatial"][:, 1] -= min_y
            
            if "spatialdata_attrs" in sub_table.uns:
                del sub_table.uns["spatialdata_attrs"]
                
            new_tables[name] = TableModel.parse(
                sub_table,
                region=sub_table.obs['region'].iloc[0] if len(sub_table) > 0 else "cells",
                region_key="region",
                instance_key="instance_id"
            )

    new_sdata = sd.SpatialData(images=new_images, labels=new_labels, tables=new_tables)
    
    if output_path.exists():
        shutil.rmtree(output_path)
    new_sdata.write(output_path)
    print(f"Saved {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdata", required=True, help="Path to input SpatialData zarr")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--prefix", required=True, help="Prefix for output files")
    parser.add_argument("--margin", type=int, default=100, help="Crop margin")
    args = parser.parse_args()

    sdata_path = Path(args.sdata)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {sdata_path}...")
    sdata = sd.read_zarr(sdata_path)

    if "tissue_mask" not in sdata.labels:
        print("Error: 'tissue_mask' not found in sdata.labels")
        sys.exit(1)
        
    lbl = sdata.labels["tissue_mask"]
    if hasattr(lbl, 'keys') and 'scale0' in lbl:
         ds = lbl['scale0']
         var_name = list(ds.data_vars.keys())[0]
         mask = ds[var_name].values
    else:
         mask = lbl.values

    if mask.ndim == 3:
        mask = mask[0]

    # Clean mask: remove small noise first
    # This prevents noise from affecting watershed or component counts
    mask = mask > 0 # Binary
    
    print("Initial component analysis...")
    components, labeled_mask = get_centroids_and_areas(mask)
    
    # Filter noise immediately
    if components:
        max_area = max(c['area'] for c in components)
        noise_threshold = max_area * 0.05
        mask = np.isin(labeled_mask, [c['label'] for c in components if c['area'] > noise_threshold]).astype(int)
        # Re-label after noise removal
        components, labeled_mask = get_centroids_and_areas(mask)
        print(f"After noise filtering: {len(components)} components.")
    
    # Check if splitting is needed
    if len(components) < 3:
        print(f"Found fewer than 3 components ({len(components)}). Applying Watershed Splitting...")
        split_labels = split_merged_components(mask, min_distance=250)
        
        # Re-calculate components from split labels
        # Note: split_labels has new label IDs
        # We need to re-run get_centroids to get the dict structure
        components_split = []
        slices = ndimage.find_objects(split_labels)
        for i, sl in enumerate(slices):
            if sl is None: continue
            label_id = i + 1
            y_slice, x_slice = sl
            sub_mask = (split_labels[y_slice, x_slice] == label_id)
            area = np.sum(sub_mask)
            # Filter tiny bits from watershed
            if area < 1000: continue 
            
            cy_local, cx_local = ndimage.center_of_mass(sub_mask)
            cy = cy_local + y_slice.start
            cx = cx_local + x_slice.start
            components_split.append({
                'label': label_id,
                'centroid': (cy, cx),
                'area': area,
                'bbox': (y_slice.start, x_slice.start, y_slice.stop, x_slice.stop),
                'slice': sl
            })
        
        components = components_split
        print(f"After Watershed: {len(components)} components.")

    # Select Top 3
    if len(components) < 3:
        print(f"Warning: Still found fewer than 3 kidneys ({len(components)}).")
        sorted_components = sorted(components, key=lambda c: c['centroid'][1])
        names = [f"kidney{i+1}" for i in range(len(sorted_components))]
    else:
        by_area = sorted(components, key=lambda c: c['area'], reverse=True)
        top_3 = by_area[:3]
        sorted_components = sort_kidneys(top_3)
        names = ["kidney1", "kidney2", "kidney3"]
        
        print("Identified Kidneys:")
        print(f"  Kidney 1 (Top):   Centroid {sorted_components[0]['centroid']}")
        print(f"  Kidney 2 (Left):  Centroid {sorted_components[1]['centroid']}")
        print(f"  Kidney 3 (Right): Centroid {sorted_components[2]['centroid']}")

    # Process
    for comp, kid_name in zip(sorted_components, names):
        out_name = f"{args.prefix}_{kid_name}.sdata.zarr"
        out_path = out_dir / out_name
        print(f"Extracting {kid_name} -> {out_name}")
        crop_and_save(sdata, comp, out_path, margin=args.margin)

if __name__ == "__main__":
    main()
