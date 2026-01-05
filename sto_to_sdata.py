#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy import ndimage

import scanpy as sc
import spatialdata as sd
from spatialdata.models import (
    Image2DModel,
    Labels2DModel,
    TableModel,
)


def load_image_as_cyx(path: str):
    """Load TIFF and return (c, y, x), dims tuple."""
    img = tiff.imread(path)
    if img.ndim == 2:
        # (y, x) -> add channel axis
        img = img[None, :, :]
        dims = ("c", "y", "x")
    elif img.ndim == 3:
        # assume (y, x, c)
        if img.shape[2] <= 4:  # typical microscopy
            img = np.transpose(img, (2, 0, 1))  # -> (c, y, x)
            dims = ("c", "y", "x")
        else:
            # fallback: assume already (c, y, x)
            dims = ("c", "y", "x")
    else:
        raise ValueError(f"Unexpected image shape {img.shape}")
    return img, dims


def load_labels_yx(path: str):
    """Load mask TIFF and return (y, x)."""
    lab = tiff.imread(path)
    if lab.ndim == 3:
        # If z-stack, take first plane
        lab = lab[0]
    if lab.ndim != 2:
        raise ValueError(f"Labels should be 2D, got shape {lab.shape}")
    return lab


def main():
    p = argparse.ArgumentParser(
        description=(
            "Build SpatialData from STOmics AnnData (H5AD) + ssDNA image + masks.\n"
            "Links the AnnData table to a logical 'cells' region; "
            "ssDNA / tissue masks are stored as label layers.\n"
            "Cells are filtered by tissue mask only, and both a global Zarr and per-tissue Zarrs are written.\n"
            "Tissues are assigned consistent semantic labels across chips based on position "
            "(top / left / right when three components are present)."
        )
    )
    p.add_argument("--h5ad", required=True, help="Input AnnData (from sto_export_h5ad.py)")
    p.add_argument("--image", required=True, help="Path to ssDNA (registered) TIFF")
    p.add_argument("--labels", required=True, help="Path to ssDNA / segmentation mask TIFF")
    p.add_argument(
        "--sample-id",
        required=True,
        help="Sample / tissue slide ID (e.g. C04689E2). Used for naming table and output.",
    )
    p.add_argument(
        "--out",
        help="Output SpatialData Zarr for the global object. If not provided, uses <sample-id>.sdata.zarr",
    )
    p.add_argument(
        "--no-pyramid",
        action="store_true",
        help="Disable multi-scale pyramids (no scale_factors).",
    )
    p.add_argument(
        "--tissue-mask",
        help="Path to tissue mask TIFF (binary / labeled). Optional.",
    )

    args = p.parse_args()
    out_path = args.out or f"{args.sample_id}.sdata.zarr"

    # ---------- 1) Read AnnData ----------
    print(f"[sdata] Reading AnnData: {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)

    if "spatial" not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] not found. Check StereoPy export.")

    coords = adata.obsm["spatial"]
    if coords.shape[1] != 2:
        raise ValueError(
            f"Expected adata.obsm['spatial'] to have shape (n_obs, 2), got {coords.shape}"
        )

    print(f"[sdata] AnnData obs: {adata.n_obs}, genes: {adata.n_vars}")
    print(f"[sdata] Spatial coords shape: {coords.shape}")

    # ---------- 2) Read image ----------
    print(f"[sdata] Reading image: {args.image}")
    img, img_dims = load_image_as_cyx(args.image)
    print(f"[sdata] Image shape (c,y,x): {img.shape}")

    scale_factors = None if args.no_pyramid else (2, 2, 2)
    print(f"[sdata] Parsing image with scale_factors={scale_factors}")
    img_model = Image2DModel.parse(
        img,
        dims=img_dims,
        scale_factors=scale_factors,
    )

    # ---------- 3) Read label masks (ssDNA mask + optional tissue mask) ----------
    print(f"[sdata] Reading labels (mask): {args.labels}")
    labels_array = load_labels_yx(args.labels)
    print(f"[sdata] Labels shape (y,x): {labels_array.shape}, dtype={labels_array.dtype}")

    print(f"[sdata] Parsing ssDNA mask labels with scale_factors={scale_factors}")
    ssdna_labels_model = Labels2DModel.parse(
        labels_array,
        dims=("y", "x"),
        scale_factors=scale_factors,
    )

    labels_dict = {"ssdna_mask": ssdna_labels_model}

    tissue_array = None
    tissue_components = None
    n_components = 0
    raw_to_canonical_id = {}
    raw_to_name = {}

    if args.tissue_mask is not None:
        print(f"[sdata] Reading tissue mask: {args.tissue_mask}")
        tissue_array = load_labels_yx(args.tissue_mask)
        print(f"[sdata] Tissue mask shape (y,x): {tissue_array.shape}, dtype={tissue_array.dtype}")

        if tissue_array.shape != labels_array.shape:
            print(
                f"[sdata][warning] Tissue mask shape {tissue_array.shape} != ssDNA labels shape {labels_array.shape}. "
                "Assuming they are in the same coordinate system but be cautious."
            )

        tissue_labels_model = Labels2DModel.parse(
            tissue_array,
            dims=("y", "x"),
            scale_factors=scale_factors,
        )
        labels_dict["tissue_mask"] = tissue_labels_model

        # Label connected components in the tissue mask
        tissue_binary = tissue_array > 0
        tissue_components, n_components = ndimage.label(tissue_binary)
        print(f"[sdata] Found {n_components} tissue components in tissue mask.")

        if n_components > 0:
            # Compute centroids for each raw tissue component
            centroids = {}
            for comp_id in range(1, n_components + 1):
                ys_comp, xs_comp = np.where(tissue_components == comp_id)
                if ys_comp.size == 0:
                    continue
                cy = ys_comp.mean()
                cx = xs_comp.mean()
                centroids[comp_id] = (cx, cy)  # (x, y)

            # Define consistent semantics if we have exactly 3 components:
            # top = smallest y; remaining two: left (smallest x), right (largest x)
            if len(centroids) == 3:
                comp_ids = list(centroids.keys())
                # pick top
                top_id = min(comp_ids, key=lambda k: centroids[k][1])  # smallest y
                rest = [k for k in comp_ids if k != top_id]
                # among remaining: left/right by x
                left_id, right_id = sorted(rest, key=lambda k: centroids[k][0])  # x coordinate

                mapping_order = [
                    ("top", top_id),
                    ("left", left_id),
                    ("right", right_id),
                ]
                print("[sdata] Assigning semantic tissue labels based on position:")
                for idx, (name, raw_id) in enumerate(mapping_order, start=1):
                    raw_to_canonical_id[raw_id] = idx
                    raw_to_name[raw_id] = name
                    cx, cy = centroids[raw_id]
                    print(f"  - {name}: raw_id={raw_id}, canonical_id={idx}, centroid=(x={cx:.1f}, y={cy:.1f})")
            else:
                # Fallback: generic naming
                print(
                    f"[sdata] Number of components != 3 (n={len(centroids)}). "
                    "Using generic tissue IDs tissue1, tissue2, ..."
                )
                for raw_id in centroids:
                    raw_to_canonical_id[raw_id] = raw_id
                    raw_to_name[raw_id] = f"tissue{raw_id}"
        else:
            print("[sdata] No nonzero components found in tissue mask.")
    else:
        print("[sdata] No tissue mask provided. No mask-based filtering or per-tissue Zarrs will be created.")

    # ---------- 4) Assign cells to tissues (if tissue mask) and filter ----------
    print("[sdata] Assigning cell centroids to tissue components (if provided)...")

    xs = np.round(coords[:, 0]).astype(int)
    ys = np.round(coords[:, 1]).astype(int)

    # Use the ssDNA image size as reference for bounds (assumed same as tissue mask)
    h, w = labels_array.shape
    inside_bounds = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)

    # Start with everything valid
    valid = np.ones_like(inside_bounds, dtype=bool)
    tissue_ids_raw = np.zeros(coords.shape[0], dtype=int)

    if tissue_components is not None and n_components > 0:
        # For cells that fall within the image bounds, read the tissue component label
        tissue_ids_raw[inside_bounds] = tissue_components[ys[inside_bounds], xs[inside_bounds]]
        # Valid = cells that belong to any tissue component (>0)
        valid = tissue_ids_raw > 0
        filter_desc = "inside tissue components"
    else:
        # No tissue mask: keep all cells, tissue_ids_raw stay 0
        filter_desc = "no tissue mask provided (no mask-based filtering)"

    n_valid = valid.sum()
    print(f"[sdata] Cells kept ({filter_desc}): {n_valid}/{adata.n_obs}")

    if tissue_components is not None and n_components > 0 and n_valid == 0:
        raise RuntimeError("No cell centroids inside tissue mask components. Check coordinate alignment.")

    # subset to valid cells only
    adata = adata[valid].copy()
    coords_valid = np.asarray(coords[valid], dtype=float)
    tissue_ids_raw = tissue_ids_raw[valid]

    # Map raw component IDs to canonical IDs and names
    if tissue_components is not None and n_components > 0:
        tissue_ids = np.array(
            [raw_to_canonical_id.get(r, 0) for r in tissue_ids_raw],
            dtype=int,
        )
        tissue_names = np.array(
            [raw_to_name.get(r, "unknown") for r in tissue_ids_raw],
            dtype=object,
        )
    else:
        tissue_ids = np.zeros_like(tissue_ids_raw, dtype=int)
        tissue_names = np.array(["none"] * tissue_ids_raw.shape[0], dtype=object)

    # ---------- 5) Build GLOBAL table linked to a logical 'cells' region ----------
    print("[sdata] Building global table (all tissues, with tissue_id and tissue_name)...")

    adata.obsm["spatial"] = coords_valid
    adata.obs["sample_id"] = args.sample_id
    adata.obs["tissue_id"] = tissue_ids
    adata.obs["tissue_name"] = tissue_names

    region_name = "cells"
    adata.obs["region"] = pd.Categorical([region_name] * adata.n_obs)
    adata.obs["instance_id"] = np.arange(adata.n_obs, dtype=int)

    table_global = TableModel.parse(
        adata,
        region=region_name,
        region_key="region",
        instance_key="instance_id",
    )

    # ---------- 6) Assemble GLOBAL SpatialData (one Zarr with all tissues) ----------
    print("[sdata] Assembling global SpatialData object (all tissues)...")
    sdata_global = sd.SpatialData(
        images={"ssDNA": img_model},
        labels=labels_dict,
        tables={args.sample_id: table_global},
    )

    print(sdata_global)

    print(f"[sdata] Writing global SpatialData to {out_path}")
    sdata_global.write(out_path, overwrite=True)
    print("[sdata] Wrote global Zarr.")

    # ---------- 7) Per-tissue SpatialData Zarrs ----------
    if tissue_components is not None and n_components > 0:
        unique_tissues = np.unique(tissue_ids)
        unique_tissues = unique_tissues[unique_tissues > 0]  # ignore 0 (background)

        print(f"[sdata] Writing per-tissue SpatialData Zarrs for tissue IDs: {list(unique_tissues)}")

        for t_id in unique_tissues:
            mask_t = tissue_ids == t_id
            adata_t = adata[mask_t].copy()
            coords_t = coords_valid[mask_t]
            names_t = tissue_names[mask_t]

            # all entries in names_t should be the same
            tissue_label = str(names_t[0]) if names_t.size > 0 else f"tissue{t_id}"
            tissue_sample_id = f"{args.sample_id}_{tissue_label}"

            print(f"[sdata] Tissue {t_id} ({tissue_label}): {adata_t.n_obs} cells -> {tissue_sample_id}.sdata.zarr")

            adata_t.obsm["spatial"] = coords_t
            adata_t.obs["sample_id"] = tissue_sample_id
            # tissue_id and tissue_name already present

            adata_t.obs["region"] = pd.Categorical([region_name] * adata_t.n_obs)
            adata_t.obs["instance_id"] = np.arange(adata_t.n_obs, dtype=int)

            if "spatialdata_attrs" in adata_t.uns:
                del adata_t.uns["spatialdata_attrs"]

            table_t = TableModel.parse(
                adata_t,
                region=region_name,
                region_key="region",
                instance_key="instance_id",
            )

            sdata_t = sd.SpatialData(
                images={"ssDNA": img_model},
                labels=labels_dict,
                tables={tissue_sample_id: table_t},
            )

            out_path_t = f"{tissue_sample_id}.sdata.zarr"
            print(f"[sdata] Writing per-tissue SpatialData to {out_path_t}")
            sdata_t.write(out_path_t, overwrite=True)

    print("[sdata] Done.")


if __name__ == "__main__":
    main()