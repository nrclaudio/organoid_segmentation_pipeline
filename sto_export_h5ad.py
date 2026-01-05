#!/usr/bin/env python

import argparse
import stereo as st  # stereopy


def main():
    p = argparse.ArgumentParser(
        description="Convert STOmics GEF to AnnData (H5AD) with spatial coords."
    )
    p.add_argument(
        "--gef",
        required=True,
        help="Path to *.gef (raw.gef or *.cellbin.gef)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output .h5ad file",
    )
    args = p.parse_args()

    print(f"[stereo] Reading GEF: {args.gef}")

    # Detect cellbin vs raw by filename
    if args.gef.endswith("cellbin.gef"):
        data = st.io.read_gef(
            file_path=args.gef,
            bin_type="cell_bins",
        )
    else:
        # raw / square-bin mode (raw.gef, tissue.gef, etc.)
        data = st.io.read_gef(
            file_path=args.gef,
            bin_size=1,
            # you can add bin_size=100 here if you want coarser bins
        )

    print("[stereo] Creating raw checkpoint...")
    data.tl.raw_checkpoint()

    print("[stereo] Converting to AnnData...")
    adata = st.io.stereo_to_anndata(
        data,
        flavor="scanpy",
        output=None,
    )

    print(f"[stereo] Writing AnnData to {args.out}")
    adata.write(args.out)
    print("[stereo] Done.")


if __name__ == "__main__":
    main()