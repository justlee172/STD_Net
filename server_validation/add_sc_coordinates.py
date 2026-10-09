#!/usr/bin/env python3
"""Create per-cell coordinate files from Lohoff affine metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("original_data/Lohoff_seqFISH/metadata_affine.tsv"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("final_data"))
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata, sep="\t", dtype={"uniqueID": str})
    required = {
        "uniqueID",
        "x_global_affine",
        "y_global_affine",
    }
    missing_columns = required.difference(metadata.columns)
    if missing_columns:
        raise ValueError(
            f"Metadata is missing columns: {sorted(missing_columns)}"
        )
    metadata = metadata.set_index("uniqueID")

    datasets = sorted(
        path for path in args.data_root.iterdir() if path.is_dir()
    )
    if not datasets:
        raise FileNotFoundError(f"No datasets found under {args.data_root}")

    for dataset in datasets:
        label_file = dataset / "sc_data" / "sc_label.tsv"
        if not label_file.is_file():
            continue
        labels = pd.read_csv(label_file, sep="\t", dtype=str)
        cell_id_column = labels.columns[0]
        cell_ids = labels[cell_id_column].astype(str)
        missing_cells = cell_ids[~cell_ids.isin(metadata.index)]
        if not missing_cells.empty:
            print(
                f"SKIP {dataset.name}: {len(missing_cells)} cells are absent "
                "from the Lohoff metadata"
            )
            continue
        coordinates = metadata.loc[
            cell_ids, ["x_global_affine", "y_global_affine"]
        ].copy()
        coordinates.columns = ["x", "y"]
        coordinates.index.name = "cell_id"
        output_file = dataset / "sc_data" / "sc_coordinates.csv"
        temporary_file = output_file.with_suffix(".csv.tmp")
        coordinates.to_csv(temporary_file)
        temporary_file.replace(output_file)
        print(f"WROTE {output_file} ({len(coordinates)} cells)")


if __name__ == "__main__":
    main()
