#!/usr/bin/env python3
"""Fail-fast checks before expensive STD-Net server runs."""

from __future__ import annotations

import ast
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_files = [
        repo_root / "Run_STD.py",
        repo_root / "STD_Net" / "STD_Net.py",
        repo_root / "STD_Net" / "DGAT.py",
        repo_root / "STD_Net" / "adjacency_matrix.py",
        repo_root / "STD_Net" / "stdg.py",
        repo_root / "STD_Net" / "optimization.py",
    ]
    for source_file in source_files:
        ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))

    import anndata
    import numpy
    import pandas
    import scanpy
    import scipy
    import sklearn
    import torch
    import torch_geometric

    datasets = sorted(
        path for path in (repo_root / "final_data").iterdir() if path.is_dir()
    )
    raw_counts_path = (
        repo_root / "original_data" / "Lohoff_seqFISH" / "counts.txt"
    )
    if not raw_counts_path.is_file():
        raise FileNotFoundError(f"Missing raw single-cell counts: {raw_counts_path}")

    import pandas as pd

    raw_header = pd.read_csv(raw_counts_path, sep="\t", index_col=0, nrows=1)
    raw_cell_ids = set(raw_header.columns.astype(str))
    if len(datasets) != 6:
        raise RuntimeError(f"Expected 6 seqFISH slices, found {len(datasets)}")
    for dataset in datasets:
        required = [
            dataset / "sc_data" / "sc_data.tsv",
            dataset / "sc_data" / "sc_label.tsv",
            dataset / "sc_data" / "sc_coordinates.csv",
            dataset / "st_data" / "ST_data.tsv",
            dataset / "st_data" / "coordinates.csv",
            dataset / "st_data" / "ST_ground_truth.tsv",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing files: " + ", ".join(missing))
        label_table = pd.read_csv(
            dataset / "sc_data" / "sc_label.tsv", sep="\t", dtype=str
        )
        missing_raw_cells = set(label_table.iloc[:, 0]).difference(raw_cell_ids)
        if missing_raw_cells:
            first_missing = sorted(missing_raw_cells)[0]
            raise RuntimeError(
                f"{dataset.name}: {len(missing_raw_cells)} labeled cells are "
                f"absent from raw counts; first missing cell: {first_missing}"
            )

    print(f"Python: {torch.__version__=}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyG: {torch_geometric.__version__}")
    print(f"scanpy: {scanpy.__version__}; anndata: {anndata.__version__}")
    print(
        f"numpy: {numpy.__version__}; pandas: {pandas.__version__}; "
        f"scipy: {scipy.__version__}; sklearn: {sklearn.__version__}"
    )
    print("PREFLIGHT_OK")


if __name__ == "__main__":
    main()
