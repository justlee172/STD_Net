#!/usr/bin/env python3
"""Fail early when benchmark inputs or baseline installations are incomplete."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline_validation.common import (  # noqa: E402
    CELL_TYPES_18,
    DATASETS,
    LOW_QUALITY,
    canonical_cell_type,
    parse_csv,
)


def validate_data(repo_root: Path) -> None:
    data_root = repo_root / "final_data"
    raw_path = repo_root / "original_data" / "Lohoff_seqFISH" / "counts.txt"
    if not raw_path.is_file():
        raise FileNotFoundError(f"Missing raw reference counts: {raw_path}")
    raw_header = pd.read_csv(raw_path, sep="\t", nrows=0).columns.astype(str)
    raw_cells = set(raw_header[1:]) if raw_header[0].startswith("Unnamed") else set(raw_header)
    truth_union = set()
    total_low_only = 0
    for dataset in DATASETS:
        root = data_root / dataset
        required = (
            root / "sc_data" / "sc_label.tsv",
            root / "st_data" / "ST_data.tsv",
            root / "st_data" / "ST_ground_truth.tsv",
            root / "st_data" / "coordinates.csv",
        )
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"Missing input: {path}")
        labels = pd.read_csv(required[0], sep="\t", dtype=str)
        missing_cells = set(labels.iloc[:, 0]).difference(raw_cells)
        if missing_cells:
            raise ValueError(
                f"{dataset}: {len(missing_cells)} reference cells are absent from counts.txt"
            )
        truth = pd.read_csv(required[2], sep="\t", index_col=0)
        biological_columns = []
        for column in truth.columns:
            label = canonical_cell_type(column)
            if label not in {"cell_num", "coor_X", "coor_Y"}:
                truth_union.add(label)
                if label != LOW_QUALITY:
                    biological_columns.append(column)
        total_low_only += int((truth[biological_columns].sum(axis=1) == 0).sum())
        st_index = pd.read_csv(required[1], sep="\t", usecols=[0]).iloc[:, 0].astype(str)
        coordinate_index = pd.read_csv(required[3], usecols=[0]).iloc[:, 0].astype(str)
        if list(st_index) != list(coordinate_index):
            raise ValueError(f"{dataset}: ST and coordinate spot order does not match")
    biological_union = truth_union.difference({LOW_QUALITY})
    if biological_union != set(CELL_TYPES_18):
        raise ValueError(
            "Fixed ontology does not equal the six-slice biological label union: "
            f"missing={sorted(biological_union.difference(CELL_TYPES_18))}, "
            f"extra={sorted(set(CELL_TYPES_18).difference(biological_union))}"
        )
    print(
        f"DATA_READY datasets={len(DATASETS)} ontology={len(CELL_TYPES_18)} "
        f"low_quality_only_spots={total_low_only}"
    )


def validate_nls(repo_root: Path) -> None:
    vendor = repo_root / "baseline_validation" / "vendor" / "NLSDeconv_3bbe0a53"
    for name in ("preprocess.py", "deconv.py"):
        if not (vendor / name).is_file():
            raise FileNotFoundError(f"Missing NLSDeconv vendor file: {vendor / name}")
    __import__("scanpy")
    __import__("anndata")
    __import__("torch")
    print("NLSDECONV_READY")


def validate_card(repo_root: Path, rscript: str) -> None:
    executable = shutil.which(rscript)
    if executable is None:
        raise FileNotFoundError(f"Rscript executable not found: {rscript}")
    library_dir = repo_root / "baseline_validation" / "R_library"
    environment = os.environ.copy()
    environment["R_LIBS_USER"] = str(library_dir)
    process = subprocess.run(
        [
            executable,
            "-e",
            "suppressPackageStartupMessages(library(CARD));cat('CARD_READY\\n')",
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stdout + process.stderr)
    print(process.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="nlsdeconv,card")
    parser.add_argument("--rscript", default=os.environ.get("RSCRIPT_BIN", "Rscript"))
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    methods = [value.lower() for value in parse_csv(args.methods)]
    validate_data(repo_root)
    if "nlsdeconv" in methods:
        validate_nls(repo_root)
    if "card" in methods:
        validate_card(repo_root, args.rscript)


if __name__ == "__main__":
    main()

