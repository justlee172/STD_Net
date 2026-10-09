#!/usr/bin/env python3
"""Run NLSDeconv and CARD under standard, slice, and animal protocols."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline_validation.common import (  # noqa: E402
    CELL_TYPES_18,
    DATASETS,
    available_datasets,
    balanced_subsample,
    evaluate_prediction,
    load_reference_labels,
    parse_csv,
    reference_datasets,
    unique_output_root,
    write_json,
    write_metric_summaries,
)


NLSDECONV_COMMIT = "3bbe0a53a8b0f6798eacf7eab9f7298742fd27fd"
CARD_COMMIT = "2d64b91abb5cdd0c7f576b1c5d4727c84e7c93a0"


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw_counts(path: Path) -> pd.DataFrame:
    print(f"Loading raw single-cell counts once: {path}", flush=True)
    # ``dtype`` cannot be applied during parsing because the first, unnamed
    # column contains gene identifiers. Cast only after it becomes the index.
    counts = pd.read_csv(path, sep="\t", index_col=0).astype(np.float32)
    counts.index = counts.index.astype(str)
    counts.columns = counts.columns.astype(str)
    if counts.index.has_duplicates or counts.columns.has_duplicates:
        raise ValueError("Raw count matrix has duplicate gene or cell identifiers")
    if (counts.to_numpy() < 0).any():
        raise ValueError("Raw count matrix contains negative values")
    print(f"Raw counts loaded: genes={counts.shape[0]} cells={counts.shape[1]}")
    return counts


def prepare_task(
    *,
    data_root: Path,
    raw_counts: pd.DataFrame,
    protocol: str,
    dataset: str,
    seed: int,
    quick: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    references = reference_datasets(protocol, dataset, DATASETS)
    labels = load_reference_labels(data_root, references)
    labels = balanced_subsample(labels, 40 if quick else None, seed)
    missing_cells = labels.index.difference(raw_counts.columns)
    if len(missing_cells):
        raise ValueError(
            f"Raw count matrix is missing {len(missing_cells)} reference cells; "
            f"first missing cell: {missing_cells[0]}"
        )

    st_path = data_root / dataset / "st_data" / "ST_data.tsv"
    st_expression = pd.read_csv(st_path, sep="\t", index_col=0)
    st_expression.index = st_expression.index.astype(str)
    st_expression.columns = st_expression.columns.astype(str)
    if quick:
        st_expression = st_expression.iloc[:50].copy()
    common_genes = sorted(set(st_expression.columns).intersection(raw_counts.index))
    if len(common_genes) < 2:
        raise ValueError(f"Too few common genes for {dataset}: {len(common_genes)}")

    sc_expression = raw_counts.loc[common_genes, labels.index].T.copy()
    st_expression = st_expression.loc[:, common_genes].copy()
    coordinates = pd.read_csv(
        data_root / dataset / "st_data" / "coordinates.csv", index_col=0
    )
    coordinates.index = coordinates.index.astype(str)
    coordinates = coordinates.loc[st_expression.index]
    if coordinates.shape[1] < 2:
        raise ValueError(f"Coordinates require two columns for {dataset}")
    coordinates = coordinates.iloc[:, :2].copy()
    coordinates.columns = ["x", "y"]
    return sc_expression, labels, st_expression, coordinates, references


def run_nlsdeconv(
    *,
    vendor_root: Path,
    sc_expression: pd.DataFrame,
    labels: pd.DataFrame,
    st_expression: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not (vendor_root / "preprocess.py").is_file():
        raise FileNotFoundError(
            f"NLSDeconv is not installed at {vendor_root}; run the installer first"
        )
    vendor_text = str(vendor_root)
    if vendor_text not in sys.path:
        sys.path.insert(0, vendor_text)
    from anndata import AnnData
    from deconv import Deconv
    from preprocess import Preprocess

    sc_adata = AnnData(
        X=sc_expression.to_numpy(dtype=np.float32),
        obs=labels.loc[sc_expression.index, ["celltype"]].copy(),
        var=pd.DataFrame(index=sc_expression.columns),
    )
    st_adata = AnnData(
        X=st_expression.to_numpy(dtype=np.float32),
        obs=pd.DataFrame(index=st_expression.index),
        var=pd.DataFrame(index=st_expression.columns),
    )
    start = time.perf_counter()
    st_processed, sc_processed = Preprocess(
        st_adata,
        sc_adata,
        celltype_key="celltype",
        cellcount_min=2,
        gene_top=200,
        cellcount_norm=True,
    ).preprocess()
    values, method_seconds, output_types = Deconv(
        sc_processed,
        st_processed,
        celltype_key="celltype",
        normalization=False,
    ).SLS()
    elapsed = time.perf_counter() - start
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    prediction = pd.DataFrame(
        np.asarray(values, dtype=float),
        index=st_processed.obs_names.astype(str),
        columns=[str(value) for value in output_types],
    )
    metadata = {
        "algorithm": "NLSDeconv SLS",
        "official_commit": NLSDECONV_COMMIT,
        "preprocessed_gene_count": int(sc_processed.n_vars),
        "method_reported_seconds": float(method_seconds),
        "wall_seconds": float(elapsed),
    }
    return prediction, metadata


def run_card(
    *,
    rscript: str,
    r_runner: Path,
    sc_expression: pd.DataFrame,
    labels: pd.DataFrame,
    st_expression: pd.DataFrame,
    coordinates: pd.DataFrame,
    seed: int,
    task_root: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    with tempfile.TemporaryDirectory(prefix="stdnet_card_") as temporary:
        work_root = Path(temporary)
        sc_path = work_root / "sc_expression.tsv"
        labels_path = work_root / "sc_labels.tsv"
        st_path = work_root / "st_expression.tsv"
        coordinates_path = work_root / "coordinates.csv"
        prediction_path = work_root / "predict_result.csv"
        sc_expression.to_csv(sc_path, sep="\t")
        labels.loc[sc_expression.index, ["celltype", "sampleInfo"]].to_csv(
            labels_path, sep="\t"
        )
        st_expression.to_csv(st_path, sep="\t")
        coordinates.loc[st_expression.index].to_csv(coordinates_path)
        command = [
            rscript,
            str(r_runner),
            "--sc-expression",
            str(sc_path),
            "--sc-labels",
            str(labels_path),
            "--st-expression",
            str(st_path),
            "--coordinates",
            str(coordinates_path),
            "--output",
            str(prediction_path),
            "--seed",
            str(seed),
        ]
        log_path = task_root / "card_console.log"
        start = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.run(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        elapsed = time.perf_counter() - start
        if process.returncode:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            raise RuntimeError(
                f"CARD exited with code {process.returncode}. Last output:\n"
                + "\n".join(tail[-30:])
            )
        prediction = pd.read_csv(prediction_path, index_col=0)
    metadata = {
        "algorithm": "CARD reference-based deconvolution",
        "official_commit": CARD_COMMIT,
        "official_internal_seed": 20200107,
        "minCountGene": 1,
        "minCountSpot": 1,
        "wall_seconds": float(elapsed),
        "console_log": str(log_path),
    }
    return prediction, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("final_data"))
    parser.add_argument(
        "--raw-sc-counts",
        type=Path,
        default=Path("original_data/Lohoff_seqFISH/counts.txt"),
    )
    parser.add_argument("--methods", default="nlsdeconv,card")
    parser.add_argument("--protocols", default="standard,slice,animal")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--nls-vendor",
        type=Path,
        default=Path("baseline_validation/vendor/NLSDeconv_3bbe0a53"),
    )
    parser.add_argument("--rscript", default=os.environ.get("RSCRIPT_BIN", "Rscript"))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    data_root = (repo_root / args.data_root).resolve()
    raw_path = (repo_root / args.raw_sc_counts).resolve()
    nls_vendor = (repo_root / args.nls_vendor).resolve()
    methods = [item.lower() for item in parse_csv(args.methods)]
    protocols = [item.lower() for item in parse_csv(args.protocols)]
    datasets = parse_csv(args.datasets) if args.datasets else available_datasets(data_root)
    seeds = [int(value) for value in parse_csv(args.seeds)]
    invalid_methods = sorted(set(methods).difference({"nlsdeconv", "card"}))
    invalid_protocols = sorted(
        set(protocols).difference({"standard", "slice", "animal"})
    )
    if invalid_methods or invalid_protocols:
        raise ValueError(
            f"Invalid methods={invalid_methods}, protocols={invalid_protocols}"
        )
    unknown_datasets = sorted(set(datasets).difference(DATASETS))
    if unknown_datasets:
        raise ValueError(f"Unknown datasets: {unknown_datasets}")
    if not seeds:
        raise ValueError("At least one seed is required")
    if len(seeds) > 1:
        print(
            "WARNING: NLSDeconv SLS is deterministic and official CARD resets "
            "its internal seed to 20200107; multiple seeds are expected to duplicate results.",
            file=sys.stderr,
        )
    if args.quick:
        protocols = protocols[:1]
        datasets = datasets[:1]
        seeds = seeds[:1]

    output_root = (
        args.output.resolve()
        if args.output
        else unique_output_root(repo_root / "baseline_runs", "run")
    )
    if args.output:
        output_root.mkdir(parents=True, exist_ok=False)
    raw_counts = load_raw_counts(raw_path)
    r_runner = repo_root / "baseline_validation" / "run_card.R"
    configuration = {
        "methods": methods,
        "protocols": protocols,
        "datasets": datasets,
        "seeds": seeds,
        "quick": args.quick,
        "raw_sc_counts": str(raw_path),
        "raw_sc_counts_sha256": sha256_file(raw_path),
        "ontology": list(CELL_TYPES_18),
        "low_quality_rule": (
            "Low quality cells are removed from references. Ground-truth spots "
            "with no remaining biological mass are excluded from every method."
        ),
        "protocol_note": (
            "Slice and animal settings hold out the target single-cell reference "
            "slice or animal. Target ST expression and coordinates remain available, "
            "so these are transductive reference-holdout evaluations."
        ),
        "commits": {
            "NLSDeconv": NLSDECONV_COMMIT,
            "CARD": CARD_COMMIT,
        },
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    write_json(output_root / "run_configuration.json", configuration)

    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for method in methods:
        for protocol in protocols:
            for seed in seeds:
                set_seed(seed)
                for dataset in datasets:
                    task_root = (
                        output_root
                        / method
                        / protocol
                        / f"seed_{seed}"
                        / dataset
                    )
                    task_root.mkdir(parents=True, exist_ok=False)
                    print(
                        f"START method={method} protocol={protocol} "
                        f"dataset={dataset} seed={seed}",
                        flush=True,
                    )
                    try:
                        (
                            sc_expression,
                            labels,
                            st_expression,
                            coordinates,
                            references,
                        ) = prepare_task(
                            data_root=data_root,
                            raw_counts=raw_counts,
                            protocol=protocol,
                            dataset=dataset,
                            seed=seed,
                            quick=args.quick,
                        )
                        if method == "nlsdeconv":
                            prediction, runtime = run_nlsdeconv(
                                vendor_root=nls_vendor,
                                sc_expression=sc_expression,
                                labels=labels,
                                st_expression=st_expression,
                            )
                        else:
                            prediction, runtime = run_card(
                                rscript=args.rscript,
                                r_runner=r_runner,
                                sc_expression=sc_expression,
                                labels=labels,
                                st_expression=st_expression,
                                coordinates=coordinates,
                                seed=seed,
                                task_root=task_root,
                            )
                        prediction_path = task_root / "predict_result.csv"
                        prediction.to_csv(prediction_path)
                        metrics = evaluate_prediction(
                            prediction_path,
                            data_root
                            / dataset
                            / "st_data"
                            / "ST_ground_truth.tsv",
                            require_complete=not args.quick,
                        )
                        metrics.update(
                            {
                                "method": "NLSDeconv" if method == "nlsdeconv" else "CARD",
                                "protocol": protocol,
                                "dataset": dataset,
                                "seed": seed,
                            }
                        )
                        task_metadata = {
                            "method": metrics["method"],
                            "protocol": protocol,
                            "dataset": dataset,
                            "seed": seed,
                            "reference_datasets": references,
                            "reference_cell_count": int(len(sc_expression)),
                            "reference_cell_types": sorted(labels["celltype"].unique()),
                            "gene_count": int(sc_expression.shape[1]),
                            "target_spot_count": int(len(st_expression)),
                            "target_ground_truth_used_by_method": False,
                            "runtime": runtime,
                        }
                        write_json(task_root / "task_metadata.json", task_metadata)
                        write_json(task_root / "final_metrics_fixed18.json", metrics)
                        records.append(metrics)
                        print(f"DONE {metrics}", flush=True)
                    except Exception as error:  # continue independent tasks
                        failure = {
                            "method": method,
                            "protocol": protocol,
                            "dataset": dataset,
                            "seed": seed,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                        failures.append(failure)
                        write_json(task_root / "failure.json", failure)
                        print(f"FAILED {failure}", file=sys.stderr, flush=True)

    if records:
        write_metric_summaries(pd.DataFrame(records), output_root)
    write_json(output_root / "failures.json", failures)
    if failures:
        raise SystemExit(
            f"{len(failures)} baseline tasks failed; see {output_root / 'failures.json'}"
        )
    print(f"ALL BASELINE RUNS COMPLETED: {output_root}")


if __name__ == "__main__":
    main()
