#!/usr/bin/env python3
"""Re-evaluate completed STD-Net predictions on the fixed 18-type ontology."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline_validation.common import (  # noqa: E402
    DATASETS,
    evaluate_prediction,
    parse_csv,
    unique_output_root,
    write_json,
    write_metric_summaries,
)


def locate_prediction(seed_root: Path, dataset: str) -> Path:
    candidates = sorted((seed_root / dataset).rglob("predict_result.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one prediction for {dataset}, found {len(candidates)} "
            f"under {seed_root}"
        )
    return candidates[0]


def discover_run(runs_root: Path, protocol: str) -> Path:
    candidates = []
    for run_root in runs_root.glob("run_*"):
        config_path = run_root / "run_configuration.json"
        failures_path = run_root / "failures.json"
        if not config_path.is_file() or not failures_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        failures = json.loads(failures_path.read_text(encoding="utf-8"))
        if config.get("protocol") != protocol or failures:
            continue
        datasets = config.get("datasets", [])
        seeds = config.get("seeds", [])
        if set(datasets) != set(DATASETS) or not seeds:
            continue
        try:
            for seed in seeds:
                for dataset in DATASETS:
                    locate_prediction(run_root / f"seed_{seed}", dataset)
        except FileNotFoundError:
            continue
        candidates.append(run_root)
    if not candidates:
        raise FileNotFoundError(f"No complete STD-Net run found for {protocol}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def parse_run_overrides(value: str) -> dict[str, Path]:
    result = {}
    if not value:
        return result
    for item in parse_csv(value):
        if "=" not in item:
            raise ValueError("--runs entries must use protocol=path")
        protocol, path = item.split("=", 1)
        result[protocol.strip()] = Path(path.strip()).resolve()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("final_data"))
    parser.add_argument(
        "--runs-root", type=Path, default=Path("paper_aligned_runs")
    )
    parser.add_argument(
        "--protocols", default="standard,slice,animal"
    )
    parser.add_argument(
        "--runs",
        default="",
        help="Optional comma list such as standard=/path/run_1,slice=/path/run_2",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_root = (repo_root / args.data_root).resolve()
    runs_root = (repo_root / args.runs_root).resolve()
    protocols = parse_csv(args.protocols)
    invalid = sorted(set(protocols).difference({"standard", "slice", "animal"}))
    if invalid:
        raise ValueError(f"Unknown protocols: {invalid}")
    overrides = parse_run_overrides(args.runs)
    run_roots = {
        protocol: overrides.get(protocol) or discover_run(runs_root, protocol)
        for protocol in protocols
    }
    output_root = (
        args.output.resolve()
        if args.output
        else unique_output_root(runs_root, "fixed18_reevaluation")
    )
    if args.output:
        output_root.mkdir(parents=True, exist_ok=False)

    records = []
    source_manifest = {}
    for protocol, run_root in run_roots.items():
        config = json.loads(
            (run_root / "run_configuration.json").read_text(encoding="utf-8")
        )
        source_manifest[protocol] = str(run_root)
        for seed in config["seeds"]:
            for dataset in DATASETS:
                prediction_path = locate_prediction(
                    run_root / f"seed_{seed}", dataset
                )
                metrics = evaluate_prediction(
                    prediction_path,
                    data_root / dataset / "st_data" / "ST_ground_truth.tsv",
                )
                metrics.update(
                    {
                        "method": "STD-Net",
                        "protocol": protocol,
                        "dataset": dataset,
                        "seed": int(seed),
                        "prediction_path": str(prediction_path),
                    }
                )
                records.append(metrics)
                print(
                    f"DONE method=STD-Net protocol={protocol} "
                    f"dataset={dataset} seed={seed}",
                    flush=True,
                )

    frame = pd.DataFrame(records)
    write_metric_summaries(frame, output_root)
    write_json(
        output_root / "reevaluation_manifest.json",
        {
            "source_runs": source_manifest,
            "ontology": "18 biological cell types; Low quality removed",
            "spot_rule": (
                "Spots containing no biological ground-truth mass after removal "
                "of Low quality are excluded for every method."
            ),
            "metric_rule": (
                "RMSE, Jensen-Shannon distance, squared Jensen-Shannon distance "
                "(divergence), and Spearman correlation are averaged over spots."
            ),
        },
    )
    print(f"FIXED-18 RE-EVALUATION COMPLETED: {output_root}")


if __name__ == "__main__":
    main()

