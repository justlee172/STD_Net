#!/usr/bin/env python3
"""Collect the newest complete v2 run for each profile/protocol pair."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "rmse",
    "jensen_shannon_distance",
    "jensen_shannon_divergence",
    "spearman",
    "predicted_entropy",
    "predicted_effective_cell_types",
    "support_f1_at_0_01",
    "rare_rmse",
    "rare_mae",
    "rare_average_precision",
]


def load_complete_runs(runs_root: Path) -> list[dict]:
    candidates = []
    for run_dir in sorted(runs_root.glob("run_*")):
        config_file = run_dir / "run_configuration.json"
        metrics_file = run_dir / "all_seed_metrics.csv"
        failures_file = run_dir / "failures.json"
        if not config_file.exists() or not metrics_file.exists():
            continue
        if failures_file.exists():
            failures = json.loads(failures_file.read_text(encoding="utf-8"))
            if failures:
                continue
        config = json.loads(config_file.read_text(encoding="utf-8"))
        frame = pd.read_csv(metrics_file)
        if frame.empty:
            continue
        profile = config.get("profile", "legacy")
        protocol = config.get("protocol", "standard")
        frame["profile"] = profile
        frame["protocol"] = protocol
        frame["run_id"] = run_dir.name
        candidates.append(
            {
                "run_dir": run_dir,
                "profile": profile,
                "protocol": protocol,
                "frame": frame,
            }
        )
    return candidates


def newest_by_configuration(candidates: list[dict]) -> list[dict]:
    newest = {}
    for candidate in candidates:
        key = (candidate["profile"], candidate["protocol"])
        if key not in newest or candidate["run_dir"].name > newest[key]["run_dir"].name:
            newest[key] = candidate
    return sorted(newest.values(), key=lambda item: (item["protocol"], item["profile"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root", type=Path, default=Path("paper_aligned_runs")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(".ecc/benchmarks/stdnet_pre_v2_20260813.json"),
    )
    args = parser.parse_args()

    selected = newest_by_configuration(load_complete_runs(args.runs_root))
    if not selected:
        raise SystemExit("No complete paper_aligned_runs were found")
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or Path("paper_result_tables") / f"v2_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)

    all_rows = pd.concat([item["frame"] for item in selected], ignore_index=True)
    available_metrics = [metric for metric in METRICS if metric in all_rows.columns]
    all_rows.to_csv(output / "selected_complete_runs.csv", index=False)

    summary = all_rows.groupby(["profile", "protocol"])[available_metrics].agg(
        ["mean", "std", "count"]
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(output / "macro_mean_sd.csv", index=False)

    per_dataset = all_rows.groupby(
        ["profile", "protocol", "dataset"]
    )[available_metrics].agg(["mean", "std", "count"])
    per_dataset.columns = ["_".join(column) for column in per_dataset.columns]
    per_dataset.reset_index().to_csv(
        output / "dataset_mean_sd.csv", index=False
    )

    if args.baseline.exists():
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        regression_rows = []
        full_rows = all_rows[all_rows["profile"] == "full_v2"]
        for protocol, protocol_rows in full_rows.groupby("protocol"):
            for metric, direction in baseline["metric_directions"].items():
                if metric not in protocol_rows or protocol not in baseline["STD-Net"]:
                    continue
                current = float(protocol_rows[metric].mean())
                historical = float(baseline["STD-Net"][protocol][metric])
                comparator = float(baseline["best_new_baseline"][protocol][metric])
                historical_delta = current - historical
                comparator_delta = current - comparator
                improved = (
                    historical_delta < 0 if direction == "lower"
                    else historical_delta > 0
                )
                beats_comparator = (
                    comparator_delta < 0 if direction == "lower"
                    else comparator_delta > 0
                )
                regression_rows.append(
                    {
                        "protocol": protocol,
                        "metric": metric,
                        "direction": direction,
                        "historical_stdnet": historical,
                        "current_full_v2": current,
                        "delta_vs_historical": historical_delta,
                        "improved_vs_historical": improved,
                        "best_new_baseline": comparator,
                        "delta_vs_best_new_baseline": comparator_delta,
                        "beats_best_new_baseline": beats_comparator,
                    }
                )
        pd.DataFrame(regression_rows).to_csv(
            output / "before_after_regression.csv", index=False
        )

    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "selection_rule": "newest complete run for each profile/protocol pair",
        "selected_runs": [str(item["run_dir"]) for item in selected],
        "metrics": available_metrics,
        "baseline_file": str(args.baseline) if args.baseline.exists() else None,
        "nonfinite_counts": {
            metric: int((~np.isfinite(all_rows[metric].to_numpy(dtype=float))).sum())
            for metric in available_metrics
        },
    }
    (output / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"PAPER TABLES READY: {output.resolve()}")


if __name__ == "__main__":
    main()
