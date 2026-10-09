#!/usr/bin/env python3
"""Combine fixed-18 STD-Net and new-baseline metrics into paper-ready CSVs."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline_validation.common import (  # noqa: E402
    DATASETS,
    METRIC_COLUMNS,
    unique_output_root,
)


LOWER_IS_BETTER = {
    "rmse",
    "jensen_shannon_distance",
    "jensen_shannon_divergence",
}
BOOTSTRAP_REPLICATES = 20000


def latest_with_file(parent: Path, pattern: str, filename: str) -> Path:
    candidates = sorted(
        path for path in parent.glob(pattern) if (path / filename).is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"No {filename} found below {parent} with {pattern}")
    return candidates[-1]


def format_mean_sd(mean: float, std: float, count: int) -> str:
    if count <= 1 or not np.isfinite(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def summarize(frame: pd.DataFrame, groups: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = frame.groupby(groups)[list(METRIC_COLUMNS)].agg(["mean", "std", "count"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    formatted = summary.loc[:, groups].copy()
    for metric in METRIC_COLUMNS:
        formatted[metric] = summary.apply(
            lambda row: format_mean_sd(
                row[f"{metric}_mean"], row[f"{metric}_std"], int(row[f"{metric}_count"])
            ),
            axis=1,
        )
    return summary, formatted


def exact_sign_flip_p(effects: np.ndarray) -> float:
    """Two-sided exact paired randomization test on the mean effect."""
    effects = np.asarray(effects, dtype=float)
    effects = effects[np.isfinite(effects) & (effects != 0)]
    if not len(effects):
        return 1.0
    observed = abs(float(effects.mean()))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(effects)):
        permuted = abs(float(np.mean(effects * np.asarray(signs))))
        extreme += permuted >= observed - 1e-15
        total += 1
    return float(extreme / total)


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Holm family-wise error correction with monotone adjusted values."""
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return pd.Series(adjusted, index=p_values.index)


def _bootstrap_effects(
    seed_effects: pd.DataFrame,
    *,
    unit_level: str,
    replicates: int,
    random_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    dataset_values = {
        dataset: group["improvement"].to_numpy(dtype=float)
        for dataset, group in seed_effects.groupby("dataset")
    }
    dataset_animals = {
        dataset: dataset.split("_z", 1)[0] for dataset in dataset_values
    }
    if unit_level == "slice":
        units = sorted(dataset_values)
        unit_datasets = {unit: [unit] for unit in units}
    elif unit_level == "animal":
        units = sorted(set(dataset_animals.values()))
        unit_datasets = {
            unit: sorted(
                dataset
                for dataset, animal in dataset_animals.items()
                if animal == unit
            )
            for unit in units
        }
    else:
        raise ValueError(f"Unknown comparison unit: {unit_level}")

    seed_counts = {len(values) for values in dataset_values.values()}
    if len(seed_counts) != 1:
        raise ValueError("Every dataset must have the same STD-Net seed count")
    seed_count = seed_counts.pop()
    datasets_per_unit = {len(unit_datasets[unit]) for unit in units}
    if len(datasets_per_unit) != 1:
        raise ValueError("Every comparison unit must contain the same slice count")
    dataset_count = datasets_per_unit.pop()
    effect_matrix = np.stack(
        [
            np.stack([dataset_values[dataset] for dataset in unit_datasets[unit]])
            for unit in units
        ]
    )
    unit_indices = rng.integers(0, len(units), size=(replicates, len(units)))
    seed_indices = rng.integers(
        0,
        seed_count,
        size=(replicates, len(units), dataset_count),
    )
    dataset_indices = np.broadcast_to(
        np.arange(dataset_count),
        (replicates, len(units), dataset_count),
    )
    sampled = effect_matrix[
        unit_indices[:, :, None],
        dataset_indices,
        seed_indices,
    ]
    return sampled.mean(axis=(1, 2))


def paired_statistical_comparisons(
    combined: pd.DataFrame,
    *,
    unit_level: str,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare stochastic STD-Net with deterministic baselines without pseudoreplication."""
    summaries = []
    effect_records = []
    baselines = ("CARD", "NLSDeconv")
    protocols = ("standard", "slice", "animal")
    for baseline_index, baseline_method in enumerate(baselines):
        for protocol_index, protocol in enumerate(protocols):
            stdnet = combined.loc[
                (combined["method"] == "STD-Net")
                & (combined["protocol"] == protocol)
            ]
            baseline = combined.loc[
                (combined["method"] == baseline_method)
                & (combined["protocol"] == protocol)
            ]
            for metric_index, metric in enumerate(METRIC_COLUMNS):
                baseline_values = baseline.groupby("dataset")[metric].agg(
                    ["mean", "nunique"]
                )
                if (baseline_values["nunique"] != 1).any():
                    raise ValueError(
                        f"{baseline_method} is not deterministic for {protocol}/{metric}"
                    )
                baseline_values = baseline_values["mean"].rename("baseline_value")
                seed_effects = stdnet[["dataset", "seed", metric]].merge(
                    baseline_values,
                    left_on="dataset",
                    right_index=True,
                    validate="many_to_one",
                )
                if set(seed_effects["dataset"]) != set(DATASETS):
                    raise ValueError(
                        f"Incomplete paired datasets for {baseline_method}/{protocol}/{metric}"
                    )
                if metric in LOWER_IS_BETTER:
                    seed_effects["improvement"] = (
                        seed_effects["baseline_value"] - seed_effects[metric]
                    )
                    direction = "positive means lower STD-Net error"
                else:
                    seed_effects["improvement"] = (
                        seed_effects[metric] - seed_effects["baseline_value"]
                    )
                    direction = "positive means higher STD-Net correlation"

                dataset_effects = seed_effects.groupby("dataset").agg(
                    stdnet_value=(metric, "mean"),
                    baseline_value=("baseline_value", "first"),
                    improvement=("improvement", "mean"),
                    stdnet_seed_sd=(metric, "std"),
                    stdnet_seed_count=("seed", "nunique"),
                ).reset_index()
                dataset_effects["animal"] = dataset_effects["dataset"].str.split(
                    "_z", n=1
                ).str[0]
                if unit_level == "slice":
                    unit_effects = dataset_effects.assign(
                        unit=dataset_effects["dataset"]
                    )
                else:
                    unit_effects = dataset_effects.groupby("animal", as_index=False).agg(
                        stdnet_value=("stdnet_value", "mean"),
                        baseline_value=("baseline_value", "mean"),
                        improvement=("improvement", "mean"),
                    ).rename(columns={"animal": "unit"})

                for row in unit_effects.itertuples(index=False):
                    effect_records.append(
                        {
                            "comparison_unit": unit_level,
                            "comparator": baseline_method,
                            "protocol": protocol,
                            "metric": metric,
                            "unit": row.unit,
                            "stdnet_value": row.stdnet_value,
                            "comparator_value": row.baseline_value,
                            "improvement": row.improvement,
                        }
                    )

                effects = unit_effects["improvement"].to_numpy(dtype=float)
                bootstrap = _bootstrap_effects(
                    seed_effects,
                    unit_level=unit_level,
                    replicates=bootstrap_replicates,
                    random_seed=(
                        20260908
                        + baseline_index * 1000
                        + protocol_index * 100
                        + metric_index
                    ),
                )
                baseline_mean = float(unit_effects["baseline_value"].mean())
                mean_improvement = float(effects.mean())
                summaries.append(
                    {
                        "comparison_unit": unit_level,
                        "stdnet_method": "STD-Net",
                        "comparator": baseline_method,
                        "protocol": protocol,
                        "metric": metric,
                        "direction": direction,
                        "independent_unit_count": int(len(unit_effects)),
                        "stdnet_seed_count": int(
                            seed_effects.groupby("dataset")["seed"].nunique().min()
                        ),
                        "stdnet_mean": float(unit_effects["stdnet_value"].mean()),
                        "comparator_mean": baseline_mean,
                        "mean_improvement": mean_improvement,
                        "relative_improvement_percent": (
                            100.0 * mean_improvement / abs(baseline_mean)
                            if baseline_mean != 0
                            else np.nan
                        ),
                        "bootstrap_95_ci_lower": float(np.quantile(bootstrap, 0.025)),
                        "bootstrap_95_ci_upper": float(np.quantile(bootstrap, 0.975)),
                        "stdnet_better_unit_count": int((effects > 0).sum()),
                        "exact_sign_flip_p_two_sided": exact_sign_flip_p(effects),
                    }
                )

    summary = pd.DataFrame(summaries)
    summary["holm_p_all_24_comparisons"] = holm_adjust(
        summary["exact_sign_flip_p_two_sided"]
    )
    summary["holm_p_within_protocol"] = summary.groupby("protocol", group_keys=False)[
        "exact_sign_flip_p_two_sided"
    ].apply(holm_adjust)
    return summary, pd.DataFrame(effect_records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdnet", type=Path)
    parser.add_argument("--baselines", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Only for development tests; never use for manuscript tables",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    stdnet_root = args.stdnet or latest_with_file(
        repo_root / "paper_aligned_runs",
        "fixed18_reevaluation_*",
        "all_metrics_fixed18.csv",
    )
    baseline_root = args.baselines or latest_with_file(
        repo_root / "baseline_runs", "run_*", "all_metrics_fixed18.csv"
    )
    baseline_root = Path(baseline_root)
    baseline_config = json.loads(
        (baseline_root / "run_configuration.json").read_text(encoding="utf-8")
    )
    if baseline_config.get("quick") and not args.allow_incomplete:
        raise ValueError("Refusing to build manuscript tables from a quick baseline run")
    frames = [
        pd.read_csv(Path(stdnet_root) / "all_metrics_fixed18.csv"),
        pd.read_csv(baseline_root / "all_metrics_fixed18.csv"),
    ]
    combined = pd.concat(frames, axis=0, ignore_index=True)
    key = ["method", "protocol", "dataset", "seed"]
    if combined.duplicated(key).any():
        duplicates = combined.loc[combined.duplicated(key, keep=False), key]
        raise ValueError(f"Duplicate result keys detected:\n{duplicates}")
    if not args.allow_incomplete:
        expected_protocols = {"standard", "slice", "animal"}
        expected_methods = {"NLSDeconv", "CARD", "STD-Net"}
        if set(combined["protocol"]) != expected_protocols:
            raise ValueError("All three protocols are required for manuscript tables")
        if set(combined["method"]) != expected_methods:
            raise ValueError("STD-Net, NLSDeconv, and CARD are all required")
        for (method, protocol), group in combined.groupby(["method", "protocol"]):
            if set(group["dataset"]) != set(DATASETS):
                raise ValueError(f"Incomplete datasets for {method}, {protocol}")
    output_root = (
        args.output.resolve()
        if args.output
        else unique_output_root(repo_root / "paper_result_tables", "fixed18")
    )
    if args.output:
        output_root.mkdir(parents=True, exist_ok=False)
    csv_options = {"index": False, "encoding": "utf-8-sig"}
    combined.to_csv(output_root / "combined_all_metrics_fixed18.csv", **csv_options)

    by_dataset, by_dataset_formatted = summarize(
        combined, ["method", "protocol", "dataset"]
    )
    by_dataset.to_csv(output_root / "comparison_by_dataset.csv", **csv_options)
    by_dataset_formatted.to_csv(
        output_root / "comparison_by_dataset_formatted.csv", **csv_options
    )
    macro_by_seed = (
        combined.groupby(["method", "protocol", "seed"])[list(METRIC_COLUMNS)]
        .mean()
        .reset_index()
    )
    macro, macro_formatted = summarize(macro_by_seed, ["method", "protocol"])
    macro.to_csv(output_root / "comparison_macro.csv", **csv_options)
    macro_formatted.to_csv(
        output_root / "comparison_macro_formatted.csv", **csv_options
    )
    counts = (
        combined.groupby(["dataset"])[
            [
                "truth_spot_count",
                "evaluable_spot_count",
                "excluded_low_quality_only_spot_count",
                "cell_type_count",
            ]
        ]
        .nunique()
    )
    if (counts > 1).any().any():
        raise ValueError("Evaluation spot or ontology counts differ between methods")
    combined.groupby("dataset")[[
        "truth_spot_count",
        "evaluable_spot_count",
        "excluded_low_quality_only_spot_count",
        "cell_type_count",
    ]].first().reset_index().to_csv(output_root / "evaluation_counts.csv", **csv_options)
    for unit_level in ("slice", "animal"):
        statistical_summary, unit_effects = paired_statistical_comparisons(
            combined, unit_level=unit_level
        )
        statistical_summary.to_csv(
            output_root / f"paired_statistical_comparisons_{unit_level}.csv",
            **csv_options,
        )
        unit_effects.to_csv(
            output_root / f"paired_effects_{unit_level}.csv",
            **csv_options,
        )
    (output_root / "statistical_analysis_manifest.json").write_text(
        json.dumps(
            {
                "deterministic_baselines": ["CARD", "NLSDeconv"],
                "baseline_seed_policy": (
                    "One run per deterministic baseline; repeated identical seeds "
                    "are not treated as replicates."
                ),
                "stdnet_seed_policy": (
                    "Five STD-Net seeds are averaged within each target dataset "
                    "for the paired point estimate."
                ),
                "effect_direction": "Positive values favor STD-Net.",
                "confidence_interval": (
                    "20,000-replicate hierarchical bootstrap. Independent units "
                    "are resampled with replacement and one STD-Net seed is sampled "
                    "within each selected dataset."
                ),
                "paired_test": (
                    "Exact two-sided sign-flip randomization test on paired unit "
                    "effects, with Holm correction both within protocol and across "
                    "all 24 comparisons."
                ),
                "unit_interpretation": {
                    "slice": "Six target slices; exploratory benchmark inference.",
                    "animal": (
                        "Three embryo clusters after averaging their two slices; "
                        "conservative primary biological inference."
                    ),
                },
                "animal_power_limit": (
                    "With three independent animals, the minimum attainable exact "
                    "two-sided sign-flip p value is 0.25."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"PAPER TABLES COMPLETED: {output_root}")


if __name__ == "__main__":
    main()
