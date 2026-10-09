#!/usr/bin/env python3
"""Shared protocol, data-alignment, and fixed-ontology evaluation code."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr


DATASETS = (
    "embryo1_z2",
    "embryo1_z5",
    "embryo2_z2",
    "embryo2_z5",
    "embryo3_z2",
    "embryo3_z5",
)

# The union across all six seqFISH slices after removing the technical label
# "Low quality". Keep this order fixed for every method and protocol.
CELL_TYPES_18 = (
    "Allantois",
    "Cardiomyocytes",
    "Cranial mesoderm",
    "Dermomyotome",
    "Endothelium",
    "Erythroid",
    "Forebrain_Midbrain_Hindbrain",
    "Gut tube",
    "Haematoendothelial progenitors",
    "Intermediate mesoderm",
    "Lateral plate mesoderm",
    "Mixed mesenchymal mesoderm",
    "Neural crest",
    "Presomitic mesoderm",
    "Sclerotome",
    "Spinal cord",
    "Splanchnic mesoderm",
    "Surface ectoderm",
)
LOW_QUALITY = "Low quality"
METRIC_COLUMNS = (
    "rmse",
    "jensen_shannon_distance",
    "jensen_shannon_divergence",
    "spearman",
)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def canonical_cell_type(value: object) -> str:
    """Canonicalize known punctuation variants without changing biology."""
    label = " ".join(str(value).strip().split())
    aliases = {
        "Forebrain/Midbrain/Hindbrain": "Forebrain_Midbrain_Hindbrain",
        "Forebrain-Midbrain-Hindbrain": "Forebrain_Midbrain_Hindbrain",
        "Low-quality": LOW_QUALITY,
        "Low_quality": LOW_QUALITY,
    }
    return aliases.get(label, label)


def canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [canonical_cell_type(column) for column in result.columns]
    if result.columns.has_duplicates:
        result = result.T.groupby(level=0, sort=False).sum().T
    return result


def available_datasets(data_root: Path) -> list[str]:
    found = sorted(path.name for path in data_root.iterdir() if path.is_dir())
    missing = sorted(set(DATASETS).difference(found))
    if missing:
        raise FileNotFoundError(f"Missing benchmark datasets: {missing}")
    return list(DATASETS)


def reference_datasets(
    protocol: str, target: str, datasets: Iterable[str] = DATASETS
) -> list[str]:
    datasets = list(datasets)
    if target not in datasets:
        raise ValueError(f"Unknown target dataset: {target}")
    if protocol == "standard":
        return [target]
    if protocol == "slice":
        return [dataset for dataset in datasets if dataset != target]
    if protocol == "animal":
        animal = target.split("_z", 1)[0]
        return [
            dataset
            for dataset in datasets
            if dataset.split("_z", 1)[0] != animal
        ]
    raise ValueError(f"Unknown protocol: {protocol}")


def load_reference_labels(
    data_root: Path, references: Iterable[str]
) -> pd.DataFrame:
    tables = []
    for reference in references:
        path = data_root / reference / "sc_data" / "sc_label.tsv"
        table = pd.read_csv(path, sep="\t", dtype=str)
        if table.shape[1] < 2:
            raise ValueError(f"Reference labels require two columns: {path}")
        table = table.iloc[:, :2].copy()
        table.columns = ["cell_id", "celltype"]
        table["celltype"] = table["celltype"].map(canonical_cell_type)
        table["sampleInfo"] = reference
        table = table.loc[table["celltype"] != LOW_QUALITY]
        tables.append(table)
    result = pd.concat(tables, axis=0, ignore_index=True)
    if result["cell_id"].duplicated().any():
        duplicated = result.loc[result["cell_id"].duplicated(), "cell_id"].iloc[0]
        raise ValueError(f"Duplicate reference cell ID: {duplicated}")
    unexpected = sorted(set(result["celltype"]).difference(CELL_TYPES_18))
    if unexpected:
        raise ValueError(f"Unexpected reference cell types: {unexpected}")
    return result.set_index("cell_id")


def balanced_subsample(
    labels: pd.DataFrame, maximum_per_type: Optional[int], seed: int
) -> pd.DataFrame:
    if maximum_per_type is None:
        return labels
    sampled = []
    for _, group in labels.groupby("celltype", sort=True):
        if len(group) > maximum_per_type:
            group = group.sample(n=maximum_per_type, random_state=seed)
        sampled.append(group)
    return pd.concat(sampled, axis=0).sort_index()


def load_truth(path: Path) -> pd.DataFrame:
    truth = pd.read_csv(path, sep="\t", index_col=0)
    truth = truth.drop(columns=["cell_num", "coor_X", "coor_Y"], errors="ignore")
    return canonicalize_columns(truth)


def _validate_numeric(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result = frame.apply(pd.to_numeric, errors="raise").astype(float)
    values = result.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    if (values < -1e-10).any():
        raise ValueError(f"{name} contains negative proportions")
    return result.clip(lower=0.0)


def align_fixed_ontology(
    frame: pd.DataFrame, *, name: str
) -> pd.DataFrame:
    frame = canonicalize_columns(frame)
    frame = frame.drop(columns=[LOW_QUALITY], errors="ignore")
    frame = frame.reindex(columns=CELL_TYPES_18, fill_value=0.0)
    return _validate_numeric(frame, name)


def evaluate_frames(
    prediction: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    require_complete: bool = True,
) -> dict[str, Union[float, int]]:
    """Evaluate on the same 18 biological types and evaluable spots.

    Spots whose truth is entirely the technical ``Low quality`` class cannot be
    renormalized over the 18 biological types and are therefore excluded for
    every method. Their count is returned explicitly.
    """
    prediction = align_fixed_ontology(prediction, name="prediction")
    truth = align_fixed_ontology(truth, name="truth")
    truth_sums = truth.sum(axis=1)
    evaluable = truth_sums > 0
    evaluable_truth = truth.loc[evaluable]
    missing = evaluable_truth.index.difference(prediction.index)
    if require_complete and len(missing):
        raise ValueError(
            f"Prediction is missing {len(missing)} evaluable target spots; "
            f"first missing spot: {missing[0]}"
        )
    common = evaluable_truth.index.intersection(prediction.index, sort=False)
    if not len(common):
        raise ValueError("No evaluable prediction and truth spots overlap")
    truth_eval = evaluable_truth.loc[common]
    prediction_eval = prediction.loc[common]
    prediction_sums = prediction_eval.sum(axis=1)
    zero_predictions = prediction_sums <= 0
    if zero_predictions.any():
        spot = prediction_sums.index[zero_predictions][0]
        raise ValueError(f"Prediction has zero biological mass at spot {spot}")

    truth_values = truth_eval.to_numpy(dtype=float)
    prediction_values = prediction_eval.to_numpy(dtype=float)
    truth_values /= truth_values.sum(axis=1, keepdims=True)
    prediction_values /= prediction_values.sum(axis=1, keepdims=True)

    rmse_per_spot = np.sqrt(
        np.mean((truth_values - prediction_values) ** 2, axis=1)
    )
    js_distance = np.asarray(
        [
            jensenshannon(true_row, predicted_row)
            for true_row, predicted_row in zip(truth_values, prediction_values)
        ],
        dtype=float,
    )
    correlations = []
    for true_row, predicted_row in zip(truth_values, prediction_values):
        correlation = spearmanr(true_row, predicted_row).statistic
        correlations.append(0.0 if np.isnan(correlation) else float(correlation))

    return {
        "truth_spot_count": int(len(truth)),
        "evaluable_spot_count": int(evaluable.sum()),
        "excluded_low_quality_only_spot_count": int((~evaluable).sum()),
        "prediction_spot_count": int(len(prediction)),
        "evaluated_spot_count": int(len(common)),
        "missing_evaluable_spot_count": int(len(missing)),
        "cell_type_count": len(CELL_TYPES_18),
        "rmse": float(rmse_per_spot.mean()),
        "jensen_shannon_distance": float(js_distance.mean()),
        "jensen_shannon_divergence": float(np.square(js_distance).mean()),
        "spearman": float(np.mean(correlations)),
    }


def evaluate_prediction(
    prediction_path: Path,
    truth_path: Path,
    *,
    require_complete: bool = True,
) -> dict[str, Union[float, int]]:
    prediction = pd.read_csv(prediction_path, index_col=0)
    truth = load_truth(truth_path)
    return evaluate_frames(prediction, truth, require_complete=require_complete)


def unique_output_root(parent: Path, prefix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = parent / f"{prefix}_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{prefix}_{stamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_metric_summaries(frame: pd.DataFrame, output_root: Path) -> None:
    frame.to_csv(output_root / "all_metrics_fixed18.csv", index=False)
    group_columns = ["method", "protocol", "dataset"]
    dataset_summary = frame.groupby(group_columns)[list(METRIC_COLUMNS)].agg(
        ["mean", "std", "count"]
    )
    dataset_summary.columns = ["_".join(column) for column in dataset_summary.columns]
    dataset_summary.reset_index().to_csv(
        output_root / "metrics_by_dataset_fixed18.csv", index=False
    )
    macro_by_seed = (
        frame.groupby(["method", "protocol", "seed"])[list(METRIC_COLUMNS)]
        .mean()
        .reset_index()
    )
    macro_by_seed.to_csv(output_root / "macro_by_seed_fixed18.csv", index=False)
    macro_summary = macro_by_seed.groupby(["method", "protocol"])[
        list(METRIC_COLUMNS)
    ].agg(["mean", "std", "count"])
    macro_summary.columns = ["_".join(column) for column in macro_summary.columns]
    macro_summary.reset_index().to_csv(
        output_root / "macro_mean_sd_fixed18.csv", index=False
    )
