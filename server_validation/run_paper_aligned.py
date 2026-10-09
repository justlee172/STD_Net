#!/usr/bin/env python3
"""Run the paper-aligned STD-Net with a target-label firewall."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw_counts(path: Path):
    import numpy as np
    import pandas as pd

    counts = pd.read_csv(path, sep="\t", index_col=0).astype(np.float32)
    counts.index = counts.index.astype(str)
    counts.columns = counts.columns.astype(str)
    if counts.index.has_duplicates or counts.columns.has_duplicates:
        raise ValueError("Raw count matrix has duplicate gene or cell identifiers")
    values = counts.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Raw count matrix contains invalid values")
    return counts


def load_filtered_reference_tables(reference_root: Path):
    """Load aligned reference metadata after removing technical labels."""
    import pandas as pd

    label_path = reference_root / "sc_label.tsv"
    coordinate_path = reference_root / "sc_coordinates.csv"
    labels = pd.read_csv(label_path, sep="\t", dtype=str)
    if labels.shape[1] < 2:
        raise ValueError(f"Reference labels require two columns: {label_path}")
    id_column = labels.columns[0]
    normalized_columns = {
        str(column).strip().lower(): column for column in labels.columns
    }
    label_column = next(
        (
            normalized_columns[name]
            for name in ("cell_type", "celltype", "cell_type_label", "label")
            if name in normalized_columns
        ),
        labels.columns[1],
    )
    normalized_labels = labels[label_column].map(
        lambda value: " ".join(str(value).strip().split()).casefold()
    )
    low_quality_labels = {"low quality", "low-quality", "low_quality"}
    keep = ~normalized_labels.isin(low_quality_labels)
    excluded_count = int((~keep).sum())
    labels = labels.loc[keep].copy()
    labels[id_column] = labels[id_column].astype(str)
    cell_ids = labels[id_column].tolist()
    if not cell_ids:
        raise ValueError(f"No biological reference cells remain in {label_path}")
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError(f"Duplicate reference cell IDs in {label_path}")

    coordinates = pd.read_csv(coordinate_path, index_col=0)
    coordinates.index = coordinates.index.astype(str)
    if coordinates.index.has_duplicates:
        raise ValueError(f"Duplicate reference cell IDs in {coordinate_path}")
    missing_coordinates = [
        cell_id for cell_id in cell_ids if cell_id not in coordinates.index
    ]
    if missing_coordinates:
        raise ValueError(
            f"Reference coordinates are missing {len(missing_coordinates)} cells "
            f"in {coordinate_path}; first missing cell: {missing_coordinates[0]}"
        )
    return labels, coordinates.loc[cell_ids].copy(), cell_ids, excluded_count


def raw_expression_for_cells(raw_counts, cell_ids: list[str], dataset: str):
    missing_cells = [cell_id for cell_id in cell_ids if cell_id not in raw_counts.columns]
    if missing_cells:
        raise ValueError(
            f"Raw counts are missing {len(missing_cells)} cells for {dataset}; "
            f"first missing cell: {missing_cells[0]}"
        )
    return raw_counts.loc[:, cell_ids].T.copy()


def stage_dataset(
    source_root: Path,
    staged_root: Path,
    dataset: str,
    raw_counts,
) -> None:
    source = source_root / dataset
    destination = staged_root / dataset
    required = ["st_data/ST_data.tsv", "st_data/coordinates.csv"]
    for relative_path in required:
        input_file = source / relative_path
        if not input_file.is_file():
            raise FileNotFoundError(f"Missing required input: {input_file}")
        link_or_copy(input_file, destination / relative_path)
    labels, coordinates, cell_ids, excluded_count = load_filtered_reference_tables(
        source / "sc_data"
    )
    (destination / "sc_data").mkdir(parents=True, exist_ok=True)
    labels.to_csv(
        destination / "sc_data" / "sc_label.tsv", sep="\t", index=False
    )
    coordinates.to_csv(destination / "sc_data" / "sc_coordinates.csv")
    raw_expression_for_cells(raw_counts, cell_ids, dataset).to_csv(
        destination / "sc_data" / "sc_data.tsv", sep="\t"
    )
    marker_file = source / "sc_data" / "marker_genes.tsv"
    if marker_file.is_file():
        link_or_copy(marker_file, destination / "sc_data" / marker_file.name)
    (destination / "reference_filter.json").write_text(
        json.dumps(
            {
                "excluded_reference_label": "Low quality",
                "excluded_reference_cell_count": excluded_count,
                "retained_reference_cell_count": len(cell_ids),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if (destination / "st_data" / "ST_ground_truth.tsv").exists():
        raise RuntimeError("Target-label firewall failed during staging")


def stage_grouped_dataset(
    source_root: Path,
    staged_root: Path,
    target: str,
    reference_datasets: list[str],
    raw_counts,
) -> None:
    """Pool source single cells while keeping the target ST graph unlabeled."""
    import pandas as pd

    destination = staged_root / target
    label_tables = []
    coordinate_tables = []
    excluded_low_quality_count = 0
    for reference in reference_datasets:
        reference_root = source_root / reference / "sc_data"
        labels, coordinates, _, excluded_count = load_filtered_reference_tables(
            reference_root
        )
        label_tables.append(labels)
        coordinate_tables.append(coordinates)
        excluded_low_quality_count += excluded_count
    pooled_labels = pd.concat(label_tables, axis=0, ignore_index=True)
    pooled_coordinates = pd.concat(coordinate_tables, axis=0)
    label_id_column = pooled_labels.columns[0]
    pooled_cell_ids = pooled_labels[label_id_column].astype(str).tolist()
    if len(pooled_cell_ids) != len(set(pooled_cell_ids)):
        raise ValueError(f"Duplicate reference cell IDs for target {target}")
    pooled_expression = raw_expression_for_cells(
        raw_counts, pooled_cell_ids, target
    )
    pooled_labels = pooled_labels.set_index(label_id_column).loc[
        pooled_cell_ids
    ].reset_index()
    pooled_coordinates = pooled_coordinates.loc[pooled_expression.index]

    (destination / "sc_data").mkdir(parents=True, exist_ok=True)
    pooled_expression.to_csv(
        destination / "sc_data" / "sc_data.tsv", sep="\t"
    )
    pooled_labels.to_csv(
        destination / "sc_data" / "sc_label.tsv", sep="\t", index=False
    )
    pooled_coordinates.to_csv(
        destination / "sc_data" / "sc_coordinates.csv"
    )
    for relative_path in ["st_data/ST_data.tsv", "st_data/coordinates.csv"]:
        link_or_copy(
            source_root / target / relative_path,
            destination / relative_path,
        )
    metadata = {
        "target": target,
        "reference_datasets": reference_datasets,
        "target_ground_truth_staged": False,
        "excluded_reference_label": "Low quality",
        "excluded_reference_cell_count": excluded_low_quality_count,
        "retained_reference_cell_count": len(pooled_cell_ids),
    }
    (destination / "reference_holdout.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def set_seed(seed: int, torch_module) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np

    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False


def evaluate_prediction(prediction_file: Path, truth_file: Path) -> dict[str, float]:
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score
    from baseline_validation.common import (
        CELL_TYPES_18,
        LOW_QUALITY,
        canonicalize_columns,
    )

    prediction = canonicalize_columns(pd.read_csv(prediction_file, index_col=0))
    truth = pd.read_csv(truth_file, sep="\t", index_col=0)
    truth = truth.drop(
        columns=["cell_num", "coor_X", "coor_Y"], errors="ignore"
    )
    truth = canonicalize_columns(truth)
    prediction = prediction.drop(columns=[LOW_QUALITY], errors="ignore")
    truth = truth.drop(columns=[LOW_QUALITY], errors="ignore")
    prediction = prediction.reindex(columns=CELL_TYPES_18, fill_value=0.0)
    truth = truth.reindex(columns=CELL_TYPES_18, fill_value=0.0)
    truth_sums = truth.sum(axis=1)
    evaluable = truth_sums > 0
    evaluable_truth = truth.loc[evaluable]
    common_spots = evaluable_truth.index.intersection(prediction.index, sort=False)
    if len(common_spots) != len(evaluable_truth):
        raise ValueError(
            f"Prediction covers {len(common_spots)} of "
            f"{len(evaluable_truth)} evaluable target spots"
        )
    truth = evaluable_truth.loc[common_spots]
    prediction = prediction.loc[common_spots]
    prediction_sums = prediction.sum(axis=1)
    zero_predictions = prediction_sums <= 0
    if zero_predictions.any():
        spot = prediction_sums.index[zero_predictions][0]
        raise ValueError(f"Prediction has zero biological mass at spot {spot}")

    raw_truth_values = np.clip(truth.to_numpy(dtype=float), 0, None)
    raw_prediction_values = np.clip(
        prediction.to_numpy(dtype=float), 0, None
    )
    truth_values = np.clip(raw_truth_values, 1e-10, None)
    prediction_values = np.clip(raw_prediction_values, 1e-10, None)
    truth_values /= truth_values.sum(axis=1, keepdims=True)
    prediction_values /= prediction_values.sum(axis=1, keepdims=True)

    rmse = np.sqrt(
        np.mean((truth_values - prediction_values) ** 2, axis=1)
    )
    js_distance = np.array(
        [
            jensenshannon(true_row, predicted_row)
            for true_row, predicted_row in zip(
                truth_values, prediction_values
            )
        ]
    )
    correlations = []
    for true_row, predicted_row in zip(truth_values, prediction_values):
        correlation = spearmanr(true_row, predicted_row).statistic
        correlations.append(0.0 if np.isnan(correlation) else correlation)
    predicted_entropy = -np.sum(
        prediction_values * np.log(prediction_values), axis=1
    )
    true_entropy = -np.sum(truth_values * np.log(truth_values), axis=1)
    support_threshold = 0.01
    predicted_support = prediction_values >= support_threshold
    true_support = raw_truth_values > 0
    true_positive = np.logical_and(predicted_support, true_support).sum()
    false_positive = np.logical_and(predicted_support, ~true_support).sum()
    false_negative = np.logical_and(~predicted_support, true_support).sum()
    support_precision = true_positive / max(true_positive + false_positive, 1)
    support_recall = true_positive / max(true_positive + false_negative, 1)
    support_f1 = 2 * support_precision * support_recall / max(
        support_precision + support_recall, 1e-12
    )

    tissue_abundance = truth_values.mean(axis=0)
    rare_columns = (
        (raw_truth_values.sum(axis=0) > 0) & (tissue_abundance < 0.05)
    )
    if np.any(rare_columns):
        rare_truth = truth_values[:, rare_columns]
        rare_prediction = prediction_values[:, rare_columns]
        rare_rmse = float(np.sqrt(np.mean((rare_truth - rare_prediction) ** 2)))
        rare_mae = float(np.mean(np.abs(rare_truth - rare_prediction)))
        rare_average_precision = float(
            average_precision_score(
                (raw_truth_values[:, rare_columns] > 0).ravel().astype(int),
                rare_prediction.ravel(),
            )
        )
    else:
        rare_rmse = float('nan')
        rare_mae = float('nan')
        rare_average_precision = float('nan')

    return {
        "spot_count": int(len(common_spots)),
        "truth_spot_count": int(len(truth_sums)),
        "excluded_low_quality_only_spot_count": int((~evaluable).sum()),
        "cell_type_count": int(len(CELL_TYPES_18)),
        "rmse": float(rmse.mean()),
        "jensen_shannon_distance": float(js_distance.mean()),
        "jensen_shannon_divergence": float((js_distance ** 2).mean()),
        "spearman": float(np.mean(correlations)),
        "predicted_entropy": float(predicted_entropy.mean()),
        "true_entropy": float(true_entropy.mean()),
        "predicted_effective_cell_types": float(np.exp(predicted_entropy).mean()),
        "true_effective_cell_types": float(np.exp(true_entropy).mean()),
        "support_precision_at_0_01": float(support_precision),
        "support_recall_at_0_01": float(support_recall),
        "support_f1_at_0_01": float(support_f1),
        "rare_rmse": rare_rmse,
        "rare_mae": rare_mae,
        "rare_average_precision": rare_average_precision,
    }


def locate_prediction(seed_root: Path, dataset: str) -> Path:
    candidates = sorted((seed_root / dataset).rglob("predict_result.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one prediction for {dataset}, found {len(candidates)}"
        )
    return candidates[0]


PROFILE_CHOICES = [
    "legacy",
    "pseudo_panel",
    "sparse_supervision",
    "reconstruction",
    "full_v2",
    "no_reconstruction",
    "no_alignment",
    "no_spatial",
    "no_sparse_head",
    "no_platform_match",
    "no_full_panel",
    "uniform_pseudo",
    "support_calibrated",
    "support_rare_balanced",
    "niche_calibrated",
    "domain_adaptive",
    "niche_domain",
    "niche_signature",
]


def configure_runner(runner, profile: str, epochs: int, pseudo_spots: int) -> None:
    """Apply an explicit optimization/ablation profile to the shared runner."""
    runner.spatial_deconv_params["method"] = "STD_Net"
    legacy = profile == "legacy"
    legacy_architecture = profile in {"legacy", "pseudo_panel"}
    support_calibrated = profile in {
        "support_calibrated", "support_rare_balanced",
        "niche_calibrated", "niche_signature"
    }
    rare_balanced = profile == "support_rare_balanced"
    domain_adaptive = profile in {"domain_adaptive", "niche_domain"}
    support_calibrated = support_calibrated or domain_adaptive
    niche_calibrated = profile in {
        "niche_calibrated", "niche_domain", "niche_signature"
    }
    niche_signature = profile == "niche_signature"
    runner.find_marker_genes_paras.update(
        {
            "preserve_full_targeted_panel": profile not in {
                "legacy", "no_full_panel"
            },
            "targeted_panel_gene_limit": 2000,
        }
    )
    runner.pseudo_spot_simulation_paras.update(
        {
            "spot_num": pseudo_spots,
            "enable_low_abundance_boost": profile != "uniform_pseudo",
            "require_spatial_coordinates": profile != "uniform_pseudo",
            "spot_ratio_to_real": None if legacy else 8.0,
            "minimum_spot_count": 100,
            "candidate_spot_probability": (
                0.0 if profile == "uniform_pseudo" else (1.0 if legacy else 0.25)
            ),
            "probability_mode_weights": (
                (0.0, 1.0, 0.0, 0.0)
                if profile == "uniform_pseudo"
                else (0.25, 0.30, 0.30, 0.15)
            ),
            "cell_type_count_weights": (
                None
                if profile in {"legacy", "uniform_pseudo"}
                else (0.70, 0.22, 0.06, 0.02)
            ),
            "dirichlet_alpha": (
                None if profile in {"legacy", "uniform_pseudo"} else 0.35
            ),
            "platform_match": profile not in {"legacy", "no_platform_match"},
            "platform_ratio_clip": (0.25, 4.0),
            "target_complexity_calibration": (
                {"temperature": 0.02, "pseudocount": 0.02}
                if niche_calibrated
                else False
            ),
        }
    )
    runner.inter_exp_adj_paras["corr_dist_neighbors"] = 20 if legacy else 8
    runner.real_intra_exp_adj_paras["corr_dist_neighbors"] = 10 if legacy else 8
    runner.pseudo_intra_exp_adj_paras["corr_dist_neighbors"] = 20 if legacy else 8
    runner.spatial_adj_paras.update(
        {
            "space_dist_neighbors": 27 if legacy else 8,
            "n_clusters": 7,
            "boundary_threshold_percent": 0.1,
            "boundary_decay": 0.3,
        }
    )

    objective_enabled = profile not in {"legacy", "pseudo_panel"}
    objective = {
        "enabled": objective_enabled,
        "composition": {
            "js_weight": 1.0,
            "rmse_weight": 1.0,
            "support_weight": 0.20,
            "entropy_weight": 0.10,
            "rare_weight_power": 0.75 if rare_balanced else 0.5,
            "rare_ranking_weight": 0.05 if rare_balanced else 0.0,
            "rare_prevalence_threshold": 0.10,
            "rare_ranking_margin": 0.05,
            "rare_ranking_pairs": 64,
            "support_threshold": 1e-4,
            "focal_gamma": 2.0,
        },
        "reconstruction_weight": (
            0.05
            if profile in {
                "reconstruction", "full_v2", "niche_signature",
                "no_alignment", "no_spatial",
                "no_sparse_head", "no_platform_match", "no_full_panel",
                "uniform_pseudo", "support_calibrated",
                "support_rare_balanced",
            }
            else 0.0
        ),
        "prototype_weight": (
            0.05
            if profile in {
                "full_v2", "no_reconstruction", "no_spatial", "no_sparse_head",
                "no_platform_match", "no_full_panel", "uniform_pseudo",
                "support_calibrated", "support_rare_balanced",
            }
            else 0.0
        ),
        "spatial_weight": (
            0.02
            if profile in {
                "full_v2", "no_reconstruction", "no_alignment", "no_sparse_head",
                "no_platform_match", "no_full_panel", "uniform_pseudo",
                "support_calibrated", "support_rare_balanced",
            }
            else 0.0
        ),
        "domain_adversarial_weight": 0.05 if domain_adaptive else 0.0,
        "adaptive_signature_weight": 0.01 if niche_signature else 0.0,
        "confidence_threshold": 0.35,
        "early_stopping_min_delta": 1e-5,
    }
    runner.DGAT_paras.update(
        {
            "optimization_profile": profile,
            "epoch_n": epochs,
            "dim": 80 if legacy_architecture else 64,
            "common_hid_layers_num": 1 if legacy_architecture else 2,
            "dropout": 0.5 if legacy_architecture else 0.2,
            "early_stopping_patience": 15 if legacy_architecture else 60,
            "clip_grad_max_norm": 0.5 if legacy_architecture else 1.0,
            "use_gatv2": True,
            "gat_heads": 4,
            "use_sparse_graph": not legacy_architecture,
            "output_transform": (
                "softmax" if profile in {"legacy", "pseudo_panel", "no_sparse_head"}
                else "sparsemax"
            ),
            "support_smoothing_epsilon": 0.02 if support_calibrated else 0.0,
            "domain_adversarial": domain_adaptive,
            "domain_adversarial_lambda": 1.0,
            "use_pseudo_spatial_graph": niche_calibrated,
            "pseudo_spatial_neighbors": 4,
            "pseudo_spatial_edge_scale": 0.25,
            "adaptive_signature": niche_signature,
            "adaptive_signature_rank": 8,
            "gated_fusion": not legacy_architecture,
            "use_gene_weights": not legacy,
            "optimizer": "SGD" if legacy_architecture else "AdamW",
            "learning_rate": 0.2 if legacy_architecture else 1e-3,
            "weight_decay_SGD": 3e-4 if legacy_architecture else 1e-4,
            "objective": objective,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("final_data"))
    parser.add_argument(
        "--raw-sc-counts",
        type=Path,
        default=Path("original_data/Lohoff_seqFISH/counts.txt"),
    )
    parser.add_argument("--datasets", default="")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--pseudo-spots", type=int, default=30000)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--protocol",
        choices=["standard", "slice", "animal"],
        default="standard",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default="full_v2",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    repo_root_text = str(repo_root)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)
    os.chdir(repo_root)
    data_root = args.data_root.resolve()
    raw_counts_path = args.raw_sc_counts.resolve()
    raw_counts = load_raw_counts(raw_counts_path)
    available_datasets = sorted(
        path.name for path in data_root.iterdir() if path.is_dir()
    )
    datasets = parse_csv(args.datasets) if args.datasets else available_datasets.copy()
    seeds = [int(seed) for seed in parse_csv(args.seeds)]
    if args.quick:
        datasets = datasets[:1]
        seeds = seeds[:1]
        args.epochs = min(args.epochs, 2)
        args.pseudo_spots = min(args.pseudo_spots, 100)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        args.output.resolve()
        if args.output
        else repo_root / "paper_aligned_runs" / f"run_{timestamp}"
    )
    output_root.mkdir(parents=True, exist_ok=False)

    staged_root = output_root / "training_input_without_ground_truth"
    for dataset in datasets:
        if args.protocol == "standard":
            stage_dataset(data_root, staged_root, dataset, raw_counts)
        else:
            target_animal = dataset.split("_z", 1)[0]
            if args.protocol == "slice":
                references = [
                    item for item in available_datasets if item != dataset
                ]
            else:
                references = [
                    item
                    for item in available_datasets
                    if item.split("_z", 1)[0] != target_animal
                ]
            if not references:
                raise ValueError(f"No reference datasets for target {dataset}")
            stage_grouped_dataset(
                data_root, staged_root, dataset, references, raw_counts
            )

    import numpy as np
    import pandas as pd
    import torch
    from tqdm import tqdm as terminal_tqdm

    from baseline_validation.common import CELL_TYPES_18
    import Run_STD as runner
    import STD_Net.process_utils as process_utils
    import STD_Net.XAI_visual as xai_visual

    process_utils.tqdm = terminal_tqdm
    xai_visual.tqdm = terminal_tqdm
    configure_runner(runner, args.profile, args.epochs, args.pseudo_spots)
    runner.runtime_n_jobs = args.jobs

    configuration = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "datasets": datasets,
        "seeds": seeds,
        "epochs": args.epochs,
        "pseudo_spots": args.pseudo_spots,
        "jobs": args.jobs,
        "protocol": args.protocol,
        "profile": args.profile,
        "optimization_configuration": runner.DGAT_paras,
        "pseudo_spot_configuration": runner.pseudo_spot_simulation_paras,
        "raw_sc_counts": str(raw_counts_path),
        "raw_sc_counts_sha256": sha256_file(raw_counts_path),
        "single_cell_expression_scale": "raw counts",
        "target_ground_truth_in_training_workspace": False,
        "reference_label_exclusion": "Low quality",
        "evaluation_ontology": list(CELL_TYPES_18),
        "metric_note": (
            "scipy.spatial.distance.jensenshannon is reported as distance; "
            "its square is additionally reported as divergence"
        ),
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "source_sha256": {
            str(path.relative_to(repo_root)): sha256_file(path)
            for path in [
                repo_root / "Run_STD.py",
                repo_root / "STD_Net" / "STD_Net.py",
                repo_root / "STD_Net" / "DGAT.py",
                repo_root / "STD_Net" / "stdg.py",
                repo_root / "STD_Net" / "optimization.py",
                repo_root / "baseline_validation" / "common.py",
            ]
        },
    }
    (output_root / "run_configuration.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    records = []
    failures = []
    for seed in seeds:
        set_seed(seed, torch)
        runner.pseudo_spot_simulation_paras["random_seed"] = seed
        seed_root = output_root / f"seed_{seed}"
        seed_root.mkdir()
        for dataset in datasets:
            print(f"START dataset={dataset} seed={seed}", flush=True)
            success = runner.process_single_dataset(
                str(staged_root), str(seed_root), dataset
            )
            if not success:
                failures.append({"dataset": dataset, "seed": seed})
                continue
            prediction_file = locate_prediction(seed_root, dataset)
            metrics = evaluate_prediction(
                prediction_file,
                data_root / dataset / "st_data" / "ST_ground_truth.tsv",
            )
            metrics.update(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "protocol": args.protocol,
                    "profile": args.profile,
                }
            )
            records.append(metrics)
            (prediction_file.parent / "final_metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"DONE {metrics}", flush=True)

    if records:
        frame = pd.DataFrame(records)
        frame.to_csv(output_root / "all_seed_metrics.csv", index=False)
        metric_columns = [
            "rmse",
            "jensen_shannon_distance",
            "jensen_shannon_divergence",
            "spearman",
            "predicted_entropy",
            "true_entropy",
            "predicted_effective_cell_types",
            "true_effective_cell_types",
            "support_precision_at_0_01",
            "support_recall_at_0_01",
            "support_f1_at_0_01",
            "rare_rmse",
            "rare_mae",
            "rare_average_precision",
        ]
        summary = frame.groupby("dataset")[metric_columns].agg(
            ["mean", "std", "count"]
        )
        summary.columns = ["_".join(column) for column in summary.columns]
        summary.reset_index().to_csv(
            output_root / "metrics_mean_sd.csv", index=False
        )
    (output_root / "failures.json").write_text(
        json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if failures:
        raise SystemExit(f"{len(failures)} runs failed; see failures.json")
    print(f"ALL RUNS COMPLETED: {output_root}")


if __name__ == "__main__":
    main()
