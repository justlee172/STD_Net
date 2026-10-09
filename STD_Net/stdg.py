"""Paper-aligned spatially topology-enhanced data generation (STDG)."""

from __future__ import annotations

import multiprocessing
import random
from collections.abc import Sequence

import anndata
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


def identify_low_density_candidates(
    spatial_coordinates: np.ndarray,
    bandwidth: float = 0.1,
    quantile: float = 0.20,
) -> np.ndarray:
    """Identify KDE low-density sampling candidates.

    These candidates are a computational proxy and are not anatomical
    tissue-boundary annotations.
    """
    coordinates = np.asarray(spatial_coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("spatial coordinates must be an n_cells x 2 array")
    if not np.isfinite(coordinates).all():
        raise ValueError("spatial coordinates contain non-finite values")
    if bandwidth <= 0:
        raise ValueError("KDE bandwidth must be positive")
    if not 0 < quantile < 1:
        raise ValueError("low-density quantile must be between 0 and 1")

    kde = KernelDensity(bandwidth=bandwidth).fit(coordinates)
    densities = np.exp(kde.score_samples(coordinates))
    return densities < np.quantile(densities, quantile)


def _dense_row(expression) -> np.ndarray:
    if hasattr(expression, "toarray"):
        expression = expression.toarray()
    return np.asarray(expression).ravel()


def augment_low_abundance_candidates(
    single_cell_data: anndata.AnnData,
    candidate_mask: np.ndarray,
    low_abundance_types: Sequence[str],
    oversample_ratio: float = 3.0,
    alpha_min: float = 0.3,
    alpha_max: float = 0.7,
    random_seed: int = 0,
) -> tuple[anndata.AnnData, np.ndarray]:
    """Interpolate same-type rare cells inside the candidate subset."""
    mask = np.asarray(candidate_mask, dtype=bool)
    if mask.shape[0] != single_cell_data.n_obs:
        raise ValueError("candidate mask length does not match single-cell data")
    if not 0 <= oversample_ratio:
        raise ValueError("oversample_ratio must be non-negative")
    if not 0 <= alpha_min <= alpha_max <= 1:
        raise ValueError("interpolation alpha bounds must lie in [0, 1]")

    candidate_metadata = single_cell_data.obs.loc[mask]
    cells_by_type = {
        cell_type: candidate_metadata.index[
            candidate_metadata["cell_type"] == cell_type
        ].tolist()
        for cell_type in low_abundance_types
    }
    cells_by_type = {
        cell_type: cells
        for cell_type, cells in cells_by_type.items()
        if len(cells) >= 2
    }
    eligible_count = sum(len(cells) for cells in cells_by_type.values())
    synthetic_count = int(eligible_count * oversample_ratio)
    if synthetic_count == 0:
        data = single_cell_data.copy()
        data.obs["is_synthetic"] = False
        return data, mask

    rng = np.random.default_rng(random_seed)
    eligible_types = list(cells_by_type)
    type_weights = np.array(
        [len(cells_by_type[cell_type]) for cell_type in eligible_types],
        dtype=float,
    )
    type_weights /= type_weights.sum()

    expression_rows = []
    metadata_rows = []
    synthetic_names = []
    for synthetic_index in range(synthetic_count):
        cell_type = str(rng.choice(eligible_types, p=type_weights))
        first_id, second_id = rng.choice(
            cells_by_type[cell_type], size=2, replace=False
        )
        alpha = float(rng.uniform(alpha_min, alpha_max))
        expression_rows.append(
            alpha * _dense_row(single_cell_data[first_id].X)
            + (1 - alpha) * _dense_row(single_cell_data[second_id].X)
        )
        metadata = single_cell_data.obs.loc[first_id].copy()
        metadata["cell_type"] = cell_type
        metadata["is_synthetic"] = True
        metadata_rows.append(metadata)
        synthetic_names.append(f"stdg_synthetic_{synthetic_index}")

    synthetic_data = anndata.AnnData(
        X=np.vstack(expression_rows),
        obs=pd.DataFrame(metadata_rows, index=synthetic_names),
        var=single_cell_data.var.copy(),
    )
    original_data = single_cell_data.copy()
    original_data.obs["is_synthetic"] = False
    combined_data = anndata.concat(
        [original_data, synthetic_data], join="inner", index_unique=None
    )
    combined_mask = np.concatenate(
        [mask, np.ones(synthetic_count, dtype=bool)]
    )
    return combined_data, combined_mask


def _sampling_probabilities(cell_counts: np.ndarray) -> dict[str, np.ndarray]:
    counts = np.asarray(cell_counts, dtype=float)
    type_count = len(counts)
    median_count = np.median(counts)
    sigmoid = 1 / (1 + np.exp(-0.2 * (counts - median_count)))
    curvature = (1 / np.sqrt(counts + 1e-6)) * sigmoid
    curvature /= curvature.sum()
    return {
        "unbalance": counts / counts.sum(),
        "balance": np.ones(type_count) / type_count,
        "sqrt": np.sqrt(counts) / np.sqrt(counts).sum(),
        "curve": curvature,
    }


def generate_one_pseudo_spot(
    single_cell_data: anndata.AnnData,
    min_cells: int,
    max_cells: int,
    max_cell_types: int,
    generation_method: str,
    candidate_mask: np.ndarray | None,
    candidate_cell_fraction: float,
    global_probability_weight: float,
    probability_mode_weights: Sequence[float],
    random_seed: int,
    cell_type_count_weights: Sequence[float] | None = None,
    dirichlet_alpha: float | None = None,
    candidate_spot_probability: float = 1.0,
) -> anndata.AnnData:
    """Generate one pseudo-spot with a strict cell-type-count limit."""
    python_rng = random.Random(random_seed)
    numpy_rng = np.random.default_rng(random_seed)
    cell_number = python_rng.randint(min_cells, max_cells)

    if generation_method == "cell":
        cell_ids = single_cell_data.obs_names.tolist()
        return single_cell_data[python_rng.choices(cell_ids, k=cell_number)]
    if generation_method != "celltype":
        raise ValueError('generation_method must be "cell" or "celltype"')

    cell_type_counts = single_cell_data.obs["cell_type"].value_counts()
    all_types = cell_type_counts.index.tolist()
    probability_components = _sampling_probabilities(
        cell_type_counts.to_numpy(dtype=float)
    )
    probability_mode = python_rng.choices(
        list(probability_components), weights=probability_mode_weights, k=1
    )[0]
    global_probabilities = probability_components[probability_mode]
    global_probabilities = (global_probabilities + 1e-6) / (
        1 + 1e-6 * len(all_types)
    )
    global_probabilities /= global_probabilities.sum()

    mask = None if candidate_mask is None else np.asarray(candidate_mask, dtype=bool)
    candidate_probabilities = np.zeros(len(all_types), dtype=float)
    if mask is not None:
        candidate_counts = (
            single_cell_data.obs.loc[mask, "cell_type"]
            .value_counts()
            .reindex(all_types, fill_value=0)
            .to_numpy(dtype=float)
        )
        if candidate_counts.sum() > 0:
            candidate_probabilities = candidate_counts / candidate_counts.sum()
    combined_probabilities = (
        global_probability_weight * global_probabilities
        + (1 - global_probability_weight) * candidate_probabilities
    )
    combined_probabilities /= combined_probabilities.sum()

    available_type_count = min(max_cell_types, len(all_types), cell_number)
    if cell_type_count_weights is None:
        selected_type_count = python_rng.randint(1, available_type_count)
    else:
        count_probabilities = np.asarray(
            list(cell_type_count_weights)[:available_type_count], dtype=float
        )
        if np.any(count_probabilities < 0) or count_probabilities.sum() <= 0:
            raise ValueError("cell_type_count_weights must be non-negative")
        count_probabilities /= count_probabilities.sum()
        selected_type_count = int(
            numpy_rng.choice(
                np.arange(1, available_type_count + 1),
                p=count_probabilities,
            )
        )
    selected_types = []
    if not 0 <= candidate_spot_probability <= 1:
        raise ValueError("candidate_spot_probability must lie in [0, 1]")
    if (
        candidate_probabilities.sum() > 0
        and numpy_rng.random() < candidate_spot_probability
    ):
        selected_types.append(
            str(numpy_rng.choice(all_types, p=candidate_probabilities))
        )
    remaining_types = [item for item in all_types if item not in selected_types]
    while len(selected_types) < selected_type_count:
        remaining_indices = [all_types.index(item) for item in remaining_types]
        remaining_probabilities = combined_probabilities[remaining_indices]
        remaining_probabilities /= remaining_probabilities.sum()
        chosen_type = str(
            numpy_rng.choice(remaining_types, p=remaining_probabilities)
        )
        selected_types.append(chosen_type)
        remaining_types.remove(chosen_type)

    global_pools = {
        cell_type: single_cell_data.obs_names[
            single_cell_data.obs["cell_type"] == cell_type
        ].tolist()
        for cell_type in selected_types
    }
    candidate_pools = {
        cell_type: single_cell_data.obs_names[
            mask & (single_cell_data.obs["cell_type"].to_numpy() == cell_type)
        ].tolist()
        if mask is not None
        else []
        for cell_type in selected_types
    }

    selected_indices = np.array(
        [all_types.index(cell_type) for cell_type in selected_types]
    )
    if dirichlet_alpha is not None:
        if dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive")
        selected_weights = numpy_rng.dirichlet(
            np.full(selected_type_count, float(dirichlet_alpha))
        )
    else:
        selected_weights = combined_probabilities[selected_indices]
        selected_weights /= selected_weights.sum()

    # Include every selected type at least once and allocate the remaining
    # cells from a sparse Dirichlet composition.  This makes the simulated
    # support distribution match low-cell-count imaging spots more closely.
    type_cell_counts = np.ones(selected_type_count, dtype=int)
    remaining_cell_count = cell_number - selected_type_count
    if remaining_cell_count > 0:
        type_cell_counts += numpy_rng.multinomial(
            remaining_cell_count, selected_weights
        )

    picked_cells = []
    for chosen_type, chosen_count in zip(selected_types, type_cell_counts):
        candidate_pool = candidate_pools[chosen_type]
        candidate_count = 0
        if candidate_pool:
            candidate_count = min(
                int(chosen_count),
                int(round(float(chosen_count) * candidate_cell_fraction)),
            )
        for _ in range(candidate_count):
            picked_cells.append(python_rng.choice(candidate_pool))
        for _ in range(int(chosen_count) - candidate_count):
            picked_cells.append(python_rng.choice(global_pools[chosen_type]))
    python_rng.shuffle(picked_cells)
    return single_cell_data[picked_cells]


def pseudo_spot_generation_aligned(
    single_cell_data: anndata.AnnData,
    index_to_cell_type: dict[int, str],
    spot_count: int,
    min_cells: int,
    max_cells: int,
    max_cell_types: int,
    generation_method: str,
    spatial_coordinates: np.ndarray | None,
    enable_low_abundance_boost: bool = True,
    require_spatial_coordinates: bool = True,
    low_density_quantile: float = 0.20,
    kde_bandwidth: float = 0.1,
    low_abundance_threshold: float = 0.05,
    oversample_ratio: float = 3.0,
    candidate_cell_fraction: float = 0.20,
    global_probability_weight: float = 0.70,
    probability_mode_weights: Sequence[float] = (0.25, 0.30, 0.30, 0.15),
    cell_type_count_weights: Sequence[float] | None = None,
    dirichlet_alpha: float | None = None,
    candidate_spot_probability: float = 1.0,
    interpolation_alpha_min: float = 0.3,
    interpolation_alpha_max: float = 0.7,
    random_seed: int = 0,
    n_jobs: int = -1,
) -> anndata.AnnData:
    """Generate paper-aligned STDG pseudo-spots deterministically."""
    if enable_low_abundance_boost and spatial_coordinates is None:
        if require_spatial_coordinates:
            raise ValueError(
                "STDG requires sc_data/sc_coordinates.csv with x and y columns"
            )
        candidate_mask = None
    elif enable_low_abundance_boost:
        candidate_mask = identify_low_density_candidates(
            spatial_coordinates,
            bandwidth=kde_bandwidth,
            quantile=low_density_quantile,
        )
        type_counts = single_cell_data.obs["cell_type"].value_counts()
        low_abundance_types = type_counts[
            type_counts / type_counts.sum() < low_abundance_threshold
        ].index.tolist()
        single_cell_data, candidate_mask = augment_low_abundance_candidates(
            single_cell_data,
            candidate_mask,
            low_abundance_types,
            oversample_ratio=oversample_ratio,
            alpha_min=interpolation_alpha_min,
            alpha_max=interpolation_alpha_max,
            random_seed=random_seed,
        )
    else:
        candidate_mask = None

    seed_sequence = np.random.SeedSequence(random_seed)
    spot_seeds = [
        int(seed.generate_state(1)[0]) for seed in seed_sequence.spawn(spot_count)
    ]
    arguments = [
        (
            single_cell_data,
            min_cells,
            max_cells,
            max_cell_types,
            generation_method,
            candidate_mask,
            candidate_cell_fraction,
            global_probability_weight,
            probability_mode_weights,
            spot_seeds[index],
            cell_type_count_weights,
            dirichlet_alpha,
            candidate_spot_probability,
        )
        for index in range(spot_count)
    ]
    worker_count = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
    if worker_count == 1:
        generated_spots = [generate_one_pseudo_spot(*args) for args in arguments]
    else:
        with multiprocessing.Pool(processes=worker_count) as pool:
            generated_spots = pool.starmap(generate_one_pseudo_spot, arguments)

    cell_type_count = len(index_to_cell_type)
    pseudo_expression = np.zeros(
        (spot_count, single_cell_data.n_vars), dtype=float
    )
    pseudo_fractions = np.zeros((spot_count, cell_type_count), dtype=float)
    for index, pseudo_spot in enumerate(generated_spots):
        pseudo_expression[index] = np.asarray(
            pseudo_spot.X.sum(axis=0)
        ).ravel()
        for type_index in pseudo_spot.obs["cell_type_idx"].to_numpy(dtype=int):
            pseudo_fractions[index, type_index] += 1

    pseudo_data = anndata.AnnData(
        X=pseudo_expression,
        var=pd.DataFrame(index=single_cell_data.var_names.copy()),
    )
    type_names = [index_to_cell_type[index] for index in range(cell_type_count)]
    fraction_table = pd.DataFrame(
        pseudo_fractions,
        index=pseudo_data.obs_names.copy(),
        columns=type_names,
    )
    fraction_table["cell_num"] = fraction_table.sum(axis=1)
    fraction_table[type_names] = fraction_table[type_names].div(
        fraction_table["cell_num"], axis=0
    )
    pseudo_data.obs = fraction_table
    return pseudo_data
