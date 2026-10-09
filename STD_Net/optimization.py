"""Optimization utilities for the publication-oriented STD-Net v2 profile.

The functions in this module are deliberately independent from the training
driver.  This keeps the legacy implementation reproducible while allowing the
v2 ablations to switch individual components on and off from a configuration
file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


EPS = 1e-8


def as_dense(matrix) -> np.ndarray:
    """Return a finite two-dimensional float array."""
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("expression matrix must be two-dimensional")
    if not np.isfinite(array).all():
        raise ValueError("expression matrix contains non-finite values")
    return array


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax transformation with exact zeros and unit row sums.

    This is the Euclidean projection onto the probability simplex described by
    Martins and Astudillo (2016).  It avoids the systematically dense mixtures
    produced by softmax while remaining differentiable almost everywhere.
    """
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    sorted_logits = torch.sort(shifted, dim=dim, descending=True).values
    dimension = logits.size(dim)
    ranks_shape = [1] * logits.dim()
    ranks_shape[dim] = dimension
    ranks = torch.arange(
        1, dimension + 1, device=logits.device, dtype=logits.dtype
    ).view(ranks_shape)
    cumulative = sorted_logits.cumsum(dim)
    support = 1 + ranks * sorted_logits > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp(min=1)
    threshold = (
        cumulative.gather(dim, support_size - 1) - 1
    ) / support_size.to(logits.dtype)
    probabilities = torch.clamp(shifted - threshold, min=0)
    return probabilities / probabilities.sum(dim=dim, keepdim=True).clamp_min(EPS)


def support_calibrated_sparsemax(
    logits: torch.Tensor,
    epsilon: float = 0.0,
    dim: int = -1,
) -> torch.Tensor:
    """Sparsemax with a small differentiable probability floor during training.

    Exact sparsemax has zero probability and zero local gradient outside its
    active support.  That is desirable for the final composition map, but it
    can permanently remove a rare cell type during optimization.  Mixing a
    small softmax component keeps every class reachable while retaining the
    sparsemax geometry.  Set ``epsilon=0`` to recover exact sparsemax.
    """
    epsilon = float(epsilon)
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0, 1]")
    hard = sparsemax(logits, dim=dim)
    if epsilon == 0.0:
        return hard
    soft = F.softmax(logits, dim=dim)
    return (1.0 - epsilon) * hard + epsilon * soft


class SparseCompositionalLoss(nn.Module):
    """Pseudo-spot supervision matched to compositional evaluation metrics."""

    def __init__(
        self,
        js_weight: float = 1.0,
        rmse_weight: float = 1.0,
        support_weight: float = 0.20,
        entropy_weight: float = 0.10,
        rare_weight_power: float = 0.5,
        rare_ranking_weight: float = 0.0,
        rare_prevalence_threshold: float = 0.10,
        rare_ranking_margin: float = 0.05,
        rare_ranking_pairs: int = 64,
        support_threshold: float = 1e-4,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.js_weight = float(js_weight)
        self.rmse_weight = float(rmse_weight)
        self.support_weight = float(support_weight)
        self.entropy_weight = float(entropy_weight)
        self.rare_weight_power = float(rare_weight_power)
        self.rare_ranking_weight = float(rare_ranking_weight)
        self.rare_prevalence_threshold = float(rare_prevalence_threshold)
        self.rare_ranking_margin = float(rare_ranking_margin)
        self.rare_ranking_pairs = max(1, int(rare_ranking_pairs))
        self.support_threshold = float(support_threshold)
        self.focal_gamma = float(focal_gamma)
        if self.rare_ranking_weight < 0:
            raise ValueError("rare_ranking_weight must be non-negative")
        if not 0.0 < self.rare_prevalence_threshold <= 1.0:
            raise ValueError("rare_prevalence_threshold must lie in (0, 1]")
        if self.rare_ranking_margin < 0:
            raise ValueError("rare_ranking_margin must be non-negative")

    @staticmethod
    def _probabilities(log_predictions: torch.Tensor) -> torch.Tensor:
        probabilities = log_predictions.exp().clamp_min(0)
        return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(EPS)

    def components(
        self, log_predictions: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        targets = targets.clamp_min(0)
        targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(EPS)
        predictions = self._probabilities(log_predictions)

        midpoint = 0.5 * (predictions + targets)
        js = 0.5 * (
            targets
            * (targets.clamp_min(EPS).log() - midpoint.clamp_min(EPS).log())
        ).sum(dim=1)
        js = js + 0.5 * (
            predictions
            * (
                predictions.clamp_min(EPS).log()
                - midpoint.clamp_min(EPS).log()
            )
        ).sum(dim=1)

        abundance = targets.mean(dim=0).clamp_min(EPS)
        class_weights = abundance.pow(-self.rare_weight_power)
        class_weights = class_weights / class_weights.mean().clamp_min(EPS)
        squared_error = (predictions - targets).pow(2) * class_weights
        rmse = torch.sqrt(squared_error.mean() + EPS)

        support = (targets > self.support_threshold).to(predictions.dtype)
        prevalence = support.mean(dim=0).clamp(0.01, 0.99)
        positive_weight = ((1 - prevalence) / prevalence).clamp(1.0, 20.0)
        binary_cross_entropy = -(
            positive_weight * support * predictions.clamp_min(EPS).log()
            + (1 - support) * (1 - predictions).clamp_min(EPS).log()
        )
        focal = torch.where(
            support.bool(),
            (1 - predictions).pow(self.focal_gamma),
            predictions.pow(self.focal_gamma),
        )
        support_loss = (focal * binary_cross_entropy).mean()

        predicted_entropy = -(
            predictions * predictions.clamp_min(EPS).log()
        ).sum(dim=1)
        target_entropy = -(
            targets * targets.clamp_min(EPS).log()
        ).sum(dim=1)
        entropy = F.smooth_l1_loss(predicted_entropy, target_entropy)

        # Average precision depends on the ordering of positive and negative
        # spots for each rare type.  The compositional losses above constrain
        # mass and binary support, but do not directly constrain this ordering.
        # Use a bounded number of deterministic pairs per low-prevalence type
        # so the objective remains cheap for the 30k pseudo-spot batches.
        if self.rare_ranking_weight > 0.0:
            rare_ranking_terms = []
            rare_ranking_weights = []
            rare_prevalence = support.mean(dim=0)
            rare_types = torch.nonzero(
                (rare_prevalence > 0.0)
                & (rare_prevalence <= self.rare_prevalence_threshold),
                as_tuple=False,
            ).flatten()
            for type_index in rare_types.tolist():
                positive_indices = torch.nonzero(
                    support[:, type_index] > 0.5, as_tuple=False
                ).flatten()
                negative_indices = torch.nonzero(
                    support[:, type_index] <= 0.5, as_tuple=False
                ).flatten()
                pair_count = min(
                    self.rare_ranking_pairs,
                    int(positive_indices.numel()),
                    int(negative_indices.numel()),
                )
                if pair_count < 2:
                    continue

                # Evenly spaced indices avoid relying on the order of the
                # pseudo-spot generator while keeping the operation fully
                # deterministic under a fixed seed.
                positive_take = torch.linspace(
                    0,
                    positive_indices.numel() - 1,
                    pair_count,
                    device=positive_indices.device,
                ).round().long()
                negative_take = torch.linspace(
                    0,
                    negative_indices.numel() - 1,
                    pair_count,
                    device=negative_indices.device,
                ).round().long()
                positive_scores = log_predictions[
                    positive_indices[positive_take], type_index
                ]
                negative_scores = log_predictions[
                    negative_indices[negative_take], type_index
                ]
                pairwise_margin = positive_scores - negative_scores
                rare_ranking_terms.append(
                    F.softplus(
                        self.rare_ranking_margin - pairwise_margin
                    ).mean()
                )
                rare_ranking_weights.append(
                    rare_prevalence[type_index]
                    .clamp_min(EPS)
                    .pow(-self.rare_weight_power)
                )

            if rare_ranking_terms:
                rare_ranking_weights = torch.stack(rare_ranking_weights)
                rare_ranking_weights = rare_ranking_weights / (
                    rare_ranking_weights.mean().clamp_min(EPS)
                )
                rare_ranking = (
                    torch.stack(rare_ranking_terms) * rare_ranking_weights
                ).mean()
            else:
                rare_ranking = predictions.sum() * 0.0
        else:
            rare_ranking = predictions.sum() * 0.0

        return {
            "js": js.mean(),
            "rmse": rmse,
            "support": support_loss,
            "entropy": entropy,
            "rare_ranking": rare_ranking,
        }

    def forward(
        self, log_predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        values = self.components(log_predictions, targets)
        return (
            self.js_weight * values["js"]
            + self.rmse_weight * values["rmse"]
            + self.support_weight * values["support"]
            + self.entropy_weight * values["entropy"]
            + self.rare_ranking_weight * values["rare_ranking"]
        )


def negative_binomial_nll(
    observed: torch.Tensor,
    mean: torch.Tensor,
    dispersion: torch.Tensor,
) -> torch.Tensor:
    """Mean negative-binomial log likelihood under a mean/dispersion form."""
    observed = observed.clamp_min(0)
    mean = mean.clamp_min(EPS)
    dispersion = dispersion.clamp_min(1e-4)
    log_probability = (
        torch.lgamma(observed + dispersion)
        - torch.lgamma(dispersion)
        - torch.lgamma(observed + 1)
        + dispersion * (dispersion.log() - (dispersion + mean).log())
        + observed * (mean.log() - (dispersion + mean).log())
    )
    return -log_probability.mean()


def reconstruct_counts(
    probabilities: torch.Tensor,
    signature: torch.Tensor,
    library_sizes: torch.Tensor,
    gene_scale: torch.Tensor | None = None,
    ambient_fraction: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reconstruct raw spot counts from proportions and signatures.

    ``signature`` may be a shared ``(cell_type, gene)`` matrix or a
    spot-specific ``(spot, cell_type, gene)`` tensor for the optional
    niche-adaptive decoder.
    """
    if signature.dim() == 2:
        mixture = probabilities @ signature
        ambient = signature.mean(dim=0, keepdim=True)
    elif signature.dim() == 3:
        mixture = torch.einsum("nc,ncg->ng", probabilities, signature)
        ambient = signature.mean(dim=1)
    else:
        raise ValueError("signature must have two or three dimensions")
    if gene_scale is not None:
        mixture = mixture * gene_scale.unsqueeze(0)
    mixture = mixture.clamp_min(EPS)
    mixture = mixture / mixture.sum(dim=1, keepdim=True).clamp_min(EPS)
    if ambient_fraction is not None:
        ambient = ambient / ambient.sum(dim=1, keepdim=True).clamp_min(EPS)
        fraction = ambient_fraction.clamp(0, 0.25)
        mixture = (1 - fraction) * mixture + fraction * ambient
    return library_sizes.reshape(-1, 1).clamp_min(1) * mixture


def conditional_prototype_loss(
    source_embeddings: torch.Tensor,
    source_targets: torch.Tensor,
    target_embeddings: torch.Tensor,
    target_probabilities: torch.Tensor,
    confidence_threshold: float = 0.35,
) -> torch.Tensor:
    """Align class prototypes without treating all target spots as one domain."""
    source_targets = source_targets.clamp_min(0)
    source_targets = source_targets / source_targets.sum(
        dim=1, keepdim=True
    ).clamp_min(EPS)
    target_probabilities = target_probabilities.clamp_min(0)
    target_probabilities = target_probabilities / target_probabilities.sum(
        dim=1, keepdim=True
    ).clamp_min(EPS)
    confidence = target_probabilities.max(dim=1).values
    adaptive_threshold = min(
        float(confidence_threshold),
        max(0.15, 2.0 / max(target_probabilities.shape[1], 1)),
    )
    confident_mask = confidence >= adaptive_threshold
    if not torch.any(confident_mask):
        # Keep the objective alive during early epochs without accepting every
        # uncertain target spot.  The highest-confidence decile is used only as
        # a fallback and remains softly weighted by its confidence.
        fallback_count = max(1, target_probabilities.shape[0] // 10)
        confident_mask = torch.zeros_like(confidence, dtype=torch.bool)
        confident_mask[torch.topk(confidence, fallback_count).indices] = True
    target_weights = (
        target_probabilities
        * confidence.unsqueeze(1)
        * confident_mask.to(target_probabilities.dtype).unsqueeze(1)
    )

    source_mass = source_targets.sum(dim=0)
    target_mass = target_weights.sum(dim=0)
    valid = (source_mass > EPS) & (target_mass > EPS)
    if not torch.any(valid):
        return source_embeddings.sum() * 0

    source_prototypes = (
        source_targets.transpose(0, 1) @ source_embeddings
    ) / source_mass.unsqueeze(1).clamp_min(EPS)
    target_prototypes = (
        target_weights.transpose(0, 1) @ target_embeddings
    ) / target_mass.unsqueeze(1).clamp_min(EPS)
    source_prototypes = F.normalize(source_prototypes[valid], dim=1)
    target_prototypes = F.normalize(target_prototypes[valid], dim=1)
    return (1 - (source_prototypes * target_prototypes).sum(dim=1)).mean()


def spatial_total_variation(
    probabilities: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    """Robust spatial smoothing over measured real-spot neighbors only."""
    if edge_index.numel() == 0:
        return probabilities.sum() * 0
    source, target = edge_index
    differences = F.smooth_l1_loss(
        probabilities[source], probabilities[target], reduction="none"
    ).mean(dim=1)
    weights = edge_weight.to(differences.dtype).clamp_min(0)
    return (differences * weights).sum() / weights.sum().clamp_min(EPS)


def build_reference_signature(
    single_cell_expression,
    cell_labels: Sequence[str],
    cell_types: Sequence[str],
) -> np.ndarray:
    """Build non-negative, row-normalized cell-type signatures."""
    expression = np.clip(as_dense(single_cell_expression), 0, None)
    labels = np.asarray(cell_labels, dtype=str)
    signatures = []
    for cell_type in cell_types:
        mask = labels == str(cell_type)
        if not np.any(mask):
            raise ValueError(f"no reference cells found for {cell_type}")
        signatures.append(expression[mask].mean(axis=0))
    signature = np.vstack(signatures) + EPS
    signature /= signature.sum(axis=1, keepdims=True)
    return signature.astype(np.float32)


def estimate_target_type_count_weights(
    target_expression,
    reference_signature,
    max_cell_types: int,
    temperature: float = 0.05,
    pseudocount: float = 0.05,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Estimate a target mixture-complexity prior without target labels.

    The reference signature is compared with each observed target profile in a
    cosine-normalized expression space.  The resulting soft type scores are
    converted to an effective number of types and used only to calibrate the
    *distribution* of pseudo-spot support sizes.  No target annotations enter
    this calculation, so it remains a transductive, unsupervised prior.

    This is intentionally a weak calibration signal rather than a replacement
    for the learned deconvolution head: it corrects the common mismatch where
    pseudo spots are almost all single-type while the target contains mixed
    neighborhoods.
    """
    if max_cell_types < 1:
        raise ValueError("max_cell_types must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if pseudocount < 0:
        raise ValueError("pseudocount must be non-negative")

    target = np.clip(as_dense(target_expression), 0, None)
    signature = np.clip(as_dense(reference_signature), 0, None)
    if target.shape[1] != signature.shape[1]:
        raise ValueError(
            "target expression and reference signature must share the gene dimension"
        )
    if target.shape[0] == 0 or signature.shape[0] == 0:
        raise ValueError("target expression and reference signature must be non-empty")

    target_l1 = target / np.maximum(target.sum(axis=1, keepdims=True), EPS)
    signature_l1 = signature / np.maximum(signature.sum(axis=1, keepdims=True), EPS)
    target_l2 = target_l1 / np.maximum(
        np.linalg.norm(target_l1, axis=1, keepdims=True), EPS
    )
    signature_l2 = signature_l1 / np.maximum(
        np.linalg.norm(signature_l1, axis=1, keepdims=True), EPS
    )
    cosine = np.clip(target_l2 @ signature_l2.T, -1.0, 1.0)
    scaled = (cosine - cosine.max(axis=1, keepdims=True)) / float(temperature)
    scaled = np.clip(scaled, -60.0, 0.0)
    type_probabilities = np.exp(scaled)
    type_probabilities /= np.maximum(
        type_probabilities.sum(axis=1, keepdims=True), EPS
    )
    entropy = -np.sum(
        type_probabilities * np.log(np.maximum(type_probabilities, EPS)), axis=1
    )
    effective_types = np.exp(entropy)
    estimated_counts = np.clip(
        np.rint(effective_types).astype(int), 1, min(max_cell_types, signature.shape[0])
    )

    count_probabilities = np.bincount(
        estimated_counts,
        minlength=min(max_cell_types, signature.shape[0]) + 1,
    )[1:].astype(float)
    if pseudocount:
        count_probabilities = count_probabilities.astype(float) + float(pseudocount)
    count_probabilities /= np.maximum(count_probabilities.sum(), EPS)
    diagnostics = pd.DataFrame(
        {
            "target_proxy_entropy": entropy,
            "target_proxy_effective_type_count": effective_types,
            "target_proxy_estimated_type_count": estimated_counts,
            "target_proxy_max_type_probability": type_probabilities.max(axis=1),
        }
    )
    diagnostics.attrs["cell_type_count_weights"] = count_probabilities.tolist()
    return count_probabilities.astype(np.float32), diagnostics


def compute_discriminative_gene_weights(
    single_cell_expression,
    cell_labels: Sequence[str],
    cell_types: Sequence[str],
) -> np.ndarray:
    """Estimate bounded between-type/within-type gene weights."""
    expression = np.clip(as_dense(single_cell_expression), 0, None)
    library = expression.sum(axis=1, keepdims=True)
    normalized = np.log1p(expression / np.maximum(library, 1) * 1e4)
    labels = np.asarray(cell_labels, dtype=str)
    group_means = []
    within_variance = np.zeros(normalized.shape[1], dtype=np.float64)
    for cell_type in cell_types:
        group = normalized[labels == str(cell_type)]
        if len(group) == 0:
            continue
        group_means.append(group.mean(axis=0))
        within_variance += group.var(axis=0)
    if not group_means:
        return np.ones(normalized.shape[1], dtype=np.float32)
    within_variance /= len(group_means)
    between_variance = np.var(np.vstack(group_means), axis=0)
    score = np.log1p(between_variance / (within_variance + 1e-4))
    positive = score[score > 0]
    if positive.size:
        lower, upper = np.quantile(positive, [0.05, 0.95])
        score = np.clip(score, lower, max(upper, lower + EPS))
    score = score / max(score.mean(), EPS)
    return np.clip(score, 0.25, 4.0).astype(np.float32)


def infer_pseudo_coordinates(
    real_features,
    pseudo_features,
    real_coordinates,
    neighbors: int = 4,
) -> np.ndarray:
    """Assign pseudo spots soft coordinates from nearby observed ST spots.

    A pseudo spot has no measured location.  Matching it to a weighted local
    barycenter of expression-nearest real spots creates a spatial bridge while
    keeping the target labels completely unused.  The caller can opt into the
    resulting pseudo spatial graph; the historical self-loop-only behavior is
    retained by default.
    """
    real = as_dense(real_features).astype(np.float32)
    pseudo = as_dense(pseudo_features).astype(np.float32)
    coordinates = as_dense(real_coordinates).astype(np.float32)
    if len(real) == 0 or len(pseudo) == 0:
        return np.empty((len(pseudo), 2), dtype=np.float32)
    if coordinates.shape != (len(real), 2):
        raise ValueError("real_coordinates must have shape (n_real, 2)")
    if real.shape[1] != pseudo.shape[1]:
        raise ValueError("real and pseudo features must share the feature dimension")
    neighbor_count = max(1, min(int(neighbors), len(real)))
    model = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric="cosine",
        n_jobs=-1,
    )
    model.fit(real)
    distances, indices = model.kneighbors(pseudo)
    positive = distances[distances > 0]
    scale = np.median(positive) if positive.size else 1.0
    weights = np.exp(-np.square(distances / max(scale, EPS)))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), EPS)
    return np.einsum("ij,ijk->ik", weights, coordinates[indices]).astype(
        np.float32
    )


def platform_match_pseudo_spots(
    pseudo_data: anndata.AnnData,
    target_data: anndata.AnnData,
    random_seed: int = 0,
    ratio_clip: tuple[float, float] = (0.25, 4.0),
) -> tuple[anndata.AnnData, pd.DataFrame]:
    """Match pseudo spots to target library sizes and damped gene efficiencies.

    Target labels are never used.  Only the observed target count matrix is
    used, which makes this a transductive, unsupervised platform correction.
    """
    target_gene_set = set(map(str, target_data.var_names))
    common_genes = [
        str(gene) for gene in pseudo_data.var_names if str(gene) in target_gene_set
    ]
    if not common_genes:
        raise ValueError("pseudo and target data have no common genes")
    pseudo = pseudo_data[:, common_genes].copy()
    target = target_data[:, common_genes].copy()
    pseudo_expression = np.clip(as_dense(pseudo.X), 0, None)
    target_expression = np.clip(as_dense(target.X), 0, None)

    pseudo_mean = pseudo_expression.mean(axis=0) + EPS
    target_mean = target_expression.mean(axis=0) + EPS
    efficiency = np.sqrt(target_mean / pseudo_mean)
    efficiency = np.clip(efficiency, ratio_clip[0], ratio_clip[1])
    adjusted = pseudo_expression * efficiency

    target_libraries = target_expression.sum(axis=1)
    target_libraries = target_libraries[target_libraries > 0]
    if target_libraries.size == 0:
        raise ValueError("target count matrix has zero library size in every spot")
    rng = np.random.default_rng(random_seed)
    sampled_libraries = rng.choice(target_libraries, size=pseudo.n_obs, replace=True)
    integer_like = (
        np.max(np.abs(target_expression - np.rint(target_expression))) < 1e-4
        and np.max(np.abs(pseudo_expression - np.rint(pseudo_expression))) < 1e-4
    )
    calibrated = np.zeros_like(adjusted, dtype=np.float64)
    for row_index, row in enumerate(adjusted):
        row_sum = row.sum()
        if row_sum <= 0:
            continue
        probabilities = row / row_sum
        target_total = max(1, int(round(float(sampled_libraries[row_index]))))
        if integer_like:
            calibrated[row_index] = rng.multinomial(target_total, probabilities)
        else:
            calibrated[row_index] = probabilities * target_total
    pseudo.X = calibrated.astype(np.float32)
    diagnostics = pd.DataFrame(
        {
            "gene": common_genes,
            "pseudo_mean_before": pseudo_mean,
            "target_mean": target_mean,
            "platform_efficiency": efficiency,
        }
    )
    return pseudo, diagnostics


def _append_knn_edges(
    rows: list[np.ndarray],
    columns: list[np.ndarray],
    weights: list[np.ndarray],
    query: np.ndarray,
    reference: np.ndarray,
    query_offset: int,
    reference_offset: int,
    neighbors: int,
    metric: str = "cosine",
    exclude_same_index: bool = False,
) -> None:
    if len(query) == 0 or len(reference) == 0 or neighbors <= 0:
        return
    extra = 1 if exclude_same_index else 0
    k = min(len(reference), neighbors + extra)
    model = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1)
    model.fit(reference)
    distances, indices = model.kneighbors(query)
    query_ids = np.repeat(np.arange(len(query)), k)
    reference_ids = indices.reshape(-1)
    flat_distances = distances.reshape(-1)
    if exclude_same_index:
        keep = query_ids != reference_ids
        query_ids = query_ids[keep]
        reference_ids = reference_ids[keep]
        flat_distances = flat_distances[keep]
    if metric == "cosine":
        edge_weights = np.clip(1 - flat_distances, 0.05, 1.0)
    else:
        positive = flat_distances[flat_distances > 0]
        scale = np.median(positive) if positive.size else 1.0
        edge_weights = np.exp(-np.square(flat_distances / max(scale, EPS)))
    rows.append(query_ids + query_offset)
    columns.append(reference_ids + reference_offset)
    weights.append(edge_weights.astype(np.float32))


def _coalesced_normalized_graph(
    node_count: int,
    rows: Iterable[np.ndarray],
    columns: Iterable[np.ndarray],
    weights: Iterable[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    row = np.concatenate(list(rows)).astype(np.int64)
    column = np.concatenate(list(columns)).astype(np.int64)
    value = np.concatenate(list(weights)).astype(np.float32)
    adjacency = sp.coo_matrix(
        (value, (row, column)), shape=(node_count, node_count)
    ).tocsr()
    adjacency = adjacency.maximum(adjacency.transpose())
    adjacency = adjacency + sp.eye(node_count, dtype=np.float32, format="csr")
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_root = np.power(np.maximum(degree, EPS), -0.5)
    adjacency = sp.diags(inverse_root) @ adjacency @ sp.diags(inverse_root)
    coo = adjacency.tocoo()
    edge_index = torch.from_numpy(
        np.vstack([coo.row, coo.col]).astype(np.int64)
    )
    edge_weight = torch.from_numpy(coo.data.astype(np.float32))
    return edge_index, edge_weight


@dataclass(frozen=True)
class SparseDualGraph:
    expression: tuple[torch.Tensor, torch.Tensor]
    spatial: tuple[torch.Tensor, torch.Tensor]
    real_spatial_edges: tuple[torch.Tensor, torch.Tensor]


def build_sparse_dual_graph(
    real_features,
    pseudo_features,
    real_coordinates,
    inter_neighbors: int = 8,
    real_expression_neighbors: int = 8,
    pseudo_expression_neighbors: int = 8,
    spatial_neighbors: int = 8,
    pseudo_coordinates=None,
    pseudo_spatial_edge_scale: float = 0.25,
) -> SparseDualGraph:
    """Construct O(Nk) expression and spatial graphs without boundary claims."""
    real = as_dense(real_features).astype(np.float32)
    pseudo = as_dense(pseudo_features).astype(np.float32)
    coordinates = as_dense(real_coordinates).astype(np.float32)
    if len(coordinates) != len(real) or coordinates.shape[1] != 2:
        raise ValueError("real coordinates must align with real expression rows")
    real_count = len(real)
    pseudo_count = len(pseudo)
    node_count = real_count + pseudo_count
    if pseudo_coordinates is not None:
        if not 0.0 <= float(pseudo_spatial_edge_scale) <= 1.0:
            raise ValueError("pseudo_spatial_edge_scale must lie in [0, 1]")
        pseudo_coordinates = as_dense(pseudo_coordinates).astype(np.float32)
        if pseudo_coordinates.shape != (pseudo_count, 2):
            raise ValueError(
                "pseudo_coordinates must have shape (n_pseudo, 2)"
            )

    expression_rows: list[np.ndarray] = []
    expression_columns: list[np.ndarray] = []
    expression_weights: list[np.ndarray] = []
    _append_knn_edges(
        expression_rows, expression_columns, expression_weights,
        real, real, 0, 0, real_expression_neighbors,
        metric="cosine", exclude_same_index=True,
    )
    _append_knn_edges(
        expression_rows, expression_columns, expression_weights,
        pseudo, pseudo, real_count, real_count,
        pseudo_expression_neighbors, metric="cosine", exclude_same_index=True,
    )
    _append_knn_edges(
        expression_rows, expression_columns, expression_weights,
        real, pseudo, 0, real_count, inter_neighbors, metric="cosine",
    )
    expression_graph = _coalesced_normalized_graph(
        node_count, expression_rows, expression_columns, expression_weights
    )

    spatial_rows: list[np.ndarray] = []
    spatial_columns: list[np.ndarray] = []
    spatial_weights: list[np.ndarray] = []
    if pseudo_coordinates is None:
        _append_knn_edges(
            spatial_rows, spatial_columns, spatial_weights,
            coordinates, coordinates, 0, 0, spatial_neighbors,
            metric="euclidean", exclude_same_index=True,
        )
        raw_spatial = _coalesced_normalized_graph(
            real_count, spatial_rows, spatial_columns, spatial_weights
        )
        raw_index, raw_weight = raw_spatial
        non_self = raw_index[0] != raw_index[1]
        real_spatial_edges = (raw_index[:, non_self], raw_weight[non_self])

        global_rows = [raw_index[0].numpy()]
        global_columns = [raw_index[1].numpy()]
        global_weights = [raw_weight.numpy()]
        if pseudo_count:
            pseudo_ids = np.arange(real_count, node_count, dtype=np.int64)
            global_rows.append(pseudo_ids)
            global_columns.append(pseudo_ids)
            global_weights.append(np.ones(pseudo_count, dtype=np.float32))
    else:
        all_coordinates = np.vstack([coordinates, pseudo_coordinates])
        _append_knn_edges(
            spatial_rows, spatial_columns, spatial_weights,
            all_coordinates, all_coordinates, 0, 0, spatial_neighbors,
            metric="euclidean", exclude_same_index=True,
        )
        if spatial_rows:
            spatial_rows[0] = np.asarray(spatial_rows[0])
            spatial_columns[0] = np.asarray(spatial_columns[0])
            spatial_weights[0] = np.asarray(spatial_weights[0]).copy()
            bridge_mask = (
                (spatial_rows[0] >= real_count)
                | (spatial_columns[0] >= real_count)
            )
            spatial_weights[0][bridge_mask] *= float(
                pseudo_spatial_edge_scale
            )
        raw_spatial = _coalesced_normalized_graph(
            node_count, spatial_rows, spatial_columns, spatial_weights
        )
        raw_index, raw_weight = raw_spatial
        real_edge_mask = (
            (raw_index[0] < real_count)
            & (raw_index[1] < real_count)
            & (raw_index[0] != raw_index[1])
        )
        real_spatial_edges = (
            raw_index[:, real_edge_mask],
            raw_weight[real_edge_mask],
        )
        global_rows = [raw_index[0].numpy()]
        global_columns = [raw_index[1].numpy()]
        global_weights = [raw_weight.numpy()]
    spatial_graph = _coalesced_normalized_graph(
        node_count, global_rows, global_columns, global_weights
    )
    return SparseDualGraph(
        expression=expression_graph,
        spatial=spatial_graph,
        real_spatial_edges=real_spatial_edges,
    )
