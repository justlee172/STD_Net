#!/usr/bin/env python3
"""Focused regression tests for the paper-aligned algorithm path."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata
import numpy as np
import pandas as pd
import torch

from STD_Net.DGAT import GradientReversal, conDGAT, dense_to_weighted_graph
from STD_Net.adjacency_matrix import (
    intra_dist_adj_with_boundary_decay,
    intra_dist_adj_with_clusters,
)
from STD_Net.stdg import pseudo_spot_generation_aligned
from STD_Net.optimization import (
    SparseCompositionalLoss,
    build_sparse_dual_graph,
    estimate_target_type_count_weights,
    infer_pseudo_coordinates,
    platform_match_pseudo_spots,
    sparsemax,
    support_calibrated_sparsemax,
    reconstruct_counts,
)
from server_validation.run_paper_aligned import evaluate_prediction, stage_dataset


class PaperAlignmentTests(unittest.TestCase):
    def test_condgat_forward_with_two_common_and_one_fcnn_layer(self) -> None:
        model = conDGAT(
            nfeat=4,
            nhid=8,
            common_hid_layers_num=2,
            fcnn_hid_layers_num=1,
            dropout=0.0,
            nout1=3,
        )
        model.eval()
        features = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        adjacency = torch.eye(5, dtype=torch.float32)

        with torch.no_grad():
            log_probabilities, layers = model(features, [adjacency, adjacency])

        self.assertEqual(log_probabilities.shape, (5, 3))
        torch.testing.assert_close(
            log_probabilities.exp().sum(dim=1),
            torch.ones(5),
        )
        self.assertIn("gc_out111", layers)
        self.assertNotIn("gc_out112", layers)

    def test_domain_adversarial_head_has_gradient_reversal(self) -> None:
        features = torch.randn(5, 4)
        adjacency = torch.eye(5, dtype=torch.float32)
        model = conDGAT(
            nfeat=4,
            nhid=8,
            common_hid_layers_num=1,
            fcnn_hid_layers_num=0,
            dropout=0.0,
            nout1=3,
            domain_adversarial=True,
        )
        model.eval()
        model(features, [adjacency, adjacency])
        domain_logits = model.domain_logits(model.layer_outputs['final_embedding'])
        self.assertEqual(domain_logits.shape, (5,))

        scalar = torch.ones(2, requires_grad=True)
        GradientReversal()(scalar).sum().backward()
        torch.testing.assert_close(scalar.grad, -torch.ones_like(scalar))

    def test_niche_adaptive_signature_decoder_preserves_counts(self) -> None:
        model = conDGAT(
            nfeat=4,
            nhid=8,
            common_hid_layers_num=1,
            fcnn_hid_layers_num=0,
            dropout=0.0,
            nout1=3,
            reconstruction_gene_count=5,
            adaptive_signature=True,
            support_smoothing_epsilon=0.02,
        )
        model.eval()
        features = torch.randn(4, 4)
        adjacency = torch.eye(4, dtype=torch.float32)
        log_probabilities, _ = model(features, [adjacency, adjacency])
        reference = torch.rand(3, 5)
        reference = reference / reference.sum(dim=1, keepdim=True)
        signatures, delta = model.adaptive_signatures(
            model.layer_outputs['final_embedding'], reference
        )
        self.assertEqual(signatures.shape, (4, 3, 5))
        self.assertEqual(delta.shape, (4, 3, 5))
        torch.testing.assert_close(
            signatures.sum(dim=2), torch.ones(4, 3), atol=1e-5, rtol=1e-5
        )
        reconstructed = reconstruct_counts(
            log_probabilities.exp(), signatures, torch.full((4,), 20.0)
        )
        torch.testing.assert_close(
            reconstructed.sum(dim=1), torch.full((4,), 20.0), atol=1e-4, rtol=1e-5
        )

    def test_staging_uses_raw_single_cell_counts(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source" / "slice_a"
            staged = root / "staged"
            (source / "sc_data").mkdir(parents=True)
            (source / "st_data").mkdir(parents=True)
            pd.DataFrame(
                {"cell_id": ["cell_2", "cell_1"], "celltype": ["b", "a"]}
            ).to_csv(source / "sc_data" / "sc_label.tsv", sep="\t", index=False)
            pd.DataFrame(
                {"x": [2.0, 1.0], "y": [0.0, 0.0]},
                index=["cell_2", "cell_1"],
            ).to_csv(source / "sc_data" / "sc_coordinates.csv")
            pd.DataFrame({"gene_1": [3.0]}, index=["spot_1"]).to_csv(
                source / "st_data" / "ST_data.tsv", sep="\t"
            )
            pd.DataFrame({"x": [0.0], "y": [0.0]}, index=["spot_1"]).to_csv(
                source / "st_data" / "coordinates.csv"
            )
            raw_counts = pd.DataFrame(
                {"cell_1": [1.0, 2.0], "cell_2": [3.0, 4.0]},
                index=["gene_1", "gene_2"],
            )

            stage_dataset(source.parent, staged, "slice_a", raw_counts)

            staged_expression = pd.read_csv(
                staged / "slice_a" / "sc_data" / "sc_data.tsv",
                sep="\t",
                index_col=0,
            )
            expected = raw_counts.loc[:, ["cell_2", "cell_1"]].T
            pd.testing.assert_frame_equal(staged_expression, expected)

    def test_staging_excludes_low_quality_reference_cells(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source" / "slice_a"
            staged = root / "staged"
            (source / "sc_data").mkdir(parents=True)
            (source / "st_data").mkdir(parents=True)
            pd.DataFrame(
                {
                    "cell_id": ["cell_a", "cell_low", "cell_b"],
                    "celltype": ["a", "Low quality", "b"],
                }
            ).to_csv(source / "sc_data" / "sc_label.tsv", sep="\t", index=False)
            pd.DataFrame(
                {"x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0]},
                index=["cell_a", "cell_low", "cell_b"],
            ).to_csv(source / "sc_data" / "sc_coordinates.csv")
            pd.DataFrame({"gene_1": [3.0]}, index=["spot_1"]).to_csv(
                source / "st_data" / "ST_data.tsv", sep="\t"
            )
            pd.DataFrame({"x": [0.0], "y": [0.0]}, index=["spot_1"]).to_csv(
                source / "st_data" / "coordinates.csv"
            )
            raw_counts = pd.DataFrame(
                {
                    "cell_a": [1.0, 2.0],
                    "cell_low": [3.0, 4.0],
                    "cell_b": [5.0, 6.0],
                },
                index=["gene_1", "gene_2"],
            )

            stage_dataset(source.parent, staged, "slice_a", raw_counts)

            staged_labels = pd.read_csv(
                staged / "slice_a" / "sc_data" / "sc_label.tsv", sep="\t"
            )
            staged_expression = pd.read_csv(
                staged / "slice_a" / "sc_data" / "sc_data.tsv",
                sep="\t",
                index_col=0,
            )
            staged_coordinates = pd.read_csv(
                staged / "slice_a" / "sc_data" / "sc_coordinates.csv",
                index_col=0,
            )
            self.assertEqual(staged_labels["cell_id"].tolist(), ["cell_a", "cell_b"])
            self.assertEqual(staged_expression.index.tolist(), ["cell_a", "cell_b"])
            self.assertEqual(staged_coordinates.index.tolist(), ["cell_a", "cell_b"])

    def test_prediction_metrics_use_fixed_biological_ontology(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prediction_path = root / "prediction.csv"
            truth_path = root / "truth.tsv"
            pd.DataFrame(
                {
                    "Cardiomyocytes": [1.0, 0.2, 0.0],
                    "Gut tube": [0.0, 0.8, 0.0],
                },
                index=["spot_1", "spot_2", "low_only"],
            ).to_csv(prediction_path)
            pd.DataFrame(
                {
                    "Cardiomyocytes": [1.0, 0.2, 0.0],
                    "Gut tube": [0.0, 0.8, 0.0],
                    "Low quality": [0.0, 0.0, 1.0],
                },
                index=["spot_1", "spot_2", "low_only"],
            ).to_csv(truth_path, sep="\t")

            metrics = evaluate_prediction(prediction_path, truth_path)

            self.assertEqual(metrics["cell_type_count"], 18)
            self.assertEqual(metrics["spot_count"], 2)
            self.assertEqual(metrics["excluded_low_quality_only_spot_count"], 1)
            self.assertAlmostEqual(metrics["rmse"], 0.0)

    def test_stdg_is_deterministic_and_respects_type_limit(self) -> None:
        cell_types = ["rare"] * 2 + ["a"] * 8 + ["b"] * 8 + ["c"] * 8 + ["d"] * 8
        type_names = list(dict.fromkeys(cell_types))
        type_to_index = {name: index for index, name in enumerate(type_names)}
        observations = pd.DataFrame(
            {
                "cell_type": cell_types,
                "cell_type_idx": [type_to_index[name] for name in cell_types],
            },
            index=[f"cell_{index}" for index in range(len(cell_types))],
        )
        expression = np.arange(len(cell_types) * 5, dtype=float).reshape(-1, 5) + 1
        data = anndata.AnnData(X=expression, obs=observations)
        coordinates = np.column_stack(
            [np.linspace(0, 1, len(cell_types)), np.zeros(len(cell_types))]
        )
        arguments = dict(
            single_cell_data=data,
            index_to_cell_type={index: name for name, index in type_to_index.items()},
            spot_count=20,
            min_cells=8,
            max_cells=12,
            max_cell_types=4,
            generation_method="celltype",
            spatial_coordinates=coordinates,
            random_seed=7,
            n_jobs=1,
        )
        first = pseudo_spot_generation_aligned(**arguments)
        second = pseudo_spot_generation_aligned(**arguments)
        np.testing.assert_allclose(first.X, second.X)
        np.testing.assert_allclose(first.obs.to_numpy(), second.obs.to_numpy())
        self.assertFalse(first.obs.isna().any().any())
        self.assertTrue(first.obs.index.equals(first.obs_names))
        proportion_columns = type_names
        active_types = (first.obs[proportion_columns] > 0).sum(axis=1)
        self.assertTrue((active_types <= 4).all())
        self.assertTrue(first.obs["cell_num"].between(8, 12).all())
        np.testing.assert_allclose(
            first.obs[proportion_columns].sum(axis=1).to_numpy(),
            np.ones(first.n_obs),
        )

    def test_cross_domain_interface_edges_are_attenuated(self) -> None:
        observations = pd.DataFrame(
            {
                "coor_X": [0.0, 0.2, 0.4, 0.6],
                "coor_Y": [0.0, 0.0, 0.0, 0.0],
            },
            index=[f"spot_{index}" for index in range(4)],
        )
        data = anndata.AnnData(X=np.ones((4, 2)), obs=observations)
        clusters = np.array([0, 0, 1, 1])
        boundaries = np.array([False, True, True, False])
        base = intra_dist_adj_with_clusters(
            data,
            clusters,
            link_method="soft",
            space_dist_neighbors=2,
            restrict_within_cluster=False,
        ).to_numpy()
        attenuated = intra_dist_adj_with_boundary_decay(
            data,
            clusters,
            boundaries,
            link_method="soft",
            space_dist_neighbors=2,
            boundary_decay=0.3,
        ).to_numpy()
        cross_edges = np.argwhere(
            (clusters[:, None] != clusters[None, :]) & (base > 0)
        )
        self.assertGreater(len(cross_edges), 0)
        for source, target in cross_edges:
            if boundaries[source] or boundaries[target]:
                self.assertAlmostEqual(
                    attenuated[source, target], 0.3 * base[source, target]
                )

    def test_weighted_graph_conversion_preserves_decay_value(self) -> None:
        adjacency = torch.tensor([[1.0, 0.3], [0.3, 1.0]])
        _, weights = dense_to_weighted_graph(adjacency)
        self.assertTrue(torch.any(torch.isclose(weights, torch.tensor(0.3))))

    def test_sparsemax_produces_a_sparse_simplex(self) -> None:
        logits = torch.tensor([[4.0, 1.0, -2.0], [0.2, 0.2, 0.2]])
        probabilities = sparsemax(logits)
        np.testing.assert_allclose(
            probabilities.sum(dim=1).numpy(), np.ones(2), atol=1e-6
        )
        self.assertGreater(int((probabilities[0] == 0).sum()), 0)
        self.assertTrue(torch.all(probabilities >= 0))

    def test_support_calibration_keeps_rare_type_gradient_reachable(self) -> None:
        logits = torch.tensor([[4.0, 1.0, -2.0]], requires_grad=True)
        target = torch.tensor([[0.0, 0.0, 1.0]])
        probabilities = support_calibrated_sparsemax(logits, epsilon=0.02)
        loss = -(target * probabilities.clamp_min(1e-8).log()).sum()
        loss.backward()

        self.assertTrue(torch.all(probabilities > 0))
        self.assertGreater(float(logits.grad[0, 2].abs()), 0.0)

        hard = sparsemax(logits.detach())
        self.assertEqual(float(hard[0, 2]), 0.0)

    def test_target_complexity_calibration_returns_valid_prior(self) -> None:
        reference = np.eye(4, dtype=float)
        target = np.array(
            [
                [10.0, 0.0, 0.0, 0.0],
                [5.0, 5.0, 0.0, 0.0],
                [3.0, 3.0, 3.0, 0.0],
                [2.0, 2.0, 2.0, 2.0],
            ]
        )
        weights, diagnostics = estimate_target_type_count_weights(
            target,
            reference,
            max_cell_types=4,
            temperature=0.05,
        )

        self.assertEqual(weights.shape, (4,))
        np.testing.assert_allclose(weights.sum(), 1.0)
        self.assertEqual(len(diagnostics), len(target))
        self.assertTrue(
            set(diagnostics['target_proxy_estimated_type_count'])
            .issubset({1, 2, 3, 4})
        )
        self.assertGreater(weights[1], 0.0)
        self.assertGreater(weights[2], 0.0)

    def test_composite_loss_is_finite(self) -> None:
        predictions = torch.tensor(
            [[0.70, 0.30, 0.00], [0.05, 0.05, 0.90]], dtype=torch.float32
        )
        targets = torch.tensor(
            [[0.75, 0.25, 0.00], [0.00, 0.10, 0.90]], dtype=torch.float32
        )
        loss = SparseCompositionalLoss()((predictions + 1e-8).log(), targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0)

    def test_rare_ranking_loss_has_finite_gradient(self) -> None:
        predictions = torch.tensor(
            [
                [0.70, 0.25, 0.05],
                [0.55, 0.40, 0.05],
                [0.65, 0.30, 0.05],
                [0.40, 0.55, 0.05],
                [0.80, 0.15, 0.05],
                [0.45, 0.50, 0.05],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        targets = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        loss_fn = SparseCompositionalLoss(
            rare_ranking_weight=0.1,
            rare_prevalence_threshold=0.5,
            rare_ranking_pairs=4,
        )
        components = loss_fn.components(predictions.log(), targets)
        self.assertTrue(torch.isfinite(components["rare_ranking"]))
        self.assertGreater(float(components["rare_ranking"]), 0.0)
        loss_fn(predictions.log(), targets).backward()
        self.assertTrue(torch.all(torch.isfinite(predictions.grad)))

    def test_platform_matching_uses_target_library_scale(self) -> None:
        pseudo = anndata.AnnData(
            X=np.array([[10, 2, 1], [2, 12, 1], [1, 1, 9]], dtype=float),
            obs=pd.DataFrame(
                {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 1.0]},
                index=["p1", "p2", "p3"],
            ),
            var=pd.DataFrame(index=["g1", "g2", "g3"]),
        )
        target = anndata.AnnData(
            X=np.array([[10, 10, 10], [20, 10, 10]], dtype=float),
            var=pd.DataFrame(index=["g1", "g2", "g3"]),
        )
        matched, diagnostics = platform_match_pseudo_spots(
            pseudo, target, random_seed=3
        )
        self.assertEqual(list(matched.var_names), ["g1", "g2", "g3"])
        self.assertEqual(len(diagnostics), 3)
        self.assertTrue(set(matched.X.sum(axis=1)).issubset({30.0, 40.0}))
        pd.testing.assert_frame_equal(matched.obs, pseudo.obs)

    def test_sparse_dual_graph_scales_with_neighbors(self) -> None:
        rng = np.random.default_rng(2)
        real = rng.normal(size=(12, 6))
        pseudo = rng.normal(size=(30, 6))
        coordinates = rng.normal(size=(12, 2))
        graph = build_sparse_dual_graph(
            real,
            pseudo,
            coordinates,
            inter_neighbors=3,
            real_expression_neighbors=3,
            pseudo_expression_neighbors=3,
            spatial_neighbors=3,
        )
        node_count = len(real) + len(pseudo)
        self.assertLess(graph.expression[0].shape[1], node_count ** 2)
        self.assertLess(graph.spatial[0].shape[1], node_count ** 2)
        self.assertEqual(graph.expression[0].shape[0], 2)
        self.assertEqual(graph.real_spatial_edges[0].shape[0], 2)

    def test_pseudo_coordinates_create_a_spatial_bridge(self) -> None:
        real_features = np.eye(3, dtype=float)
        pseudo_features = np.array(
            [[0.9, 0.1, 0.0], [0.0, 0.2, 0.8]], dtype=float
        )
        coordinates = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float
        )
        pseudo_coordinates = infer_pseudo_coordinates(
            real_features, pseudo_features, coordinates, neighbors=2
        )
        self.assertEqual(pseudo_coordinates.shape, (2, 2))
        self.assertLess(float(pseudo_coordinates[0, 0]), 1.0)
        self.assertGreater(float(pseudo_coordinates[1, 0]), 1.0)

        graph = build_sparse_dual_graph(
            real_features,
            pseudo_features,
            coordinates,
            inter_neighbors=1,
            real_expression_neighbors=1,
            pseudo_expression_neighbors=1,
            spatial_neighbors=1,
            pseudo_coordinates=pseudo_coordinates,
        )
        self.assertGreater(graph.spatial[0].shape[1], len(real_features) + len(pseudo_features))


if __name__ == "__main__":
    unittest.main()
