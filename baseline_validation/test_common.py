#!/usr/bin/env python3

from __future__ import annotations

import unittest

import pandas as pd

from baseline_validation.common import (
    CELL_TYPES_18,
    evaluate_frames,
    reference_datasets,
)


class FixedOntologyTests(unittest.TestCase):
    def test_identical_predictions_are_zero_error(self) -> None:
        truth = pd.DataFrame(
            {
                "Cardiomyocytes": [1.0, 0.0, 0.0],
                "Gut tube": [0.0, 1.0, 0.0],
                "Low quality": [0.0, 0.0, 1.0],
            },
            index=["spot1", "spot2", "low_only"],
        )
        prediction = pd.DataFrame(
            {
                "Cardiomyocytes": [1.0, 0.0, 0.2],
                "Gut tube": [0.0, 1.0, 0.8],
            },
            index=truth.index,
        )
        metrics = evaluate_frames(prediction, truth)
        self.assertEqual(metrics["cell_type_count"], len(CELL_TYPES_18))
        self.assertEqual(metrics["evaluated_spot_count"], 2)
        self.assertEqual(metrics["excluded_low_quality_only_spot_count"], 1)
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["jensen_shannon_divergence"], 0.0)
        self.assertAlmostEqual(metrics["spearman"], 1.0)

    def test_known_label_alias_is_merged(self) -> None:
        truth = pd.DataFrame(
            {"Forebrain_Midbrain_Hindbrain": [1.0]}, index=["spot1"]
        )
        prediction = pd.DataFrame(
            {"Forebrain/Midbrain/Hindbrain": [1.0]}, index=["spot1"]
        )
        metrics = evaluate_frames(prediction, truth)
        self.assertAlmostEqual(metrics["rmse"], 0.0)

    def test_reference_holdout_sets(self) -> None:
        self.assertEqual(reference_datasets("standard", "embryo1_z2"), ["embryo1_z2"])
        self.assertNotIn("embryo1_z2", reference_datasets("slice", "embryo1_z2"))
        animal_refs = reference_datasets("animal", "embryo1_z2")
        self.assertFalse(any(item.startswith("embryo1_") for item in animal_refs))
        self.assertEqual(len(animal_refs), 4)


if __name__ == "__main__":
    unittest.main()

