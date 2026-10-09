#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from baseline_validation.build_paper_tables import (
    exact_sign_flip_p,
    holm_adjust,
    paired_statistical_comparisons,
)
from baseline_validation.common import DATASETS, METRIC_COLUMNS


class PaperTableStatisticsTests(unittest.TestCase):
    def test_exact_sign_flip_has_expected_minimum_for_three_units(self) -> None:
        self.assertEqual(exact_sign_flip_p(np.ones(3)), 0.25)

    def test_holm_adjustment_is_monotone_in_rank_order(self) -> None:
        adjusted = holm_adjust(pd.Series([0.03, 0.01, 0.20]))
        np.testing.assert_allclose(adjusted.to_numpy(), [0.06, 0.03, 0.20])

    def test_paired_statistics_do_not_duplicate_deterministic_baselines(self) -> None:
        records = []
        for protocol in ("standard", "slice", "animal"):
            for dataset in DATASETS:
                for seed in range(5):
                    record = {
                        "method": "STD-Net",
                        "protocol": protocol,
                        "dataset": dataset,
                        "seed": seed,
                    }
                    record.update({metric: 0.5 for metric in METRIC_COLUMNS})
                    records.append(record)
                for method in ("CARD", "NLSDeconv"):
                    record = {
                        "method": method,
                        "protocol": protocol,
                        "dataset": dataset,
                        "seed": 0,
                    }
                    record.update({metric: 0.6 for metric in METRIC_COLUMNS})
                    records.append(record)
        combined = pd.DataFrame(records)

        summary, effects = paired_statistical_comparisons(
            combined, unit_level="animal", bootstrap_replicates=100
        )

        self.assertEqual(len(summary), 24)
        self.assertTrue((summary["independent_unit_count"] == 3).all())
        self.assertTrue((summary["stdnet_seed_count"] == 5).all())
        self.assertEqual(effects["unit"].nunique(), 3)


if __name__ == "__main__":
    unittest.main()
