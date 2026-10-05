"""Independent numerical/schema checks; no model training or real image data."""

import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loop_ultrasound.metrics import (  # noqa: E402
    aggregate_cases,
    classification_metrics,
    paired_brier_change,
    select_sensitivity_threshold,
)


def record(case="a", image="a1", label=1, probability=0.8, step=1,
           arm="SJ", seed=17, dice=0.7):
    return dict(case_id=case, image_id=image, label=label,
                probability=probability, step=step, arm=arm, seed=seed, dice=dice)


class AggregationTests(unittest.TestCase):
    def test_probability_mean_not_mean_logits_and_no_cross_group_mixing(self):
        rows = [
            record(image="a1", probability=0.2, dice=0.4),
            record(image="a2", probability=0.9, dice=0.8),
            record(image="a1", step=4, probability=0.9),
            record(image="a1", seed=29, probability=0.1),
            record(image="a1", arm="SC", probability=0.3),
            record(case="b", image="b1", label=0, probability=0.7, dice=None),
        ]
        aggregated = aggregate_cases(reversed(rows))
        by_key = {(r["arm"], r["seed"], r["step"], r["case_id"]): r
                  for r in aggregated}
        self.assertEqual(len(by_key), 5)
        case = by_key[("SJ", 17, 1, "a")]
        # mean(0.2, 0.9)=0.55, whereas sigmoid(mean(logit(.2),logit(.9)))=.6.
        self.assertAlmostEqual(case["probability"], 0.55)
        self.assertAlmostEqual(case["dice"], 0.6)
        self.assertEqual(case["n_images"], 2)
        self.assertEqual(case["image_ids"], ["a1", "a2"])
        self.assertEqual(by_key[("SJ", 17, 4, "a")]["probability"], 0.9)
        self.assertEqual(by_key[("SJ", 29, 1, "a")]["probability"], 0.1)
        self.assertEqual(by_key[("SC", 17, 1, "a")]["probability"], 0.3)
        self.assertIsNone(by_key[("SJ", 17, 1, "b")]["dice"])
        self.assertEqual(by_key[("SJ", 17, 1, "b")]["n_dice"], 0)
        self.assertEqual(aggregate_cases(rows), aggregated)
        json.dumps(aggregated, allow_nan=False)

    def test_inconsistent_pathology_across_seed_and_arm_is_rejected(self):
        for changed in [dict(seed=29), dict(step=4), dict(arm="SC")]:
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "Inconsistent"):
                aggregate_cases([record(), record(image="a2", label=0, **changed)])

    def test_duplicate_images_and_cross_case_assignment_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_cases([record(), record()])
        with self.assertRaisesRegex(ValueError, "different Cases"):
            aggregate_cases([record(), record(case="b")])

    def test_invalid_record_schema_and_values_rejected(self):
        invalid = [dict(record(), probability=float("nan")),
                   dict(record(), probability=1.01), dict(record(), label=0.2),
                   dict(record(), dice=255), dict(record(), step=0),
                   dict(record(), seed=True), dict(record(), arm=""),
                   dict(record(), case_id=None)]
        missing = record()
        missing.pop("image_id")
        invalid.extend([missing, None])
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                aggregate_cases([row])
        self.assertEqual(aggregate_cases([]), [])


class ClassificationTests(unittest.TestCase):
    def test_unweighted_scores_counts_and_inclusive_threshold(self):
        metrics = classification_metrics([0, 1, 0, 1], [0.1, 0.9, 0.5, 0.2])
        self.assertAlmostEqual(metrics["brier"], (0.01 + 0.01 + 0.25 + 0.64) / 4)
        self.assertAlmostEqual(metrics["nll"], -math.log(0.9 * 0.9 * 0.5 * 0.2) / 4)
        self.assertEqual([metrics[k] for k in ("tp", "tn", "fp", "fn")], [1, 1, 1, 1])
        self.assertEqual(metrics["sensitivity"], 0.5)
        self.assertEqual(metrics["specificity"], 0.5)
        self.assertEqual(metrics["auroc"], 0.75)
        json.dumps(metrics, allow_nan=False)

    def test_auroc_ties_half_credit_and_order_invariance(self):
        # Positive-negative pairs win, tie, tie, lose => 0.5.
        y, p = np.array([1, 0, 1, 0]), np.array([0.8, 0.8, 0.2, 0.2])
        self.assertEqual(classification_metrics(y, p)["auroc"], 0.5)
        permutation = [3, 0, 2, 1]
        self.assertEqual(classification_metrics(y[permutation], p[permutation])["auroc"], 0.5)
        self.assertEqual(classification_metrics([0, 1], [0.5, 0.5])["auroc"], 0.5)
        self.assertEqual(classification_metrics([0, 1], [0.9, 0.1])["auroc"], 0.0)

    def test_single_class_nulls_and_finite_boundary_nll(self):
        negative = classification_metrics([0, 0], [0.0, 1.0])
        self.assertIsNone(negative["auroc"])
        self.assertIsNone(negative["sensitivity"])
        self.assertEqual(negative["specificity"], 0.5)
        positive = classification_metrics([1, 1], [0.0, 1.0])
        self.assertIsNone(positive["specificity"])
        self.assertTrue(math.isfinite(positive["nll"]))
        json.dumps([negative, positive], allow_nan=False)

    def test_bad_vectors_and_threshold_rejected(self):
        for labels, probabilities in [([], []), ([0, 1], [0.2]),
                                      ([[0, 1]], [[0.1, 0.9]]),
                                      ([0, 2], [0.1, 0.9]),
                                      ([float("nan")], [0.2]),
                                      ([0], [float("inf")]), ([1], [-0.1])]:
            with self.subTest(labels=labels, p=probabilities), self.assertRaises(ValueError):
                classification_metrics(labels, probabilities)
        with self.assertRaises(ValueError):
            classification_metrics([0, 1], [0.2, 0.8], threshold=1.1)


class PairedBootstrapTests(unittest.TestCase):
    def test_signed_changes_match_direct_paired_case_differences(self):
        y = [0, 1, 0, 1]
        early, late = [0.8, 0.2, 0.4, 0.6], [0.2, 0.8, 0.4, 0.6]
        result = paired_brier_change(y, early, late, draws=2000, seed=17)
        self.assertAlmostEqual(result["mean"], 0.3)
        self.assertAlmostEqual(result["ci95"][0], 0.0)
        self.assertAlmostEqual(result["ci95"][1], 0.6)
        self.assertEqual(result, paired_brier_change(y, early, late, draws=2000, seed=17))
        reversed_result = paired_brier_change(y, late, early, draws=2000, seed=17)
        self.assertAlmostEqual(reversed_result["mean"], -result["mean"])
        self.assertAlmostEqual(reversed_result["ci95"][0], -result["ci95"][1])
        self.assertAlmostEqual(reversed_result["ci95"][1], -result["ci95"][0])
        json.dumps(result, allow_nan=False)

    def test_balanced_opposite_directions_have_zero_mean_not_zero_uncertainty(self):
        result = paired_brier_change([0, 0], [0.9, 0.1], [0.1, 0.9], draws=1000)
        self.assertAlmostEqual(result["mean"], 0.0)
        self.assertLess(result["ci95"][0], 0.0)
        self.assertGreater(result["ci95"][1], 0.0)

    def test_identity_has_zero_interval_and_input_validation(self):
        result = paired_brier_change([0, 1], [0.3, 0.7], [0.3, 0.7], draws=10)
        self.assertEqual(result["mean"], 0.0)
        self.assertEqual(result["ci95"], [0.0, 0.0])
        for arguments in [dict(p_late=[0.2]), dict(draws=1), dict(seed=-1),
                          dict(p_late=[0.2, float("nan")])]:
            kwargs = dict(labels=[0, 1], p_early=[0.2, 0.8], p_late=[0.3, 0.7])
            kwargs.update(arguments)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                paired_brier_change(**kwargs)


class CalibrationThresholdTests(unittest.TestCase):
    def test_largest_observed_threshold_and_tie_sensitive_rule(self):
        result = select_sensitivity_threshold(
            [1, 1, 1, 1, 0, 0], [0.2, 0.4, 0.4, 0.9, 0.3, 0.8],
            target_sensitivity=0.75, source_split="calibration")
        self.assertEqual(result["threshold"], 0.4)
        self.assertEqual(result["achieved_sensitivity"], 0.75)
        self.assertEqual(result["specificity"], 0.5)
        # Higher observed cutoffs no longer meet the target.
        for threshold in [0.8, 0.9]:
            metrics = classification_metrics([1, 1, 1, 1, 0, 0],
                                             [0.2, 0.4, 0.4, 0.9, 0.3, 0.8], threshold)
            self.assertLess(metrics["sensitivity"], 0.75)
        json.dumps(result, allow_nan=False)

    def test_threshold_requires_calibration_not_self_test_selection(self):
        with self.assertRaises(TypeError):
            select_sensitivity_threshold([0, 1], [0.1, 0.9])
        for source in ["test", "train", "tune", "external", "Calibration", None]:
            with self.subTest(source=source), self.assertRaises(ValueError):
                select_sensitivity_threshold([0, 1], [0.1, 0.9], source_split=source)
        calibration = select_sensitivity_threshold([1, 1, 0], [0.2, 0.8, 0.4],
                                                   source_split="calibration")
        held_out = classification_metrics([1, 0], [0.1, 0.9], calibration["threshold"])
        self.assertEqual(calibration["threshold"], 0.2)
        self.assertEqual(held_out["threshold"], 0.2)
        self.assertEqual(held_out["sensitivity"], 0.0)
        self.assertEqual(held_out["specificity"], 0.0)

    def test_no_positives_invalid_target_and_integer_target_rounding(self):
        with self.assertRaises(ValueError):
            select_sensitivity_threshold([0, 0], [0.2, 0.8], source_split="calibration")
        for target in [0.0, 1.01, float("nan")]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                select_sensitivity_threshold([1], [0.3], target, source_split="calibration")
        # Twenty positives: 95% permits one false negative, not zero.
        result = select_sensitivity_threshold([1] * 20, np.arange(20) / 20,
                                              source_split="calibration")
        self.assertEqual(result["threshold"], 0.05)
        self.assertEqual(result["achieved_sensitivity"], 0.95)
        tiny = select_sensitivity_threshold([1], [0.3], np.nextafter(0.0, 1.0),
                                            source_split="calibration")
        self.assertEqual(tiny["threshold"], 0.3)


if __name__ == "__main__":
    unittest.main()
