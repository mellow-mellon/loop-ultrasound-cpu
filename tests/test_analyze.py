"""Independent four-arm alignment, provenance, privacy and contrast checks."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loop_ultrasound.analyze import analyze_runs, main  # noqa: E402


ARMS = ("SC", "SJ", "UC", "UJ")
FIELDS = ("case_id", "image_id", "label", "probability", "step", "arm", "seed", "dice")


def write_csv(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def build_runs(root):
    """Two held-out Cases, with two views of Case 0 and one of Case 1.

    Every step-2 probability is .5. Step-4 Case probabilities give D:
    SC=-.055, SJ=.24, UC=.16, UJ=.09. Hence theta=.365.
    The per-Case theta values are .31 and .42, giving a [.31,.42] interval.
    """
    late = {"SC": (0.5, 0.4), "SJ": (0.1, 0.9), "UC": (0.3, 0.7), "UJ": (0.4, 0.6)}
    dice_change = {"SC": 0.0, "SJ": 0.4, "UC": 0.2, "UJ": 0.1}
    directories = []
    for arm in ARMS:
        directory = root / arm
        directory.mkdir()
        records = []
        for step in range(1, 5):
            for case, count in enumerate([2, 1]):
                probability = late[arm][case] if step == 4 else 0.5
                for image in range(count):
                    # Nonidentical view probabilities require actual mean
                    # aggregation, not keeping only the first or last image.
                    offset = (-0.1 if image == 0 else 0.1) if count == 2 else 0.0
                    records.append(dict(case_id=f"private_case_{case}",
                                        image_id=f"private_image_{case}_{image}", label=case,
                                        probability=probability + offset, step=step,
                                        arm=arm, seed=17,
                                        dice=0.4 + (dice_change[arm] if step == 4 else 0.0)))
        training = [dict(case_id=f"private_training_{case}", image_id=f"training_image_{case}",
                         label=case, probability=.5, step=step, arm=arm, seed=17, dice=.3)
                    for step in range(1, 5) for case in range(2)]
        summary = {
            "status": "executed_cpu_engineering_pilot", "clinical_validation": False,
            "patient_mapping": "unverified_Case_grouped_only", "device": "cpu",
            "config": {"arm": arm, "steps": 4, "seed": 17, "epochs": 30, "batch_size": 2,
                       "learning_rate": .001, "segmentation_weight": 1.0,
                       "cached_frozen_features": True, "augment": False,
                       "unknown_private_field": "private_local_path_or_identifier"},
            "encoder_sha256": "a" * 64, "manifest_sha256": "b" * 64,
            "n_training_cases": 2, "n_training_images": 2,
            "development_subset": {str(step): {"n_cases": 2} for step in range(1, 5)},
        }
        (directory / "summary.json").write_text(json.dumps(summary))
        write_csv(directory / "development_predictions.csv", records)
        write_csv(directory / "training_predictions.csv", training)
        directories.append(directory)
    return directories


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runs = build_runs(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def change_summary(self, index, update):
        path = self.runs[index] / "summary.json"
        value = json.loads(path.read_text())
        update(value)
        path.write_text(json.dumps(value))

    def change_records(self, index, update, filename="development_predictions.csv"):
        path = self.runs[index] / filename
        records = read_csv(path)
        update(records)
        write_csv(path, records)

    def test_explicit_alignment_macrocase_deltas_and_shared_bootstrap(self):
        # Reverse one CSV and directory order: no positional alignment allowed.
        self.change_records(1, lambda rows: rows.reverse())
        report = analyze_runs(list(reversed(self.runs)))
        expected = {"SC": (-.055, .0), "SJ": (.24, .4), "UC": (.16, .2), "UJ": (.09, .1)}
        for arm, (diagnostic, segmentation) in expected.items():
            self.assertAlmostEqual(report["arms"][arm]["D"], diagnostic)
            self.assertAlmostEqual(report["arms"][arm]["S"], segmentation)
            self.assertEqual(report["arms"][arm]["by_step"]["2"]["brier"], .25)
        self.assertAlmostEqual(report["theta_D"]["estimate"], .365)
        self.assertAlmostEqual(report["theta_D"]["ci95"][0], .31)
        self.assertAlmostEqual(report["theta_D"]["ci95"][1], .42)
        self.assertEqual(report["n_development_cases"], 2)
        self.assertEqual(report["n_development_images"], 3)
        self.assertEqual(report, analyze_runs(self.runs))

    def test_public_output_has_no_identifiers_paths_or_individual_labels(self):
        report = analyze_runs(self.runs, draws=100)
        encoded = json.dumps(report, allow_nan=False)
        for private_value in ["private_case", "private_image", "private_training",
                              "training_image_0", "private_local_path_or_identifier", str(self.root)]:
            self.assertNotIn(private_value, encoded)
        self.assertNotIn("unknown_private_field", report["config"])
        self.assertIs(report["clinical_validation"], False)
        self.assertIs(report["mechanism_identified"], False)

    def test_requires_four_distinct_completed_arms(self):
        for dirs in [self.runs[:3], self.runs + self.runs[:1], self.runs[:3] + self.runs[:1]]:
            with self.subTest(dirs=dirs), self.assertRaises(ValueError):
                analyze_runs(dirs)
        self.change_summary(0, lambda s: s.update(status="proposed_not_executed"))
        with self.assertRaisesRegex(ValueError, "completed"):
            analyze_runs(self.runs)

    def test_missing_step_and_missing_case_are_rejected_without_intersection(self):
        self.change_records(0, lambda rows: rows.__setitem__(slice(None),
                            [r for r in rows if r["step"] != "3"]))
        with self.assertRaisesRegex(ValueError, "Case sets"):
            analyze_runs(self.runs)

    def test_missing_case_at_all_steps_is_rejected(self):
        self.change_records(0, lambda rows: rows.__setitem__(slice(None),
                            [r for r in rows if r["case_id"] != "private_case_1"]))
        self.change_summary(0, lambda s: s.update(development_subset={str(k): {"n_cases": 1} for k in range(1, 5)}))
        with self.assertRaisesRegex(ValueError, "Case sets"):
            analyze_runs(self.runs)

    def test_view_sets_must_match_even_if_case_counts_match(self):
        self.change_records(0, lambda rows: rows.__setitem__(slice(None),
                            [r for r in rows if r["image_id"] != "private_image_0_1"]))
        with self.assertRaisesRegex(ValueError, "view sets"):
            analyze_runs(self.runs)

    def test_conflicting_labels_across_arms_raise(self):
        def flip(rows):
            for row in rows:
                if row["case_id"] == "private_case_0":
                    row["label"] = 1
        self.change_records(0, flip)
        with self.assertRaisesRegex(ValueError, "labels conflict"):
            analyze_runs(self.runs)

    def test_duplicate_image_and_missing_dice_are_rejected(self):
        self.change_records(0, lambda rows: rows.append(dict(rows[0])))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze_runs(self.runs)

    def test_dice_required_for_every_view_and_step(self):
        self.change_records(0, lambda rows: rows[0].update(dice=""))
        with self.assertRaisesRegex(ValueError, "Dice"):
            analyze_runs(self.runs)

    def test_seed_or_arm_in_trace_must_match_summary(self):
        self.change_records(0, lambda rows: rows[0].update(seed=29))
        with self.assertRaisesRegex(ValueError, "arm, seed, or step"):
            analyze_runs(self.runs)

    def test_config_and_provenance_must_match(self):
        # Restore the source between independent provenance alterations.
        path = self.runs[0] / "summary.json"
        original = path.read_text()
        for update in [lambda s: s["config"].update(seed=29),
                       lambda s: s["config"].update(learning_rate=.002),
                       lambda s: s.update(encoder_sha256="c" * 64),
                       lambda s: s.update(manifest_sha256="d" * 64),
                       lambda s: s.update(n_training_cases=3)]:
            path.write_text(original)
            self.change_summary(0, update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                analyze_runs(self.runs)

    def test_equal_training_counts_do_not_hide_different_training_cases(self):
        def rename(rows):
            for row in rows:
                if row["case_id"] == "private_training_0":
                    row["case_id"] = "different_private_training_case"
        self.change_records(0, rename, filename="training_predictions.csv")
        with self.assertRaisesRegex(ValueError, "Case sets"):
            analyze_runs(self.runs)

    def test_training_development_overlap_rejected(self):
        def overlap(rows):
            for row in rows:
                if row["case_id"] == "private_training_0":
                    row["case_id"] = "private_case_0"
        self.change_records(0, overlap, filename="training_predictions.csv")
        with self.assertRaisesRegex(ValueError, "overlap"):
            analyze_runs(self.runs)

    def test_summary_counts_not_trusted_without_matching_trace(self):
        self.change_summary(0, lambda s: s["development_subset"]["4"].update(n_cases=3))
        with self.assertRaisesRegex(ValueError, "count disagrees"):
            analyze_runs(self.runs)

    def test_malformed_development_summary_is_rejected(self):
        self.change_summary(0, lambda s: s["development_subset"].update({"4": "invalid"}))
        with self.assertRaisesRegex(ValueError, "metric mappings"):
            analyze_runs(self.runs)

    def test_cli_writes_aggregate_report_and_prints_no_local_paths(self):
        destination = self.root / "analysis.json"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = main(["--run-dirs", *map(str, self.runs), "--output", str(destination), "--draws", "100"])
        self.assertEqual(result, 0)
        report = json.loads(destination.read_text())
        self.assertEqual(report["theta_D"]["bootstrap"]["draws"], 100)
        self.assertNotIn(str(self.root), stdout.getvalue())
        self.assertNotIn("private_case", destination.read_text())

    def test_cli_error_sanitizes_prediction_identifiers(self):
        self.change_records(0, lambda rows: rows.append(dict(rows[0])))
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            main(["--run-dirs", *map(str, self.runs)])
        self.assertNotIn("private_case", stderr.getvalue())
        self.assertNotIn("private_image", stderr.getvalue())

    def test_cli_does_not_overwrite_existing_analysis(self):
        destination = self.root / "existing_analysis.json"
        original = '{"preserve": "completed analysis"}\n'
        destination.write_text(original)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main(["--run-dirs", *map(str, self.runs), "--output", str(destination)])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(destination.read_text(), original)
        self.assertIn("already exists", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
