"""Twelve-run cohort alignment, seed pairing, baseline and privacy checks."""
import contextlib
import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest

from loop_ultrasound.analyze_expanded import (_fingerprint, _paired_case_interval,
                                             analyze_expanded, main)


ARMS = ("SC", "SJ", "UC", "UJ")
SEEDS = (17, 29, 43)
FIELDS = ("case_id", "image_id", "label", "probability", "step", "arm", "seed", "dice", "split")


def csv_write(path, records, fields=FIELDS):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def csv_edit(path, update):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, records = reader.fieldnames, list(reader)
    update(records)
    csv_write(path, records, fields)


def summary_edit(path, update):
    value = json.loads(path.read_text())
    update(value)
    path.write_text(json.dumps(value))


def index_fixture(split):
    prefix = "private_training" if split == "train" else "private_development"
    counts = (1, 1) if split == "train" else (2, 1)
    return {f"{prefix}_{case}": {"label": case, "n_images": count,
                               "image_ids": [f"{prefix}_image_{case}_{image}" for image in range(count)]}
            for case, count in enumerate(counts)}


@pytest.fixture
def experiment(tmp_path):
    runs = []
    training, development = index_fixture("train"), index_fixture("tune")
    selection = {"sampling": "proportional", "selection_seed": 20261004,
                 "train_fingerprint": _fingerprint(training), "tune_fingerprint": _fingerprint(development)}
    for seed in SEEDS:
        for arm in ARMS:
            directory = tmp_path / f"{arm}-{seed}"
            directory.mkdir()
            summary = {
                "status": "executed_cpu_engineering_pilot", "clinical_validation": False,
                "patient_mapping": "unverified_Case_grouped_only", "device": "cpu",
                "evaluation_split": "tune_development", "cohort_selection": selection,
                "model_selection": "fixed_final_epoch_no_best_epoch_selection",
                "config": {"arm": arm, "seed": seed, "steps": 4, "epochs": 50,
                           "batch_size": 2, "learning_rate": .0003, "segmentation_weight": 1.,
                           "cached_frozen_features": True, "augment": False,
                           "selection_seed": 20261004, "sampling": "proportional", "eval_every": 10,
                           "unknown_private_field": "private_local_path_or_identifier"},
                "encoder_sha256": "a"*64, "manifest_sha256": "b"*64,
                "n_training_cases": 2, "n_training_images": 2,
                "development_subset": {str(step): {"n_cases": 2} for step in range(1, 5)},
            }
            (directory / "summary.json").write_text(json.dumps(summary))
            for split, index, filename in (("train", training, "training_predictions.csv"),
                                          ("tune", development, "development_predictions.csv")):
                records = []
                for step in range(1, 5):
                    for case, row in index.items():
                        label = row["label"]
                        probability = .5
                        if step == 4 and split == "tune":
                            late = {"SC": (.5, .4), "SJ": ((.1, .2, .3)[SEEDS.index(seed)],
                                                          (.9, .8, .7)[SEEDS.index(seed)]),
                                    "UC": (.3, .7), "UJ": (.4, .6)}
                            probability = late[arm][label]
                        for view, image_id in enumerate(row["image_ids"]):
                            offset = (-.05 if view == 0 else .05) if row["n_images"] == 2 else 0.
                            dice = .4 + ({"SC": 0, "SJ": .4, "UC": .2, "UJ": .1}[arm] if step == 4 else 0.)
                            records.append(dict(case_id=case, image_id=image_id, label=label,
                                                probability=probability+offset, step=step, arm=arm,
                                                seed=seed, dice=dice, split=split))
                csv_write(directory / filename, records)
            runs.append(directory)
    return runs


def analyze(runs, **kwargs):
    return analyze_expanded(runs, expected_training_cases=2, expected_development_cases=2, **kwargs)


def write_baselines(root):
    root.mkdir()
    summary = {
        "status": "executed_cpu_preexperiment_baselines", "clinical_validation": False,
        "device": "cpu", "patient_mapping": "unverified_Case_grouped_only",
        "sources": {"encoder_sha256": "a"*64, "manifest_sha256": "b"*64},
        "cohort_fingerprints": {split: _fingerprint(index_fixture(split)) for split in ("train", "tune")},
    }
    (root / "baselines_summary.json").write_text(json.dumps(summary))
    rows = []
    for name in ("constant", "logistic_regression"):
        for split in ("train", "tune"):
            for case, item in index_fixture(split).items():
                rows.append(dict(baseline=name, split=split, seed=17, case_id=case,
                                 label=item["label"], n_images=item["n_images"],
                                 probability=.5 if name == "constant" else (.3, .7)[item["label"]]))
    csv_write(root / "baseline_case_predictions.csv", rows,
              ("baseline", "split", "seed", "case_id", "label", "probability", "n_images"))
    return root


def test_known_seed_means_sample_std_and_exact_case_bootstrap(experiment):
    report = analyze(list(reversed(experiment)))
    sj = report["arms"]["SJ"]
    expected_D = np.array([.24, .21, .16])
    assert sj["D"]["estimate"] == pytest.approx(expected_D.mean())
    assert sj["D"]["std_across_training_seeds"] == pytest.approx(expected_D.std(ddof=1))
    assert sj["D"]["positive_seed_count"] == 3
    assert sj["S"]["estimate"] == pytest.approx(.4)
    expected_theta = expected_D+.125
    assert report["theta_D"]["estimate"] == pytest.approx(expected_theta.mean())
    assert report["theta_D"]["ci95"] == pytest.approx([expected_D.mean()+.07, expected_D.mean()+.18])
    assert report["theta_D"]["bootstrap"]["n_cases"] == 2
    assert report["theta_D"]["bootstrap"]["includes_training_seed_sampling_uncertainty"] is False
    assert [item["theta_D"] for item in report["seed_results"]] == pytest.approx(expected_theta.tolist())
    assert report == analyze(experiment)


def test_seed_metrics_are_averaged_not_predictions(experiment):
    report = analyze(experiment, draws=100)
    expected = np.mean([.1**2, .2**2, .3**2])
    assert report["arms"]["SJ"]["by_step"]["4"]["brier"]["mean"] == pytest.approx(expected)
    assert expected != pytest.approx(.2**2)


def test_case_bootstrap_does_not_triple_cases_when_seed_effects_repeat():
    per_case = np.array([0., 1.])
    values = np.stack([per_case, per_case, per_case])
    interval = _paired_case_interval(values, draws=2000, bootstrap_seed=17)
    # Case sampling has support {0, .5, 1}; an incorrect flattened bootstrap
    # of six observations would produce a narrower [.1667,.8333] interval.
    assert interval["ci95"] == [0., 1.]
    assert interval["bootstrap"]["n_cases"] == 2


def test_public_output_contains_no_case_ids_paths_or_label_vectors(experiment, tmp_path):
    report = analyze(experiment, baseline_dir=write_baselines(tmp_path/"baselines"), draws=100)
    encoded = json.dumps(report, allow_nan=False)
    for private in ("private_training", "private_development", "private_local_path", str(tmp_path)):
        assert private not in encoded
    assert "unknown_private_field" not in report["config"]
    assert report["clinical_validation"] is False
    assert report["decision_rule_automatically_applied"] is False


def test_baseline_pairing_uses_identical_cases_and_fixed_fit(experiment, tmp_path):
    report = analyze(experiment, baseline_dir=write_baselines(tmp_path/"baselines"), draws=100)
    const = report["baselines"]["constant"]
    assert const["fixed_fit_shared_across_all_training_seeds"] is True
    assert const["model_minus_baseline_brier"]["SJ"]["4"]["estimate"] == pytest.approx(-np.mean([.24,.21,.16]))
    linear = report["baselines"]["logistic_regression"]
    assert linear["tune_metrics"]["brier"] == pytest.approx(.09)
    # Third seed ties this baseline exactly; zero is not an improvement.
    assert linear["model_minus_baseline_brier"]["SJ"]["4"]["negative_seed_count"] == 2


@pytest.mark.parametrize("bad_runs", ["missing", "duplicate"])
def test_requires_twelve_distinct_expected_arm_seed_runs(experiment, bad_runs):
    dirs = experiment[:-1] if bad_runs == "missing" else experiment[:-1]+experiment[:1]
    with pytest.raises(ValueError, match="twelve distinct"):
        analyze(dirs)


@pytest.mark.parametrize("update", [lambda s: s["config"].update(seed=71),
                                   lambda s: s["config"].update(seed=17, arm="SC")])
def test_unexpected_or_duplicate_arm_seed_pair_rejected(experiment, update):
    summary_edit(experiment[-1]/"summary.json", update)
    with pytest.raises(ValueError, match="expected seed"):
        analyze(experiment)


@pytest.mark.parametrize("update", [lambda s: s["config"].update(selection_seed=99),
                                   lambda s: s["config"].update(learning_rate=.002),
                                   lambda s: s.update(encoder_sha256="c"*64),
                                   lambda s: s.update(manifest_sha256="c"*64)])
def test_config_and_source_discrepancies_across_seeds_rejected(experiment, update):
    summary_edit(experiment[-1]/"summary.json", update)
    with pytest.raises(ValueError, match="configs"):
        analyze(experiment)


def test_equal_counts_cannot_hide_different_cases_across_seeds(experiment):
    def rename(rows):
        for row in rows:
            if row["case_id"] == "private_development_0":
                row["case_id"] = "other_private_development"
    csv_edit(experiment[-1]/"development_predictions.csv", rename)
    with pytest.raises(ValueError, match="Case sets"):
        analyze(experiment)


def test_equal_counts_cannot_hide_different_views(experiment):
    def rename(rows):
        for row in rows:
            if row["image_id"] == "private_development_image_0_0":
                row["image_id"] = "different_private_view"
    csv_edit(experiment[-1]/"development_predictions.csv", rename)
    with pytest.raises(ValueError, match="view sets"):
        analyze(experiment)


def test_conflicting_pathology_across_seeds_rejected(experiment):
    def flip(rows):
        for row in rows:
            if row["case_id"] == "private_development_0":
                row["label"] = 1
    csv_edit(experiment[-1]/"development_predictions.csv", flip)
    with pytest.raises(ValueError, match="labels conflict"):
        analyze(experiment)


@pytest.mark.parametrize("split", ["cal", "calibration", "test", "train", ""])
def test_calibration_test_or_other_non_tune_development_records_rejected(experiment, split):
    csv_edit(experiment[-1]/"development_predictions.csv", lambda rows: rows[0].update(split=split))
    with pytest.raises(ValueError, match="train/tune split"):
        analyze(experiment)


def test_false_source_provenance_cannot_be_hidden_by_filename(experiment):
    summary_edit(experiment[-1]/"summary.json", lambda s: s.update(evaluation_split="test"))
    with pytest.raises(ValueError, match="tune-only"):
        analyze(experiment)


def test_best_development_epoch_selection_is_not_a_fixed_final_epoch(experiment):
    summary_edit(experiment[-1]/"summary.json", lambda s: s.update(model_selection="best_tune_epoch"))
    with pytest.raises(ValueError, match="prespecified final epoch"):
        analyze(experiment)


def test_cohort_fingerprint_must_match_actual_prediction_records(experiment):
    summary_edit(experiment[-1]/"summary.json", lambda s: s["cohort_selection"].update(train_fingerprint="f"*64))
    with pytest.raises(ValueError, match="fingerprints"):
        analyze(experiment)


def test_declared_counts_cannot_be_substituted_for_full_cohort(experiment):
    with pytest.raises(ValueError, match="Case counts"):
        analyze_expanded(experiment)


@pytest.mark.parametrize("field", ["label", "n_images", "split"])
def test_baseline_trace_must_match_cases_labels_counts_and_split(experiment, tmp_path, field):
    directory = write_baselines(tmp_path/"baselines")
    def change(rows):
        row = next(r for r in rows if r["split"] == "tune")
        row[field] = {"label": 1, "n_images": 7, "split": "test"}[field]
    csv_edit(directory/"baseline_case_predictions.csv", change)
    with pytest.raises(ValueError):
        analyze(experiment, baseline_dir=directory)


def test_baseline_fingerprints_and_model_source_must_match(experiment, tmp_path):
    directory = write_baselines(tmp_path/"baselines")
    summary_edit(directory/"baselines_summary.json", lambda s: s["sources"].update(encoder_sha256="f"*64))
    with pytest.raises(ValueError, match="digests"):
        analyze(experiment, baseline_dir=directory)


def test_cli_writes_immutable_aggregate_only_report(experiment, tmp_path):
    output = tmp_path/"result.json"
    args = ["--run-dirs", *map(str, experiment), "--output", str(output), "--draws", "100",
            "--expected-training-cases", "2", "--expected-development-cases", "2"]
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        assert main(args) == 0
    first = output.read_text()
    assert "private_development" not in first
    assert str(tmp_path) not in stdout.getvalue()
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
        main(args)
    assert output.read_text() == first
