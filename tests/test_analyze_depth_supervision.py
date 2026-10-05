"""Endpoint-vs-readout contrasts, exact paired provenance and aggregate privacy."""
import copy
import csv
import json

import numpy as np
import pytest

from loop_ultrasound.analyze_depth_supervision import (
    DEPTHS, FIXED_CONFIG, SEEDS, SUPERVISIONS, analyze_depth_supervision, main,
)
from loop_ultrasound.analyze_expanded import _fingerprint
from loop_ultrasound.metrics import aggregate_cases, classification_metrics


FIELDS = ("case_id", "image_id", "label", "probability", "step", "arm", "seed", "dice", "split")


def write_csv(path, records, fields=FIELDS):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def edit_csv(path, change):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, records = reader.fieldnames, list(reader)
    change(records)
    write_csv(path, records, fields)


def edit_json(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def cohort(split):
    prefix = f"private_{split}"
    counts = (1, 1) if split == "train" else (2, 1)
    return {f"{prefix}_case{label}": {"label": label, "n_images": count,
                                     "image_ids": [f"{prefix}_image{label}_{view}" for view in range(count)]}
            for label, count in enumerate(counts)}


def digest(seed):
    return f"{seed:064x}"


def records_for(seed, depth, supervision, split):
    records = []
    for step in range(1, depth + 1):
        error = .45
        if step == depth:
            errors = {(2, "all"): (.3, .4, .5), (4, "all"): (.2, .3, .4),
                      (2, "terminal"): (.4, .4, .4), (4, "terminal"): (.1, .2, .3)}
            error = errors[(depth, supervision)][SEEDS.index(seed)]
        # A depth4 model's intermediate output deliberately differs from a
        # separately fitted depth2 endpoint. Confusing them changes the sign.
        if depth == 4 and step == 2:
            error = .05 if supervision == "all" else .6
        for case, row in cohort(split).items():
            for view, image in enumerate(row["image_ids"]):
                offset = ((-.025, .025)[view] if row["n_images"] == 2 else 0.)
                probability = (error if row["label"] == 0 else 1.-error) + offset
                records.append({"case_id": case, "image_id": image, "label": row["label"],
                                "probability": probability, "step": step, "arm": "SJ", "seed": seed,
                                "dice": .5+.02*step, "split": split})
    return records


def observed_metrics(records, depth):
    rows = aggregate_cases(records)
    result = {}
    for step in range(1, depth+1):
        selected = [row for row in rows if row["step"] == step]
        metric = classification_metrics([row["label"] for row in selected],
                                        [row["probability"] for row in selected])
        metric["mean_case_dice"] = float(np.mean([row["dice"] for row in selected]))
        metric["dice_space"] = "224x224_letterbox_valid_pixels"
        result[str(step)] = metric
    return result


@pytest.fixture
def experiment(tmp_path):
    protocol = {**{key: value for key, value in FIXED_CONFIG.items() if key != "arm"},
                "status": "fixed_before_depth_supervision_training", "clinical_validation": False,
                "device": "cpu", "patient_mapping": "unverified_Case_grouped_only",
                "calibration_and_test_predictions": False,
                "model_selection": "fixed_final_epoch_no_best_epoch_selection",
                "arms": ["SJ"], "seeds": list(SEEDS), "depths": list(DEPTHS),
                "supervisions": list(SUPERVISIONS), "warm_starts": False,
                "training_updates_per_run": 50,
                "encoder_sha256": "a"*64, "manifest_sha256": "b"*64, "training_source_sha256": "c"*64,
                "initialization_sha256_by_seed": {str(seed): digest(seed) for seed in SEEDS}}
    for split in ("train", "tune"):
        index = cohort(split)
        protocol[split] = {"cases": len(index), "images": sum(row["n_images"] for row in index.values()),
                           "benign": 1, "malignant": 1, "fingerprint": _fingerprint(index)}
    path = tmp_path / "locked_protocol.json"
    path.write_text(json.dumps(protocol))
    runs = []
    for seed in SEEDS:
        for depth in DEPTHS:
            for supervision in SUPERVISIONS:
                directory = tmp_path / f"SJ-depth{depth}-{supervision}-seed{seed}"
                directory.mkdir()
                tune_records = records_for(seed, depth, supervision, "tune")
                metrics = observed_metrics(tune_records, depth)
                summary = {"status": "executed_cpu_engineering_pilot", "clinical_validation": False,
                           "patient_mapping": "unverified_Case_grouped_only", "device": "cpu",
                           "evaluation_split": "tune_development",
                           "model_selection": "fixed_final_epoch_no_best_epoch_selection",
                           "config": {**FIXED_CONFIG, "seed": seed, "steps": depth,
                                      "supervision": supervision, "unknown_field": "private_local_filename"},
                           "cohort_selection": {"sampling": "proportional", "selection_seed": 20261004,
                                                "train_fingerprint": protocol["train"]["fingerprint"],
                                                "tune_fingerprint": protocol["tune"]["fingerprint"]},
                           "encoder_sha256": "a"*64, "manifest_sha256": "b"*64,
                           "training_source_sha256": "c"*64, "initialization_sha256": digest(seed),
                           "readout_supervision": {"mode": supervision, "directly_supervised_steps":
                                                   list(range(1, depth+1)) if supervision == "all" else [depth]},
                           "n_training_cases": 2, "n_training_images": 2,
                           "timing": {"updates": 50}, "development_subset": metrics,
                           "development_history": [{"epoch": epoch, "by_step": copy.deepcopy(metrics)}
                                                   for epoch in range(5, 51, 5)]}
                (directory / "summary.json").write_text(json.dumps(summary))
                write_csv(directory / "training_predictions.csv", records_for(seed, depth, supervision, "train"))
                write_csv(directory / "development_predictions.csv", tune_records)
                runs.append(directory)
    return runs, path


def analyze(experiment, **kwargs):
    runs, protocol = experiment
    return analyze_depth_supervision(runs, protocol_path=protocol, expected_training_cases=2,
                                    expected_development_cases=2, draws=100, **kwargs)


def baseline(directory):
    directory.mkdir()
    summary = {"status": "executed_cpu_preexperiment_baselines", "clinical_validation": False,
               "device": "cpu", "patient_mapping": "unverified_Case_grouped_only",
               "sources": {"encoder_sha256": "a"*64, "manifest_sha256": "b"*64},
               "cohort_fingerprints": {split: _fingerprint(cohort(split)) for split in ("train", "tune")}}
    (directory / "baselines_summary.json").write_text(json.dumps(summary))
    rows = []
    for name in ("constant", "logistic_regression"):
        for split in ("train", "tune"):
            for case, row in cohort(split).items():
                rows.append({"baseline": name, "split": split, "seed": 17, "case_id": case,
                             "label": row["label"], "n_images": row["n_images"],
                             "probability": .5 if name == "constant" else (.2, .8)[row["label"]]})
    write_csv(directory / "baseline_case_predictions.csv", rows,
              ("baseline", "split", "seed", "case_id", "label", "n_images", "probability"))
    return directory


def test_primary_uses_separately_fitted_endpoints_not_intermediate_readouts(experiment):
    report = analyze(experiment)
    all_benefit = np.array([.3**2-.2**2, .4**2-.3**2, .5**2-.4**2])
    terminal_benefit = np.array([.4**2-.1**2, .4**2-.2**2, .4**2-.3**2])
    assert report["primary_depth_benefit"]["all"]["estimate"] == pytest.approx(all_benefit.mean())
    assert report["primary_depth_benefit"]["terminal"]["estimate"] == pytest.approx(terminal_benefit.mean())
    assert report["interaction_B"]["estimate"] == pytest.approx((terminal_benefit-all_benefit).mean())
    assert report["primary_depth_benefit"]["all"]["std_across_training_seeds"] == pytest.approx(all_benefit.std(ddof=1))
    assert report["primary_depth_benefit"]["all"]["positive_seed_count"] == 3
    assert report["secondary_within_depth4"]["all"]["Brier_change"]["estimate"] < 0
    assert report["secondary_within_depth4"]["terminal"]["Brier_change"]["estimate"] > 0
    assert report["secondary_within_depth4"]["terminal"]["intermediate_readout_directly_supervised"] is False
    assert report["secondary_within_depth4"]["all"]["Dice_change"]["estimate"] == pytest.approx(.04)


def test_metrics_average_fitted_seed_metrics_not_ensemble_probabilities(experiment):
    report = analyze(experiment)
    expected = np.mean(np.array([.2, .3, .4])**2)
    assert report["regimes"]["depth4_all"]["endpoint"]["brier"]["mean"] == pytest.approx(expected)
    assert expected != pytest.approx(.3**2)
    assert report["regimes"]["depth2_terminal"]["by_step"].keys() == {"1", "2"}
    assert [item["seed"] for item in report["seed_results"]] == list(SEEDS)


def test_case_bootstrap_averages_seeds_inside_each_case_and_is_deterministic(experiment):
    report = analyze(experiment)
    effect = report["primary_depth_benefit"]["all"]
    # The two synthetic Cases have equal paired effects; all bootstrap draws
    # must produce the seed-mean effect, not resample six Case/seed entries.
    assert effect["ci95"] == pytest.approx([effect["estimate"], effect["estimate"]])
    assert effect["bootstrap"]["n_cases"] == 2
    assert effect["bootstrap"]["fitted_seed_effects_averaged_within_case"] is True
    assert effect["bootstrap"]["includes_training_seed_sampling_uncertainty"] is False
    assert report == analyze((list(reversed(experiment[0])), experiment[1]))


def test_fixed_baselines_pair_exact_endpoint_cases_and_no_private_output(experiment, tmp_path):
    report = analyze(experiment, baseline_dir=baseline(tmp_path / "baselines"))
    expected = np.mean(np.array([.2, .3, .4])**2)-.25
    assert report["baselines"]["constant"]["model_minus_baseline_brier"]["depth4_all"]["estimate"] == pytest.approx(expected)
    assert report["baselines"]["logistic_regression"]["tune_metrics"]["brier"] == pytest.approx(.04)
    serialized = json.dumps(report, allow_nan=False)
    for private in ("private_train", "private_tune", "private_local_filename", str(tmp_path)):
        assert private not in serialized
    assert "unknown_field" not in report["config"]
    assert report["clinical_validation"] is False
    assert report["decision_rule_automatically_applied"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_exactly_twelve_unique_directories_required(experiment, mutation):
    runs, protocol = experiment
    candidate = runs[:-1] if mutation == "missing" else runs[:-1]+runs[:1]
    with pytest.raises(ValueError, match="twelve distinct"):
        analyze((candidate, protocol))


@pytest.mark.parametrize("change", [lambda x: x["config"].update(seed=99),
                                    lambda x: x["config"].update(steps=3),
                                    lambda x: x["config"].update(supervision="none"),
                                    lambda x: x["config"].update(arm="SC"),
                                    lambda x: x["config"].update(epochs=49),
                                    lambda x: x["config"].update(augment=True),
                                    lambda x: x["config"].update(batch_size=True)])
def test_config_outside_locked_design_rejected(experiment, change):
    edit_json(experiment[0][-1]/"summary.json", change)
    with pytest.raises(ValueError, match="locked design|fixed SJ"):
        analyze(experiment)


def test_duplicate_depth_supervision_seed_combination_rejected(experiment):
    edit_json(experiment[0][-1]/"summary.json", lambda x: x["config"].update(supervision="all"))
    with pytest.raises(ValueError, match="supervised readouts|Exactly one"):
        analyze(experiment)


@pytest.mark.parametrize("key", ["encoder_sha256", "manifest_sha256", "training_source_sha256"])
def test_source_digest_must_match_locked_protocol(experiment, key):
    edit_json(experiment[0][-1]/"summary.json", lambda x: x.update({key: "e"*64}))
    with pytest.raises(ValueError, match="digests"):
        analyze(experiment)


def test_init_digest_must_match_protocol_and_seed_pairing(experiment):
    edit_json(experiment[0][-1]/"summary.json", lambda x: x.update(initialization_sha256="f"*64))
    with pytest.raises(ValueError, match="Initial model parameters"):
        analyze(experiment)


@pytest.mark.parametrize("change", [lambda x: x.update(status="training_in_progress"),
                                    lambda x: x.update(evaluation_split="test"),
                                    lambda x: x.update(model_selection="best_tune_epoch"),
                                    lambda x: x["timing"].update(updates=49),
                                    lambda x: x["readout_supervision"].update(directly_supervised_steps=[1,2,3,4])])
def test_completion_evaluation_or_supervision_provenance_rejected(experiment, change):
    edit_json(experiment[0][-1]/"summary.json", change)
    with pytest.raises(ValueError, match="completed|final epoch|updates|supervised readouts"):
        analyze(experiment)


@pytest.mark.parametrize("split", ["train", "test", "calibration", "", "cal"])
def test_development_records_require_explicit_tune_origin(experiment, split):
    edit_csv(experiment[0][-1]/"development_predictions.csv", lambda rows: rows[0].update(split=split))
    with pytest.raises(ValueError, match="train/tune split"):
        analyze(experiment)


@pytest.mark.parametrize("change,match", [
    (lambda rows: rows.append(copy.deepcopy(rows[0])), "duplicate"),
    (lambda rows: rows[0].update(probability="nan"), "invalid"),
    (lambda rows: rows[0].update(dice=""), "Dice"),
    (lambda rows: rows[0].update(step=5), "readout step"),
    (lambda rows: rows[0].update(seed=17), "seed"),
    (lambda rows: rows[0].update(arm="UJ"), "arm"),
])
def test_invalid_prediction_schema_and_values_rejected(experiment, change, match):
    edit_csv(experiment[0][-1]/"development_predictions.csv", change)
    with pytest.raises(ValueError, match=match):
        analyze(experiment)


@pytest.mark.parametrize("field,value,match", [("case_id", "private_different_case", "Case sets"),
                                              ("image_id", "private_different_view", "view sets"),
                                              ("label", 1, "labels conflict")])
def test_equal_counts_cannot_hide_changed_cases_views_or_labels(experiment, field, value, match):
    def change(rows):
        for row in rows:
            if row["case_id"] == "private_tune_case0":
                if field != "image_id" or row["image_id"] == "private_tune_image0_0":
                    row[field] = value
    edit_csv(experiment[0][-1]/"development_predictions.csv", change)
    with pytest.raises(ValueError, match=match):
        analyze(experiment)


def test_missing_readout_rejected_without_partial_intersection(experiment):
    edit_csv(experiment[0][0]/"development_predictions.csv", lambda rows: rows.__setitem__(slice(None),
             [row for row in rows if row["step"] != "2"]))
    with pytest.raises(ValueError, match="Case sets"):
        analyze(experiment)


@pytest.mark.parametrize("change", [lambda x: x["development_subset"]["4"].update(brier=.123),
                                    lambda x: x["development_subset"]["4"].update(n_cases=3),
                                    lambda x: x["development_history"][-1]["by_step"]["4"].update(auroc=0.),
                                    lambda x: x["development_history"].pop(),
                                    lambda x: x.update(n_training_images=3)])
def test_summary_and_fixed_final_history_must_match_traces(experiment, change):
    edit_json(experiment[0][-1]/"summary.json", change)
    with pytest.raises(ValueError, match="metrics|history|count"):
        analyze(experiment)


@pytest.mark.parametrize("change", [lambda x: x.update(status="not_locked"),
                                    lambda x: x.update(depths=[2,3]),
                                    lambda x: x.update(warm_starts=True),
                                    lambda x: x.update(training_updates_per_run=49),
                                    lambda x: x["tune"].update(fingerprint="e"*64),
                                    lambda x: x["initialization_sha256_by_seed"].pop("43"),
                                    lambda x: x["train"].update(images=3)])
def test_required_locked_protocol_rejects_inconsistent_design_or_cohort(experiment, change):
    edit_json(experiment[1], change)
    with pytest.raises(ValueError):
        analyze(experiment)


def test_baseline_with_equal_counts_but_different_cases_rejected(experiment, tmp_path):
    directory = baseline(tmp_path/"baseline")
    def change(rows):
        for row in rows:
            if row["case_id"] == "private_tune_case0":
                row["case_id"] = "private_other_case"
    edit_csv(directory/"baseline_case_predictions.csv", change)
    with pytest.raises(ValueError, match="Case sets"):
        analyze(experiment, baseline_dir=directory)


def test_cli_publishes_aggregate_only_and_refuses_overwrite(experiment, tmp_path, capsys):
    runs, protocol = experiment
    output = tmp_path/"aggregate.json"
    args = ["--run-dirs", *map(str,runs), "--protocol", str(protocol), "--output", str(output),
            "--expected-training-cases", "2", "--expected-development-cases", "2", "--draws", "50"]
    assert main(args) == 0
    before = output.read_bytes()
    assert json.loads(before)["status"] == "executed_cpu_depth_supervision_analysis"
    assert "private_tune" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(args)
    assert output.read_bytes() == before


def test_missing_protocol_is_rejected_before_output_creation(experiment, tmp_path):
    runs, _ = experiment
    output = tmp_path/"not_created.json"
    with pytest.raises(SystemExit):
        main(["--run-dirs", *map(str,runs), "--protocol", str(tmp_path/"missing.json"), "--output", str(output)])
    assert not output.exists()
