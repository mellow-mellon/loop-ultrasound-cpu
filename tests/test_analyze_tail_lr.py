"""Cloned-state provenance, exact replay and paired endpoint effects."""
import copy
import csv
import hashlib
import json

import numpy as np
import pytest

from loop_ultrasound.analyze_tail_lr import FIXED_CONFIG, RATES, SEEDS, analyze_tail_lr, main
from loop_ultrasound.analyze_expanded import _fingerprint
from loop_ultrasound.analyze_depth_supervision import _metrics, _trace


FIELDS = ("case_id", "image_id", "label", "probability", "step", "arm", "seed", "dice", "split")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)


def _json_edit(path, change):
    value = json.loads(path.read_text()); change(value); path.write_text(json.dumps(value))


def _csv_edit(path, change):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    change(rows); _write_csv(path, rows)


def _cohort(split):
    if split == "train":
        return {f"private_train_case{i}": {"label": i//2, "image_ids": [f"private_train_view{i}"]}
                for i in range(4)}
    return {f"private_tune_case{i}": {"label": i, "image_ids": [f"private_tune_view{i}_{j}" for j in range(2 if i == 0 else 1)]}
            for i in range(2)}


def _rows(seed, depth, split, error):
    rows = []
    for step in range(1, depth+1):
        current_error = error if step == depth else (.02 if depth == 4 and step == 2 else .45)
        for case, value in _cohort(split).items():
            for view, image in enumerate(value["image_ids"]):
                offset = (-.01, .01)[view] if len(value["image_ids"]) == 2 else 0.
                rows.append({"case_id": case, "image_id": image, "label": value["label"],
                             "probability": (current_error if value["label"] == 0 else 1-current_error)+offset,
                             "step": step, "arm": "SJ", "seed": seed, "dice": .6+.02*step, "split": split})
    return rows


def _write_predictions(directory, seed, depth, error):
    for split, filename in (("train", "training_predictions.csv"), ("tune", "development_predictions.csv")):
        _write_csv(directory/filename, _rows(seed, depth, split, error))
    trace = _trace(directory/"development_predictions.csv", split="tune", seed=seed, depth=depth, require_dice=True)
    return {str(step): _metrics(trace[step]) for step in range(1, depth+1)}


@pytest.fixture
def experiment(tmp_path):
    root, references = tmp_path/"tails", tmp_path/"references"
    root.mkdir(); references.mkdir()
    source = {"encoder_sha256": "a"*64, "manifest_sha256": "b"*64, "training_source_sha256": "c"*64}
    init = {str(seed): f"{seed:064x}" for seed in SEEDS}
    cohorts = {}
    for split in ("train", "tune"):
        values = _cohort(split)
        cohorts[split] = {"cases": len(values), "images": sum(len(row["image_ids"]) for row in values.values()),
                          "benign": sum(row["label"] == 0 for row in values.values()),
                          "malignant": sum(row["label"] for row in values.values()), "fingerprint": _fingerprint(values)}
    reference_protocol = {**{key: value for key, value in FIXED_CONFIG.items() if key not in {"arm", "supervision"}},
                          **source, **cohorts, "status": "fixed_before_case_expansion_seed_replication_training",
                          "seeds": list(SEEDS), "depths": [2, 4], "supervisions": ["terminal"], "arms": ["SJ"],
                          "clinical_validation": False, "device": "cpu", "calibration_and_test_predictions": False,
                          "patient_mapping": "unverified_Case_grouped_only", "model_selection": "fixed_final_epoch_no_best_epoch_selection",
                          "initialization_sha256_by_seed": init, "training_updates_per_run": 50}
    reference_path = tmp_path/"reference_protocol.json"
    reference_path.write_text(json.dumps(reference_protocol))
    protocol = {**reference_protocol, "status": "fixed_before_cpu_tail_lr_control",
                "phase": "exploratory_tail_optimizer_intervention_on_observed_tune", "policies": ["constant", "low_tail"],
                "prefix_epochs": 20, "tail_epochs": 5, "reference_epochs": 25, "tail_learning_rates": dict(RATES),
                "prefix_updates": 40, "tail_updates": 10, "total_actual_new_updates": 600,
                "full_state_fork": True, "warm_starts": False, "shared_prefixes": 10,
                "effective_endpoint_trajectories": 20, "updates_per_epoch": 2,
                "launcher_source_sha256": "d"*64, "reference_protocol_sha256": _sha(reference_path),
                "unknown_private_field": "private_local_filename"}
    protocol_path = tmp_path/"protocol.json"; protocol_path.write_text(json.dumps(protocol))
    runs, reference_runs = [], []
    bindings = {}
    for i, seed in enumerate(SEEDS):
        for depth in (2, 4):
            prefix = root/f"prefix-depth{depth}-seed{seed}"; prefix.mkdir()
            observed_prefix = _write_predictions(prefix, seed, depth, .45 if depth == 2 else .4)
            checkpoint = prefix/"checkpoint.pt"; checkpoint.write_bytes(f"synthetic full state {seed} {depth}".encode())
            state = hashlib.sha256(f"full-state-{seed}-{depth}".encode()).hexdigest()
            prefix_sampling = hashlib.sha256(f"prefix-views-{seed}".encode()).hexdigest()
            tail_sampling = hashlib.sha256(f"tail-views-{seed}".encode()).hexdigest()
            objective = [{"epoch": epoch, "mean_training_objective": 1./(epoch+1)} for epoch in range(1, 26)]
            history_prefix = [{"epoch": epoch, "by_step": copy.deepcopy(observed_prefix)} for epoch in range(5, 21, 5)]
            common = {**source, "clinical_validation": False, "device": "cpu", "patient_mapping": "unverified_Case_grouped_only",
                      "evaluation_split": "tune_development", "model_selection": "fixed_final_epoch_no_best_epoch_selection",
                      "initialization_sha256": init[str(seed)], "n_training_cases": 4, "n_training_images": 4,
                      "cohort_selection": {"selection_seed": 20261004, "sampling": "proportional",
                                           "train_fingerprint": cohorts["train"]["fingerprint"], "tune_fingerprint": cohorts["tune"]["fingerprint"]},
                      "readout_supervision": {"mode": "terminal", "directly_supervised_steps": [depth]},
                      "unknown_private_field": "private_local_filename"}
            prefix_summary = {**common, "status": "executed_cpu_tail_lr_prefix", "launcher_source_sha256": "d"*64,
                              "config": {**FIXED_CONFIG, "seed": seed, "steps": depth, "prefix_updates": 40},
                              "checkpoint_sha256": _sha(checkpoint), "state_sha256": state,
                              "training_predictions_sha256": _sha(prefix/"training_predictions.csv"),
                              "development_predictions_sha256": _sha(prefix/"development_predictions.csv"),
                              "timing": {"updates": 40}, "sampling_trace_sha256": prefix_sampling,
                              "development_subset": observed_prefix, "development_history": history_prefix}
            (prefix/"prefix_summary.json").write_text(json.dumps(prefix_summary))
            (prefix/"history.json").write_text(json.dumps(objective[:20]))
            constant_error = .4 if depth == 2 else (.2, .25, .3, .35, .4)[i]
            low_error = .35 if depth == 2 else (.15, .2, .28, .36, .45)[i]
            refdir = references/f"SJ-depth{depth}-terminal-seed{seed}"; refdir.mkdir()
            observed_ref = _write_predictions(refdir, seed, depth, constant_error)
            reference_summary = {**common, "status": "executed_cpu_engineering_pilot",
                                 "config": {**FIXED_CONFIG, "seed": seed, "steps": depth}, "timing": {"updates": 50},
                                 "development_subset": observed_ref,
                                 "development_history": history_prefix+[{"epoch": 25, "by_step": observed_ref}]}
            (refdir/"summary.json").write_text(json.dumps(reference_summary))
            (refdir/"history.json").write_text(json.dumps(objective)); reference_runs.append(refdir)
            (refdir/"checkpoint.pt").write_bytes(f"original fitted model {seed} {depth}".encode())
            bindings[f"seed{seed}_depth{depth}"] = {field: _sha(refdir/filename) for field, filename in (
                ("checkpoint_sha256", "checkpoint.pt"), ("summary_sha256", "summary.json"),
                ("training_predictions_sha256", "training_predictions.csv"),
                ("development_predictions_sha256", "development_predictions.csv"))}
            constant_dir = None
            for policy, error in (("constant", constant_error), ("low_tail", low_error)):
                directory = root/f"SJ-depth{depth}-{policy}-seed{seed}"; directory.mkdir()
                observed = _write_predictions(directory, seed, depth, error)
                branch = {**common, "status": "executed_cpu_tail_lr_control", "launcher_source_sha256": "d"*64,
                          "config": {**FIXED_CONFIG, "seed": seed, "steps": depth, "tail_policy": policy,
                                     "prefix_updates": 40, "tail_learning_rate": RATES[policy]},
                          "prefix_reused": True, "timing": {"updates": 50, "new_updates_in_branch": 10},
                          "branch_provenance": {"prefix_checkpoint_sha256": _sha(checkpoint), "prefix_state_sha256": state,
                                                "resumed_state_sha256": state, "prefix_updates": 40, "restored_sampler_epoch": 20,
                                                "restored_loader_generator": True, "restored_rng": True},
                          "sampling_trace_sha256": {"prefix": prefix_sampling, "tail": tail_sampling},
                          "development_subset": observed, "development_history": history_prefix+[{"epoch": 25, "by_step": observed}]}
                if policy == "constant":
                    constant_dir = directory
                    branch["constant_replay_verification"] = {"status": "exact_replay_verified", "reference_summary_sha256": _sha(refdir/"summary.json"),
                        "model_state_tensors_equal": True, "prediction_records_equal": True,
                        "training_objective_history_equal": True, "development_history_equal": True}
                else:
                    branch["constant_replay_verification"] = {"status": "paired_constant_replay_verified",
                        "reference_summary_sha256": _sha(refdir/"summary.json"), "constant_summary_sha256": _sha(constant_dir/"summary.json")}
                (directory/"summary.json").write_text(json.dumps(branch))
                (directory/"history.json").write_text(json.dumps(objective)); runs.append(directory)
    protocol["reference_run_artifact_bindings"] = bindings
    protocol_path.write_text(json.dumps(protocol))
    return {"runs": runs, "protocol": protocol_path, "reference_runs": reference_runs,
            "reference_protocol": reference_path, "root": root}


def _analyze(e, **kwargs):
    return analyze_tail_lr(e["runs"], protocol_path=e["protocol"], reference_run_dirs=e["reference_runs"],
                           reference_protocol_path=e["reference_protocol"], expected_training_cases=4,
                           expected_development_cases=2, draws=100, **kwargs)


def test_known_paired_H_and_absolute_depth_gains(experiment):
    report = _analyze(experiment)
    constant = .4**2-np.array([.2, .25, .3, .35, .4])**2
    low = .35**2-np.array([.15, .2, .28, .36, .45])**2
    assert report["primary_H"]["estimate"] == pytest.approx((low-constant).mean())
    assert report["depth_benefit_Brier"]["constant"]["mean"] == pytest.approx(constant.mean())
    assert report["depth_benefit_Brier"]["low_tail"]["mean"] == pytest.approx(low.mean())
    assert [row["H"] for row in report["seed_results"]] == pytest.approx(low-constant)
    assert report["primary_H"]["bootstrap"]["n_cases"] == 2
    assert report["primary_H"]["bootstrap"]["n_fitted_seeds"] == 5
    assert report["primary_H"]["bootstrap"]["includes_training_seed_sampling_uncertainty"] is False


def test_means_are_fit_metrics_not_ensemble_and_work_is_not_double_counted(experiment):
    report = _analyze(experiment)
    errors = np.array([.15, .2, .28, .36, .45])
    assert report["regimes"]["depth4_low_tail"]["endpoint"]["brier"]["mean"] == pytest.approx(np.mean(errors**2))
    assert np.mean(errors**2) != pytest.approx(errors.mean()**2)
    assert report["compute_budget"]["total_actual_new_updates"] == 600
    assert report["compute_budget"]["total_actual_new_updates"] != 20*50
    assert report["alignment_checks"]["constant_prediction_csv_bytes_replay_reference"] is True


def test_same_depth_policy_effect_and_within_tail_change_use_matched_prefix(experiment):
    report = _analyze(experiment)
    assert report["policy_benefit_by_depth"]["2"]["Brier_benefit"]["mean"] == pytest.approx(.4**2-.35**2)
    assert report["within_tail_changes"]["depth2_low_tail"]["Brier_benefit"]["mean"] == pytest.approx(.45**2-.35**2)
    assert report["learning_curves"][-1]["updates"] == 50
    assert report["learning_curves"][-2]["updates"] == 40


def test_runner_embedded_prefix_objective_histories_are_supported(experiment):
    for prefix in experiment["root"].glob("prefix-*"):
        history = json.loads((prefix/"history.json").read_text())
        _json_edit(prefix/"prefix_summary.json", lambda value: value.update(history=history))
        (prefix/"history.json").unlink()
    assert _analyze(experiment)["alignment_checks"]["full_prefix_state_restoration_certified"] is True


def test_enriched_new_diagnostics_replay_bare_old_objective_history(experiment):
    """New histories add diagnostics absent from the original bare histories."""
    def enrich(history, policy="constant"):
        return [{**row, "mean_training_classification": row["mean_training_objective"]*.6,
                 "mean_training_segmentation": row["mean_training_objective"]*.4,
                 "core_preclip_norm_mean": .7, "core_clip_fraction": .1,
                 "mask_preclip_norm_mean": .8, "mask_clip_fraction": .2,
                 "learning_rate": RATES[policy] if row["epoch"] > 20 else .0003}
                for row in history]
    for prefix in experiment["root"].glob("prefix-*"):
        history = enrich(json.loads((prefix/"history.json").read_text()))
        _json_edit(prefix/"prefix_summary.json", lambda summary: summary.update(history=history))
        (prefix/"history.json").unlink()
    for branch in experiment["runs"]:
        policy = json.loads((branch/"summary.json").read_text())["config"]["tail_policy"]
        history = enrich(json.loads((branch/"history.json").read_text()), policy)
        (branch/"history.json").write_text(json.dumps(history))
    assert set(json.loads((experiment["reference_runs"][0]/"history.json").read_text())[0]) == {"epoch", "mean_training_objective"}
    report = _analyze(experiment)
    assert report["alignment_checks"]["constant_objective_and_development_histories_replay_reference"] is True


def test_reference_checkpoint_bytes_are_pinned(experiment):
    (experiment["reference_runs"][0]/"checkpoint.pt").write_bytes(b"unbound replacement")
    with pytest.raises(ValueError, match="locked protocol bindings"):
        _analyze(experiment)


def test_same_policy_sampling_cannot_differ_between_depths(experiment):
    constant = experiment["root"]/"SJ-depth4-constant-seed53"
    low = experiment["root"]/"SJ-depth4-low_tail-seed53"
    for directory in (constant, low):
        _json_edit(directory/"summary.json", lambda value: value["sampling_trace_sha256"].update(tail="0"*64))
    _json_edit(low/"summary.json", lambda value: value["constant_replay_verification"].update(constant_summary_sha256=_sha(constant/"summary.json")))
    with pytest.raises(ValueError, match="Paired depths"):
        _analyze(experiment)


def test_output_contains_only_aggregates_and_known_configuration(experiment, tmp_path):
    report = _analyze(experiment)
    text = json.dumps(report, allow_nan=False)
    for value in ("private_train", "private_tune", "private_local_filename", str(tmp_path)):
        assert value not in text
    assert "unknown_private_field" not in report["config"]
    assert report["clinical_validation"] is False
    assert report["decision_rule_automatically_applied"] is False


@pytest.mark.parametrize("change", ["rate", "prefix_updates", "actual_branch_updates", "state", "rng", "sampler_epoch", "tail_sampling", "source", "replay_status", "direct_supervision"])
def test_rejects_wrong_intervention_or_unrestored_state(experiment, change):
    directory = experiment["root"]/"SJ-depth2-low_tail-seed53"
    def alter(s):
        if change == "rate": s["config"]["tail_learning_rate"] = .0001
        elif change == "prefix_updates": s["config"]["prefix_updates"] = 39
        elif change == "actual_branch_updates": s["timing"]["new_updates_in_branch"] = 50
        elif change == "state": s["branch_provenance"]["resumed_state_sha256"] = "0"*64
        elif change == "rng": s["branch_provenance"]["restored_rng"] = False
        elif change == "sampler_epoch": s["branch_provenance"]["restored_sampler_epoch"] = 19
        elif change == "tail_sampling": s["sampling_trace_sha256"]["tail"] = "0"*64
        elif change == "source": s["launcher_source_sha256"] = "0"*64
        elif change == "replay_status": s["constant_replay_verification"]["status"] = "unchecked"
        elif change == "direct_supervision": s["readout_supervision"]["directly_supervised_steps"] = [1, 2]
    _json_edit(directory/"summary.json", alter)
    with pytest.raises(ValueError): _analyze(experiment)


@pytest.mark.parametrize("change", ["constant_csv", "constant_history", "prefix_checkpoint", "prefix_csv", "low_binding", "prefix_history"])
def test_rejects_replay_or_prefix_artifact_mismatch(experiment, change):
    constant = experiment["root"]/"SJ-depth2-constant-seed53"
    low = experiment["root"]/"SJ-depth2-low_tail-seed53"
    prefix = experiment["root"]/"prefix-depth2-seed53"
    if change == "constant_csv":
        # A byte-only change leaves the mathematical probabilities unchanged;
        # replay certification must still fail its exact artifact contract.
        path = constant/"development_predictions.csv"; path.write_bytes(path.read_bytes()+b"\n")
    elif change == "constant_history":
        _json_edit(constant/"history.json", lambda h: h[-1].update(mean_training_objective=.123))
    elif change == "prefix_checkpoint": (prefix/"checkpoint.pt").write_bytes(b"changed state")
    elif change == "prefix_csv":
        _csv_edit(prefix/"development_predictions.csv", lambda rows: rows[0].update(probability=.3))
    elif change == "low_binding":
        _json_edit(low/"summary.json", lambda s: s["constant_replay_verification"].update(constant_summary_sha256="0"*64))
    elif change == "prefix_history":
        _json_edit(prefix/"history.json", lambda h: h[0].update(mean_training_objective=.123))
    with pytest.raises(ValueError): _analyze(experiment)


@pytest.mark.parametrize("change", ["missing_case", "duplicate_view", "split", "label", "summary_metric"])
def test_rejects_raw_case_or_view_misalignment(experiment, change):
    directory = experiment["root"]/"SJ-depth2-low_tail-seed53"
    if change == "missing_case":
        _csv_edit(directory/"development_predictions.csv", lambda rows: rows.__setitem__(slice(None), [r for r in rows if r["case_id"] != "private_tune_case1"]))
    elif change == "duplicate_view": _csv_edit(directory/"development_predictions.csv", lambda rows: rows.append(dict(rows[0])))
    elif change == "split": _csv_edit(directory/"development_predictions.csv", lambda rows: rows[0].update(split="calibration"))
    elif change == "label": _csv_edit(directory/"development_predictions.csv", lambda rows: rows[0].update(label=1))
    elif change == "summary_metric": _json_edit(directory/"summary.json", lambda s: s["development_subset"]["2"].update(brier=.1))
    with pytest.raises(ValueError): _analyze(experiment)


def test_rejects_partial_branch_grid_or_reference_protocol_drift(experiment):
    with pytest.raises(ValueError):
        analyze_tail_lr(experiment["runs"][:-1], protocol_path=experiment["protocol"],
                        reference_run_dirs=experiment["reference_runs"], reference_protocol_path=experiment["reference_protocol"],
                        expected_training_cases=4, expected_development_cases=2)
    _json_edit(experiment["reference_protocol"], lambda p: p.update(learning_rate=.001))
    with pytest.raises(ValueError, match="protocol bytes"): _analyze(experiment)


def test_cli_does_not_overwrite_analysis(experiment, tmp_path):
    output = tmp_path/"analysis.json"; output.write_text("preserve")
    with pytest.raises(SystemExit):
        main(["--run-dirs", *map(str, experiment["runs"]), "--protocol", str(experiment["protocol"]),
              "--reference-run-dirs", *map(str, experiment["reference_runs"]),
              "--reference-protocol", str(experiment["reference_protocol"]), "--output", str(output)])
    assert output.read_text() == "preserve"
