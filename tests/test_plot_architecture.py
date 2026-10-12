"""Aggregate architecture validation, effect signs and pathology denominators."""
import copy
from hashlib import sha256
import json

import numpy as np
import pytest

from loop_ultrasound.plot_architecture import (
    ARMS, DEPTHS, POLICIES, SEEDS, PARAMETERS, ENCODER_PARAMETERS,
    FIXED_CONFIG, RATES, SOURCE_KEYS, load_reports, _seed_effects, regime_name,
)


def _stats(values):
    return {"mean": float(np.mean(values)), "std_across_training_seeds": float(np.std(values, ddof=1)),
            "min": float(np.min(values)), "max": float(np.max(values))}


def _effect(values):
    return {**_stats(values), "estimate": float(np.mean(values)),
            "by_seed": [{"seed": seed, "value": float(value)} for seed, value in zip(SEEDS, values)]}


def make_fixture(tmp_path):
    reports = tmp_path/"reports"; reports.mkdir()
    protocol = {**{key: value for key, value in FIXED_CONFIG.items() if key != "supervision"},
                "status": "fixed_before_cpu_architecture_comparison",
                "phase": "exploratory_sharing_depth_comparison_on_observed_tune",
                "arms": list(ARMS), "depths": list(DEPTHS), "policies": list(POLICIES), "seeds": list(SEEDS),
                "device": "cpu", "clinical_validation": False, "patient_mapping": "unverified_Case_grouped_only",
                "calibration_and_test_predictions": False, "model_selection": "fixed_final_epoch_no_best_epoch_selection",
                "training_updates_per_run": 3200, "tail_updates": 640, "tail_learning_rates": dict(RATES),
                "encoder_frozen_parameters": ENCODER_PARAMETERS,
                "error_diagnostics": {"classification_threshold": .5, "high_confidence_false_negative_max": .05,
                                      "high_confidence_false_positive_min": .95, "bounds": "inclusive", "unit": "Case"},
                "train": {"cases": 256, "images": 445, "benign": 174, "malignant": 82, "fingerprint": "a"*64},
                "tune": {"cases": 159, "images": 281, "benign": 108, "malignant": 51, "fingerprint": "b"*64},
                "parameters_by_arm_depth": {key: {"trainable": value} for key, value in PARAMETERS.items()},
                "common_initialization_sha256_by_seed": {str(seed): f"{seed:064x}" for seed in SEEDS},
                "initialization_sha256_by_seed_depth": {f"seed{seed}_depth{depth}": f"{seed*depth:064x}" for seed in SEEDS for depth in DEPTHS}}
    for index, key in enumerate(SOURCE_KEYS): protocol[key] = f"{index+100:064x}"
    protocol["shared_launcher_source_sha256"] = protocol["helper_source_sha256"]
    path = tmp_path/"protocol.json"; path.write_text(json.dumps(protocol))
    runs, seed_records = {}, []
    for i, seed in enumerate(SEEDS):
        seed_record = {"seed": seed, "regimes": {arm: {} for arm in ARMS}}
        for arm in ARMS:
            for depth in DEPTHS:
                for policy in POLICIES:
                    deep, shared, low = depth == 4, arm == "SJ", policy == "low_tail"
                    brier = .20 + (0 if shared else .02) + i*.005
                    if deep: brier -= (.03 if low else .01)+(.005 if shared else 0)+(i*.001 if shared else -i*.001)
                    if low: brier -= .015
                    fn = 18+i+(0 if shared else 4)-(2 if deep else 0)-(3 if low else 0)
                    fp = 13-i+(0 if shared else 3)+(1 if deep else 0)-(1 if low else 0)
                    hc_fn, hc_fp = max(0, fn//2-(1 if low else 0)), max(0, fp//3-(1 if low else 0))
                    endpoint = {"n_cases": 159, "n_positive": 51, "n_negative": 108, "threshold": .5,
                                "fn": fn, "fp": fp, "tp": 51-fn, "tn": 108-fp,
                                "brier": brier, "auroc": .77+(.025 if shared else 0)+(.01 if deep else 0)+(.015 if low else 0)+i*.004,
                                "nll": .95+(.0 if shared else .15)-(.1 if deep else 0)-(.12 if low else 0)+i*.015,
                                "mean_case_dice": .7+(.01 if deep else 0)+i*.004}
                    source = {key: protocol[key] for key in ("encoder_sha256", "manifest_sha256", "training_source_sha256")}
                    source["launcher_source_sha256"] = protocol["launcher_source_sha256"] if arm == "UJ" else protocol["shared_launcher_source_sha256"]
                    init = protocol["initialization_sha256_by_seed_depth"][f"seed{seed}_depth{depth}"] if arm == "UJ" else protocol["common_initialization_sha256_by_seed"][str(seed)]
                    summary = {**source, "status": "executed_cpu_architecture_control" if arm == "UJ" else "executed_cpu_tail_lr_control",
                               "clinical_validation": False, "device": "cpu", "evaluation_split": "tune_development",
                               "patient_mapping": "unverified_Case_grouped_only", "model_selection": protocol["model_selection"],
                               "n_training_cases": 256, "n_training_images": 445, "initialization_sha256": init,
                               "config": {**FIXED_CONFIG, "arm": arm, "steps": depth, "seed": seed,
                                          "tail_policy": policy, "tail_learning_rate": RATES[policy]},
                               "cohort_selection": {"sampling": "proportional", "selection_seed": 20261004,
                                                    "train_fingerprint": "a"*64, "tune_fingerprint": "b"*64},
                               "timing": {"updates": 3200, "new_updates_in_branch": 640},
                               "readout_supervision": {"mode": "terminal", "directly_supervised_steps": [depth]},
                               "parameters": {"trainable": PARAMETERS[f"{arm}{depth}"], "encoder_frozen": ENCODER_PARAMETERS},
                               "development_subset": {str(step): copy.deepcopy(endpoint) for step in range(1, depth+1)},
                               "development_history": [{"epoch": epoch, "by_step": {str(step): copy.deepcopy(endpoint) for step in range(1, depth+1)}} for epoch in (5, 10, 15, 20, 25)]}
                    if arm == "UJ": summary.update(helper_source_sha256=protocol["helper_source_sha256"], common_initialization_sha256=protocol["common_initialization_sha256_by_seed"][str(seed)])
                    (reports/f"{arm}-depth{depth}-{policy}-seed{seed}.json").write_text(json.dumps(summary))
                    runs[arm, depth, policy, seed] = summary
                    name = regime_name(depth, policy)
                    pathology = {"malignant": {"n_cases": 51, "error_count": fn, "error_rate": fn/51,
                                                "high_confidence_error_count": hc_fn, "high_confidence_error_rate": hc_fn/51},
                                 "benign": {"n_cases": 108, "error_count": fp, "error_rate": fp/108,
                                            "high_confidence_error_count": hc_fp, "high_confidence_error_rate": hc_fp/108}}
                    seed_record["regimes"][arm][name] = {"endpoint": endpoint, "pathology": pathology,
                        "endpoint_diagnostics": {"high_confidence_false_negatives": hc_fn, "high_confidence_false_positives": hc_fp}}
        seed_records.append(seed_record)
    regimes = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for depth in DEPTHS:
            for policy in POLICIES:
                name = regime_name(depth, policy)
                regimes[arm][name] = {"arm": arm, "depth": depth, "tail_policy": policy,
                    "endpoint": {metric: _stats([runs[arm, depth, policy, seed]["development_subset"][str(depth)][metric] for seed in SEEDS]) for metric in ("brier", "auroc", "nll", "mean_case_dice")},
                    "pathology": {truth: {"n_cases": 51 if truth == "malignant" else 108,
                        **{key: _stats([row["regimes"][arm][name]["pathology"][truth][key] for row in seed_records]) for key in ("error_count", "error_rate", "high_confidence_error_count", "high_confidence_error_rate")}} for truth in ("malignant", "benign")}}
    sharing, gains, theta = _seed_effects(runs)
    analysis = {"status": "executed_cpu_architecture_analysis", "clinical_validation": False,
                "training_seeds": list(SEEDS), "arms": list(ARMS), "depths": list(DEPTHS), "policies": list(POLICIES),
                "n_training_cases": 256, "n_development_cases": 159, "evaluation_split": "tune_development",
                "sources": {**{key: protocol[key] for key in SOURCE_KEYS if key != "shared_launcher_source_sha256"}, "protocol_sha256": sha256(path.read_bytes()).hexdigest()},
                "cohort_fingerprints": {"train": "a"*64, "tune": "b"*64}, "regimes": regimes, "seed_results": seed_records,
                "sharing_benefit_by_depth_policy": {regime_name(depth, policy): {"Brier_benefit": _effect(sharing[depth, policy])} for depth in DEPTHS for policy in POLICIES},
                "depth_benefit_by_arm_policy": {arm: {policy: {"Brier_benefit": _effect(gains[arm, policy])} for policy in POLICIES} for arm in ARMS},
                "theta_by_policy": {policy: _effect(theta[policy]) for policy in POLICIES}}
    (reports/"architecture_analysis.json").write_text(json.dumps(analysis))
    return reports, path


def _edit(path, change):
    value = json.loads(path.read_text()); change(value); path.write_text(json.dumps(value))


def test_accepts_forty_summaries_and_analysis_with_unequal_parameter_counts(tmp_path):
    reports, protocol = make_fixture(tmp_path)
    runs, seed_rows = load_reports(reports, protocol_path=protocol)
    assert len(runs) == 40 and len(seed_rows) == 5
    assert runs["UJ", 4, "constant", 53]["parameters"]["trainable"] > runs["SJ", 4, "constant", 53]["parameters"]["trainable"]


def test_sharing_depth_and_theta_signs(tmp_path):
    reports, protocol = make_fixture(tmp_path)
    runs, _ = load_reports(reports, protocol_path=protocol)
    sharing, gains, theta = _seed_effects(runs)
    np.testing.assert_allclose(sharing[2, "constant"], [.02]*5)
    np.testing.assert_allclose(sharing[4, "constant"], [.025, .027, .029, .031, .033])
    np.testing.assert_allclose(gains["SJ", "constant"], [.015, .016, .017, .018, .019])
    np.testing.assert_allclose(gains["UJ", "constant"], [.01, .009, .008, .007, .006])
    np.testing.assert_allclose(theta["constant"], [.005, .007, .009, .011, .013])
    np.testing.assert_allclose(theta["constant"], sharing[4, "constant"]-sharing[2, "constant"])


@pytest.mark.parametrize("change", ["parameters", "encoder", "status", "updates", "seed", "runner", "source", "cohort", "threshold", "final_metric"])
def test_rejects_changed_summary_provenance_or_fit(tmp_path, change):
    reports, protocol = make_fixture(tmp_path)
    target = reports/"UJ-depth4-low_tail-seed53.json"
    def mutate(value):
        if change == "parameters": value["parameters"]["trainable"] = 714530
        elif change == "encoder": value["parameters"]["encoder_frozen"] = 0
        elif change == "status": value["status"] = "executed_cpu_tail_lr_control"
        elif change == "updates": value["timing"]["updates"] = 2560
        elif change == "seed": value["config"]["seed"] = 17
        elif change == "runner": value["launcher_source_sha256"] = "0"*64
        elif change == "source": value["training_source_sha256"] = "0"*64
        elif change == "cohort": value["cohort_selection"]["tune_fingerprint"] = "0"*64
        elif change == "threshold": value["development_subset"]["4"]["threshold"] = .3
        elif change == "final_metric": value["development_subset"]["4"]["brier"] = .001
    _edit(target, mutate)
    with pytest.raises(ValueError): load_reports(reports, protocol_path=protocol)


@pytest.mark.parametrize("change", ["false_negative", "highconf_bounds", "highconf_diagnostic", "wrong_rate_denominator", "wrong_class_count", "aggregate_error_mean", "endpoint_mean", "theta_sign", "raw_case_ids"])
def test_rejects_inconsistent_error_aggregates_or_effects(tmp_path, change):
    reports, protocol = make_fixture(tmp_path)
    target = reports/"architecture_analysis.json"
    def mutate(value):
        pathology = value["seed_results"][0]["regimes"]["SJ"]["depth2_constant"]["pathology"]["malignant"]
        if change == "false_negative": pathology["error_count"] += 1
        elif change == "highconf_bounds": pathology["high_confidence_error_count"] = pathology["error_count"]+1
        elif change == "highconf_diagnostic": value["seed_results"][0]["regimes"]["SJ"]["depth2_constant"]["endpoint_diagnostics"]["high_confidence_false_negatives"] += 1
        elif change == "wrong_rate_denominator": pathology["error_rate"] = pathology["error_count"]/108
        elif change == "wrong_class_count": pathology["n_cases"] = 159
        elif change == "aggregate_error_mean": value["regimes"]["SJ"]["depth2_constant"]["pathology"]["malignant"]["error_count"]["mean"] += 1
        elif change == "endpoint_mean": value["regimes"]["SJ"]["depth2_constant"]["endpoint"]["brier"]["mean"] = .001
        elif change == "theta_sign": value["theta_by_policy"]["constant"]["by_seed"][0]["value"] *= -1
        elif change == "raw_case_ids": value["case_id"] = "private_identifier"
    _edit(target, mutate)
    with pytest.raises(ValueError): load_reports(reports, protocol_path=protocol)


def test_incomplete_grid_and_old_seed_files_are_handled_explicitly(tmp_path):
    reports, protocol = make_fixture(tmp_path)
    (reports/"SJ-depth2-constant-seed17.json").write_text("ignored old seed, deliberately invalid")
    assert len(load_reports(reports, protocol_path=protocol)[0]) == 40
    (reports/"UJ-depth4-low_tail-seed131.json").unlink()
    with pytest.raises(ValueError, match="Missing aggregate"): load_reports(reports, protocol_path=protocol)


def test_protocol_binding_change_is_rejected(tmp_path):
    reports, protocol = make_fixture(tmp_path)
    _edit(protocol, lambda value: value.update(launcher_source_sha256="0"*64))
    with pytest.raises(ValueError): load_reports(reports, protocol_path=protocol)
