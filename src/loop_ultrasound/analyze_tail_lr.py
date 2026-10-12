"""Paired-Case analysis of a cloned-prefix CPU tail-learning-rate control.

Only aggregate results are exported. The low-rate and constant-rate branches
must share the complete prefix state, sampler trajectory, Cases and views.
The constant branch must replay the existing fit exactly. Case bootstrap
intervals condition on the five fitted seeds and already observed tune cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .analyze import _hash, _integer, _match_cases
from .analyze_case_expansion import _new_runs
from .analyze_depth_supervision import (FINAL_SELECTION, SOURCE_KEYS, _declared_metrics,
                                       _json, _metrics, _trace)
from .analyze_expanded import AGGREGATE_METRICS
from .analyze_seed_replication import (_cohort_check, _effect, _high_confidence_errors,
                                      _nll_per_case, _stats, _transitions)


SEEDS = (53, 71, 89, 107, 131)
DEPTHS = (2, 4)
POLICIES = ("constant", "low_tail")
RATES = {"constant": .0003, "low_tail": .00003}
FIXED_CONFIG = {"arm": "SJ", "epochs": 25, "batch_size": 2,
                "learning_rate": .0003, "segmentation_weight": 1.,
                "cached_frozen_features": True, "augment": False,
                "selection_seed": 20261004, "sampling": "proportional", "eval_every": 5,
                "supervision": "terminal"}


def regime_name(depth, policy):
    return f"depth{depth}_{policy}"


def _file_hash(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        raise ValueError("Required bound local artifact is unreadable.") from None


def _exact_config(config, expected, message):
    if (not isinstance(config, dict)
            or any(config.get(key) != value or type(config.get(key)) is not type(value)
                   for key, value in expected.items())):
        raise ValueError(message)


def _cohort_declaration(value, count):
    if not isinstance(value, dict) or _integer(value.get("cases"), "locked Cases", 2) != count:
        raise ValueError("Protocol cohort counts disagree with the expected fixed experiment.")
    _integer(value.get("images"), "locked images", count)
    benign, malignant = (_integer(value.get(key), "locked pathology count") for key in ("benign", "malignant"))
    if benign+malignant != count:
        raise ValueError("Protocol pathology counts are inconsistent.")
    _hash(value.get("fingerprint"), "locked cohort fingerprint")


def _protocol(path, training_cases, development_cases):
    p = _json(path, "A readable locked tail-LR protocol is required.")
    if (p.get("status") != "fixed_before_cpu_tail_lr_control"
            or p.get("phase") != "exploratory_tail_optimizer_intervention_on_observed_tune"
            or p.get("seeds") != list(SEEDS) or p.get("depths") != list(DEPTHS)
            or p.get("policies") != list(POLICIES) or p.get("arms") != ["SJ"]
            or p.get("clinical_validation") is not False or p.get("device") != "cpu"
            or p.get("patient_mapping") != "unverified_Case_grouped_only"
            or p.get("model_selection") != FINAL_SELECTION
            or p.get("calibration_and_test_predictions") is not False
            or p.get("full_state_fork") is not True or p.get("warm_starts") is not False
            or p.get("supervisions") != ["terminal"] or p.get("shared_prefixes") != 10
            or p.get("effective_endpoint_trajectories") != 20
            or p.get("prefix_epochs") != 20 or p.get("tail_epochs") != 5
            or p.get("reference_epochs") != 25 or p.get("tail_learning_rates") != RATES):
        raise ValueError("Protocol must lock the five-seed, terminal-only, cloned-prefix CPU LR control.")
    _exact_config(p, {key: value for key, value in FIXED_CONFIG.items() if key not in {"arm", "supervision"}},
                  "Protocol differs from the fixed CPU training recipe.")
    updates_per_epoch = math.ceil(training_cases/FIXED_CONFIG["batch_size"])
    if p.get("updates_per_epoch") != updates_per_epoch:
        raise ValueError("Declared optimizer updates per epoch disagree with the locked cohort and batch.")
    budgets = {"prefix_updates": 20*updates_per_epoch, "tail_updates": 5*updates_per_epoch,
               "training_updates_per_run": 25*updates_per_epoch,
               "total_actual_new_updates": len(SEEDS)*len(DEPTHS)*30*updates_per_epoch}
    for key, expected in budgets.items():
        if _integer(p.get(key), "locked update budget", 1) != expected:
            raise ValueError("Prefix, branch and actual-new-work update budgets are inconsistent.")
    for key in SOURCE_KEYS + ("launcher_source_sha256", "reference_protocol_sha256"):
        _hash(p.get(key), "locked source/artifact digest")
    init = p.get("initialization_sha256_by_seed")
    if not isinstance(init, dict) or set(init) != {str(seed) for seed in SEEDS}:
        raise ValueError("Protocol must lock the initial parameters of all five paired seeds.")
    for value in init.values():
        _hash(value, "initialization digest")
    for split, count in (("train", training_cases), ("tune", development_cases)):
        _cohort_declaration(p.get(split), count)
    bindings = p.get("reference_run_artifact_bindings")
    expected_keys = {f"seed{seed}_depth{depth}" for seed in SEEDS for depth in DEPTHS}
    fields = {"checkpoint_sha256", "summary_sha256", "training_predictions_sha256", "development_predictions_sha256"}
    if not isinstance(bindings, dict) or set(bindings) != expected_keys:
        raise ValueError("Protocol must pin all ten reference replay artifact groups.")
    for item in bindings.values():
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError("Reference replay bindings must lock checkpoints, summaries and both traces.")
        for value in item.values():
            _hash(value, "reference artifact binding")
    return p


def _source_summary(summary, p, *, status):
    if (summary.get("status") != status or summary.get("clinical_validation") is not False
            or summary.get("device") != "cpu" or summary.get("patient_mapping") != "unverified_Case_grouped_only"
            or summary.get("evaluation_split") != "tune_development"
            or summary.get("model_selection") != FINAL_SELECTION):
        raise ValueError("Only completed CPU, fixed-final, tune-only control summaries are accepted.")
    config = summary.get("config")
    _exact_config(config, FIXED_CONFIG, "Control config differs from the locked terminal-only recipe.")
    seed, depth = _integer(config.get("seed"), "seed"), _integer(config.get("steps"), "depth", 1)
    if seed not in SEEDS or depth not in DEPTHS:
        raise ValueError("Control fit is outside the five-seed/two-depth design.")
    for key in SOURCE_KEYS + ("launcher_source_sha256",):
        if _hash(summary.get(key), key) != p[key]:
            raise ValueError("Control source digests disagree with the locked protocol.")
    if _hash(summary.get("initialization_sha256"), "initialization") != p["initialization_sha256_by_seed"][str(seed)]:
        raise ValueError("Control initial parameters disagree with the paired locked seed.")
    for key, expected in (("n_training_cases", p["train"]["cases"]), ("n_training_images", p["train"]["images"])):
        if _integer(summary.get(key), key, 2) != expected:
            raise ValueError("Control training counts disagree with the locked cohort.")
    selection = {"selection_seed": p["selection_seed"], "sampling": p["sampling"],
                 "train_fingerprint": p["train"]["fingerprint"], "tune_fingerprint": p["tune"]["fingerprint"]}
    if summary.get("cohort_selection") != selection:
        raise ValueError("Declared control cohort selection differs from the locked protocol.")
    readout = summary.get("readout_supervision")
    if (not isinstance(readout, dict) or readout.get("mode") != "terminal"
            or readout.get("directly_supervised_steps") != [depth]):
        raise ValueError("Only the final trained readout may receive direct supervision.")
    return seed, depth


def _objective_values(history, epochs):
    if (not isinstance(history, list) or len(history) != epochs
            or not all(isinstance(row, dict) for row in history)
            or [row.get("epoch") for row in history] != list(range(1, epochs+1))):
        raise ValueError("Objective history must record every prespecified epoch exactly once.")
    for row in history:
        value = row.get("mean_training_objective")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Objective histories require finite nonnegative values.")
    return history


def _objective_history(path, epochs):
    try:
        history = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        raise ValueError("A readable full objective history is required.") from None
    return _objective_values(history, epochs)


def _reference_objective_projection(history):
    """Compare the fields recorded by both the old and enriched new trainer."""
    return [{"epoch": row["epoch"], "mean_training_objective": row["mean_training_objective"]}
            for row in history]


def _history(summary, metrics, *, final_epoch):
    history = summary.get("development_history")
    if (not isinstance(history, list) or not all(isinstance(row, dict) for row in history)
            or [row.get("epoch") for row in history] != list(range(5, final_epoch+1, 5))):
        raise ValueError("Development histories must record the complete fixed evaluation schedule.")
    for entry in history:
        by_step = entry.get("by_step")
        if not isinstance(by_step, dict) or set(by_step) != set(metrics):
            raise ValueError("Every development snapshot must record all configured readouts.")
        for step, observed in metrics.items():
            candidate = by_step[step]
            if not isinstance(candidate, dict):
                raise ValueError("Development readout snapshots must be metric mappings.")
            for key in ("n_cases", "n_positive", "n_negative"):
                if _integer(candidate.get(key), "development snapshot count") != observed[key]:
                    raise ValueError("Recorded development cohort counts change across checkpoints.")
            for metric in AGGREGATE_METRICS:
                value = candidate.get(metric)
                if value is None and observed[metric] is None:
                    continue
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value < 0 or (metric != "nll" and value > 1)):
                    raise ValueError("Development histories contain invalid/nonfinite endpoint metrics.")
    _declared_metrics(summary.get("development_subset"), metrics, steps=range(1, summary["config"]["steps"]+1))
    _declared_metrics(history[-1].get("by_step"), metrics, steps=range(1, summary["config"]["steps"]+1))
    return history


def _reference(run_dirs, protocol_path, p):
    if _file_hash(protocol_path) != p["reference_protocol_sha256"]:
        raise ValueError("Reference protocol bytes disagree with the locked replay artifact.")
    reference = _json(protocol_path, "A readable bound reference protocol is required.")
    if (reference.get("status") != "fixed_before_case_expansion_seed_replication_training"
            or reference.get("seeds") != list(SEEDS) or reference.get("depths") != list(DEPTHS)
            or reference.get("supervisions") != ["terminal"] or reference.get("arms") != ["SJ"]
            or reference.get("clinical_validation") is not False or reference.get("device") != "cpu"
            or reference.get("patient_mapping") != "unverified_Case_grouped_only"
            or reference.get("model_selection") != FINAL_SELECTION
            or reference.get("calibration_and_test_predictions") is not False):
        raise ValueError("Replay reference must be the completed, fixed-final 256-Case five-seed experiment.")
    _exact_config(reference, {key: value for key, value in FIXED_CONFIG.items() if key not in {"arm", "supervision"}},
                  "Replay reference recipe differs from the common-prefix recipe.")
    for key in SOURCE_KEYS + ("train", "tune", "initialization_sha256_by_seed", "training_updates_per_run"):
        if reference.get(key) != p[key]:
            raise ValueError("Replay reference source, cohorts, initialization or budget differ from the control.")
    loaded = _new_runs(run_dirs, reference)
    directories = {}
    for raw in run_dirs:
        directory = Path(raw)
        summary = _json(directory/"summary.json", "Readable replay reference summaries are required.")
        key = (summary["config"]["seed"], summary["config"]["steps"])
        directories[key] = directory
        binding = p["reference_run_artifact_bindings"][f"seed{key[0]}_depth{key[1]}"]
        for field, filename in (("checkpoint_sha256", "checkpoint.pt"), ("summary_sha256", "summary.json"),
                                ("training_predictions_sha256", "training_predictions.csv"),
                                ("development_predictions_sha256", "development_predictions.csv")):
            if _file_hash(directory/filename) != binding[field]:
                raise ValueError("Original replay artifacts differ from their locked protocol bindings.")
        _objective_history(directory/"history.json", 25)
    loaded["directories"] = directories
    return loaded


def analyze_tail_lr(run_dirs, *, protocol_path, reference_run_dirs, reference_protocol_path,
                    prefix_root=None, draws=2000, bootstrap_seed=17,
                    expected_training_cases=256, expected_development_cases=159):
    draws = _integer(draws, "bootstrap draws", 2)
    bootstrap_seed = _integer(bootstrap_seed, "bootstrap seed")
    p = _protocol(protocol_path, expected_training_cases, expected_development_cases)
    reference = _reference(reference_run_dirs, reference_protocol_path, p)
    directories = [Path(value) for value in run_dirs]
    if len(directories) != 20 or len({value.resolve() for value in directories}) != 20:
        raise ValueError("Supply exactly twenty distinct completed policy/depth/seed branches.")
    if prefix_root is None:
        parents = {value.parent.resolve() for value in directories}
        if len(parents) != 1:
            raise ValueError("A shared prefix-root is required when branches have different parents.")
        prefix_root = next(iter(parents))
    prefix_root = Path(prefix_root)
    runs, training, development, metrics, branch_dirs = {}, {}, {}, {}, {}
    common_config = None
    for directory in directories:
        summary = _json(directory/"summary.json", "Every branch requires a readable completed summary.")
        seed, depth = _source_summary(summary, p, status="executed_cpu_tail_lr_control")
        config, timing = summary["config"], summary.get("timing", {})
        policy = config.get("tail_policy")
        if policy not in POLICIES:
            raise ValueError("Tail policy is outside the locked constant/low-tail design.")
        _exact_config(config, {"prefix_updates": p["prefix_updates"], "tail_learning_rate": RATES[policy]},
                      "Tail learning rate or prefix budget differs from the locked intervention.")
        if (not isinstance(timing, dict) or timing.get("updates") != p["training_updates_per_run"]
                or timing.get("new_updates_in_branch") != p["tail_updates"]
                or summary.get("prefix_reused", timing.get("prefix_reused")) is not True):
            raise ValueError("Branch conceptual trajectory and actual-new-work budgets are inconsistent.")
        key = (seed, depth, policy)
        if key in runs:
            raise ValueError("Duplicate policy/depth/seed branch; all twenty combinations are required.")
        comparable = {key: value for key, value in config.items()
                      if key not in {"seed", "steps", "tail_policy", "tail_learning_rate"}}
        if common_config is not None and comparable != common_config:
            raise ValueError("Branch configurations differ beyond the locked depth/seed/LR factors.")
        common_config = comparable
        runs[key], branch_dirs[key] = summary, directory
        training[key] = _trace(directory/"training_predictions.csv", split="train", seed=seed, depth=depth, require_dice=True)
        development[key] = _trace(directory/"development_predictions.csv", split="tune", seed=seed, depth=depth, require_dice=True)
        metrics[key] = {str(step): _metrics(development[key][step]) for step in range(1, depth+1)}
        _history(summary, metrics[key], final_epoch=25)
        _objective_history(directory/"history.json", 25)
    if set(runs) != {(seed, depth, policy) for seed in SEEDS for depth in DEPTHS for policy in POLICIES}:
        raise ValueError("All twenty locked branch combinations are required.")
    train_reference, tune_reference = reference["train_reference"], reference["tune_reference"]
    train_images = _cohort_check(train_reference, p["train"])
    tune_images = _cohort_check(tune_reference, p["tune"])
    if set(train_reference)&set(tune_reference) or train_images&tune_images:
        raise ValueError("Training and development Cases or views overlap.")
    for key in runs:
        for step in range(1, key[1]+1):
            _match_cases(train_reference, training[key][step])
            _match_cases(tune_reference, development[key][step])
    prefixes, prefix_metrics = {}, {}
    for seed in SEEDS:
        for depth in DEPTHS:
            directory = prefix_root/f"prefix-depth{depth}-seed{seed}"
            summary = _json(directory/"prefix_summary.json", "Each pair requires its completed common-prefix summary.")
            if _source_summary(summary, p, status="executed_cpu_tail_lr_prefix") != (seed, depth):
                raise ValueError("Prefix summary belongs to a different seed/depth pair.")
            if (summary["config"].get("prefix_updates") != p["prefix_updates"]
                    or summary.get("timing", {}).get("updates") != p["prefix_updates"]):
                raise ValueError("Prefix summary must record exactly the common twenty-epoch update budget.")
            for field, filename in (("checkpoint_sha256", "checkpoint.pt"),
                                    ("training_predictions_sha256", "training_predictions.csv"),
                                    ("development_predictions_sha256", "development_predictions.csv")):
                if _hash(summary.get(field), "bound prefix artifact") != _file_hash(directory/filename):
                    raise ValueError("Prefix artifact bytes disagree with the declared binding.")
            _hash(summary.get("state_sha256"), "full prefix state")
            prefix_sampling = _hash(summary.get("sampling_trace_sha256"), "prefix sampler trace")
            tr = _trace(directory/"training_predictions.csv", split="train", seed=seed, depth=depth, require_dice=True)
            dev = _trace(directory/"development_predictions.csv", split="tune", seed=seed, depth=depth, require_dice=True)
            for step in range(1, depth+1):
                _match_cases(train_reference, tr[step]); _match_cases(tune_reference, dev[step])
            observed = {str(step): _metrics(dev[step]) for step in range(1, depth+1)}
            _history(summary, observed, final_epoch=20)
            prefix_objective = (_objective_values(summary["history"], 20) if "history" in summary
                                else _objective_history(directory/"history.json", 20))
            prefixes[seed, depth], prefix_metrics[seed, depth] = dev, observed
            for policy in POLICIES:
                branch = runs[seed, depth, policy]
                provenance = branch.get("branch_provenance", {})
                required = {"prefix_checkpoint_sha256": summary["checkpoint_sha256"],
                            "prefix_state_sha256": summary["state_sha256"], "resumed_state_sha256": summary["state_sha256"],
                            "prefix_updates": p["prefix_updates"], "restored_sampler_epoch": 20,
                            "restored_loader_generator": True, "restored_rng": True}
                _exact_config(provenance, required, "Branch does not certify restoration of the complete paired prefix state.")
                sampling = branch.get("sampling_trace_sha256", {})
                if not isinstance(sampling, dict) or sampling.get("prefix") != prefix_sampling:
                    raise ValueError("Branch prefix sampling trace disagrees with its common prefix.")
                _hash(sampling.get("tail"), "tail sampler trace")
                if branch["development_history"][:-1] != summary["development_history"]:
                    raise ValueError("Pre-intervention development histories differ across paired branches.")
                objective = _objective_history(branch_dirs[seed, depth, policy]/"history.json", 25)
                if objective[:20] != prefix_objective:
                    raise ValueError("Pre-intervention training objective histories differ across paired branches.")
            constant, low = runs[seed, depth, "constant"], runs[seed, depth, "low_tail"]
            if constant["sampling_trace_sha256"] != low["sampling_trace_sha256"]:
                raise ValueError("Policy branches must share identical sampled-view/order trajectories.")
            refdir = reference["directories"][seed, depth]
            refsummary = reference["runs"][seed, depth]
            verify = constant.get("constant_replay_verification", {})
            expected_replay = {"status": "exact_replay_verified", "reference_summary_sha256": _file_hash(refdir/"summary.json"),
                               "model_state_tensors_equal": True, "prediction_records_equal": True,
                               "training_objective_history_equal": True, "development_history_equal": True}
            _exact_config(verify, expected_replay, "The constant-LR branch must certify an exact reference replay.")
            for filename in ("training_predictions.csv", "development_predictions.csv"):
                if _file_hash(branch_dirs[seed, depth, "constant"]/filename) != _file_hash(refdir/filename):
                    raise ValueError("Constant replay prediction CSV bytes differ from the bound original fit.")
            if (_reference_objective_projection(_objective_history(branch_dirs[seed, depth, "constant"]/"history.json", 25))
                    != _reference_objective_projection(_objective_history(refdir/"history.json", 25))
                    or constant["development_history"] != refsummary["development_history"]):
                raise ValueError("Constant replay training or development histories differ from the original fit.")
            low_replay = low.get("constant_replay_verification", {})
            _exact_config(low_replay, {"status": "paired_constant_replay_verified",
                                      "reference_summary_sha256": _file_hash(refdir/"summary.json"),
                                      "constant_summary_sha256": _file_hash(branch_dirs[seed, depth, "constant"]/"summary.json")},
                          "The low-rate branch must bind its verified paired constant replay.")
        for field in ("prefix", "tail"):
            if runs[seed, 2, "constant"]["sampling_trace_sha256"][field] != runs[seed, 4, "constant"]["sampling_trace_sha256"][field]:
                raise ValueError("Paired depths must share the same Case-view sampling trajectory.")
    return _aggregate(p, protocol_path, runs, training, development, metrics, prefixes, prefix_metrics,
                      train_reference, tune_reference, draws=draws, bootstrap_seed=bootstrap_seed)


def _aggregate(p, protocol_path, runs, training, development, metrics, prefixes, prefix_metrics,
               train_reference, tune_reference, *, draws, bootstrap_seed):
    case_ids = sorted(tune_reference)
    labels = np.array([tune_reference[case]["label"] for case in case_ids])
    probabilities = {(policy, depth): np.array([[development[seed, depth, policy][depth][case]["probability"]
                     for case in case_ids] for seed in SEEDS]) for policy in POLICIES for depth in DEPTHS}
    prefix_probabilities = {depth: np.array([[prefixes[seed, depth][depth][case]["probability"]
                          for case in case_ids] for seed in SEEDS]) for depth in DEPTHS}
    brier = {key: (value-labels)**2 for key, value in probabilities.items()}
    gains = {policy: brier[policy, 2]-brier[policy, 4] for policy in POLICIES}
    regimes, within_tail = {}, {}
    for policy in POLICIES:
        for depth in DEPTHS:
            name = regime_name(depth, policy)
            by_step = {str(step): {metric: _stats([metrics[seed, depth, policy][str(step)][metric] for seed in SEEDS])
                                  for metric in AGGREGATE_METRICS} for step in range(1, depth+1)}
            regimes[name] = {"depth": depth, "tail_policy": policy, "by_step": by_step, "endpoint": by_step[str(depth)]}
            delta_brier = (prefix_probabilities[depth]-labels)**2-brier[policy, depth]
            delta_nll = _nll_per_case(labels, probabilities[policy, depth])-_nll_per_case(labels, prefix_probabilities[depth])
            within_tail[name] = {"Brier_benefit": _effect(delta_brier, SEEDS, draws=draws, bootstrap_seed=bootstrap_seed),
                                 "NLL_worsening": _effect(delta_nll, SEEDS, draws=draws, bootstrap_seed=bootstrap_seed),
                                 "probability_change": _stats((probabilities[policy, depth]-prefix_probabilities[depth]).mean(axis=1)),
                                 "absolute_probability_change": _stats(np.abs(probabilities[policy, depth]-prefix_probabilities[depth]).mean(axis=1))}
    H = gains["low_tail"]-gains["constant"]
    depth_effects = {}
    for policy in POLICIES:
        dice = {depth: np.array([[development[seed, depth, policy][depth][case]["dice"] for case in case_ids]
                                 for seed in SEEDS]) for depth in DEPTHS}
        depth_effects[policy] = {
            "Brier_benefit": _effect(gains[policy], SEEDS, draws=draws, bootstrap_seed=bootstrap_seed),
            "NLL_worsening": _effect(_nll_per_case(labels, probabilities[policy, 4])-
                                      _nll_per_case(labels, probabilities[policy, 2]), SEEDS, draws=draws, bootstrap_seed=bootstrap_seed),
            "AUROC_change": _stats([metrics[seed, 4, policy]["4"]["auroc"]-
                                     metrics[seed, 2, policy]["2"]["auroc"] for seed in SEEDS]),
            "Dice_change": _effect(dice[4]-dice[2], SEEDS, draws=draws, bootstrap_seed=bootstrap_seed)}
    policy_effects = {}
    for depth in DEPTHS:
        policy_effects[str(depth)] = {
            "Brier_benefit": _effect(brier["constant", depth]-brier["low_tail", depth], SEEDS, draws=draws, bootstrap_seed=bootstrap_seed),
            "NLL_worsening": _effect(_nll_per_case(labels, probabilities["low_tail", depth])-
                                      _nll_per_case(labels, probabilities["constant", depth]), SEEDS, draws=draws, bootstrap_seed=bootstrap_seed),
            "AUROC_change": _stats([metrics[seed, depth, "low_tail"][str(depth)]["auroc"]-
                                     metrics[seed, depth, "constant"][str(depth)]["auroc"] for seed in SEEDS]),
            "Dice_change": _stats([metrics[seed, depth, "low_tail"][str(depth)]["mean_case_dice"]-
                                    metrics[seed, depth, "constant"][str(depth)]["mean_case_dice"] for seed in SEEDS])}
    seed_results = []
    for i, seed in enumerate(SEEDS):
        seed_results.append({"seed": seed, "H": float(H[i].mean()),
            "depth_benefit_Brier": {policy: float(gains[policy][i].mean()) for policy in POLICIES},
            "regimes": {regime_name(depth, policy): {
                "depth": depth, "tail_policy": policy, "endpoint": metrics[seed, depth, policy][str(depth)],
                "prefix_endpoint": prefix_metrics[seed, depth][str(depth)],
                "endpoint_diagnostics": _high_confidence_errors(labels, probabilities[policy, depth][i]),
                "within_tail_classification_transitions": _transitions(labels, prefix_probabilities[depth][i], probabilities[policy, depth][i])}
                for policy in POLICIES for depth in DEPTHS}})
    history = []
    for epoch in range(5, 26, 5):
        history.append({"epoch": epoch, "updates": epoch*math.ceil(p["train"]["cases"]/2),
                        "regimes": {regime_name(depth, policy): {metric: _stats([
                            next(entry for entry in runs[seed, depth, policy]["development_history"] if entry["epoch"] == epoch)
                            ["by_step"][str(depth)][metric] for seed in SEEDS]) for metric in AGGREGATE_METRICS}
                            for policy in POLICIES for depth in DEPTHS}})
    return {"status": "executed_cpu_tail_lr_analysis", "clinical_validation": False, "mechanism_identified": False,
            "decision_rule_automatically_applied": False, "phase": p["phase"], "evaluation_split": "tune_development",
            "patient_mapping": "unverified_Case_grouped_only", "model_selection": FINAL_SELECTION,
            "training_seeds": list(SEEDS), "policies": list(POLICIES), "depths": list(DEPTHS),
            "n_training_cases": len(train_reference), "n_development_cases": len(tune_reference),
            "n_training_images": p["train"]["images"], "n_development_images": p["tune"]["images"],
            "sources": {**{key: p[key] for key in SOURCE_KEYS+("launcher_source_sha256", "reference_protocol_sha256")},
                        "protocol_sha256": _file_hash(protocol_path)},
            "cohort_fingerprints": {split: p[split]["fingerprint"] for split in ("train", "tune")},
            "config": {**FIXED_CONFIG, "prefix_updates": p["prefix_updates"], "tail_updates": p["tail_updates"],
                       "tail_learning_rates": dict(RATES)},
            "compute_budget": {"prefix_fits": 10, "tail_branches": 20, "updates_per_prefix": p["prefix_updates"],
                               "new_updates_per_tail_branch": p["tail_updates"], "full_trajectory_updates_per_endpoint": p["training_updates_per_run"],
                               "total_actual_new_updates": p["total_actual_new_updates"], "four_steps_use_more_block_applications": True},
            "alignment_checks": {"exact_twenty_policy_depth_seed_branches": True, "same_case_view_label_sets": True,
                                 "all_prefix_artifact_digests_verified": True, "full_prefix_state_restoration_certified": True,
                                 "paired_sampling_trajectories_match": True, "constant_prediction_csv_bytes_replay_reference": True,
                                 "constant_objective_and_development_histories_replay_reference": True,
                                 "low_branch_binds_verified_constant_summary": True, "no_train_development_overlap": True},
            "definitions": {"G": "Brier_depth2_endpoint - Brier_depth4_endpoint within a tail policy; positive favors depth4",
                            "H": "G_low_tail - G_constant; positive means low LR increases the extra-depth Brier benefit",
                            "policy_Brier_benefit": "Brier_constant - Brier_low_tail at the same trained depth",
                            "within_tail_Brier_benefit": "Brier_prefix2560 - Brier_endpoint3200",
                            "NLL_worsening": "NLL_later_or_low - NLL_prefix_or_constant; positive is worse",
                            "case_bootstrap": "mean seed-specific paired effects within each Case, then resample Cases; fitted seeds held fixed",
                            "seed_SD": "sample SD ddof=1 across five fits; not a confidence interval"},
            "regimes": regimes, "primary_H": _effect(H, SEEDS, draws=draws, bootstrap_seed=bootstrap_seed),
            "depth_benefit_Brier": {policy: _effect(gains[policy], SEEDS, draws=draws, bootstrap_seed=bootstrap_seed) for policy in POLICIES},
            "depth_effects_by_policy": depth_effects,
            "policy_benefit_by_depth": policy_effects, "within_tail_changes": within_tail,
            "seed_results": seed_results, "learning_curves": history,
            "limitations": ["The tune cohort was already observed; this is an exploratory optimizer intervention, not independent validation.",
                            "H measures a paired tail-LR interaction; a positive H alone does not establish positive absolute depth benefit.",
                            "AdamW LR changes both gradient-step and decoupled weight-decay magnitudes; this intervention does not isolate a pure gradient-size mechanism.",
                            "The full-state restoration and tensor replay assertions are certified by the isolated runner; this analyzer verifies artifact bindings and raw prediction/history replay.",
                            "Case bootstrap intervals condition on fitted seeds and fixed Cases; they do not cover training-seed sampling uncertainty or establish patient independence.",
                            "Frozen features, no augmentation, fixed uncalibrated threshold 0.5 and resized Dice limit clinical interpretation.",
                            "Four-step models use additional block applications; matched optimizer updates do not imply equal FLOPs.",
                            "No calibration or test predictions enter this LR-control analysis; it is separate from temperature calibration."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--reference-run-dirs", nargs="+", required=True)
    parser.add_argument("--reference-protocol", required=True)
    parser.add_argument("--prefix-root")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument("--expected-training-cases", type=int, default=256)
    parser.add_argument("--expected-development-cases", type=int, default=159)
    parser.add_argument("--output", default="outputs/cpu-tail-lr256/tail_lr_analysis.json")
    args = parser.parse_args(argv)
    destination = Path(args.output)
    if destination.exists():
        parser.error("Analysis output exists; choose a fresh output path.")
    try:
        result = analyze_tail_lr(args.run_dirs, protocol_path=args.protocol, reference_run_dirs=args.reference_run_dirs,
                                 reference_protocol_path=args.reference_protocol, prefix_root=args.prefix_root,
                                 draws=args.draws, bootstrap_seed=args.bootstrap_seed,
                                 expected_training_cases=args.expected_training_cases,
                                 expected_development_cases=args.expected_development_cases)
    except ValueError as error:
        parser.error(str(error))
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    except FileExistsError:
        parser.error("Analysis output exists; choose a fresh output path.")
    print(json.dumps({"status": result["status"], "primary_H": result["primary_H"],
                      "depth_benefit_Brier": result["depth_benefit_Brier"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
