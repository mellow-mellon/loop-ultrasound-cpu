"""Paired-Case analysis of separately fitted loop depth and supervision regimes.

Only aggregate metrics leave this module. The primary comparison uses each
separately trained model's endpoint; intermediate outputs of terminal-only
models are descriptive readouts, not directly supervised predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .analyze import _hash, _integer, _match_cases, _read_predictions
from .analyze_expanded import (AGGREGATE_METRICS, BASELINES, SEEDS, _baseline_data,
                               _fingerprint, _paired_case_interval, _stats)
from .metrics import aggregate_cases, classification_metrics


DEPTHS = (2, 4)
SUPERVISIONS = ("all", "terminal")
FIXED_CONFIG = {"arm": "SJ", "epochs": 50, "batch_size": 2,
                "learning_rate": .0003, "segmentation_weight": 1.,
                "cached_frozen_features": True, "augment": False,
                "selection_seed": 20261004, "sampling": "proportional", "eval_every": 5}
FINAL_SELECTION = "fixed_final_epoch_no_best_epoch_selection"
SOURCE_KEYS = ("encoder_sha256", "manifest_sha256", "training_source_sha256")


def regime_name(depth, supervision):
    return f"depth{depth}_{supervision}"


def _json(path, message):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        raise ValueError(message) from None
    if not isinstance(value, dict):
        raise ValueError(message)
    return value


def _protocol(path, training_cases, development_cases):
    protocol = _json(path, "A readable locked experiment protocol is required.")
    if (protocol.get("status") != "fixed_before_depth_supervision_training"
            or protocol.get("clinical_validation") is not False or protocol.get("device") != "cpu"
            or protocol.get("patient_mapping") != "unverified_Case_grouped_only"
            or protocol.get("calibration_and_test_predictions") is not False
            or protocol.get("model_selection") != FINAL_SELECTION):
        raise ValueError("The protocol must declare a CPU, Case-grouped, fixed-final development screen.")
    if (protocol.get("seeds") != list(SEEDS) or protocol.get("depths") != list(DEPTHS)
            or protocol.get("supervisions") != list(SUPERVISIONS)
            or protocol.get("arms") != ["SJ"]):
        raise ValueError("The protocol must lock SJ, depths 2/4, all/terminal supervision and all three seeds.")
    for key, expected in FIXED_CONFIG.items():
        if key == "arm":
            continue
        if protocol.get(key) != expected or type(protocol.get(key)) is not type(expected):
            raise ValueError("The protocol differs from the fixed training configuration.")
    for key in SOURCE_KEYS:
        protocol[key] = _hash(protocol.get(key), f"protocol {key}")
    initialization = protocol.get("initialization_sha256_by_seed")
    if not isinstance(initialization, dict) or set(initialization) != {str(seed) for seed in SEEDS}:
        raise ValueError("The protocol must lock an initial parameter digest for each fitted seed.")
    protocol["initialization_sha256_by_seed"] = {
        str(seed): _hash(initialization[str(seed)], "protocol initialization digest") for seed in SEEDS}
    if protocol.get("warm_starts") is not False:
        raise ValueError("The paired-depth experiment requires fresh fits without warm starts.")
    expected_updates = math.ceil(training_cases / FIXED_CONFIG["batch_size"]) * FIXED_CONFIG["epochs"]
    if protocol.get("training_updates_per_run") != expected_updates:
        raise ValueError("The protocol must lock all 50 epochs' optimizer updates for each fresh model.")
    for split, expected in (("train", training_cases), ("tune", development_cases)):
        item = protocol.get(split)
        if not isinstance(item, dict) or _integer(item.get("cases"), f"protocol {split} Cases", 2) != expected:
            raise ValueError("Protocol Case counts disagree with the declared experiment.")
        _integer(item.get("images"), f"protocol {split} images", expected)
        _integer(item.get("benign"), f"protocol {split} benign")
        _integer(item.get("malignant"), f"protocol {split} malignant")
        if item["benign"] + item["malignant"] != expected:
            raise ValueError("Protocol pathology counts are inconsistent.")
        item["fingerprint"] = _hash(item.get("fingerprint"), f"protocol {split} cohort fingerprint")
    return protocol


def _summary(directory):
    summary = _json(directory / "summary.json", "Each run requires a readable completed summary.")
    if (summary.get("status") != "executed_cpu_engineering_pilot"
            or summary.get("clinical_validation") is not False or summary.get("device") != "cpu"
            or summary.get("patient_mapping") != "unverified_Case_grouped_only"):
        raise ValueError("Only completed CPU Case-grouped engineering runs are accepted.")
    if summary.get("evaluation_split") != "tune_development" or summary.get("model_selection") != FINAL_SELECTION:
        raise ValueError("Runs must use tune-only development evaluation at the prespecified final epoch.")
    config = summary.get("config")
    if not isinstance(config, dict):
        raise ValueError("Training config must be a mapping.")
    for key, expected in FIXED_CONFIG.items():
        if config.get(key) != expected or type(config.get(key)) is not type(expected):
            raise ValueError("Source config differs from the fixed SJ training configuration.")
    depth = _integer(config.get("steps"), "trained depth", 1)
    seed = _integer(config.get("seed"), "training seed")
    supervision = config.get("supervision")
    if depth not in DEPTHS or seed not in SEEDS or supervision not in SUPERVISIONS:
        raise ValueError("Run depth, supervision or training seed is outside the locked design.")
    for key in SOURCE_KEYS + ("initialization_sha256",):
        summary[key] = _hash(summary.get(key), key)
    declared_supervision = summary.get("readout_supervision")
    expected_steps = list(range(1, depth + 1)) if supervision == "all" else [depth]
    if (not isinstance(declared_supervision, dict) or declared_supervision.get("mode") != supervision
            or declared_supervision.get("directly_supervised_steps") != expected_steps):
        raise ValueError("Declared directly supervised readouts disagree with the training regime.")
    for key in ("n_training_cases", "n_training_images"):
        _integer(summary.get(key), key, 2)
    timing = summary.get("timing")
    if not isinstance(timing, dict):
        raise ValueError("Completed optimizer update timing is required.")
    updates = timing.get("updates")
    expected_updates = math.ceil(summary["n_training_cases"] / config["batch_size"]) * config["epochs"]
    if _integer(updates, "completed optimizer updates", 1) != expected_updates:
        raise ValueError("Completed optimizer updates disagree with 50 full fixed-cohort epochs.")
    return summary


def _trace(path, *, split, seed, depth, require_dice):
    try:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if "split" not in (reader.fieldnames or []):
                raise ValueError("Traces require an explicit train/tune split column.")
            if any(row.get("split") != split for row in reader):
                raise ValueError("Prediction records must originate only from their declared train/tune split.")
    except OSError:
        raise ValueError("Each run requires readable train/development prediction traces.") from None
    records = _read_predictions(path)
    steps = tuple(range(1, depth + 1))
    if any(row["arm"] != "SJ" or row["seed"] != seed or row["step"] not in steps for row in records):
        raise ValueError("Prediction arm, seed or readout step disagrees with the trained model.")
    try:
        cases = aggregate_cases(records)
    except ValueError:
        raise ValueError("Trace values, Case labels, view assignments or duplicate images are invalid.") from None
    indexed = {step: {} for step in steps}
    for row in cases:
        if require_dice and (row["dice"] is None or row["n_dice"] != row["n_images"]):
            raise ValueError("Every development image and readout requires a finite Dice value.")
        indexed[row["step"]][row["case_id"]] = row
    if not indexed[1]:
        raise ValueError("Each trace requires every configured readout step.")
    for step in steps:
        _match_cases(indexed[1], indexed[step])
    return indexed


def _metrics(index):
    cases = sorted(index)
    result = classification_metrics([index[case]["label"] for case in cases],
                                    [index[case]["probability"] for case in cases])
    result["mean_case_dice"] = float(np.mean([index[case]["dice"] for case in cases]))
    result["dice_space"] = "224x224_letterbox_valid_pixels"
    return result


def _declared_metrics(declared, observed, *, steps):
    """Refuse summary counts or endpoint metrics inconsistent with local traces."""
    if not isinstance(declared, dict) or set(declared) != {str(step) for step in steps}:
        raise ValueError("The summary must declare exactly the configured readout steps.")
    for step in steps:
        candidate, reference = declared[str(step)], observed[str(step)]
        if not isinstance(candidate, dict):
            raise ValueError("Declared readout metrics must be mappings.")
        for key, expected in reference.items():
            actual = candidate.get(key)
            if expected is None or isinstance(expected, str):
                equal = actual == expected
            elif isinstance(expected, int):
                equal = not isinstance(actual, bool) and isinstance(actual, int) and actual == expected
            else:
                equal = (not isinstance(actual, bool) and isinstance(actual, (int, float))
                         and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10))
            if not equal:
                raise ValueError("Declared development metrics disagree with the paired prediction traces.")


def _effect(values, *, draws, bootstrap_seed):
    seed_values = np.asarray(values).mean(axis=1).tolist()
    return {**_stats(seed_values), **_paired_case_interval(values, draws=draws, bootstrap_seed=bootstrap_seed),
            "by_seed": [{"seed": seed, "value": value} for seed, value in zip(SEEDS, seed_values)],
            "positive_seed_count": sum(value > 0 for value in seed_values),
            "negative_seed_count": sum(value < 0 for value in seed_values)}


def analyze_depth_supervision(run_dirs, *, protocol_path, baseline_dir=None, draws=2000,
                             bootstrap_seed=17, expected_training_cases=128,
                             expected_development_cases=159):
    """Validate twelve freshly fitted models against a required locked protocol.

    Primary Brier contrasts compare separate depth-2/depth-4 model endpoints.
    Case bootstrap intervals condition on the fitted seeds; seeds are averaged
    within Cases rather than treated as independent additional patients.
    """
    draws = _integer(draws, "bootstrap draws", 2)
    bootstrap_seed = _integer(bootstrap_seed, "bootstrap seed")
    expected_training_cases = _integer(expected_training_cases, "expected training Cases", 2)
    expected_development_cases = _integer(expected_development_cases, "expected development Cases", 2)
    protocol = _protocol(protocol_path, expected_training_cases, expected_development_cases)
    directories = [Path(path) for path in run_dirs]
    if len(directories) != 12 or len({path.resolve() for path in directories}) != 12:
        raise ValueError("Supply twelve distinct completed depth/supervision/seed runs.")
    runs, training, development = {}, {}, {}
    reference_config, initialization = None, {}
    for directory in directories:
        summary = _summary(directory)
        config = summary["config"]
        key = (config["seed"], config["steps"], config["supervision"])
        if key in runs:
            raise ValueError("Exactly one run per expected depth/supervision/seed combination is required.")
        common = {name: value for name, value in config.items() if name not in ("seed", "steps", "supervision")}
        if reference_config is not None and common != reference_config:
            raise ValueError("All source configs must match apart from depth, supervision and seed.")
        reference_config = common
        if any(summary[name] != protocol[name] for name in SOURCE_KEYS):
            raise ValueError("Source encoder, manifest and training-code digests must match the locked protocol.")
        if summary["initialization_sha256"] != protocol["initialization_sha256_by_seed"][str(key[0])]:
            raise ValueError("Initial model parameters must match the seed-specific digest locked before training.")
        if key[0] in initialization and initialization[key[0]] != summary["initialization_sha256"]:
            raise ValueError("All four separately fitted models within each seed require identical initial parameters.")
        initialization[key[0]] = summary["initialization_sha256"]
        runs[key] = summary
        training[key] = _trace(directory / "training_predictions.csv", split="train", seed=key[0],
                               depth=key[1], require_dice=False)
        development[key] = _trace(directory / "development_predictions.csv", split="tune", seed=key[0],
                                  depth=key[1], require_dice=True)
    expected_keys = {(seed, depth, supervision) for seed in SEEDS for depth in DEPTHS for supervision in SUPERVISIONS}
    if set(runs) != expected_keys:
        raise ValueError("All twelve expected depth/supervision/seed combinations are required.")
    reference_training = training[(17, 2, "all")][1]
    reference_development = development[(17, 2, "all")][1]
    train_images = {image for row in reference_training.values() for image in row["image_ids"]}
    dev_images = {image for row in reference_development.values() for image in row["image_ids"]}
    if set(reference_training) & set(reference_development) or train_images & dev_images:
        raise ValueError("Training/development Cases or images overlap.")
    observed_cohorts = {}
    for split, index, images in (("train", reference_training, train_images), ("tune", reference_development, dev_images)):
        labels = [row["label"] for row in index.values()]
        observed_cohorts[split] = {"cases": len(index), "images": len(images), "benign": labels.count(0),
                                   "malignant": sum(labels), "fingerprint": _fingerprint(index)}
        if any(protocol[split][key] != value for key, value in observed_cohorts[split].items()):
            raise ValueError("Observed Case/view/pathology cohort disagrees with the locked protocol.")
    seed_metrics = {}
    for key, summary in runs.items():
        steps = range(1, key[1] + 1)
        for step in steps:
            _match_cases(reference_training, training[key][step])
            _match_cases(reference_development, development[key][step])
        if (summary["n_training_cases"] != len(reference_training)
                or summary["n_training_images"] != len(train_images)):
            raise ValueError("Training Case/image count disagrees with the source summary.")
        expected_selection = {"sampling": "proportional", "selection_seed": 20261004,
                              "train_fingerprint": protocol["train"]["fingerprint"],
                              "tune_fingerprint": protocol["tune"]["fingerprint"]}
        if summary.get("cohort_selection") != expected_selection:
            raise ValueError("Declared cohort selection disagrees with the locked and observed cohorts.")
        metrics = {str(step): _metrics(development[key][step]) for step in steps}
        _declared_metrics(summary.get("development_subset"), metrics, steps=steps)
        history = summary.get("development_history")
        if (not isinstance(history, list) or [item.get("epoch") for item in history if isinstance(item, dict)]
                != list(range(5, 51, 5))):
            raise ValueError("Development history must record all fixed evaluation epochs through epoch 50.")
        _declared_metrics(history[-1].get("by_step"), metrics, steps=steps)
        seed_metrics[key] = metrics
    case_ids = sorted(reference_development)
    labels = np.array([reference_development[case]["label"] for case in case_ids])
    probabilities, dices, regimes = {}, {}, {}
    for depth in DEPTHS:
        for supervision in SUPERVISIONS:
            name = regime_name(depth, supervision)
            steps = range(1, depth + 1)
            probabilities[name] = {step: np.array([[development[(seed, depth, supervision)][step][case]["probability"]
                                                    for case in case_ids] for seed in SEEDS]) for step in steps}
            dices[name] = {step: np.array([[development[(seed, depth, supervision)][step][case]["dice"]
                                          for case in case_ids] for seed in SEEDS]) for step in steps}
            by_step = {str(step): {metric: _stats([seed_metrics[(seed, depth, supervision)][str(step)][metric]
                                                 for seed in SEEDS]) for metric in AGGREGATE_METRICS} for step in steps}
            regimes[name] = {"depth": depth, "supervision": supervision, "by_step": by_step,
                             "endpoint": by_step[str(depth)],
                             "intermediate_readouts_directly_supervised": supervision == "all"}
    primary, benefits, secondary = {}, {}, {}
    for supervision in SUPERVISIONS:
        early, late = regime_name(2, supervision), regime_name(4, supervision)
        benefits[supervision] = (probabilities[early][2] - labels) ** 2 - (probabilities[late][4] - labels) ** 2
        primary[supervision] = _effect(benefits[supervision], draws=draws, bootstrap_seed=bootstrap_seed)
        within = (probabilities[late][2] - labels) ** 2 - (probabilities[late][4] - labels) ** 2
        secondary[supervision] = {
            "Brier_change": _effect(within, draws=draws, bootstrap_seed=bootstrap_seed),
            "Dice_change": _effect(dices[late][4] - dices[late][2], draws=draws, bootstrap_seed=bootstrap_seed),
            "intermediate_readout_directly_supervised": supervision == "all"}
    baseline_report = None
    if baseline_dir is not None:
        baseline = _baseline_data(baseline_dir, sources=(protocol["encoder_sha256"], protocol["manifest_sha256"]),
                                  training=reference_training, development=reference_development)
        baseline_report = {}
        for name in BASELINES:
            p = np.array([baseline[name]["tune"][case]["probability"] for case in case_ids])
            baseline_report[name] = {"fit_seed": 17, "fixed_fit_shared_across_all_training_seeds": True,
                                     "tune_metrics": classification_metrics(labels, p), "model_minus_baseline_brier": {}}
            for regime, values in probabilities.items():
                depth = regimes[regime]["depth"]
                differences = (values[depth] - labels) ** 2 - (p - labels) ** 2
                baseline_report[name]["model_minus_baseline_brier"][regime] = _effect(
                    differences, draws=draws, bootstrap_seed=bootstrap_seed)
    safe_config = dict(FIXED_CONFIG)
    return {
        "status": "executed_cpu_depth_supervision_analysis", "clinical_validation": False,
        "mechanism_identified": False, "patient_mapping": "unverified_Case_grouped_only",
        "evaluation_split": "tune_development", "model_selection": FINAL_SELECTION,
        "decision_rule_automatically_applied": False, "training_seeds": list(SEEDS),
        "depths": list(DEPTHS), "supervisions": list(SUPERVISIONS), "config": safe_config,
        "sources": {**{name: protocol[name] for name in SOURCE_KEYS},
                    "protocol_sha256": hashlib.sha256(Path(protocol_path).read_bytes()).hexdigest()},
        "cohort_fingerprints": {split: protocol[split]["fingerprint"] for split in ("train", "tune")},
        "n_training_cases": len(reference_training), "n_training_images": len(train_images),
        "n_training_positive": int(protocol["train"]["malignant"]),
        "n_development_cases": len(reference_development), "n_development_images": len(dev_images),
        "n_development_positive": int(labels.sum()), "n_development_negative": int(labels.size-labels.sum()),
        "paired_initialization": {"identical_within_each_seed": True,
                                   "by_seed": [{"seed": seed, "initialization_sha256": initialization[seed]} for seed in SEEDS]},
        "alignment_checks": {"exactly_twelve_expected_depth_supervision_seed_runs": True,
                             "separately_fitted_endpoints": True, "same_training_case_view_label_sets_all_runs": True,
                             "same_development_case_view_label_sets_all_runs_and_steps": True,
                             "matching_config_except_depth_supervision_seed": True,
                             "matching_locked_source_digests_and_cohort": True,
                             "explicit_train_tune_only_record_provenance": True,
                             "no_training_development_case_or_image_overlap": True,
                             "fixed_50_epochs_and_completed_updates": True,
                             "summary_metrics_match_prediction_traces": True,
                             "paired_initial_parameters_within_seed": True},
        "definitions": {
            "primary_depth_benefit": "Brier_separately_fitted_depth2_endpoint - Brier_separately_fitted_depth4_endpoint; positive favors depth4",
            "interaction_B": "primary_depth_benefit_terminal - primary_depth_benefit_all",
            "secondary_within_depth4_Brier_change": "Brier_readout2 - Brier_readout4 within the same fitted depth4 model; terminal readout2 is not directly supervised",
            "secondary_within_depth4_Dice_change": "Dice_readout4 - Dice_readout2 within the same fitted depth4 model",
            "case_probability": "arithmetic mean of evaluated view probabilities; metrics computed separately per fitted seed",
            "case_dice": "arithmetic mean of image Dice within Case on 224x224 letterbox valid pixels",
            "seed_std": "sample SD ddof=1 over three fitted seeds; not a confidence interval",
            "case_bootstrap": "average seed-specific paired effects within each aligned Case, then resample Cases",
            "model_minus_baseline_brier": "Brier_model_endpoint - Brier_baseline; negative favors model",
        },
        "regimes": regimes, "primary_depth_benefit": primary,
        "interaction_B": _effect(benefits["terminal"] - benefits["all"], draws=draws, bootstrap_seed=bootstrap_seed),
        "secondary_within_depth4": secondary,
        "seed_results": [{"seed": seed,
                          "regimes": {regime_name(depth, supervision): {
                              "depth": depth, "supervision": supervision,
                              "by_step": seed_metrics[(seed, depth, supervision)],
                              "endpoint": seed_metrics[(seed, depth, supervision)][str(depth)]}
                              for depth in DEPTHS for supervision in SUPERVISIONS}}
                         for seed in SEEDS],
        "baselines": baseline_report,
        "limitations": [
            "Development-only screening does not establish clinical utility, novelty or a causal mechanism.",
            "The tune cohort was observed in the earlier CPU screen; this follow-up is exploratory, not an untouched validation study.",
            "Primary depth comparisons use separately fitted model endpoints; secondary intermediate outputs are different comparisons.",
            "Terminal-only intermediate readouts are not directly supervised and cannot be treated as standalone trained depth2 models.",
            "Equal epochs and optimizer updates do not equalize total computation: depth4 uses more recurrent block applications.",
            "Case bootstrap intervals condition on the fitted seeds and fixed cohort; they exclude training-seed sampling uncertainty.",
            "Three seeds support descriptive direction and spread only; patient independence is unverified.",
            "Calibration/test predictions are absent; thresholds are fixed at 0.5 and are not clinically optimized.",
            "Frozen features, no augmentation and resized mask Dice limit the scope of this screen.",
        ]}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dirs", nargs="+", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--baseline-dir")
    p.add_argument("--output", default="outputs/cpu-depth-supervision128/depth_supervision_analysis.json")
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=17)
    p.add_argument("--expected-training-cases", type=int, default=128)
    p.add_argument("--expected-development-cases", type=int, default=159)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    destination = Path(args.output)
    if destination.exists():
        p.error("Analysis output already exists; choose a fresh output path.")
    try:
        report = analyze_depth_supervision(args.run_dirs, protocol_path=args.protocol,
                                          baseline_dir=args.baseline_dir, draws=args.draws,
                                          bootstrap_seed=args.bootstrap_seed,
                                          expected_training_cases=args.expected_training_cases,
                                          expected_development_cases=args.expected_development_cases)
    except ValueError as error:
        p.error(str(error))
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    except FileExistsError:
        p.error("Analysis output already exists; choose a fresh output path.")
    print(json.dumps({"status": report["status"], "n_development_cases": report["n_development_cases"],
                      "primary_depth_benefit": report["primary_depth_benefit"],
                      "interaction_B": report["interaction_B"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
