"""Aggregate-only, paired-Case analysis of a fixed-cohort 12-run CPU screen.

The four arms at seeds 17, 29 and 43 must use identical train/development
Cases and views. Bootstrap samples are Cases: the three fitted seeds are
averaged inside each Case, never counted as three independent patients.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .analyze import (ARMS, STEPS, SAFE_CONFIG_FIELDS, _hash, _index, _integer,
                      _match_cases, _read_predictions, _summary, analyze_runs)
from .metrics import classification_metrics


SEEDS = (17, 29, 43)
BASELINES = ("constant", "logistic_regression")
AGGREGATE_METRICS = ("brier", "auroc", "nll", "sensitivity", "specificity", "mean_case_dice")


def _stats(values):
    """Descriptive sample SD of three seed-level metrics, not a seed CI."""
    if any(value is None for value in values):
        return {"mean": None, "std_across_training_seeds": None, "min": None, "max": None}
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(SEEDS),) or not np.isfinite(values).all():
        raise ValueError("Seed-level aggregates must contain three finite numbers.")
    return {"mean": float(values.mean()), "std_across_training_seeds": float(values.std(ddof=1)),
            "min": float(values.min()), "max": float(values.max())}


def _paired_case_interval(seed_by_case, *, draws, bootstrap_seed):
    """Mean fitted-seed effect, with one shared Case resample for all seeds.

    A seed-by-Case array is reduced over seeds FIRST. This is algebraically
    equivalent to applying the same sampled Case indices to every fitted seed.
    A flattened seed/Case resample would incorrectly triple the sample size.
    """
    values = np.asarray(seed_by_case, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(SEEDS) or values.shape[1] == 0:
        raise ValueError("Use an aligned array of three seeds by nonempty Cases.")
    if not np.isfinite(values).all():
        raise ValueError("Paired Case effects must be finite.")
    per_case = values.mean(axis=0)
    bootstrap_means = np.empty(draws, dtype=np.float64)
    rng = np.random.default_rng(bootstrap_seed)
    for start in range(0, draws, 256):
        stop = min(start + 256, draws)
        indices = rng.integers(0, per_case.size, size=(stop-start, per_case.size))
        bootstrap_means[start:stop] = per_case[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, (.025, .975))
    return {"estimate": float(per_case.mean()), "ci95": [float(low), float(high)],
            "bootstrap": {"unit": "Case", "n_cases": int(per_case.size), "draws": draws,
                          "seed": bootstrap_seed, "method": "percentile",
                          "same_case_resample_across_seeds_arms_and_steps": True,
                          "fitted_seed_effects_averaged_within_case": True,
                          "includes_training_seed_sampling_uncertainty": False}}


def _trace(path, *, split, arm, seed, require_dice):
    # Check the explicit provenance column, which the original tiny-run reader
    # deliberately did not require. Merely naming a file development is not
    # evidence that its records originated from tune.
    try:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if "split" not in (reader.fieldnames or []):
                raise ValueError("Expanded traces require an explicit train/tune split column.")
            if any(row.get("split") != split for row in reader):
                raise ValueError("Prediction records must originate only from their declared train/tune split.")
    except OSError:
        raise ValueError("Each run requires readable train and development prediction traces.") from None
    return _index(_read_predictions(path), arm, seed, require_dice=require_dice)


def _fingerprint(index):
    triples = sorted([[str(case), str(image), int(row["label"])]
                      for case, row in index.items() for image in row["image_ids"]])
    payload = json.dumps(triples, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _baseline_data(directory, *, sources, training, development):
    """Check local baseline Case probabilities against the exact trained cohort."""
    directory = Path(directory)
    try:
        summary = json.loads((directory / "baselines_summary.json").read_text())
    except (OSError, ValueError):
        raise ValueError("Baselines require a readable completed summary.") from None
    if not isinstance(summary, dict) or summary.get("status") != "executed_cpu_preexperiment_baselines":
        raise ValueError("Baselines must be completed CPU engineering runs.")
    if (summary.get("clinical_validation") is not False or summary.get("device") != "cpu"
            or summary.get("patient_mapping") != "unverified_Case_grouped_only"):
        raise ValueError("Baseline provenance must declare the same CPU and Case-grouped pilot scope.")
    baseline_sources = summary.get("sources")
    if not isinstance(baseline_sources, dict):
        raise ValueError("Baseline summaries require explicit source digests.")
    if (_hash(baseline_sources.get("encoder_sha256"), "baseline encoder digest"),
            _hash(baseline_sources.get("manifest_sha256"), "baseline manifest digest")) != sources:
        raise ValueError("Baseline and model encoder/manifest digests must match.")
    fingerprints = summary.get("cohort_fingerprints")
    expected = {"train": _fingerprint(training), "tune": _fingerprint(development)}
    if not isinstance(fingerprints, dict) or fingerprints != expected:
        raise ValueError("Baseline training/development Case, view and label fingerprints differ.")
    indexed = {name: {"train": {}, "tune": {}} for name in BASELINES}
    required = {"baseline", "split", "seed", "case_id", "label", "probability", "n_images"}
    try:
        with (directory / "baseline_case_predictions.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("Baseline prediction schema is incomplete.")
            for row in reader:
                name, split = row["baseline"], row["split"]
                if name not in BASELINES or split not in ("train", "tune") or int(row["seed"]) != 17:
                    raise ValueError("Baseline traces require the two fixed fits and only train/tune records.")
                case = row["case_id"]
                if not case or case in indexed[name][split]:
                    raise ValueError("Baseline Case identifiers are invalid or duplicated.")
                label, probability, n_images = float(row["label"]), float(row["probability"]), int(row["n_images"])
                if (label not in (0, 1) or not math.isfinite(probability)
                        or not 0 <= probability <= 1 or n_images < 1):
                    raise ValueError("Baseline values must be finite probabilities and binary Case labels.")
                indexed[name][split][case] = {"label": label, "probability": probability, "n_images": n_images}
    except (OSError, KeyError, TypeError, ValueError):
        raise ValueError("Baseline predictions are invalid, duplicated, or not train/tune-only.") from None
    for name in BASELINES:
        for split, reference in (("train", training), ("tune", development)):
            candidate = indexed[name][split]
            if set(candidate) != set(reference):
                raise ValueError("Baseline and model Case sets differ; intersections are forbidden.")
            if any(candidate[case]["label"] != row["label"]
                   or candidate[case]["n_images"] != row["n_images"] for case, row in reference.items()):
                raise ValueError("Baseline and model pathology labels or image counts differ.")
    return indexed


def analyze_expanded(run_dirs, *, baseline_dir=None, draws=2000, bootstrap_seed=17,
                     expected_training_cases=128, expected_development_cases=159):
    """Validate and summarize the fixed twelve-run CPU experiment.

    All individual predictions remain local. The returned dictionary contains
    aggregate numbers, approved config fields and source/cohort digests only.
    """
    draws = _integer(draws, "bootstrap draws", 2)
    bootstrap_seed = _integer(bootstrap_seed, "bootstrap seed")
    expected_training_cases = _integer(expected_training_cases, "expected_training_cases", 2)
    expected_development_cases = _integer(expected_development_cases, "expected_development_cases", 2)
    directories = [Path(path) for path in run_dirs]
    if len(directories) != 12 or len({path.resolve() for path in directories}) != 12:
        raise ValueError("Supply twelve distinct completed run directories, four arms at each of seeds 17, 29 and 43.")
    runs, training, development = {}, {}, {}
    reference_config, sources = None, None
    for directory in directories:
        summary = _summary(directory / "summary.json")
        config = summary["config"]
        key = (config["seed"], config["arm"])
        if key[0] not in SEEDS or key in runs:
            raise ValueError("Exactly one run per arm and expected seed is required.")
        common = {k: v for k, v in config.items() if k not in ("arm", "seed")}
        candidate_sources = (summary["encoder_sha256"], summary["manifest_sha256"])
        if reference_config is None:
            reference_config, sources = common, candidate_sources
        elif common != reference_config or candidate_sources != sources:
            raise ValueError("All twelve training configs apart from arm/seed and source digests must match.")
        if summary.get("evaluation_split") != "tune_development":
            raise ValueError("Expanded source summaries must declare tune-only development evaluation.")
        if summary.get("model_selection") != "fixed_final_epoch_no_best_epoch_selection":
            raise ValueError("Expanded source runs must use the prespecified final epoch rather than a best development epoch.")
        if config.get("sampling") != "proportional":
            raise ValueError("This expanded screen requires proportionally sampled fixed training Cases.")
        _integer(config.get("selection_seed"), "selection_seed")
        runs[key] = (directory, summary)
        training[key] = _trace(directory / "training_predictions.csv", split="train",
                               arm=key[1], seed=key[0], require_dice=False)
        development[key] = _trace(directory / "development_predictions.csv", split="tune",
                                  arm=key[1], seed=key[0], require_dice=True)
    if set(runs) != {(seed, arm) for seed in SEEDS for arm in ARMS}:
        raise ValueError("All four arms at all three expected seeds are required.")
    reference_training, reference_development = training[(17, "SC")][1], development[(17, "SC")][1]
    if len(reference_training) != expected_training_cases or len(reference_development) != expected_development_cases:
        raise ValueError("Observed training/development Case counts disagree with the declared experiment.")
    if set(reference_training) & set(reference_development):
        raise ValueError("Training and development Case sets overlap.")
    train_images = {image for row in reference_training.values() for image in row["image_ids"]}
    dev_images = {image for row in reference_development.values() for image in row["image_ids"]}
    if train_images & dev_images:
        raise ValueError("Training and development image sets overlap.")
    for key in runs:
        for step in STEPS:
            _match_cases(reference_training, training[key][step])
            _match_cases(reference_development, development[key][step])
        selection = runs[key][1].get("cohort_selection")
        expected_selection = {"sampling": reference_config["sampling"],
                              "selection_seed": reference_config["selection_seed"],
                              "train_fingerprint": _fingerprint(reference_training),
                              "tune_fingerprint": _fingerprint(reference_development)}
        if not isinstance(selection, dict) or any(selection.get(k) != value for k, value in expected_selection.items()):
            raise ValueError("Declared cohort selection disagrees with observed Case, view and label fingerprints.")
    # This existing analyzer supplies the independent per-seed four-arm checks
    # of summary counts, paired initialization config, complete readout steps,
    # pathology labels, finite values and duplicate image rejection.
    by_seed = {seed: analyze_runs([runs[(seed, arm)][0] for arm in ARMS],
                                 draws=draws, bootstrap_seed=bootstrap_seed) for seed in SEEDS}
    case_ids = sorted(reference_development)
    labels = np.array([reference_development[case]["label"] for case in case_ids], dtype=np.float64)
    probabilities, dices, diagnostic, segmentation = {}, {}, {}, {}
    arm_reports = {}
    for arm in ARMS:
        probabilities[arm] = {step: np.array([[development[(seed, arm)][step][case]["probability"]
                                              for case in case_ids] for seed in SEEDS]) for step in STEPS}
        dices[arm] = {step: np.array([[development[(seed, arm)][step][case]["dice"]
                                     for case in case_ids] for seed in SEEDS]) for step in STEPS}
        diagnostic[arm] = ((probabilities[arm][2] - labels) ** 2 - (probabilities[arm][4] - labels) ** 2)
        segmentation[arm] = dices[arm][4] - dices[arm][2]
        seed_D = [float(values.mean()) for values in diagnostic[arm]]
        seed_S = [float(values.mean()) for values in segmentation[arm]]
        arm_reports[arm] = {
            "by_step": {str(step): {metric: _stats([by_seed[seed]["arms"][arm]["by_step"][str(step)][metric]
                                                   for seed in SEEDS]) for metric in AGGREGATE_METRICS}
                        for step in STEPS},
            "D": {**_stats(seed_D), **_paired_case_interval(diagnostic[arm], draws=draws,
                                                           bootstrap_seed=bootstrap_seed),
                  "positive_seed_count": sum(value > 0 for value in seed_D)},
            "S": {**_stats(seed_S), **_paired_case_interval(segmentation[arm], draws=draws,
                                                           bootstrap_seed=bootstrap_seed),
                  "positive_seed_count": sum(value > 0 for value in seed_S)},
        }
    theta = (diagnostic["SJ"] - diagnostic["SC"]) - (diagnostic["UJ"] - diagnostic["UC"])
    theta_seeds = [float(values.mean()) for values in theta]
    baseline_report = None
    if baseline_dir is not None:
        baseline = _baseline_data(baseline_dir, sources=sources, training=reference_training,
                                  development=reference_development)
        baseline_report = {}
        for name in BASELINES:
            p = np.array([baseline[name]["tune"][case]["probability"] for case in case_ids])
            baseline_report[name] = {
                "fit_seed": 17, "fixed_fit_shared_across_all_training_seeds": True,
                "tune_metrics": classification_metrics(labels, p),
                "model_minus_baseline_brier": {},
            }
            for arm in ARMS:
                contrasts = {}
                for step in STEPS:
                    differences = (probabilities[arm][step] - labels) ** 2 - (p - labels) ** 2
                    values = differences.mean(axis=1).tolist()
                    contrasts[str(step)] = {
                        **_stats(values), **_paired_case_interval(differences, draws=draws,
                                                                 bootstrap_seed=bootstrap_seed),
                        "by_seed": [{"seed": seed, "difference": value} for seed, value in zip(SEEDS, values)],
                        "negative_seed_count": sum(value < 0 for value in values),
                    }
                baseline_report[name]["model_minus_baseline_brier"][arm] = contrasts
    safe_config = {key: reference_config[key] for key in SAFE_CONFIG_FIELDS if key != "seed"}
    if "selection_seed" in reference_config:
        safe_config["selection_seed"] = _integer(reference_config["selection_seed"], "selection_seed")
    safe_config["sampling"] = "proportional"
    if "eval_every" in reference_config:
        safe_config["eval_every"] = _integer(reference_config["eval_every"], "eval_every")
    return {
        "status": "executed_cpu_engineering_analysis", "purpose": "fixed-cohort expanded CPU research screening",
        "clinical_validation": False, "mechanism_identified": False,
        "patient_mapping": "unverified_Case_grouped_only", "evaluation_split": "tune_development",
        "decision_rule_automatically_applied": False,
        "training_seeds": list(SEEDS), "config": safe_config,
        "model_selection": "fixed_final_epoch_no_best_epoch_selection",
        "sources": {"encoder_sha256": sources[0], "manifest_sha256": sources[1]},
        "cohort_fingerprints": {"train": _fingerprint(reference_training), "tune": _fingerprint(reference_development)},
        "n_training_cases": len(reference_training), "n_training_images": len(train_images),
        "n_training_positive": sum(row["label"] for row in reference_training.values()),
        "n_development_cases": int(labels.size), "n_development_images": len(dev_images),
        "n_development_positive": int(labels.sum()), "n_development_negative": int(labels.size-labels.sum()),
        "alignment_checks": {
            "exactly_twelve_expected_arm_seed_runs": True, "same_training_case_view_label_sets_all_runs": True,
            "same_development_case_view_label_sets_all_runs_and_steps": True,
            "matching_config_except_arm_seed_and_matching_source_digests": True,
            "explicit_train_tune_only_record_provenance": True,
            "no_training_development_case_or_image_overlap": True,
        },
        "definitions": {
            "case_probability": "arithmetic mean of all evaluated view probabilities; metrics computed per seed, then summarized",
            "case_dice": "arithmetic mean of evaluated image Dice values; 224x224 letterbox valid pixels",
            "D": "Brier_step2 - Brier_step4; positive means lower step4 Brier",
            "S": "Dice_step4 - Dice_step2; positive means higher step4 Dice",
            "theta_D": "(D_SJ - D_SC) - (D_UJ - D_UC)",
            "model_minus_baseline_brier": "Brier_model - Brier_baseline; negative favors the model",
            "seed_std": "sample standard deviation with ddof=1 over the three fitted training seeds",
            "case_bootstrap": "average the three seed-specific effects within each aligned Case, then resample Cases",
        },
        "arms": arm_reports,
        "theta_D": {**_stats(theta_seeds), **_paired_case_interval(theta, draws=draws, bootstrap_seed=bootstrap_seed),
                    "positive_seed_count": sum(value > 0 for value in theta_seeds)},
        "seed_results": [{"seed": seed,
                          "arms": {arm: by_seed[seed]["arms"][arm] for arm in ARMS},
                          "theta_D": by_seed[seed]["theta_D"]["estimate"]} for seed in SEEDS],
        "baselines": baseline_report,
        "limitations": [
            "Development-only screening does not establish clinical benefit, causal mechanism or novelty.",
            "The bootstrap conditions on these fitted models and fixed observed cohort; it does not estimate training-seed sampling uncertainty.",
            "Three seeds give descriptive direction and spread only, not generalizability over training randomness.",
            "Case-to-patient mapping is unverified; patient independence is not established.",
            "Only train/tune predictions are accepted; calibration/test performance is not measured.",
            "Frozen features, no augmentation, fixed learning rate and resized Dice limit the scope of this screen.",
        ],
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dirs", nargs="+", required=True)
    p.add_argument("--baseline-dir")
    p.add_argument("--output", default="outputs/cpu-expanded128/expanded_analysis.json")
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
        report = analyze_expanded(args.run_dirs, baseline_dir=args.baseline_dir, draws=args.draws,
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
                      "theta_D": report["theta_D"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
