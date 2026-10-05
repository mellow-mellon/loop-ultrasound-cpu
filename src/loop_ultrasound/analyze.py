"""Aggregate-only acceptance analysis of one four-arm CPU engineering pilot.

This reads local prediction traces to align Cases explicitly. Its JSON contains
no image/Case identifiers, individual pathology labels, or local source paths.
The development subset supports software checks, not clinical or causal claims.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from numbers import Integral
from pathlib import Path
import re
from typing import Any

import numpy as np

from .metrics import aggregate_cases, classification_metrics


ARMS = ("SC", "SJ", "UC", "UJ")
STEPS = (1, 2, 3, 4)
SAFE_CONFIG_FIELDS = (
    "steps", "seed", "epochs", "batch_size", "learning_rate",
    "segmentation_weight", "cached_frozen_features", "augment",
)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{name} must be a SHA-256 digest.")
    return value.lower()


def _summary(path: Path) -> dict:
    try:
        summary = json.loads(path.read_text())
    except (OSError, ValueError):
        raise ValueError("Each run requires a readable, completed summary.json.") from None
    if not isinstance(summary, dict) or summary.get("status") != "executed_cpu_engineering_pilot":
        raise ValueError("Only completed CPU engineering pilot summaries are accepted.")
    if summary.get("clinical_validation") is not False or summary.get("device") != "cpu":
        raise ValueError("The source summary must declare a CPU pilot without clinical validation.")
    if summary.get("patient_mapping") != "unverified_Case_grouped_only":
        raise ValueError("This analyzer expects the declared unverified Case grouping of the pilot.")
    config = summary.get("config")
    if not isinstance(config, dict) or not set(SAFE_CONFIG_FIELDS).issubset(config):
        raise ValueError("Source training config is incomplete.")
    if config.get("arm") not in ARMS:
        raise ValueError("Source arm must be SC, SJ, UC, or UJ.")
    if _integer(config["steps"], "steps", 1) != 4:
        raise ValueError("The four-arm acceptance analysis requires four trained readout steps.")
    _integer(config["seed"], "training seed")
    _integer(config["epochs"], "epochs", 1)
    _integer(config["batch_size"], "batch_size", 1)
    for name, minimum in (("learning_rate", 0.0), ("segmentation_weight", 0.0)):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
        if value < minimum or (name == "learning_rate" and value == 0.0):
            raise ValueError(f"{name} is outside its valid range.")
    if not isinstance(config["cached_frozen_features"], bool) or not isinstance(config["augment"], bool):
        raise ValueError("Cache and augmentation flags must be booleans.")
    if config["cached_frozen_features"] and config["augment"]:
        raise ValueError("Cached image features cannot accompany pixel augmentation.")
    _integer(summary.get("n_training_cases"), "n_training_cases", 2)
    _integer(summary.get("n_training_images"), "n_training_images", 2)
    summary["encoder_sha256"] = _hash(summary.get("encoder_sha256"), "encoder_sha256")
    summary["manifest_sha256"] = _hash(summary.get("manifest_sha256"), "manifest_sha256")
    return summary


def _read_predictions(path: Path) -> list[dict]:
    required = {"case_id", "image_id", "label", "probability", "step", "arm", "seed", "dice"}
    try:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("Prediction CSV schema is incomplete.")
            records = []
            for raw in reader:
                records.append({
                    "case_id": raw["case_id"], "image_id": raw["image_id"],
                    "label": float(raw["label"]), "probability": float(raw["probability"]),
                    "step": int(raw["step"]), "arm": raw["arm"], "seed": int(raw["seed"]),
                    "dice": float(raw["dice"]) if raw["dice"].strip() else None,
                })
    except (OSError, TypeError, ValueError, KeyError):
        raise ValueError("Each run requires a readable prediction CSV with valid typed fields.") from None
    if not records:
        raise ValueError("Prediction traces cannot be empty.")
    return records


def _index(records: list[dict], arm: str, seed: int, *, require_dice: bool) -> dict:
    if any(row["arm"] != arm or row["seed"] != seed or row["step"] not in STEPS for row in records):
        raise ValueError("Prediction arm, seed, or step disagrees with the source summary.")
    try:
        cases = aggregate_cases(records)
    except ValueError:
        # Do not propagate raw Case/image identifiers from schema errors into
        # CLI logs or public aggregation failures.
        raise ValueError("Prediction trace has invalid values, duplicate images, or conflicting Case labels.") from None
    indexed = {step: {} for step in STEPS}
    for row in cases:
        if require_dice and (row["dice"] is None or row["n_dice"] != row["n_images"]):
            raise ValueError("Every development image and step requires a finite Dice value.")
        indexed[row["step"]][row["case_id"]] = row
    reference = indexed[1]
    if not reference:
        raise ValueError("Each trace requires all four readout steps.")
    for step in STEPS:
        _match_cases(reference, indexed[step])
    return indexed


def _match_cases(reference: dict, candidate: dict) -> None:
    if set(reference) != set(candidate):
        raise ValueError("Case sets differ across arms or readout steps; partial-case intersections are forbidden.")
    for case_id in reference:
        a, b = reference[case_id], candidate[case_id]
        if a["label"] != b["label"]:
            raise ValueError("Pathology labels conflict across arms or steps.")
        if a["image_ids"] != b["image_ids"]:
            raise ValueError("Case view sets differ across arms or readout steps.")


def analyze_runs(run_dirs, *, draws: int = 2000, bootstrap_seed: int = 17) -> dict:
    """Analyze exactly one seed's four completed arms; align by explicit Case ID.

    In addition to development traces, training traces verify identical training
    Cases/views and absence of training/development Case overlap. The output is
    restricted to aggregate metrics and a typed config allowlist.
    """
    draws = _integer(draws, "bootstrap draws", 2)
    bootstrap_seed = _integer(bootstrap_seed, "bootstrap seed")
    directories = [Path(path) for path in run_dirs]
    if len(directories) != 4 or len({path.resolve() for path in directories}) != 4:
        raise ValueError("Supply four distinct run directories, one per arm at the same seed.")
    summaries, development, training = {}, {}, {}
    reference_config = None
    reference_sources = None
    reference_training_counts = None
    for directory in directories:
        summary = _summary(directory / "summary.json")
        config = summary["config"]
        arm, seed = config["arm"], config["seed"]
        if arm in summaries:
            raise ValueError("Each of the four arms must appear exactly once.")
        common_config = {key: value for key, value in config.items() if key != "arm"}
        sources = (summary["encoder_sha256"], summary["manifest_sha256"])
        training_counts = (summary["n_training_cases"], summary["n_training_images"])
        if reference_config is None:
            reference_config, reference_sources = common_config, sources
            reference_training_counts = training_counts
        elif common_config != reference_config or sources != reference_sources or training_counts != reference_training_counts:
            raise ValueError("Source configs, encoder/manifest digests, and training counts must match across arms.")
        summaries[arm] = summary
        development[arm] = _index(_read_predictions(directory / "development_predictions.csv"),
                                  arm, seed, require_dice=True)
        training[arm] = _index(_read_predictions(directory / "training_predictions.csv"),
                              arm, seed, require_dice=False)
        if len(training[arm][1]) != summary["n_training_cases"]:
            raise ValueError("Training Case count disagrees with the source summary.")
        if sum(row["n_images"] for row in training[arm][1].values()) != summary["n_training_images"]:
            raise ValueError("Training image count disagrees with the source summary.")
        if set(training[arm][1]) & set(development[arm][1]):
            raise ValueError("Training and development Case sets overlap.")
        declared_development = summary.get("development_subset")
        if not isinstance(declared_development, dict) or set(declared_development) != {str(step) for step in STEPS}:
            raise ValueError("The source summary must declare development evaluation at all four steps.")
        if not all(isinstance(declared_development[str(step)], dict) for step in STEPS):
            raise ValueError("Development summary entries must be metric mappings.")
        if any(declared_development[str(step)].get("n_cases") != len(development[arm][step]) for step in STEPS):
            raise ValueError("Development Case count disagrees with the source summary.")
    if set(summaries) != set(ARMS):
        raise ValueError("All four arms SC, SJ, UC, UJ are required.")
    reference_development, reference_training = development["SC"][1], training["SC"][1]
    for arm in ARMS:
        _match_cases(reference_training, training[arm][1])
        for step in STEPS:
            _match_cases(reference_development, development[arm][step])

    # Sorting identifiers is only an internal alignment operation; IDs are never
    # included in the returned report. Every arm/step uses this same ordering.
    case_ids = sorted(reference_development)
    labels = np.array([reference_development[case]["label"] for case in case_ids], dtype=np.float64)
    diagnostic_changes, segmentation_changes, arm_results = {}, {}, {}
    for arm in ARMS:
        probabilities = {step: np.array([development[arm][step][case]["probability"] for case in case_ids])
                         for step in STEPS}
        dices = {step: np.array([development[arm][step][case]["dice"] for case in case_ids]) for step in STEPS}
        diagnostic_changes[arm] = ((probabilities[2] - labels) ** 2
                                   - (probabilities[4] - labels) ** 2)
        segmentation_changes[arm] = dices[4] - dices[2]
        by_step = {}
        for step in STEPS:
            metrics = classification_metrics(labels, probabilities[step], threshold=0.5)
            metrics["mean_case_dice"] = float(dices[step].mean())
            metrics["dice_space"] = "224x224_letterbox_valid_pixels"
            by_step[str(step)] = metrics
        arm_results[arm] = {
            "D": float(diagnostic_changes[arm].mean()),
            "S": float(segmentation_changes[arm].mean()),
            "by_step": by_step,
        }
    theta_per_case = ((diagnostic_changes["SJ"] - diagnostic_changes["SC"])
                      - (diagnostic_changes["UJ"] - diagnostic_changes["UC"]))
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 256):
        stop = min(draws, start + 256)
        indices = rng.integers(0, labels.size, size=(stop - start, labels.size))
        bootstrap_means[start:stop] = theta_per_case[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])

    return {
        "status": "executed_cpu_engineering_analysis",
        "purpose": "four-arm development-subset software acceptance",
        "clinical_validation": False,
        "mechanism_identified": False,
        "patient_mapping": "unverified_Case_grouped_only",
        "evaluation_split": "tune_development",
        "config": {key: reference_config[key] for key in SAFE_CONFIG_FIELDS},
        "sources": {"encoder_sha256": reference_sources[0], "manifest_sha256": reference_sources[1]},
        "n_training_cases": reference_training_counts[0],
        "n_training_images": reference_training_counts[1],
        "n_development_cases": int(labels.size),
        "n_development_images": sum(row["n_images"] for row in reference_development.values()),
        "n_positive": int(labels.sum()), "n_negative": int(labels.size - labels.sum()),
        "alignment_checks": {
            "same_training_case_and_view_sets": True,
            "same_development_case_and_view_sets_all_arms_and_steps": True,
            "consistent_pathology_labels": True,
            "no_training_development_case_overlap": True,
            "matching_training_config_encoder_and_manifest": True,
            "all_four_arms_and_steps_present": True,
        },
        "definitions": {
            "case_probability": "arithmetic mean of all evaluated image probabilities",
            "case_dice": "arithmetic mean of evaluated image Dice values",
            "D": "Brier_step2 - Brier_step4; positive means lower step4 Brier",
            "S": "Dice_step4 - Dice_step2; positive means higher step4 Dice",
        },
        "arms": arm_results,
        "theta_D": {
            "definition": "(D_SJ - D_SC) - (D_UJ - D_UC)",
            "estimate": float(theta_per_case.mean()), "ci95": [float(low), float(high)],
            "bootstrap": {"unit": "Case", "draws": draws, "seed": bootstrap_seed,
                          "paired_across_arms_and_steps": True, "method": "percentile"},
        },
        "limitations": [
            "This tiny development-subset analysis is an engineering acceptance check.",
            "Its descriptive contrast does not establish diagnostic benefit or a clinical mechanism.",
            "The Case bootstrap conditions on these fitted models and does not cover training-seed variability.",
            "Case-to-patient mapping is unverified, so patient independence is not established.",
            "Dice uses resized valid pixels rather than original-resolution boundary evaluation.",
        ],
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dirs", nargs="+", required=True)
    p.add_argument("--output", default="outputs/pilot_analysis.json")
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=17)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    destination = Path(args.output)
    if destination.exists():
        p.error("Analysis output already exists; choose a fresh output path.")
    try:
        report = analyze_runs(args.run_dirs, draws=args.draws, bootstrap_seed=args.bootstrap_seed)
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
