"""Run nested 256-Case terminal-depth fits at the earlier 3,200-update budget."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from .cohort import select_training_rows, training_source_fingerprint
from .data import load_manifest
from .download import file_digest
from .expanded import write_once_or_match
from .feature_cache import load_or_encode
from .models import create_encoder, make_model
from .seed_replication import counts
from .train import initialization_fingerprint, parser as train_parser, run as train_run


SEEDS = (17, 29, 43)
DEPTHS = (2, 4)
MATCHED_UPDATES = (640, 1280, 1920, 2560, 3200)


def assigned_runs(workers=1, worker_index=0):
    if workers not in (1, 2) or worker_index not in range(workers):
        raise ValueError("Use one/two workers and a valid worker index.")
    pairs = [(seed, depth) for seed in SEEDS for depth in DEPTHS]
    return [pair for index, pair in enumerate(pairs) if index % workers == worker_index]


def launcher_digest():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def nested_check(reference, expanded):
    small = {(str(r["case_id"]), str(r["image_id"])): int(r["label"]) for r in reference}
    large = {(str(r["case_id"]), str(r["image_id"])): int(r["label"]) for r in expanded}
    case_subset = {key[0] for key in small} <= {key[0] for key in large}
    image_subset = set(small) <= set(large)
    labels_match = image_subset and all(large[key] == value for key, value in small.items())
    result = {"reference_cases_subset": case_subset, "reference_images_subset": image_subset,
              "shared_view_labels_match": labels_match}
    if not all(result.values()):
        raise ValueError("The 256-Case cohort must contain every earlier Case/view with identical labels.")
    return result


def build_protocol(args):
    torch.set_num_threads(args.threads)
    path = Path(args.reference_protocol)
    old = json.loads(path.read_text())
    if (old.get("status") != "fixed_before_depth_supervision_training"
            or old.get("seeds") != list(SEEDS) or old.get("depths") != list(DEPTHS)
            or old.get("supervisions") != ["all", "terminal"] or old.get("arms") != ["SJ"]
            or old.get("clinical_validation") is not False
            or old.get("calibration_and_test_predictions") is not False):
        raise ValueError("Use the locked earlier three-seed depth/supervision protocol.")
    expected = {"device": "cpu", "epochs": 50, "batch_size": 2,
                "learning_rate": .0003, "segmentation_weight": 1., "selection_seed": 20261004,
                "sampling": "proportional", "augment": False, "cached_frozen_features": True,
                "eval_every": 5, "training_updates_per_run": 3200, "warm_starts": False,
                "model_selection": "fixed_final_epoch_no_best_epoch_selection",
                "patient_mapping": "unverified_Case_grouped_only"}
    if any(old.get(key) != value or type(old.get(key)) is not type(value) for key, value in expected.items()):
        raise ValueError("Reference recipe differs from the fixed optimization schedule.")
    rows = load_manifest(args.manifest)
    reference = select_training_rows(rows, 128, sampling="proportional", selection_seed=20261004)
    train = select_training_rows(rows, 256, sampling="proportional", selection_seed=20261004)
    tune = [r for r in rows if r["split"] == "tune"]
    if counts(reference) != old["train"] or counts(tune) != old["tune"]:
        raise ValueError("The earlier reference training and common tune cohort must match exactly.")
    if counts(train)["cases"] != 256 or counts(tune)["cases"] != 159:
        raise ValueError("Expected 256 training and 159 development Cases.")
    nesting = nested_check(reference, train)
    sources = {"encoder_sha256": file_digest(args.encoder_weights),
               "manifest_sha256": file_digest(args.manifest),
               "training_source_sha256": training_source_fingerprint()}
    if any(old.get(key) != value for key, value in sources.items()):
        raise ValueError("Training source, encoder and manifest must retain the exact earlier recipe.")
    initials = {}
    for seed in SEEDS:
        digests = []
        for depth in DEPTHS:
            torch.manual_seed(seed)
            digests.append(initialization_fingerprint(make_model("SJ", steps=depth)))
        if len(set(digests)) != 1 or digests[0] != old["initialization_sha256_by_seed"][str(seed)]:
            raise ValueError("Initial parameters must match both depths and the earlier 128-Case fits.")
        initials[str(seed)] = digests[0]
    protocol = {**old, **sources,
                "status": "fixed_before_case_expansion_training",
                "phase": "exploratory_nested_training_case_expansion_on_observed_tune",
                "threads": args.threads, "supervisions": ["terminal"], "epochs": 25,
                "reference_epochs": 50, "updates_per_epoch": 128, "reference_updates_per_epoch": 64,
                "matched_update_points": list(MATCHED_UPDATES),
                "reference_train": counts(reference), "train": counts(train),
                "nested_cohort_verified": True, "nested_cohort_check": nesting,
                "reference_protocol_sha256": file_digest(path),
                "launcher_source_sha256": launcher_digest(),
                "initialization_sha256_by_seed": initials,
                "training_exposures_per_case": 25, "reference_exposures_per_case": 50,
                "total_training_case_exposures_per_run": 6400,
                "primary_depth_benefit": "G256 = Brier_trained2_endpoint - Brier_trained4_endpoint on the common tune Cases",
                "secondary_data_size_depth_interaction": "G256 - G128 using the same three seeds and evaluated Cases",
                "supervision_loss": {"terminal": old["supervision_loss"]["terminal"]},
                "budget": "Both data sizes complete 3,200 optimizer updates and 6,400 Case exposures; expanded data trades repetition for diversity. Depth4 uses extra recurrent compute; not equal FLOPs.",
                "paired_initialization_scope": "Identical within seed across depths and training sizes; Case order/view sampling match within each size, not across sizes.",
                "secondary_metrics": ["AUROC", "NLL", "mean_Case_Dice", "train_tune_Brier_gap", "high_confidence_errors", "matched_update_learning"],
                "classification_controls": "Training-only constant prevalence and standardized mean frozen features with fixed C=1 logistic regression, refitted on 256 Cases; no hyperparameter selection.",
                "interpretation": "Descriptive fixed-budget data-diversity comparison on observed tune Cases; no new hard gate or automatic larger/GPU experiment."}
    protocol.pop("primary_interaction", None)
    return protocol, train, tune


def expected_config(seed, depth):
    return {"arm": "SJ", "steps": depth, "seed": seed, "epochs": 25, "batch_size": 2,
            "learning_rate": .0003, "segmentation_weight": 1., "cached_frozen_features": True,
            "augment": False, "selection_seed": 20261004, "sampling": "proportional",
            "eval_every": 5, "supervision": "terminal"}


def validate_completed(summary, protocol, seed, depth):
    if (summary.get("status") != "executed_cpu_engineering_pilot"
            or summary.get("clinical_validation") is not False or summary.get("device") != "cpu"
            or summary.get("evaluation_split") != "tune_development"
            or summary.get("model_selection") != "fixed_final_epoch_no_best_epoch_selection"
            or summary.get("config") != expected_config(seed, depth)
            or summary.get("initialization_sha256") != protocol["initialization_sha256_by_seed"][str(seed)]
            or summary.get("timing", {}).get("updates") != 3200
            or summary.get("n_training_cases") != 256
            or summary.get("n_training_images") != protocol["train"]["images"]):
        raise ValueError("Completed fit differs from the fixed 256-Case initialization/config/update budget.")
    for key in ("training_source_sha256", "encoder_sha256", "manifest_sha256"):
        if summary.get(key) != protocol[key]:
            raise ValueError("Completed fit source differs from the locked protocol.")
    for split in ("train", "tune"):
        if summary.get("cohort_selection", {}).get(split + "_fingerprint") != protocol[split]["fingerprint"]:
            raise ValueError("Completed fit cohort differs from the locked protocol.")
    return summary


def run(args):
    begin = time.perf_counter()
    if args.threads not in (1, 2):
        raise ValueError("Use one/two CPU threads per worker.")
    pairs = assigned_runs(args.workers, args.worker_index)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    protocol, train, tune = build_protocol(args)
    write_once_or_match(root / "protocol.json", protocol)
    if args.prepare_only:
        encoder = create_encoder(pretrained=True, weights_path=args.encoder_weights)
        train_features, _, _ = load_or_encode(encoder, train, args.encoder_weights, args.feature_cache_dir, 2)
        tune_features, _, _ = load_or_encode(encoder, tune, args.encoder_weights, args.feature_cache_dir, 2)
        from .baselines import run_baselines
        baseline_dir = root / "baselines"
        baseline_path = baseline_dir / "baselines_summary.json"
        sources = {key: protocol[key] for key in ("encoder_sha256", "manifest_sha256")}
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text())
            if (baseline.get("sources") != sources
                    or baseline.get("cohort_fingerprints") != {split: protocol[split]["fingerprint"] for split in ("train", "tune")}
                    or not (baseline_dir / "baseline_case_predictions.csv").is_file()):
                raise ValueError("Existing classification controls differ from the expanded cohort/source.")
        else:
            baseline = run_baselines(train, tune, train_features, tune_features, baseline_dir, selection_seed=20261004)
            baseline["sources"] = sources
            baseline_path.write_text(json.dumps(baseline, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"event": "expanded_baselines_ready", "baselines": baseline["baselines"]}), flush=True)
        print(json.dumps({"event": "protocol_and_cache_ready", "train": protocol["train"],
                          "tune": protocol["tune"], "nested": protocol["nested_cohort_check"]}), flush=True)
        return protocol
    completed = []
    for seed, depth in pairs:
        if (training_source_fingerprint() != protocol["training_source_sha256"]
                or launcher_digest() != protocol["launcher_source_sha256"]):
            raise ValueError("Source changed after locking the Case-expansion protocol.")
        directory = root / f"SJ-depth{depth}-terminal-seed{seed}"
        if (directory / "summary.json").exists():
            summary = validate_completed(json.loads((directory / "summary.json").read_text()), protocol, seed, depth)
        else:
            directory.mkdir(exist_ok=True)
            with (directory / "run_claim.json").open("x") as stream:
                json.dump({"worker": args.worker_index, "seed": seed, "depth": depth,
                           "supervision": "terminal", "training_cases": 256}, stream)
            command = ["--manifest", args.manifest, "--encoder-weights", args.encoder_weights,
                       "--run-dir", str(directory), "--arm", "SJ", "--steps", str(depth),
                       "--supervision", "terminal", "--seed", str(seed), "--selection-seed", "20261004",
                       "--sampling", "proportional", "--max-cases", "256", "--epochs", "25",
                       "--batch-size", "2", "--learning-rate", "0.0003", "--threads", str(args.threads),
                       "--cache-features", "--feature-cache-dir", args.feature_cache_dir,
                       "--eval-max-cases", "159", "--eval-every", "5"]
            print(json.dumps({"event": "run_start", "seed": seed, "depth": depth}), flush=True)
            summary = validate_completed(train_run(train_parser().parse_args(command)), protocol, seed, depth)
        completed.append({"seed": seed, "depth": depth, "training_seconds": summary["training_seconds"],
                          "endpoint": summary["development_subset"][str(depth)]})
        progress = {"status": "worker_complete" if len(completed) == len(pairs) else "running",
                    "workers": args.workers, "worker_index": args.worker_index, "completed_runs": completed,
                    "elapsed_seconds": time.perf_counter() - begin}
        (root / f"progress-worker{args.worker_index}.json").write_text(json.dumps(progress, indent=2) + "\n")
        print(json.dumps({"event": "run_complete", **completed[-1], "worker_completed": len(completed)}), flush=True)
    return progress


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--encoder-weights", default="data/deit_tiny_encoder.pt")
    p.add_argument("--reference-protocol", default="outputs/cpu-depth-supervision128/protocol.json")
    p.add_argument("--feature-cache-dir", default="outputs/cpu-expanded128/features")
    p.add_argument("--output-dir", default="outputs/cpu-case-expansion256")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--workers", type=int, choices=[1, 2], default=1)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--prepare-only", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
