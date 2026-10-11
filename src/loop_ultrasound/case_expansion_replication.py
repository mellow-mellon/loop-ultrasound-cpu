"""Replicate the fixed 256-Case depth comparison at five matched new seeds."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from . import case_expansion
from .cohort import select_training_rows, training_source_fingerprint
from .data import load_manifest
from .depth_supervision import validate_completed as validate_reference
from .download import file_digest
from .expanded import write_once_or_match
from .feature_cache import load_or_encode
from .models import create_encoder, make_model
from .seed_replication import counts
from .train import initialization_fingerprint, parser as train_parser, run as train_run


SEEDS = (53, 71, 89, 107, 131)
DEPTHS = (2, 4)


def assigned_runs(workers=1, worker_index=0):
    if workers not in (1, 2) or worker_index not in range(workers):
        raise ValueError("Use one/two workers and a valid worker index.")
    pairs = [(seed, depth) for seed in SEEDS for depth in DEPTHS]
    return [pair for index, pair in enumerate(pairs) if index % workers == worker_index]


def launcher_digest():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _baseline_binding(directory, protocol):
    directory = Path(directory)
    path = directory / "baselines_summary.json"
    summary = json.loads(path.read_text())
    if (summary.get("sources") != {key: protocol[key] for key in ("encoder_sha256", "manifest_sha256")}
            or summary.get("cohort_fingerprints") != {split: protocol[split]["fingerprint"] for split in ("train", "tune")}):
        raise ValueError("Existing fixed baseline cohort/source differs from the locked recipe.")
    return {"summary_sha256": file_digest(path),
            "predictions_sha256": file_digest(directory / "baseline_case_predictions.csv")}


def build_protocol(args):
    torch.set_num_threads(args.threads)
    reference_path, original_path = Path(args.reference_protocol), Path(args.original_expansion_protocol)
    old, original = json.loads(reference_path.read_text()), json.loads(original_path.read_text())
    if (old.get("status") != "fixed_before_seed_replication_training"
            or old.get("seeds") != list(SEEDS) or old.get("depths") != list(DEPTHS)
            or old.get("supervisions") != ["terminal"] or old.get("arms") != ["SJ"]):
        raise ValueError("Use the locked five-seed 128-Case replication protocol.")
    if (original.get("status") != "fixed_before_case_expansion_training"
            or original.get("seeds") != [17, 29, 43] or original.get("depths") != list(DEPTHS)
            or original.get("supervisions") != ["terminal"] or original.get("arms") != ["SJ"]):
        raise ValueError("Use the locked original three-seed 256-Case expansion protocol.")
    shared = {"device": "cpu", "batch_size": 2, "learning_rate": .0003,
              "segmentation_weight": 1., "selection_seed": 20261004, "sampling": "proportional",
              "augment": False, "cached_frozen_features": True, "eval_every": 5,
              "training_updates_per_run": 3200, "warm_starts": False,
              "model_selection": "fixed_final_epoch_no_best_epoch_selection",
              "patient_mapping": "unverified_Case_grouped_only", "clinical_validation": False,
              "calibration_and_test_predictions": False}
    for record, epochs in ((old, 50), (original, 25)):
        expected = {**shared, "epochs": epochs}
        if any(record.get(key) != value or type(record.get(key)) is not type(value) for key, value in expected.items()):
            raise ValueError("Reference optimization schedule or scope differs from the frozen recipe.")
    rows = load_manifest(args.manifest)
    reference = select_training_rows(rows, 128, sampling="proportional", selection_seed=20261004)
    train = select_training_rows(rows, 256, sampling="proportional", selection_seed=20261004)
    tune = [row for row in rows if row["split"] == "tune"]
    if (counts(reference) != old["train"] or counts(reference) != original["reference_train"]
            or counts(train) != original["train"] or counts(tune) != old["tune"]
            or counts(tune) != original["tune"] or counts(train)["cases"] != 256
            or counts(tune)["cases"] != 159):
        raise ValueError("All 128/256/tune cohorts must match their earlier locked definitions.")
    nesting = case_expansion.nested_check(reference, train)
    sources = {"encoder_sha256": file_digest(args.encoder_weights),
               "manifest_sha256": file_digest(args.manifest),
               "training_source_sha256": training_source_fingerprint()}
    if any(record.get(key) != value for record in (old, original) for key, value in sources.items()):
        raise ValueError("Training source, encoder and manifest must match both earlier experiments.")
    initials = {}
    for seed in SEEDS:
        digests = []
        for depth in DEPTHS:
            summary = json.loads((Path(args.reference_run_dir) / f"SJ-depth{depth}-terminal-seed{seed}" / "summary.json").read_text())
            validate_reference(summary, old, seed, depth, "terminal")
            if summary.get("n_training_cases") != 128 or summary.get("n_training_images") != old["train"]["images"]:
                raise ValueError("Reference fit training size differs from its locked cohort.")
            torch.manual_seed(seed)
            digests.append(initialization_fingerprint(make_model("SJ", steps=depth)))
        if len(set(digests)) != 1 or digests[0] != old["initialization_sha256_by_seed"][str(seed)]:
            raise ValueError("Initial parameters must match paired depths and each existing 128-Case fit.")
        initials[str(seed)] = digests[0]
    protocol = {**original, **sources,
                "status": "fixed_before_case_expansion_seed_replication_training",
                "phase": "exploratory_nested_case_expansion_training_randomness_replication_on_observed_tune",
                "threads": args.threads, "seeds": list(SEEDS), "reference_seeds": list(SEEDS),
                "initialization_sha256_by_seed": initials,
                "reference_protocol_sha256": file_digest(reference_path),
                "original_expansion_protocol_sha256": file_digest(original_path),
                "launcher_source_sha256": launcher_digest(),
                "shared_runner_source_sha256": case_expansion.launcher_digest(),
                "nested_cohort_check": nesting,
                "baseline_artifact_bindings": {
                    "new256": _baseline_binding(args.baseline_dir, original),
                    "reference128": _baseline_binding(args.reference_baseline_dir, old)},
                "historical_seed_scope": "original_three_secondary_only",
                "primary_depth_benefit": "G256 = Brier_trained2_endpoint - Brier_trained4_endpoint, NEW five matched seeds only",
                "secondary_data_size_depth_interaction": "G256 - G128 using the same NEW five seeds and evaluated Cases",
                "secondary_combined_scope": "All eight matched seeds may be summarized separately as descriptive secondary evidence; cannot replace the primary five-seed result.",
                "classification_controls": "Reuse the already refitted fixed 256/128 train-only controls with identical cohorts and byte-bound artifacts; no new baseline tuning.",
                "interpretation": "Five additional fitted seeds on observed tune; no new hard gate or automatic larger/GPU experiment."}
    return protocol, train, tune


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
        reused = []
        for rows in (train, tune):
            _, _, cached = load_or_encode(encoder, rows, args.encoder_weights, args.feature_cache_dir, 2)
            reused.append(cached)
        print(json.dumps({"event": "protocol_and_cache_ready", "seeds": list(SEEDS),
                          "train": protocol["train"], "tune": protocol["tune"],
                          "train_tune_cache_reused": reused}), flush=True)
        return protocol
    completed = []
    for seed, depth in pairs:
        if (training_source_fingerprint() != protocol["training_source_sha256"]
                or launcher_digest() != protocol["launcher_source_sha256"]
                or case_expansion.launcher_digest() != protocol["shared_runner_source_sha256"]):
            raise ValueError("Source changed after locking the 256-Case replication protocol.")
        directory = root / f"SJ-depth{depth}-terminal-seed{seed}"
        if (directory / "summary.json").exists():
            summary = case_expansion.validate_completed(json.loads((directory / "summary.json").read_text()), protocol, seed, depth)
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
            summary = case_expansion.validate_completed(train_run(train_parser().parse_args(command)), protocol, seed, depth)
        completed.append({"seed": seed, "depth": depth, "training_seconds": summary["training_seconds"],
                          "endpoint": summary["development_subset"][str(depth)]})
        progress = {"status": "worker_complete" if len(completed) == len(pairs) else "running",
                    "workers": args.workers, "worker_index": args.worker_index, "completed_runs": completed,
                    "elapsed_seconds": time.perf_counter() - begin}
        (root / f"progress-worker{args.worker_index}.json").write_text(json.dumps(progress, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"event": "run_complete", **completed[-1], "worker_completed": len(completed)}), flush=True)
    return progress


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--encoder-weights", default="data/deit_tiny_encoder.pt")
    p.add_argument("--reference-protocol", default="outputs/cpu-seed-replication128/protocol.json")
    p.add_argument("--reference-run-dir", default="outputs/cpu-seed-replication128")
    p.add_argument("--original-expansion-protocol", default="outputs/cpu-case-expansion256/protocol.json")
    p.add_argument("--baseline-dir", default="outputs/cpu-case-expansion256/baselines")
    p.add_argument("--reference-baseline-dir", default="outputs/cpu-expanded128/baselines")
    p.add_argument("--feature-cache-dir", default="outputs/cpu-expanded128/features")
    p.add_argument("--output-dir", default="outputs/cpu-case-expansion256-replication")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--workers", type=int, choices=[1, 2], default=1)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--prepare-only", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
