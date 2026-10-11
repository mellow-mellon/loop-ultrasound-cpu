"""Run the prespecified five-new-seed, terminal-only paired-depth CPU check."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from .cohort import cohort_fingerprint, select_training_rows, training_source_fingerprint
from .data import load_manifest
from .depth_supervision import validate_completed
from .download import file_digest
from .expanded import write_once_or_match
from .feature_cache import load_or_encode
from .models import create_encoder, make_model
from .train import initialization_fingerprint, parser as train_parser, run as train_run


SEEDS = (53, 71, 89, 107, 131)
REFERENCE_SEEDS = (17, 29, 43)
DEPTHS = (2, 4)
SCREENING_GATE = {
    "minimum_positive_seeds": 4,
    "require_all_leave_one_seed_out_means_positive": True,
    "maximum_mean_nll_worsening": 0.0,
    "definition": "All three conditions on NEW five seeds; engineering resource screen only, not significance, clinical validation, or an automatic run authorization.",
}


def assigned_runs(workers=1, worker_index=0):
    if workers not in (1, 2) or worker_index not in range(workers):
        raise ValueError("Use one/two workers and a valid worker index.")
    pairs = [(seed, depth) for seed in SEEDS for depth in DEPTHS]
    return [pair for index, pair in enumerate(pairs) if index % workers == worker_index]


def launcher_digest():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def counts(rows):
    labels = {row["case_id"]: int(row["label"]) for row in rows}
    return {"cases": len(labels), "images": len(rows),
            "benign": sum(label == 0 for label in labels.values()),
            "malignant": sum(labels.values()), "fingerprint": cohort_fingerprint(rows)}


def build_protocol(args):
    torch.set_num_threads(args.threads)
    reference_path = Path(args.reference_protocol)
    old = json.loads(reference_path.read_text())
    if (old.get("status") != "fixed_before_depth_supervision_training"
            or old.get("seeds") != list(REFERENCE_SEEDS)
            or old.get("depths") != list(DEPTHS)
            or old.get("supervisions") != ["all", "terminal"]
            or old.get("arms") != ["SJ"]
            or old.get("clinical_validation") is not False
            or old.get("calibration_and_test_predictions") is not False):
        raise ValueError("Use the locked earlier depth/supervision protocol as reference.")
    expected = {"device": "cpu", "epochs": 50, "batch_size": 2,
                "learning_rate": .0003, "segmentation_weight": 1.,
                "selection_seed": 20261004, "sampling": "proportional",
                "augment": False, "cached_frozen_features": True, "eval_every": 5,
                "training_updates_per_run": 3200, "warm_starts": False,
                "model_selection": "fixed_final_epoch_no_best_epoch_selection",
                "patient_mapping": "unverified_Case_grouped_only"}
    if any(old.get(key) != value or type(old.get(key)) is not type(value)
           for key, value in expected.items()):
        raise ValueError("Reference schedule or scope differs from the frozen recipe.")
    rows = load_manifest(args.manifest)
    train = select_training_rows(rows, 128, sampling="proportional", selection_seed=20261004)
    tune = [row for row in rows if row["split"] == "tune"]
    if counts(train) != old["train"] or counts(tune) != old["tune"]:
        raise ValueError("New seeds must retain exactly the earlier train/tune cohort.")
    if counts(train)["cases"] != 128 or counts(tune)["cases"] != 159:
        raise ValueError("Expected 128 training and 159 development Cases.")
    sources = {"encoder_sha256": file_digest(args.encoder_weights),
               "manifest_sha256": file_digest(args.manifest),
               "training_source_sha256": training_source_fingerprint()}
    if any(old.get(key) != value for key, value in sources.items()):
        raise ValueError("Training code, encoder and manifest must match the earlier recipe.")
    initials = {}
    for seed in SEEDS:
        digests = []
        for depth in DEPTHS:
            torch.manual_seed(seed)
            digests.append(initialization_fingerprint(make_model("SJ", steps=depth)))
        if len(set(digests)) != 1:
            raise ValueError("Paired shared-depth initial parameters differ.")
        initials[str(seed)] = digests[0]
    protocol = {**old, **sources,
                "status": "fixed_before_seed_replication_training",
                "phase": "exploratory_training_randomness_replication_on_observed_tune",
                "threads": args.threads, "seeds": list(SEEDS),
                "reference_seeds": list(REFERENCE_SEEDS), "supervisions": ["terminal"],
                "reference_protocol_sha256": file_digest(reference_path),
                "launcher_source_sha256": launcher_digest(),
                "initialization_sha256_by_seed": initials,
                "screening_gate": dict(SCREENING_GATE),
                "primary_depth_benefit": "Brier_trained2_endpoint - Brier_trained4_endpoint, NEW five terminal-supervision seeds only",
                "reference_results_scope": "Earlier three terminal seeds are secondary descriptive evidence; do not include them in the primary gate.",
                "supervision_loss": {"terminal": old["supervision_loss"]["terminal"]},
                "secondary_metrics": ["AUROC", "NLL", "mean_Case_Dice", "high_confidence_errors", "paired_case_corrections_and_deteriorations"],
                "interpretation": "Training-randomness robustness on already observed tune Cases; positive gate only recommends the separate Case-expansion experiment."}
    protocol.pop("primary_interaction", None)
    return protocol, train, tune


def run(args):
    begin = time.perf_counter()
    if args.threads not in (1, 2):
        raise ValueError("Only one/two CPU threads per worker.")
    pairs = assigned_runs(args.workers, args.worker_index)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    protocol, train, tune = build_protocol(args)
    write_once_or_match(root / "protocol.json", protocol)
    if args.prepare_only:
        encoder = create_encoder(pretrained=True, weights_path=args.encoder_weights)
        for rows in (train, tune):
            load_or_encode(encoder, rows, args.encoder_weights, args.feature_cache_dir, 2)
        print(json.dumps({"event": "protocol_and_cache_ready", "seeds": list(SEEDS),
                          "train": protocol["train"], "tune": protocol["tune"]}), flush=True)
        return protocol
    completed = []
    for seed, depth in pairs:
        if (training_source_fingerprint() != protocol["training_source_sha256"]
                or launcher_digest() != protocol["launcher_source_sha256"]):
            raise ValueError("Source changed after locking the replication protocol.")
        directory = root / f"SJ-depth{depth}-terminal-seed{seed}"
        if (directory / "summary.json").exists():
            summary = validate_completed(json.loads((directory / "summary.json").read_text()),
                                         protocol, seed, depth, "terminal")
        else:
            directory.mkdir(exist_ok=True)
            with (directory / "run_claim.json").open("x") as stream:
                json.dump({"worker": args.worker_index, "seed": seed, "depth": depth,
                           "supervision": "terminal"}, stream)
            command = ["--manifest", args.manifest, "--encoder-weights", args.encoder_weights,
                       "--run-dir", str(directory), "--arm", "SJ", "--steps", str(depth),
                       "--supervision", "terminal", "--seed", str(seed),
                       "--selection-seed", "20261004", "--sampling", "proportional",
                       "--max-cases", "128", "--epochs", "50", "--batch-size", "2",
                       "--learning-rate", "0.0003", "--threads", str(args.threads),
                       "--cache-features", "--feature-cache-dir", args.feature_cache_dir,
                       "--eval-max-cases", "159", "--eval-every", "5"]
            print(json.dumps({"event": "run_start", "seed": seed, "depth": depth}), flush=True)
            summary = validate_completed(train_run(train_parser().parse_args(command)),
                                         protocol, seed, depth, "terminal")
        completed.append({"seed": seed, "depth": depth,
                          "training_seconds": summary["training_seconds"],
                          "endpoint": summary["development_subset"][str(depth)]})
        progress = {"status": "worker_complete" if len(completed) == len(pairs) else "running",
                    "workers": args.workers, "worker_index": args.worker_index,
                    "completed_runs": completed, "elapsed_seconds": time.perf_counter() - begin}
        (root / f"progress-worker{args.worker_index}.json").write_text(json.dumps(progress, indent=2) + "\n")
        print(json.dumps({"event": "run_complete", **completed[-1],
                          "worker_completed": len(completed)}), flush=True)
    return progress


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--encoder-weights", default="data/deit_tiny_encoder.pt")
    p.add_argument("--reference-protocol", default="outputs/cpu-depth-supervision128/protocol.json")
    p.add_argument("--feature-cache-dir", default="outputs/cpu-expanded128/features")
    p.add_argument("--output-dir", default="outputs/cpu-seed-replication128")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--workers", type=int, choices=[1, 2], default=1)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--prepare-only", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
