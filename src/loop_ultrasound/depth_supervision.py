"""Lock and run the shared+joint depth x supervision CPU follow-up."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from .cohort import select_training_rows, cohort_fingerprint, training_source_fingerprint
from .data import load_manifest
from .download import file_digest
from .expanded import write_once_or_match
from .feature_cache import load_or_encode
from .models import create_encoder, make_model
from .train import initialization_fingerprint, parser as train_parser, run as train_run


SEEDS = (17, 29, 43)
DEPTHS = (2, 4)
SUPERVISIONS = ("all", "terminal")


def assigned_runs(workers=1, worker_index=0):
    if workers not in (1, 2) or worker_index not in range(workers):
        raise ValueError("Use one/two workers and a valid worker index.")
    combos = [(seed, depth, mode) for seed in SEEDS for depth in DEPTHS for mode in SUPERVISIONS]
    return [combo for i, combo in enumerate(combos) if i % workers == worker_index]


def launcher_digest():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def build_protocol(args):
    torch.set_num_threads(args.threads)
    rows = load_manifest(args.manifest)
    train = select_training_rows(rows, 128, sampling="proportional", selection_seed=20261004)
    tune = [r for r in rows if r["split"] == "tune"]
    def counts(selected):
        labels = {r["case_id"]: int(r["label"]) for r in selected}
        return {"cases": len(labels), "images": len(selected), "benign": sum(x == 0 for x in labels.values()),
                "malignant": sum(labels.values()), "fingerprint": cohort_fingerprint(selected)}
    initials = {}
    for seed in SEEDS:
        digests = []
        for depth in DEPTHS:
            torch.manual_seed(seed)
            digests.append(initialization_fingerprint(make_model("SJ", steps=depth)))
        if len(set(digests)) != 1:
            raise ValueError("Initial parameters must match across shared trained depths.")
        initials[str(seed)] = digests[0]
    old = json.loads(Path(args.reference_protocol).read_text())
    if counts(train) != old["train"] or counts(tune) != old["tune"]:
        raise ValueError("Follow-up must retain the exact previous train/tune cohort.")
    encoder_hash, manifest_hash = file_digest(args.encoder_weights), file_digest(args.manifest)
    if encoder_hash != old["encoder_sha256"] or manifest_hash != old["manifest_sha256"]:
        raise ValueError("Follow-up encoder/manifest must match previous experiment.")
    protocol = {
        "status": "fixed_before_depth_supervision_training",
        "clinical_validation": False, "patient_mapping": "unverified_Case_grouped_only",
        "phase": "exploratory_followup_on_previously_observed_tune",
        "device": "cpu", "threads": args.threads, "arms": ["SJ"],
        "depths": list(DEPTHS), "supervisions": list(SUPERVISIONS), "seeds": list(SEEDS),
        "selection_seed": 20261004, "sampling": "proportional",
        "epochs": 50, "batch_size": 2, "learning_rate": .0003, "segmentation_weight": 1.,
        "cached_frozen_features": True, "augment": False, "eval_every": 5,
        "model_selection": "fixed_final_epoch_no_best_epoch_selection",
        "training_updates_per_run": 3200, "warm_starts": False,
        "encoder_sha256": encoder_hash, "manifest_sha256": manifest_hash,
        "training_source_sha256": training_source_fingerprint(),
        "launcher_source_sha256": launcher_digest(),
        "initialization_sha256_by_seed": initials,
        "train": counts(train), "tune": counts(tune), "calibration_and_test_predictions": False,
        "primary_metric": "Case Brier at the final readout of each separately trained model",
        "primary_depth_benefit": "Brier_trained2_endpoint - Brier_trained4_endpoint, within each supervision mode",
        "primary_interaction": "depth_benefit_terminal - depth_benefit_all",
        "secondary_metrics": ["AUROC", "NLL", "mean_Case_Dice", "within_trained4_h2_to_h4"],
        "supervision_loss": {
            "all": "Mean identical classification+segmentation loss over every configured readout",
            "terminal": "Identical loss only on final readout; no depth divisor; full recurrent backpropagation",
        },
        "budget": "Same optimizer updates and Case exposure; trained4 uses additional compute, not equal FLOPs",
        "intermediate_terminal_predictions": "Diagnostic only; no directly trained intermediate readout",
        "interpretation": "Conditional development screening; no automatic significance, clinical, or GPU gate",
    }
    return protocol, train, tune


def expected_config(seed, depth, mode):
    return {"arm": "SJ", "steps": depth, "seed": seed, "epochs": 50, "batch_size": 2,
            "learning_rate": .0003, "segmentation_weight": 1., "cached_frozen_features": True,
            "augment": False, "selection_seed": 20261004, "sampling": "proportional",
            "eval_every": 5, "supervision": mode}


def validate_completed(summary, protocol, seed, depth, mode):
    if (summary.get("status") != "executed_cpu_engineering_pilot"
            or summary.get("clinical_validation") is not False or summary.get("device") != "cpu"
            or summary.get("model_selection") != "fixed_final_epoch_no_best_epoch_selection"
            or summary.get("evaluation_split") != "tune_development"
            or summary.get("config") != expected_config(seed, depth, mode)
            or summary.get("initialization_sha256") != protocol["initialization_sha256_by_seed"][str(seed)]
            or summary.get("timing", {}).get("updates") != 3200):
        raise ValueError("Completed run differs from the locked initialization/config/update budget.")
    for key in ("training_source_sha256", "encoder_sha256", "manifest_sha256"):
        if summary.get(key) != protocol[key]:
            raise ValueError("Completed run source differs from locked protocol.")
    for split in ("train", "tune"):
        if summary.get("cohort_selection", {}).get(split+"_fingerprint") != protocol[split]["fingerprint"]:
            raise ValueError("Completed run cohort differs.")
    return summary


def run(args):
    begin = time.perf_counter()
    if args.threads not in (1, 2):
        raise ValueError("Only one/two CPU threads per worker.")
    combos = assigned_runs(args.workers, args.worker_index)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    protocol, train, tune = build_protocol(args)
    write_once_or_match(root / "protocol.json", protocol)
    if args.prepare_only:
        encoder = create_encoder(pretrained=True, weights_path=args.encoder_weights)
        for rows in (train, tune):
            load_or_encode(encoder, rows, args.encoder_weights, args.feature_cache_dir, 2)
        print(json.dumps({"event": "protocol_and_cache_ready", "train": protocol["train"], "tune": protocol["tune"]}), flush=True)
        return protocol
    completed = []
    for seed, depth, mode in combos:
        if (training_source_fingerprint() != protocol["training_source_sha256"]
                or launcher_digest() != protocol["launcher_source_sha256"]):
            raise ValueError("Source changed after locking the follow-up.")
        directory = root / f"SJ-depth{depth}-{mode}-seed{seed}"
        if (directory / "summary.json").exists():
            summary = validate_completed(json.loads((directory / "summary.json").read_text()), protocol, seed, depth, mode)
        else:
            directory.mkdir(exist_ok=True)
            # Never let a second worker overwrite an in-progress/unfinished run.
            with (directory / "run_claim.json").open("x") as stream:
                json.dump({"worker": args.worker_index, "seed": seed, "depth": depth, "supervision": mode}, stream)
            command = ["--manifest", args.manifest, "--encoder-weights", args.encoder_weights,
                       "--run-dir", str(directory), "--arm", "SJ", "--steps", str(depth),
                       "--supervision", mode, "--seed", str(seed), "--selection-seed", "20261004",
                       "--sampling", "proportional", "--max-cases", "128", "--epochs", "50",
                       "--batch-size", "2", "--learning-rate", "0.0003", "--threads", str(args.threads),
                       "--cache-features", "--feature-cache-dir", args.feature_cache_dir,
                       "--eval-max-cases", "159", "--eval-every", "5"]
            print(json.dumps({"event": "run_start", "depth": depth, "supervision": mode, "seed": seed}), flush=True)
            summary = validate_completed(train_run(train_parser().parse_args(command)), protocol, seed, depth, mode)
        completed.append({"seed": seed, "depth": depth, "supervision": mode,
                          "training_seconds": summary["training_seconds"],
                          "endpoint": summary["development_subset"][str(depth)]})
        progress = {"status": "worker_complete" if len(completed) == len(combos) else "running",
                    "workers": args.workers, "worker_index": args.worker_index, "completed_runs": completed,
                    "elapsed_seconds": time.perf_counter()-begin}
        (root / f"progress-worker{args.worker_index}.json").write_text(json.dumps(progress, indent=2)+"\n")
        print(json.dumps({"event": "run_complete", **completed[-1], "worker_completed": len(completed)}), flush=True)
    return progress


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--encoder-weights", default="data/deit_tiny_encoder.pt")
    p.add_argument("--reference-protocol", default="configs/cpu_expanded128.json")
    p.add_argument("--feature-cache-dir", default="outputs/cpu-expanded128/features")
    p.add_argument("--output-dir", default="outputs/cpu-depth-supervision128")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--workers", type=int, choices=[1, 2], default=1)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--prepare-only", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
