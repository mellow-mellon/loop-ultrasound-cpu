"""Execute the fixed 128-Case, four-arm, three-seed CPU development screen."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch

from .cohort import cohort_fingerprint, select_training_rows, training_source_fingerprint
from .data import load_manifest
from .download import file_digest
from .feature_cache import load_or_encode
from .models import create_encoder
from .train import parser as train_parser, run as train_run

ARMS = ("SC", "SJ", "UC", "UJ")
SEEDS = (17, 29, 43)


def write_once_or_match(path, record):
    text = json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError("Existing experiment protocol differs; choose a fresh directory.")
    else:
        with path.open("x") as stream:
            stream.write(text)


def run(args):
    if args.threads not in (1, 2):
        raise ValueError("Only one/two CPU threads.")
    begin = time.perf_counter()
    torch.set_num_threads(args.threads)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    all_rows = load_manifest(args.manifest)
    train = select_training_rows(all_rows, 128, sampling="proportional", selection_seed=20261004)
    tune = [r for r in all_rows if r["split"] == "tune"]
    def counts(rows):
        labels = {r["case_id"]: int(r["label"]) for r in rows}
        return {"cases": len(labels), "images": len(rows), "benign": sum(v == 0 for v in labels.values()),
                "malignant": sum(labels.values()), "fingerprint": cohort_fingerprint(rows)}
    protocol = {
        "status": "fixed_before_expanded_training", "clinical_validation": False,
        "patient_mapping": "unverified_Case_grouped_only", "device": "cpu", "threads": args.threads,
        "arms": list(ARMS), "seeds": list(SEEDS), "selection_seed": 20261004,
        "sampling": "proportional", "epochs": 50, "batch_size": 2, "learning_rate": .0003,
        "segmentation_weight": 1., "cached_frozen_features": True, "augment": False,
        "eval_every": 5, "model_selection": "fixed_final_epoch_no_best_epoch_selection",
        "encoder_sha256": file_digest(args.encoder_weights), "manifest_sha256": file_digest(args.manifest),
        "training_source_sha256": training_source_fingerprint(),
        "train": counts(train), "tune": counts(tune),
        "calibration_and_test_predictions": False,
        "screening_rules": ["Require useful development classification relative to train-only constant/simple-feature baselines.",
                            "Inspect depth benefit and interaction direction across all three seeds; no best-seed reporting.",
                            "A positive CPU screen does not establish clinical benefit or novelty; no p-value gate."]}
    if protocol["train"]["cases"] != 128 or protocol["tune"]["cases"] != 159:
        raise ValueError("This fixed experiment requires the expected 128/159 Case cohorts.")
    write_once_or_match(root / "protocol.json", protocol)
    print(json.dumps({"event": "protocol_fixed", "train": protocol["train"], "tune": protocol["tune"]}), flush=True)
    encoder = create_encoder(pretrained=True, weights_path=args.encoder_weights)
    cache_dir = root / "features"
    train_features, _, _ = load_or_encode(encoder, train, args.encoder_weights, cache_dir, 2)
    tune_features, _, _ = load_or_encode(encoder, tune, args.encoder_weights, cache_dir, 2)
    from .baselines import run_baselines
    baseline_dir = root / "baselines"
    baseline_path = baseline_dir / "baselines_summary.json"
    if not baseline_path.exists():
        baseline = run_baselines(train, tune, train_features, tune_features, baseline_dir, selection_seed=20261004)
        baseline["sources"] = {"encoder_sha256": protocol["encoder_sha256"], "manifest_sha256": protocol["manifest_sha256"]}
        baseline_path.write_text(json.dumps(baseline, indent=2, allow_nan=False) + "\n")
    else:
        baseline = json.loads(baseline_path.read_text())
        if baseline.get("sources") != {"encoder_sha256": protocol["encoder_sha256"], "manifest_sha256": protocol["manifest_sha256"]}:
            raise ValueError("Existing baseline sources differ.")
        if baseline.get("cohort_fingerprints") != {"train": protocol["train"]["fingerprint"], "tune": protocol["tune"]["fingerprint"]}:
            raise ValueError("Existing baseline cohorts differ.")
    print(json.dumps({"event": "baselines_complete", "baselines": baseline["baselines"]}), flush=True)
    del encoder, train_features, tune_features
    completed = []
    for seed in SEEDS:
        for arm in ARMS:
            if training_source_fingerprint() != protocol["training_source_sha256"]:
                raise ValueError("Training source changed after locking the protocol.")
            directory = root / f"{arm}-seed{seed}"
            command = ["--manifest", args.manifest, "--encoder-weights", args.encoder_weights,
                       "--run-dir", str(directory), "--arm", arm, "--seed", str(seed),
                       "--selection-seed", "20261004", "--sampling", "proportional",
                       "--max-cases", "128", "--epochs", "50", "--batch-size", "2",
                       "--learning-rate", "0.0003", "--threads", str(args.threads),
                       "--cache-features", "--feature-cache-dir", str(cache_dir),
                       "--eval-max-cases", "159", "--eval-every", "5"]
            print(json.dumps({"event": "run_start", "arm": arm, "seed": seed, "completed": len(completed)}), flush=True)
            if (directory / "summary.json").exists():
                summary = json.loads((directory / "summary.json").read_text())
                expected = {"arm": arm, "steps": 4, "seed": seed, "epochs": 50, "batch_size": 2,
                            "learning_rate": .0003, "segmentation_weight": 1., "cached_frozen_features": True,
                            "augment": False, "selection_seed": 20261004, "sampling": "proportional", "eval_every": 5,
                            "supervision": "all"}
                if (summary.get("config") != expected or summary.get("encoder_sha256") != protocol["encoder_sha256"]
                        or summary.get("manifest_sha256") != protocol["manifest_sha256"]
                        or summary.get("training_source_sha256") != protocol["training_source_sha256"]
                        or summary.get("cohort_selection", {}).get("train_fingerprint") != protocol["train"]["fingerprint"]
                        or summary.get("cohort_selection", {}).get("tune_fingerprint") != protocol["tune"]["fingerprint"]):
                    raise ValueError("Existing run differs from the locked protocol.")
            else:
                summary = train_run(train_parser().parse_args(command))
            completed.append({"arm": arm, "seed": seed, "training_seconds": summary["training_seconds"],
                              "development_step4": summary["development_subset"]["4"]})
            progress = {"status": "running" if len(completed) < 12 else "training_complete",
                        "completed_runs": completed, "elapsed_seconds_this_invocation": time.perf_counter()-begin}
            (root / "progress.json").write_text(json.dumps(progress, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"event": "run_complete", **completed[-1], "n_completed": len(completed)}), flush=True)
    return progress


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--encoder-weights", default="data/deit_tiny_encoder.pt")
    p.add_argument("--output-dir", default="outputs/cpu-expanded128")
    p.add_argument("--threads", type=int, default=2)
    run(p.parse_args())


if __name__ == "__main__":
    main()
