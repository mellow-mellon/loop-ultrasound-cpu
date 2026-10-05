"""Bounded CPU timing with real training pixels, including the frozen encoder.

Timings are warm process, batch latencies, not diagnostic performance or GPU
predictions. Parameters in the timed refinement heads are newly initialized.
"""
import argparse
import json
import math
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import UltrasoundDataset, balanced_train_rows, load_manifest
from .download import file_digest
from .models import create_encoder, make_model, parameter_counts, trajectory_loss


def timing_summary(values):
    return {"n": len(values), "median_seconds": float(np.median(values)),
            "p90_seconds": float(np.quantile(values, .9)), "mean_seconds": float(np.mean(values))}


def timed(call, iterations, warmup):
    for _ in range(warmup):
        call()
    seconds = []
    for _ in range(iterations):
        begin = time.perf_counter()
        call()
        seconds.append(time.perf_counter() - begin)
    return timing_summary(seconds)


def run(args):
    if args.threads not in (1, 2) or args.batch_size < 1 or args.iterations < 1 or args.warmup < 1:
        raise ValueError("Positive batch/iterations/warmup and one/two CPU threads required.")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Use a new benchmark output to preserve measurements.")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    rows = load_manifest(args.manifest)
    training = [r for r in rows if r["split"] == "train"]
    sample = balanced_train_rows(rows, max_cases_per_class=4, seed=17)
    dataset = UltrasoundDataset(sample, augment=False)
    if args.batch_size > len(dataset):
        raise ValueError("Batch exceeds the bounded pilot subset.")

    def load_batch():
        return next(iter(DataLoader(dataset, batch_size=args.batch_size, num_workers=0)))

    batch = load_batch()
    loading = timed(load_batch, 20, args.warmup)
    encoder = create_encoder(pretrained=True, weights_path=args.encoder_weights).cpu()
    arm_results = {}
    for arm in args.arms:
        torch.manual_seed(17)
        model = make_model(arm).cpu()
        groups = model.training_parameter_groups()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)

        def update():
            model.train()
            optimizer.zero_grad(set_to_none=True)
            tokens = encoder(batch["image"])
            outputs = model(tokens, return_all=True)
            loss = trajectory_loss(outputs, batch["label"], batch["mask"], batch["valid_pixels"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite timing-run loss.")
            loss.backward()
            for parameters in groups.values():
                nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()

        training_timing = timed(update, args.iterations, args.warmup)
        model.eval()
        endpoint = {}
        with torch.no_grad():
            for depth in (1, 2, 4):
                endpoint[str(depth)] = timed(
                    lambda: model(encoder(batch["image"]), max_steps=depth, return_all=False),
                    args.iterations, args.warmup)
        epoch_updates = math.ceil(len({r["case_id"] for r in training}) / args.batch_size)
        arm_results[arm] = {"parameters": parameter_counts(model),
                            "online_encoder_training_update": training_timing,
                            "online_encoder_endpoint_inference": endpoint,
                            "rough_100_epoch_update_hours_at_this_batch":
                                training_timing["median_seconds"] * epoch_updates * 100 / 3600}
        print(json.dumps({"arm": arm, "training_update_median_seconds": training_timing["median_seconds"]}), flush=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "status": "executed_cpu_benchmark", "clinical_validation": False,
        "environment": {"python": platform.python_version(), "os": platform.system(),
                        "architecture": platform.machine(), "torch": torch.__version__,
                        "logical_cpus": __import__("os").cpu_count(), "threads": args.threads},
        "device": "cpu", "batch_size": args.batch_size, "input_size": [224, 224],
        "measured_iterations_per_operation": args.iterations, "warmup": args.warmup,
        "encoder_sha256": file_digest(args.encoder_weights),
        "encoder_frozen_parameters": sum(p.numel() for p in encoder.parameters()),
        "png_loading_preprocessing": loading, "arms": arm_results,
        "process_peak_rss_mib": rss / (1024**2 if platform.system() == "Darwin" else 1024),
        "timing_scope": {
            "training": "Frozen encoder forward + 4 refinement applications + all-step dual heads/loss + backward + separate clipping + AdamW.",
            "inference": "Frozen encoder forward + selected refinement depth + terminal classification and mask heads, no_grad.",
            "excludes_from_model_timings": ["PNG decoding/preprocessing", "network", "first model load", "augmentation"],
            "loading": "Warm OS cache, one real batch loaded/preprocessed serially; measured separately.",
            "rss": "Peak for entire process including encoder/model/optimizer/imports; cumulative across arms, not per-arm memory."},
        "limitations": ["Timing models start from random refinement heads; no accuracy claim.",
                        "Repeated fixed mini-batch is a timing workload, not a research training run.",
                        "Epoch projection counts one sampled image per training Case. It excludes loading, evaluation and scheduling.",
                        "Projection applies only to this CPU and batch size; do not extrapolate to batch 16, a GPU, or end-to-end encoder training."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--encoder-weights", default="data/deit_tiny_encoder.pt")
    p.add_argument("--output", default="outputs/cpu-benchmark.json")
    p.add_argument("--arms", nargs="+", choices=["SC", "SJ", "UC", "UJ"], default=["SC", "SJ", "UC", "UJ"])
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    run(p.parse_args())


if __name__ == "__main__":
    main()
