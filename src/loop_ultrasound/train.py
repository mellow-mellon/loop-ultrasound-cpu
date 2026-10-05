"""Bounded CPU engineering pilot using only training and development Cases.

Tiny-set fit metrics are software checks, not evidence of diagnostic benefit.
No test/calibration split is accepted by this entry point.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

from .data import UltrasoundDataset, balanced_train_rows, load_manifest
from .metrics import aggregate_cases, classification_metrics
from .models import create_encoder, make_model, trajectory_loss


class CaseViewSampler(Sampler[int]):
    """One uniformly chosen image per Case per epoch, including paired views."""
    def __init__(self, rows, seed=17):
        self.groups = {}
        for index, row in enumerate(rows):
            self.groups.setdefault(row["case_id"], []).append(index)
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return len(self.groups)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = [rng.choice(self.groups[k]) for k in sorted(self.groups)]
        rng.shuffle(indices)
        self.epoch += 1
        return iter(indices)


def case_subset(rows, max_cases, seed):
    """Label-blind, reproducible Case subset for development-only evaluation."""
    cases = sorted({r["case_id"] for r in rows})
    random.Random(seed).shuffle(cases)
    selected = set(cases[:max_cases]) if max_cases > 0 else set(cases)
    return [r for r in rows if r["case_id"] in selected]


def parameter_groups(model):
    mask = list(model.mask_head.parameters())
    mask_ids = {id(p) for p in mask}
    core = [p for p in model.parameters() if id(p) not in mask_ids]
    return core, mask


def prepare_features(encoder, dataset, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            features.append(encoder(batch["image"]))
    return torch.cat(features), time.perf_counter() - start


def mask_dice(logits, target, valid):
    pred = (logits.sigmoid() >= .5).float()
    numerator = 2 * (pred * target * valid).flatten(1).sum(1)
    denominator = ((pred + target) * valid).flatten(1).sum(1)
    return (numerator + 1e-6) / (denominator + 1e-6)


def evaluate(model, encoder, rows, args, *, cache=None):
    dataset = UltrasoundDataset(rows, image_size=224, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    records, offset = [], 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            b = len(batch["case_id"])
            tokens = cache[offset:offset+b] if cache is not None else encoder(batch["image"])
            outputs = model(tokens, max_steps=args.steps, return_all=True)
            for step, out in enumerate(outputs, 1):
                probabilities = out["cls_logits"].sigmoid().tolist()
                dice = mask_dice(out["mask_logits"], batch["mask"], batch["valid_pixels"]).tolist()
                for i in range(b):
                    records.append({
                        "case_id": batch["case_id"][i], "image_id": batch["image_id"][i],
                        "label": int(batch["label"][i]), "probability": probabilities[i],
                        "step": step, "arm": args.arm, "seed": args.seed, "dice": dice[i],
                    })
            offset += b
    case_rows = aggregate_cases(records)
    summaries = {}
    for step in range(1, args.steps + 1):
        selected = [r for r in case_rows if r["step"] == step]
        metrics = classification_metrics([r["label"] for r in selected],
                                         [r["probability"] for r in selected], threshold=.5)
        metrics["mean_case_dice"] = float(np.mean([r["dice"] for r in selected]))
        metrics["dice_space"] = "224x224_letterbox_valid_pixels"
        summaries[str(step)] = metrics
    return records, summaries


def write_records(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run(args):
    if args.epochs <= 0 or args.batch_size <= 0 or args.max_cases < 2 or args.max_cases % 2:
        raise ValueError("Use positive epochs/batch and a positive even max-cases >= 2.")
    if args.cache_features and args.augment:
        raise ValueError("Caching is for unaugmented engineering checks only.")
    if args.threads not in (1, 2) or args.eval_max_cases < 0:
        raise ValueError("Use one/two CPU threads and nonnegative eval-max-cases.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "summary.json").exists():
        raise FileExistsError("Use a fresh run directory; completed results are never overwritten.")
    all_rows = load_manifest(args.manifest)
    train_rows = balanced_train_rows(all_rows, max_cases_per_class=args.max_cases // 2, seed=args.seed)
    if len({r["case_id"] for r in train_rows}) != args.max_cases:
        raise ValueError("The requested balanced training subset is unavailable.")
    dataset = UltrasoundDataset(train_rows, image_size=224, augment=args.augment)
    encoder = create_encoder(pretrained=True, weights_path=args.encoder_weights)
    torch.manual_seed(args.seed)  # Pair core/head initialization independent of encoder creation.
    model = make_model(args.arm, steps=args.steps)
    core, mask = parameter_groups(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    cached, encoding_seconds = None, 0.
    if args.cache_features:
        cache_dataset = UltrasoundDataset(train_rows, image_size=224, augment=False)
        cached, encoding_seconds = prepare_features(encoder, cache_dataset, args.batch_size)
        feature_index = {r["image_id"]: i for i, r in enumerate(train_rows)}
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        sampler=CaseViewSampler(train_rows, args.seed), num_workers=0)
    _, initial = evaluate(model, encoder, train_rows, args, cache=cached)
    history, step_seconds = [], []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in loader:
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            if cached is None:
                tokens = encoder(batch["image"])
            else:
                tokens = cached[[feature_index[i] for i in batch["image_id"]]]
            outputs = model(tokens, max_steps=args.steps, return_all=True)
            loss = trajectory_loss(outputs, batch["label"], batch["mask"],
                                   batch["valid_pixels"], segmentation_weight=args.segmentation_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(core, 1.)
            nn.utils.clip_grad_norm_(mask, 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
            step_seconds.append(time.perf_counter() - tick)
        history.append({"epoch": epoch+1, "mean_training_objective": float(np.mean(losses))})
        if epoch == 0 or (epoch+1) % 5 == 0 or epoch+1 == args.epochs:
            print(json.dumps(history[-1]), flush=True)
    training_seconds = time.perf_counter() - start
    train_records, final = evaluate(model, encoder, train_rows, args, cache=cached)
    write_records(run_dir / "training_predictions.csv", train_records)
    torch.save({"model_state": model.state_dict(), "arm": args.arm, "steps": args.steps,
                "seed": args.seed, "encoder_sha256": hashlib.sha256(Path(args.encoder_weights).read_bytes()).hexdigest()},
               run_dir / "checkpoint.pt")
    development = None
    if args.eval_max_cases:
        tune_rows = case_subset([r for r in all_rows if r["split"] == "tune"], args.eval_max_cases, args.seed)
        if {r["case_id"] for r in tune_rows} & {r["case_id"] for r in train_rows}:
            raise ValueError("Train/tune Case overlap.")
        tune_records, development = evaluate(model, encoder, tune_rows, args)
        write_records(run_dir / "development_predictions.csv", tune_records)
    summary = {
        "status": "executed_cpu_engineering_pilot", "clinical_validation": False,
        "patient_mapping": "unverified_Case_grouped_only", "device": "cpu",
        "environment": {"python": platform.python_version(), "os": platform.system(),
                        "architecture": platform.machine(), "torch": torch.__version__, "threads": args.threads},
        "config": {"arm": args.arm, "steps": args.steps, "seed": args.seed,
                   "epochs": args.epochs, "batch_size": args.batch_size,
                   "learning_rate": args.learning_rate, "segmentation_weight": args.segmentation_weight,
                   "cached_frozen_features": args.cache_features, "augment": args.augment},
        "n_training_cases": args.max_cases, "n_training_images": len(train_rows),
        "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "encoder_sha256": hashlib.sha256(Path(args.encoder_weights).read_bytes()).hexdigest(),
        "parameters": {"encoder_frozen": sum(p.numel() for p in encoder.parameters()),
                       "core_and_class": sum(p.numel() for p in core),
                       "mask_head": sum(p.numel() for p in mask),
                       "trainable": sum(p.numel() for p in model.parameters())},
        "initial_training_fit": initial, "final_training_fit": final,
        "development_subset": development,
        "training_seconds": training_seconds, "frozen_feature_precompute_seconds": encoding_seconds,
        "timing": {"updates": len(step_seconds),
                   "median_training_step_seconds": float(np.median(step_seconds)),
                   "p90_training_step_seconds": float(np.quantile(step_seconds, .9)),
                   "includes_online_encoder": not args.cache_features,
                   "includes_data_loading": False},
        "limitations": ["Tiny training fit is a code check, not a generalization estimate.",
                        "Development subset is exploratory and not clinical validation.",
                        "Test and calibration image pixels were never used by the trainer.",
                        "Dice uses valid letterboxed pixels; original-resolution boundary metrics are pending.",
                        "Caching/no augmentation and pilot learning rate differ from the proposed scientific protocol."]
    }
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({"summary": str(run_dir / "summary.json"), "training_seconds": training_seconds}), flush=True)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--encoder-weights", default="data/deit_tiny_encoder.pt")
    p.add_argument("--run-dir", default="outputs/cpu-pilot-sj")
    p.add_argument("--arm", choices=["SC", "SJ", "UC", "UJ"], default="SJ")
    p.add_argument("--steps", type=int, choices=range(1, 5), default=4)
    p.add_argument("--max-cases", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--segmentation-weight", type=float, default=1.)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--cache-features", action="store_true")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--eval-max-cases", type=int, default=16)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
