"""Paired full-state forks for a fixed terminal-depth CPU learning-rate control."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import struct
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .analyze_case_expansion import _protocol as expansion_protocol, _new_runs
from .case_expansion import expected_config
from .cohort import select_training_rows, training_source_fingerprint
from .data import UltrasoundDataset, load_manifest
from .download import file_digest
from .expanded import write_once_or_match
from .feature_cache import load_or_encode
from .models import create_encoder, make_model, trajectory_loss
from .seed_replication import counts
from .train import CaseViewSampler, evaluate, initialization_fingerprint, parameter_groups, write_records

SEEDS = (53, 71, 89, 107, 131)
DEPTHS = (2, 4)
POLICIES = ("constant", "low_tail")
TAIL_LRS = {"constant": .0003, "low_tail": .00003}
PREFIX_UPDATES, TAIL_UPDATES = 2560, 640


def assigned_pairs(workers=1, worker_index=0):
    if workers not in (1, 2) or worker_index not in range(workers):
        raise ValueError("Use one/two workers and a valid worker index.")
    pairs = [(seed, depth) for seed in SEEDS for depth in DEPTHS]
    return [pair for i, pair in enumerate(pairs) if i % workers == worker_index]


def launcher_digest():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def state_digest(state):
    """Hash tensors and complete primitive state without pickle byte dependence."""
    digest = hashlib.sha256()
    def visit(value):
        if isinstance(value, torch.Tensor):
            t = value.detach().cpu().contiguous()
            digest.update(b"tensor" + str(t.dtype).encode() + repr(tuple(t.shape)).encode() + t.numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict")
            for key in sorted(value, key=lambda k: (type(k).__name__, repr(k))):
                visit(key); visit(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode() + str(len(value)).encode())
            for v in value: visit(v)
        elif isinstance(value, float): digest.update(b"float" + struct.pack("!d", value))
        elif value is None or isinstance(value, (str, int, bool)):
            digest.update(type(value).__name__.encode() + repr(value).encode() + b"\0")
        else: raise TypeError(f"Unsupported checkpoint state type: {type(value).__name__}")
    visit(state)
    return digest.hexdigest()


def capture_state(model, optimizer, loader):
    numpy_state = np.random.get_state()
    return {"model_state": deepcopy(model.state_dict()), "model_training": model.training,
            "optimizer_state": deepcopy(optimizer.state_dict()),
            "torch_rng_state": torch.get_rng_state().clone(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": [numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]),
                                int(numpy_state[3]), float(numpy_state[4])],
            "loader_generator_state": loader.generator.get_state().clone(),
            "sampler": {"seed": loader.sampler.seed, "epoch": loader.sampler.epoch}}


def restore_state(state, model, optimizer, loader):
    # Copy optimizer tensors: load_state_dict may otherwise alias a CPU snapshot.
    model.load_state_dict(deepcopy(state["model_state"]))
    optimizer.load_state_dict(deepcopy(state["optimizer_state"]))
    model.train(state["model_training"])
    if loader.sampler.seed != state["sampler"]["seed"]:
        raise ValueError("Checkpoint sampler seed differs from this paired branch.")
    loader.sampler.epoch = state["sampler"]["epoch"]
    loader.generator.set_state(state["loader_generator_state"])
    torch.set_rng_state(state["torch_rng_state"])
    random.setstate(state["python_rng_state"])
    n = state["numpy_rng_state"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))


def branch_config(seed, depth, policy):
    return {**expected_config(seed, depth), "tail_policy": policy,
            "prefix_updates": PREFIX_UPDATES, "tail_learning_rate": TAIL_LRS[policy]}


def build_protocol(args):
    torch.set_num_threads(args.threads)
    reference = expansion_protocol(args.reference_protocol, 256, 128, 159)
    if reference["seeds"] != list(SEEDS):
        raise ValueError("Use the five-seed 256-Case locked protocol as the flat-policy reference.")
    directories = [Path(args.reference_run_dir) / f"SJ-depth{depth}-terminal-seed{seed}"
                   for seed in SEEDS for depth in DEPTHS]
    _new_runs(directories, reference)  # Actual records, histories and endpoints, not just summaries.
    rows = load_manifest(args.manifest)
    train = select_training_rows(rows, 256, sampling="proportional", selection_seed=20261004)
    tune = [r for r in rows if r["split"] == "tune"]
    sources = {"training_source_sha256": training_source_fingerprint(),
               "encoder_sha256": file_digest(args.encoder_weights), "manifest_sha256": file_digest(args.manifest)}
    if counts(train) != reference["train"] or counts(tune) != reference["tune"]:
        raise ValueError("Tail intervention must retain exact 256/159 Case/view/label cohorts.")
    if any(reference[k] != v for k, v in sources.items()):
        raise ValueError("Training core, encoder and manifest must match existing flat references.")
    bindings = {f"seed{seed}_depth{depth}": {
        "summary_sha256": file_digest(directory / "summary.json"),
        "checkpoint_sha256": file_digest(directory / "checkpoint.pt"),
        "training_predictions_sha256": file_digest(directory / "training_predictions.csv"),
        "development_predictions_sha256": file_digest(directory / "development_predictions.csv")}
        for (seed, depth), directory in zip([(s, d) for s in SEEDS for d in DEPTHS], directories)}
    protocol = {**{k: reference[k] for k in (
        "device", "arms", "depths", "seeds", "supervisions", "epochs", "batch_size", "learning_rate",
        "segmentation_weight", "selection_seed", "sampling", "augment", "cached_frozen_features",
        "eval_every", "training_updates_per_run", "model_selection", "patient_mapping",
        "clinical_validation", "calibration_and_test_predictions", "initialization_sha256_by_seed", "train", "tune")},
        **sources, "status": "fixed_before_cpu_tail_lr_control",
        "phase": "exploratory_tail_optimizer_intervention_on_observed_tune", "threads": args.threads,
        "policies": list(POLICIES), "prefix_epochs": 20, "tail_epochs": 5,
        "prefix_updates": PREFIX_UPDATES, "tail_updates": TAIL_UPDATES,
        "tail_learning_rates": dict(TAIL_LRS), "reference_epochs": 25,
        "updates_per_epoch": 128, "total_actual_new_updates": 38400,
        "effective_endpoint_trajectories": 20, "shared_prefixes": 10,
        "launcher_source_sha256": launcher_digest(), "reference_protocol_sha256": file_digest(args.reference_protocol),
        "reference_run_artifact_bindings": bindings,
        "warm_starts": False, "full_state_fork": True,
        "primary_metric": "Case Brier at fixed update3200",
        "primary_interaction": "H = G_low_tail - G_constant; G_policy = Brier_depth2 - Brier_depth4",
        "secondary_metrics": ["G_low_tail", "NLL", "AUROC", "Dice", "prefix_to_endpoint", "gradient_clipping", "classification_and_mask_training_losses"],
        "bootstrap": {"draws": 2000, "seed": 17, "unit": "Case", "conditional_on_fitted_models": True},
        "budget": "Ten fresh prefixes at2560 plus twenty640-update tails =38400 actual new updates. Each effective trajectory completes3200; depth4 uses extra FLOPs.",
        "tail_policy_definition": "Single step reduction to3e-5 for updates2561..3200; constant branch remains3e-4. Both AdamW gradient and decoupled weight-decay step magnitudes change.",
        "interpretation": "Exploratory optimizer-policy intervention on observed tune, not a pure gradient-size cause or clinical validation; no sweep or best-epoch selection."}
    return protocol, train, tune


def train_epochs(model, optimizer, loader, cached, feature_index, args, start_epoch, stop_epoch):
    core, mask = parameter_groups(model)
    history, snapshots, durations = [], [], []
    trace = hashlib.sha256()
    for epoch in range(start_epoch, stop_epoch):
        model.train()
        losses, class_losses, core_norms, mask_norms = [], [], [], []
        for batch in loader:
            tick = time.perf_counter()
            trace.update(json.dumps(list(batch["image_id"]), separators=(",", ":")).encode() + b"\n")
            optimizer.zero_grad(set_to_none=True)
            tokens = cached[[feature_index[i] for i in batch["image_id"]]]
            outputs = model(tokens, max_steps=args.steps, return_all=False)
            loss = trajectory_loss(outputs, batch["label"], batch["mask"], batch["valid_pixels"],
                                   segmentation_weight=1., supervision="terminal")
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite training loss.")
            # Diagnostic computation does not backpropagate or consume randomness.
            cls = F.binary_cross_entropy_with_logits(outputs[-1]["cls_logits"].detach(), batch["label"])
            loss.backward()
            core_norm = nn.utils.clip_grad_norm_(core, 1.)
            mask_norm = nn.utils.clip_grad_norm_(mask, 1.)
            optimizer.step()
            losses.append(float(loss.detach())); class_losses.append(float(cls))
            core_norms.append(float(core_norm)); mask_norms.append(float(mask_norm))
            durations.append(time.perf_counter() - tick)
        history.append({"epoch": epoch + 1, "mean_training_objective": float(np.mean(losses)),
                        "mean_training_classification": float(np.mean(class_losses)),
                        "mean_training_segmentation": float(np.mean(np.asarray(losses) - class_losses)),
                        "core_preclip_norm_mean": float(np.mean(core_norms)),
                        "core_clip_fraction": float(np.mean(np.asarray(core_norms) > 1.)),
                        "mask_preclip_norm_mean": float(np.mean(mask_norms)),
                        "mask_clip_fraction": float(np.mean(np.asarray(mask_norms) > 1.)),
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        if (epoch + 1) % 5 == 0:
            _, snapshot = evaluate(model, args.encoder, args.tune_rows, args, cache=args.tune_cache)
            snapshots.append({"epoch": epoch + 1, "by_step": snapshot})
            print(json.dumps({"event": "epoch", "seed": args.seed, "depth": args.steps, "epoch": epoch + 1,
                              "learning_rate": optimizer.param_groups[0]["lr"],
                              "development_brier_endpoint": snapshot[str(args.steps)]["brier"]}), flush=True)
    return history, snapshots, durations, trace.hexdigest()


def verify_constant(directory, reference, model, history, snapshots):
    old = json.loads((reference / "summary.json").read_text())
    for name in ("training_predictions.csv", "development_predictions.csv"):
        if (directory / name).read_bytes() != (reference / name).read_bytes():
            raise ValueError("Constant-policy raw prediction replay differs from the existing fixed fit.")
    objective = [{"epoch": row["epoch"], "mean_training_objective": row["mean_training_objective"]} for row in history]
    if objective != json.loads((reference / "history.json").read_text()) or snapshots != old["development_history"]:
        raise ValueError("Constant-policy training or development history differs from its reference.")
    saved = torch.load(reference / "checkpoint.pt", map_location="cpu", weights_only=True)["model_state"]
    actual = model.state_dict()
    if set(saved) != set(actual) or any(not torch.equal(saved[k], actual[k]) for k in saved):
        raise ValueError("Constant-policy final parameter tensors differ from their exact reference replay.")
    return {"status": "exact_replay_verified", "reference_summary_sha256": file_digest(reference / "summary.json"),
            "model_state_tensors_equal": True, "prediction_records_equal": True,
            "training_objective_history_equal": True, "development_history_equal": True}


def _metadata(protocol, seed, depth):
    return {"clinical_validation": False, "patient_mapping": "unverified_Case_grouped_only", "device": "cpu",
            "evaluation_split": "tune_development", "model_selection": "fixed_final_epoch_no_best_epoch_selection",
            **{k: protocol[k] for k in ("training_source_sha256", "encoder_sha256", "manifest_sha256", "launcher_source_sha256")},
            "initialization_sha256": protocol["initialization_sha256_by_seed"][str(seed)],
            "n_training_cases": 256, "n_training_images": 445,
            "cohort_selection": {"sampling": "proportional", "selection_seed": 20261004,
                                 "train_fingerprint": protocol["train"]["fingerprint"], "tune_fingerprint": protocol["tune"]["fingerprint"]},
            "readout_supervision": {"mode": "terminal", "directly_supervised_steps": [depth],
                                   "intermediate_predictions_scope": "diagnostic_only_no_direct_intermediate_readout_loss"}}


def run_pair(args, protocol, train_rows, tune_rows, seed, depth):
    root = Path(args.output_dir)
    prefix_dir = root / f"prefix-depth{depth}-seed{seed}"
    prefix_dir.mkdir(exist_ok=True)
    with (prefix_dir / "pair_claim.json").open("x") as stream:
        json.dump({"seed": seed, "depth": depth, "worker": args.worker_index}, stream)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.use_deterministic_algorithms(True)
    encoder = create_encoder(pretrained=True, weights_path=args.encoder_weights)
    torch.manual_seed(seed)
    model = make_model("SJ", steps=depth)
    if initialization_fingerprint(model) != protocol["initialization_sha256_by_seed"][str(seed)]:
        raise ValueError("Fresh prefix initialization differs from locked reference.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
    cached, _, reused = load_or_encode(encoder, train_rows, args.encoder_weights, args.feature_cache_dir, 2)
    tune_cache, _, _ = load_or_encode(encoder, tune_rows, args.encoder_weights, args.feature_cache_dir, 2)
    dataset = UltrasoundDataset(train_rows, augment=False)
    def new_loader():
        return DataLoader(dataset, batch_size=2, sampler=CaseViewSampler(train_rows, seed), num_workers=0,
                          generator=torch.Generator().manual_seed(seed))
    loader = new_loader()
    feature_index = {r["image_id"]: i for i, r in enumerate(train_rows)}
    fit_args = argparse.Namespace(arm="SJ", seed=seed, steps=depth, batch_size=2,
                                  encoder=encoder, tune_rows=tune_rows, tune_cache=tune_cache)
    _, initial = evaluate(model, encoder, train_rows, fit_args, cache=cached)
    begin = time.perf_counter()
    history, snapshots, prefix_times, prefix_trace = train_epochs(model, optimizer, loader, cached, feature_index, fit_args, 0, 20)
    prefix_seconds = time.perf_counter() - begin
    train_records, prefix_train = evaluate(model, encoder, train_rows, fit_args, cache=cached)
    dev_records, prefix_dev = evaluate(model, encoder, tune_rows, fit_args, cache=tune_cache)
    write_records(prefix_dir / "training_predictions.csv", train_records)
    write_records(prefix_dir / "development_predictions.csv", dev_records)
    state = capture_state(model, optimizer, loader)
    state_sha = state_digest(state)
    torch.save({"state": state, "seed": seed, "depth": depth, "updates": PREFIX_UPDATES,
                "state_sha256": state_sha, "protocol_sha256": file_digest(root / "protocol.json")}, prefix_dir / "checkpoint.pt")
    prefix_sha = file_digest(prefix_dir / "checkpoint.pt")
    meta = _metadata(protocol, seed, depth)
    prefix_summary = {**meta, "status": "executed_cpu_tail_lr_prefix",
        "config": {**expected_config(seed, depth), "prefix_updates": PREFIX_UPDATES},
        "checkpoint_sha256": prefix_sha, "state_sha256": state_sha,
        "training_predictions_sha256": file_digest(prefix_dir / "training_predictions.csv"),
        "development_predictions_sha256": file_digest(prefix_dir / "development_predictions.csv"),
        "timing": {"updates": PREFIX_UPDATES}, "training_seconds": prefix_seconds,
        "development_subset": prefix_dev, "final_training_fit": prefix_train,
        "development_history": snapshots, "history": history, "sampling_trace_sha256": prefix_trace}
    (prefix_dir / "prefix_summary.json").write_text(json.dumps(prefix_summary, indent=2, allow_nan=False) + "\n")
    results = []
    for policy in POLICIES:
        if training_source_fingerprint() != protocol["training_source_sha256"] or launcher_digest() != protocol["launcher_source_sha256"]:
            raise ValueError("Training source changed after protocol lock.")
        directory = root / f"SJ-depth{depth}-{policy}-seed{seed}"
        directory.mkdir(exist_ok=True)
        payload = torch.load(prefix_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
        if (file_digest(prefix_dir / "checkpoint.pt") != prefix_sha or payload["seed"] != seed
                or payload["depth"] != depth or payload["updates"] != PREFIX_UPDATES
                or state_digest(payload["state"]) != state_sha):
            raise ValueError("Local prefix checkpoint identity/state changed before fork.")
        branch_model = make_model("SJ", steps=depth)
        branch_optimizer = torch.optim.AdamW(branch_model.parameters(), lr=.0003, weight_decay=.01)
        branch_loader = new_loader()
        restore_state(payload["state"], branch_model, branch_optimizer, branch_loader)
        resumed_sha = state_digest(capture_state(branch_model, branch_optimizer, branch_loader))
        if resumed_sha != state_sha or branch_loader.sampler.epoch != 20:
            raise ValueError("Complete branch state differs from the exact common prefix.")
        for group in branch_optimizer.param_groups: group["lr"] = TAIL_LRS[policy]
        start = time.perf_counter()
        tail_history, tail_snapshots, tail_times, tail_trace = train_epochs(
            branch_model, branch_optimizer, branch_loader, cached, feature_index, fit_args, 20, 25)
        branch_seconds = time.perf_counter() - start
        full_history, full_snapshots = deepcopy(history) + tail_history, deepcopy(snapshots) + tail_snapshots
        train_records, final = evaluate(branch_model, encoder, train_rows, fit_args, cache=cached)
        dev_records, development = evaluate(branch_model, encoder, tune_rows, fit_args, cache=tune_cache)
        write_records(directory / "training_predictions.csv", train_records)
        write_records(directory / "development_predictions.csv", dev_records)
        reference = Path(args.reference_run_dir) / f"SJ-depth{depth}-terminal-seed{seed}"
        if policy == "constant":
            verification = verify_constant(directory, reference, branch_model, full_history, full_snapshots)
        else:
            constant = root / f"SJ-depth{depth}-constant-seed{seed}" / "summary.json"
            prior = json.loads(constant.read_text())
            if prior["constant_replay_verification"]["status"] != "exact_replay_verified":
                raise ValueError("The paired flat replay must verify before low-tail analysis.")
            if prior["sampling_trace_sha256"]["tail"] != tail_trace:
                raise ValueError("Fork policies sampled different tail images/views.")
            verification = {"status": "paired_constant_replay_verified",
                            "reference_summary_sha256": file_digest(reference / "summary.json"),
                            "constant_summary_sha256": file_digest(constant)}
        torch.save({"model_state": branch_model.state_dict(), "seed": seed, "steps": depth,
                    "tail_policy": policy, "initialization_sha256": meta["initialization_sha256"]}, directory / "checkpoint.pt")
        summary = {**meta, "status": "executed_cpu_tail_lr_control", "config": branch_config(seed, depth, policy),
            "initial_training_fit": initial, "final_training_fit": final, "development_subset": development,
            "development_history": full_snapshots, "feature_cache_reused": reused,
            "parameters": {"trainable": sum(p.numel() for p in branch_model.parameters()),
                           "encoder_frozen": sum(p.numel() for p in encoder.parameters())},
            "training_seconds": prefix_seconds + branch_seconds,
            "timing": {"updates": 3200, "new_updates_in_branch": TAIL_UPDATES,
                       "prefix_reused": True, "branch_training_seconds": branch_seconds,
                       "median_training_step_seconds": float(np.median(prefix_times + tail_times))},
            "branch_provenance": {"prefix_checkpoint_sha256": prefix_sha, "prefix_state_sha256": state_sha,
                "resumed_state_sha256": resumed_sha, "prefix_updates": PREFIX_UPDATES,
                "restored_sampler_epoch": 20, "restored_loader_generator": True, "restored_rng": True},
            "sampling_trace_sha256": {"prefix": prefix_trace, "tail": tail_trace},
            "constant_replay_verification": verification,
            "limitations": ["Observed tune, Case-to-patient mapping unverified, no clinical validation.",
                             "Shared prefixes make policies paired, not independent training replications.",
                             "AdamW learning rate affects both gradient and weight-decay step magnitudes.",
                             "Four-step models use extra compute; no equal-FLOPs inference or calibration claim."]}
        (directory / "history.json").write_text(json.dumps(full_history, indent=2, allow_nan=False) + "\n")
        (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        results.append({"seed": seed, "depth": depth, "policy": policy,
                        "endpoint": development[str(depth)], "branch_training_seconds": branch_seconds})
        print(json.dumps({"event": "branch_complete", **results[-1]}), flush=True)
    return {"seed": seed, "depth": depth, "prefix_training_seconds": prefix_seconds,
            "actual_new_updates": PREFIX_UPDATES + 2 * TAIL_UPDATES, "branches": results}


def run(args):
    if args.threads != 2:
        raise ValueError("Use two CPU threads to preserve the existing reference replay recipe.")
    begin = time.perf_counter()
    root = Path(args.output_dir); root.mkdir(parents=True, exist_ok=True)
    pairs = assigned_pairs(args.workers, args.worker_index)
    protocol, train, tune = build_protocol(args)
    write_once_or_match(root / "protocol.json", protocol)
    if args.prepare_only:
        encoder = create_encoder(pretrained=True, weights_path=args.encoder_weights)
        reused = [load_or_encode(encoder, rows, args.encoder_weights, args.feature_cache_dir, 2)[2] for rows in (train, tune)]
        print(json.dumps({"event": "protocol_and_cache_ready", "train_tune_cache_reused": reused}), flush=True)
        return protocol
    completed = []
    for seed, depth in pairs:
        print(json.dumps({"event": "pair_start", "seed": seed, "depth": depth}), flush=True)
        completed.append(run_pair(args, protocol, train, tune, seed, depth))
        progress = {"status": "worker_complete" if len(completed) == len(pairs) else "running",
                    "worker_index": args.worker_index, "completed_pairs": completed,
                    "actual_new_updates": sum(p["actual_new_updates"] for p in completed),
                    "elapsed_seconds": time.perf_counter() - begin}
        (root / f"progress-worker{args.worker_index}.json").write_text(json.dumps(progress, indent=2, allow_nan=False) + "\n")
    return progress


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--encoder-weights", default="data/deit_tiny_encoder.pt")
    p.add_argument("--reference-protocol", default="outputs/cpu-case-expansion256-replication/protocol.json")
    p.add_argument("--reference-run-dir", default="outputs/cpu-case-expansion256-replication")
    p.add_argument("--feature-cache-dir", default="outputs/cpu-expanded128/features")
    p.add_argument("--output-dir", default="outputs/cpu-tail-lr256")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--workers", type=int, choices=[1, 2], default=1)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--prepare-only", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
