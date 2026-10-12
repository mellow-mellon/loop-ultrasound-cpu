"""Fresh untied controls for a locked sharing-by-depth CPU comparison.

The twenty existing shared endpoints are bound references. Only the untied
models are trained here; both tail policies fork the same complete prefix.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .analyze_tail_lr import analyze_tail_lr
from .case_expansion import expected_config
from .cohort import select_training_rows, training_source_fingerprint
from .data import UltrasoundDataset, load_manifest
from .download import file_digest
from .expanded import write_once_or_match
from .feature_cache import load_or_encode
from .models import create_encoder, make_model, parameter_counts
from .seed_replication import counts
from .tail_lr import (DEPTHS, POLICIES, PREFIX_UPDATES, SEEDS, TAIL_LRS,
                      TAIL_UPDATES, assigned_pairs, capture_state,
                      restore_state, state_digest, train_epochs)
from .train import CaseViewSampler, evaluate, initialization_fingerprint, write_records


def launcher_digest():
    return file_digest(Path(__file__))


def helper_digest():
    return file_digest(Path(__file__).with_name("tail_lr.py"))


def paired_initialization(seed, depth):
    """Match all initial values, without tying the untied parameter storage."""
    torch.manual_seed(seed)
    shared = make_model("SJ", steps=depth)
    shared_rng = torch.get_rng_state().clone()
    torch.manual_seed(seed)
    untied = make_model("UJ", steps=depth)
    if not torch.equal(shared_rng, torch.get_rng_state()):
        raise ValueError("Paired model initialization consumed different Torch randomness.")
    common = shared.state_dict()
    actual = untied.state_dict()
    for name, tensor in actual.items():
        canonical = name
        if name.startswith("blocks."):
            components = name.split(".")
            canonical = ".".join([components[0], "0", *components[2:]])
        if canonical not in common or not torch.equal(tensor, common[canonical]):
            raise ValueError("Untied initial blocks, normalization or heads differ from the shared seed.")
    pointers = [parameter.data_ptr() for block in untied.blocks for parameter in block.parameters()]
    if len(set(pointers)) != len(pointers):
        raise ValueError("Untied blocks must own distinct parameter storage.")
    fingerprint = initialization_fingerprint(shared)
    certificate = {"canonical_shared_sha256": fingerprint,
                   "all_initial_block_and_head_values_match": True,
                   "untied_parameter_storage_distinct": True,
                   "torch_rng_after_initialization_matches": True}
    return untied, initialization_fingerprint(untied), certificate


def branch_config(seed, depth, policy):
    return {**expected_config(seed, depth), "arm": "UJ", "tail_policy": policy,
            "prefix_updates": PREFIX_UPDATES, "tail_learning_rate": TAIL_LRS[policy]}


def _binding(directory, filenames):
    return {key: file_digest(directory/name) for key, name in filenames.items()}


def build_protocol(args):
    torch.set_num_threads(args.threads)
    shared_root = Path(args.shared_run_dir)
    shared_dirs = [shared_root/f"SJ-depth{d}-{policy}-seed{s}"
                   for s in SEEDS for d in DEPTHS for policy in POLICIES]
    flat_dirs = [Path(args.shared_reference_run_dir)/f"SJ-depth{d}-terminal-seed{s}"
                 for s in SEEDS for d in DEPTHS]
    analysis = analyze_tail_lr(shared_dirs, protocol_path=args.shared_protocol,
                              reference_run_dirs=flat_dirs,
                              reference_protocol_path=args.shared_reference_protocol)
    shared = json.loads(Path(args.shared_protocol).read_text())
    rows = load_manifest(args.manifest)
    train = select_training_rows(rows, 256, sampling="proportional", selection_seed=20261004)
    tune = [r for r in rows if r["split"] == "tune"]
    sources = {"training_source_sha256": training_source_fingerprint(),
               "encoder_sha256": file_digest(args.encoder_weights),
               "manifest_sha256": file_digest(args.manifest)}
    if counts(train) != shared["train"] or counts(tune) != shared["tune"]:
        raise ValueError("Architecture controls must retain exact train/tune Cases, views and labels.")
    if any(value != shared[key] for key, value in sources.items()):
        raise ValueError("Core sources, encoder or manifest differ from the bound shared reference.")
    init, common_init, pairing = {}, {}, {}
    for seed in SEEDS:
        for depth in DEPTHS:
            _, full_hash, certificate = paired_initialization(seed, depth)
            common_hash = certificate["canonical_shared_sha256"]
            if common_hash != shared["initialization_sha256_by_seed"][str(seed)]:
                raise ValueError("Initial components differ from the existing shared model.")
            init[f"seed{seed}_depth{depth}"] = full_hash
            common_init[str(seed)] = common_hash
            pairing[f"seed{seed}_depth{depth}"] = certificate
    files = {"summary_sha256": "summary.json", "checkpoint_sha256": "checkpoint.pt",
             "training_predictions_sha256": "training_predictions.csv",
             "development_predictions_sha256": "development_predictions.csv",
             "history_sha256": "history.json"}
    shared_bindings = {f"seed{s}_depth{d}_{policy}":
                       _binding(shared_root/f"SJ-depth{d}-{policy}-seed{s}", files)
                       for s in SEEDS for d in DEPTHS for policy in POLICIES}
    prefix_files = {"prefix_summary_sha256": "prefix_summary.json", "checkpoint_sha256": "checkpoint.pt",
                    "training_predictions_sha256": "training_predictions.csv",
                    "development_predictions_sha256": "development_predictions.csv"}
    prefix_bindings = {f"seed{s}_depth{d}": _binding(shared_root/f"prefix-depth{d}-seed{s}", prefix_files)
                       for s in SEEDS for d in DEPTHS}
    parameters = {f"{arm}{depth}": parameter_counts(make_model(arm, steps=depth))
                  for arm in ("SJ", "UJ") for depth in DEPTHS}
    protocol = {**{key: shared[key] for key in (
        "device", "depths", "seeds", "supervisions", "epochs", "batch_size", "learning_rate",
        "segmentation_weight", "selection_seed", "sampling", "augment", "cached_frozen_features",
        "eval_every", "training_updates_per_run", "model_selection", "patient_mapping",
        "clinical_validation", "calibration_and_test_predictions", "train", "tune")},
        **sources, "status": "fixed_before_cpu_architecture_comparison",
        "phase": "exploratory_sharing_depth_comparison_on_observed_tune",
        "arms": ["SJ", "UJ"], "new_training_arms": ["UJ"], "threads": args.threads,
        "policies": list(POLICIES), "prefix_epochs": 20, "tail_epochs": 5,
        "prefix_updates": PREFIX_UPDATES, "tail_updates": TAIL_UPDATES,
        "tail_learning_rates": dict(TAIL_LRS), "updates_per_epoch": 128,
        "total_actual_new_updates": 38400, "new_endpoint_trajectories": 20,
        "reused_shared_endpoints": 20, "effective_endpoint_trajectories": 40,
        "new_prefixes": 10, "full_state_fork": True, "warm_starts": False,
        "launcher_source_sha256": launcher_digest(), "helper_source_sha256": helper_digest(),
        "shared_launcher_source_sha256": shared["launcher_source_sha256"],
        "shared_protocol_sha256": file_digest(args.shared_protocol),
        "shared_reference_protocol_sha256": file_digest(args.shared_reference_protocol),
        "shared_run_artifact_bindings": shared_bindings,
        "shared_prefix_artifact_bindings": prefix_bindings,
        "initialization_sha256_by_seed_depth": init,
        "common_initialization_sha256_by_seed": common_init,
        "initial_component_pairing_by_seed_depth": pairing,
        "parameters_by_arm_depth": parameters, "encoder_frozen_parameters": 5717416,
        "shared_alignment_checks_before_new_training": analysis["alignment_checks"],
        "primary_metric": "raw Case Brier at fixed update3200",
        "architecture_contrast": "A_depth,policy = Brier_UJ - Brier_SJ; positive favors sharing",
        "depth_contrast": "G_arm,policy = Brier_depth2 - Brier_depth4",
        "interaction": "Theta_policy = G_SJ - G_UJ",
        "error_diagnostics": {"classification_threshold": .5, "high_confidence_false_negative_max": .05,
                              "high_confidence_false_positive_min": .95, "bounds": "inclusive",
                              "unit": "Case", "cross_model_changes_are_not_internal_correction": True},
        "bootstrap": {"draws": 2000, "seed": 17, "unit": "Case",
                      "conditional_on_fitted_models": True,
                      "truth_specific_effects_resample_within_pathology_class": True},
        "temperature_fits_or_threshold_search": False,
        "lesion_intervention_executed": False,
        "lesion_phase_review": "Qualitative review of reproducible beneficial or harmful sharing/depth patterns across seeds, policies and true labels; inspect concentration. No numerical automatic gate, no case selection or intervention in this phase.",
        "interpretation": "Same width and executed depth sharing comparison, not equal parameters, fully tuned architecture superiority, clinical validation, or same-model inference correction.",
        "initialization_policy": "Untied blocks start at the shared block's values and own independent storage; heads and readout normalization match. This controlled contrast differs from independently randomized ordinary layers.",
        "budget": "10 fresh untied prefixes x2560 plus20 untied tails x640 =38400 new updates;20 shared endpoints reused as paired references, not extra seeds."}
    return protocol, train, tune


def _metadata(protocol, seed, depth):
    key = f"seed{seed}_depth{depth}"
    return {"clinical_validation": False, "patient_mapping": "unverified_Case_grouped_only", "device": "cpu",
            "evaluation_split": "tune_development", "model_selection": "fixed_final_epoch_no_best_epoch_selection",
            **{key: protocol[key] for key in ("training_source_sha256", "encoder_sha256", "manifest_sha256",
                                             "launcher_source_sha256", "helper_source_sha256")},
            "initialization_sha256": protocol["initialization_sha256_by_seed_depth"][key],
            "common_initialization_sha256": protocol["common_initialization_sha256_by_seed"][str(seed)],
            "initial_component_pairing": protocol["initial_component_pairing_by_seed_depth"][key],
            "n_training_cases": 256, "n_training_images": 445,
            "cohort_selection": {"sampling": "proportional", "selection_seed": 20261004,
                                 "train_fingerprint": protocol["train"]["fingerprint"],
                                 "tune_fingerprint": protocol["tune"]["fingerprint"]},
            "readout_supervision": {"mode": "terminal", "directly_supervised_steps": [depth],
                                   "intermediate_predictions_scope": "diagnostic_only_no_direct_intermediate_readout_loss"}}


def verify_locked_sources(protocol):
    if (training_source_fingerprint() != protocol["training_source_sha256"]
            or launcher_digest() != protocol["launcher_source_sha256"]
            or helper_digest() != protocol["helper_source_sha256"]):
        raise ValueError("Architecture runner, reused training helper or core changed after lock.")


def run_pair(args, protocol, train_rows, tune_rows, seed, depth):
    root = Path(args.output_dir)
    prefix_dir = root/f"prefix-depth{depth}-seed{seed}"
    prefix_dir.mkdir(exist_ok=True)
    with (prefix_dir/"pair_claim.json").open("x") as stream:
        json.dump({"arm": "UJ", "seed": seed, "depth": depth, "worker": args.worker_index}, stream)
    verify_locked_sources(protocol)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.use_deterministic_algorithms(True)
    encoder = create_encoder(pretrained=True, weights_path=args.encoder_weights)
    model, init_hash, certificate = paired_initialization(seed, depth)
    key = f"seed{seed}_depth{depth}"
    if (init_hash != protocol["initialization_sha256_by_seed_depth"][key]
            or certificate != protocol["initial_component_pairing_by_seed_depth"][key]):
        raise ValueError("Untied initialization no longer matches locked component values and storage.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
    cached, _, reused = load_or_encode(encoder, train_rows, args.encoder_weights, args.feature_cache_dir, 2)
    tune_cache, _, _ = load_or_encode(encoder, tune_rows, args.encoder_weights, args.feature_cache_dir, 2)
    dataset = UltrasoundDataset(train_rows, augment=False)
    def new_loader():
        return DataLoader(dataset, batch_size=2, sampler=CaseViewSampler(train_rows, seed), num_workers=0,
                          generator=torch.Generator().manual_seed(seed))
    loader = new_loader()
    feature_index = {row["image_id"]: i for i, row in enumerate(train_rows)}
    fit_args = argparse.Namespace(arm="UJ", seed=seed, steps=depth, batch_size=2,
                                  encoder=encoder, tune_rows=tune_rows, tune_cache=tune_cache)
    _, initial = evaluate(model, encoder, train_rows, fit_args, cache=cached)
    begin = time.perf_counter()
    history, snapshots, prefix_times, prefix_trace = train_epochs(
        model, optimizer, loader, cached, feature_index, fit_args, 0, 20)
    prefix_seconds = time.perf_counter()-begin
    train_records, prefix_train = evaluate(model, encoder, train_rows, fit_args, cache=cached)
    dev_records, prefix_dev = evaluate(model, encoder, tune_rows, fit_args, cache=tune_cache)
    write_records(prefix_dir/"training_predictions.csv", train_records)
    write_records(prefix_dir/"development_predictions.csv", dev_records)
    state = capture_state(model, optimizer, loader)
    state_sha = state_digest(state)
    torch.save({"state": state, "arm": "UJ", "seed": seed, "depth": depth, "updates": PREFIX_UPDATES,
                "state_sha256": state_sha, "protocol_sha256": file_digest(root/"protocol.json")},
               prefix_dir/"checkpoint.pt")
    prefix_sha = file_digest(prefix_dir/"checkpoint.pt")
    meta = _metadata(protocol, seed, depth)
    prefix_summary = {**meta, "status": "executed_cpu_architecture_prefix",
        "config": {**expected_config(seed, depth), "arm": "UJ", "prefix_updates": PREFIX_UPDATES},
        "checkpoint_sha256": prefix_sha, "state_sha256": state_sha,
        "training_predictions_sha256": file_digest(prefix_dir/"training_predictions.csv"),
        "development_predictions_sha256": file_digest(prefix_dir/"development_predictions.csv"),
        "timing": {"updates": PREFIX_UPDATES}, "training_seconds": prefix_seconds,
        "development_subset": prefix_dev, "final_training_fit": prefix_train,
        "development_history": snapshots, "history": history, "sampling_trace_sha256": prefix_trace}
    (prefix_dir/"prefix_summary.json").write_text(json.dumps(prefix_summary, indent=2, allow_nan=False)+"\n")
    results = []
    for policy in POLICIES:
        verify_locked_sources(protocol)
        directory = root/f"UJ-depth{depth}-{policy}-seed{seed}"
        directory.mkdir(exist_ok=True)
        with (directory/"branch_claim.json").open("x") as stream:
            json.dump({"arm": "UJ", "seed": seed, "depth": depth, "policy": policy}, stream)
        payload = torch.load(prefix_dir/"checkpoint.pt", map_location="cpu", weights_only=True)
        if (file_digest(prefix_dir/"checkpoint.pt") != prefix_sha or payload["arm"] != "UJ"
                or payload["seed"] != seed or payload["depth"] != depth or payload["updates"] != PREFIX_UPDATES
                or payload["protocol_sha256"] != file_digest(root/"protocol.json")
                or state_digest(payload["state"]) != state_sha):
            raise ValueError("Prefix file, full state or identity changed before the fork.")
        branch_model = make_model("UJ", steps=depth)
        branch_optimizer = torch.optim.AdamW(branch_model.parameters(), lr=.0003, weight_decay=.01)
        branch_loader = new_loader()
        restore_state(payload["state"], branch_model, branch_optimizer, branch_loader)
        resumed_sha = state_digest(capture_state(branch_model, branch_optimizer, branch_loader))
        if resumed_sha != state_sha or branch_loader.sampler.epoch != 20:
            raise ValueError("Untied fork failed exact full-state restoration.")
        for group in branch_optimizer.param_groups: group["lr"] = TAIL_LRS[policy]
        start = time.perf_counter()
        tail_history, tail_snapshots, tail_times, tail_trace = train_epochs(
            branch_model, branch_optimizer, branch_loader, cached, feature_index, fit_args, 20, 25)
        branch_seconds = time.perf_counter()-start
        full_history, full_snapshots = deepcopy(history)+tail_history, deepcopy(snapshots)+tail_snapshots
        shared_summary = json.loads((Path(args.shared_run_dir)/f"SJ-depth{depth}-{policy}-seed{seed}"/"summary.json").read_text())
        if shared_summary["sampling_trace_sha256"] != {"prefix": prefix_trace, "tail": tail_trace}:
            raise ValueError("Untied model sampling trajectory differs from its shared architecture reference.")
        if policy == "constant":
            verification = {"status": "first_constant_branch_from_verified_prefix"}
        else:
            constant_path = root/f"UJ-depth{depth}-constant-seed{seed}"/"summary.json"
            constant = json.loads(constant_path.read_text())
            if constant["sampling_trace_sha256"] != {"prefix": prefix_trace, "tail": tail_trace}:
                raise ValueError("Untied policy branches sampled different Case/views.")
            verification = {"status": "full_state_policy_pair_verified",
                            "constant_summary_sha256": file_digest(constant_path)}
        train_records, final = evaluate(branch_model, encoder, train_rows, fit_args, cache=cached)
        dev_records, development = evaluate(branch_model, encoder, tune_rows, fit_args, cache=tune_cache)
        write_records(directory/"training_predictions.csv", train_records)
        write_records(directory/"development_predictions.csv", dev_records)
        torch.save({"model_state": branch_model.state_dict(), "arm": "UJ", "seed": seed, "steps": depth,
                    "tail_policy": policy, "initialization_sha256": init_hash}, directory/"checkpoint.pt")
        summary = {**meta, "status": "executed_cpu_architecture_control",
            "config": branch_config(seed, depth, policy), "initial_training_fit": initial,
            "final_training_fit": final, "development_subset": development,
            "development_history": full_snapshots, "feature_cache_reused": reused,
            "parameters": {**parameter_counts(branch_model),
                           "encoder_frozen": sum(p.numel() for p in encoder.parameters())},
            "training_seconds": prefix_seconds+branch_seconds,
            "timing": {"updates": 3200, "new_updates_in_branch": TAIL_UPDATES, "prefix_reused": True,
                       "branch_training_seconds": branch_seconds,
                       "median_training_step_seconds": float(np.median(prefix_times+tail_times))},
            "branch_provenance": {"prefix_checkpoint_sha256": prefix_sha, "prefix_state_sha256": state_sha,
                "resumed_state_sha256": resumed_sha, "prefix_updates": PREFIX_UPDATES,
                "restored_sampler_epoch": 20, "restored_loader_generator": True, "restored_rng": True},
            "sampling_trace_sha256": {"prefix": prefix_trace, "tail": tail_trace},
            "paired_policy_verification": verification,
            "limitations": ["Observed tune, Case-to-patient mapping unverified, no clinical validation.",
                           "Terminal readout training; intermediate predictions are diagnostic only.",
                           "Independent layers start from equal values, but have distinct parameter storage.",
                           "Same width/depth, not equal parameters, training FLOPs or optimal tuning.",
                           "Policy branches share full prefix states; endpoints are not independent seeds."]}
        (directory/"history.json").write_text(json.dumps(full_history, indent=2, allow_nan=False)+"\n")
        (directory/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        results.append({"seed": seed, "depth": depth, "policy": policy,
                        "endpoint": development[str(depth)], "branch_training_seconds": branch_seconds})
        print(json.dumps({"event": "branch_complete", **results[-1]}), flush=True)
    return {"seed": seed, "depth": depth, "prefix_training_seconds": prefix_seconds,
            "actual_new_updates": PREFIX_UPDATES+2*TAIL_UPDATES, "branches": results}


def run(args):
    if args.threads != 2:
        raise ValueError("The paired architecture controls use the reference's two CPU threads.")
    begin = time.perf_counter()
    root = Path(args.output_dir); root.mkdir(parents=True, exist_ok=True)
    pairs = assigned_pairs(args.workers, args.worker_index)
    protocol, train, tune = build_protocol(args)
    write_once_or_match(root/"protocol.json", protocol)
    if args.prepare_only:
        encoder = create_encoder(pretrained=True, weights_path=args.encoder_weights)
        reused = [load_or_encode(encoder, rows, args.encoder_weights, args.feature_cache_dir, 2)[2]
                  for rows in (train, tune)]
        print(json.dumps({"event": "architecture_protocol_and_cache_ready",
                          "train_tune_cache_reused": reused}), flush=True)
        return protocol
    completed = []
    for seed, depth in pairs:
        print(json.dumps({"event": "pair_start", "arm": "UJ", "seed": seed, "depth": depth}), flush=True)
        completed.append(run_pair(args, protocol, train, tune, seed, depth))
        progress = {"status": "worker_complete" if len(completed) == len(pairs) else "running",
                    "worker_index": args.worker_index, "completed_pairs": completed,
                    "actual_new_updates": sum(pair["actual_new_updates"] for pair in completed),
                    "elapsed_seconds": time.perf_counter()-begin}
        (root/f"progress-worker{args.worker_index}.json").write_text(json.dumps(progress, indent=2, allow_nan=False)+"\n")
    return progress


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--encoder-weights", default="data/deit_tiny_encoder.pt")
    p.add_argument("--shared-protocol", default="outputs/cpu-tail-lr256/protocol.json")
    p.add_argument("--shared-run-dir", default="outputs/cpu-tail-lr256")
    p.add_argument("--shared-reference-protocol", default="outputs/cpu-case-expansion256-replication/protocol.json")
    p.add_argument("--shared-reference-run-dir", default="outputs/cpu-case-expansion256-replication")
    p.add_argument("--feature-cache-dir", default="outputs/cpu-expanded128/features")
    p.add_argument("--output-dir", default="outputs/cpu-architecture256")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--workers", type=int, choices=[1, 2], default=1)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--prepare-only", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
