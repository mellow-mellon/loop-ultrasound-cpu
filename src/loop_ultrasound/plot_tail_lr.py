"""Aggregate-only scientific figures for the paired CPU tail-LR control.

Reads the locked protocol, aggregate analysis and twenty public summaries.
No prediction CSV, checkpoint or patient/image identifier is read. Primary
effects use five paired seeds and separately trained depth endpoints.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from .analyze_tail_lr import (DEPTHS, POLICIES, RATES, SEEDS, _protocol,
                             _source_summary, regime_name)
from .plot_seed_replication import _metric, _mean_sd, _read_json


DEFAULT_PROTOCOL = "configs/cpu_tail_lr256.json"
COLORS = {(2, "constant"): "#0072B2", (2, "low_tail"): "#56B4E9",
          (4, "constant"): "#D55E00", (4, "low_tail"): "#E69F00"}
DEPTH_COLORS = {2: "#0072B2", 4: "#D55E00"}
LABELS = {(depth, policy): f"{depth}-step: {'constant LR' if policy == 'constant' else 'lower tail LR'}"
          for policy in POLICIES for depth in DEPTHS}
DISCLAIMER = (
    "Exploratory control on an already observed tune cohort; Case-to-patient mapping unverified; no clinical validation.\n"
    "Five paired seeds; means are not ensembles; sample SD is not a confidence interval. Frozen features; no augmentation.\n"
    "Policies share the full prefix state and sampled views. Final update 3,200 fixed; four steps use more compute.\n"
    "10 prefixes x 2,560 + 20 tails x 640 = 38,400 actual new updates. Constant branches exactly replay the original fits."
)


def load_reports(report_dir, *, protocol_path=DEFAULT_PROTOCOL):
    report_dir = Path(report_dir)
    p = _protocol(protocol_path, 256, 159)
    analysis = _read_json(report_dir/"tail_lr_analysis.json")
    if (analysis.get("status") != "executed_cpu_tail_lr_analysis"
            or analysis.get("clinical_validation") is not False
            or analysis.get("training_seeds") != list(SEEDS)
            or analysis.get("depths") != list(DEPTHS) or analysis.get("policies") != list(POLICIES)
            or analysis.get("n_training_cases") != 256 or analysis.get("n_development_cases") != 159
            or analysis.get("evaluation_split") != "tune_development"):
        raise ValueError("Analysis is not the five-seed fixed-cohort CPU tail-LR control.")
    from hashlib import sha256
    source = analysis.get("sources", {})
    source_keys = ("encoder_sha256", "manifest_sha256", "training_source_sha256",
                   "launcher_source_sha256", "reference_protocol_sha256")
    if (any(source.get(key) != p[key] for key in source_keys)
            or source.get("protocol_sha256") != sha256(Path(protocol_path).read_bytes()).hexdigest()
            or analysis.get("cohort_fingerprints") != {split: p[split]["fingerprint"] for split in ("train", "tune")}):
        raise ValueError("Aggregate analysis source/cohort/protocol bindings disagree.")
    checks = analysis.get("alignment_checks", {})
    for key in ("exact_twenty_policy_depth_seed_branches", "all_prefix_artifact_digests_verified",
                "full_prefix_state_restoration_certified", "paired_sampling_trajectories_match",
                "constant_prediction_csv_bytes_replay_reference", "constant_objective_and_development_histories_replay_reference",
                "low_branch_binds_verified_constant_summary"):
        if checks.get(key) is not True:
            raise ValueError("The paired-prefix and exact-replay checks are incomplete.")
    runs = {(seed, depth, policy): _read_json(report_dir/f"SJ-depth{depth}-{policy}-seed{seed}.json")
            for seed in SEEDS for depth in DEPTHS for policy in POLICIES}
    for key, run in runs.items():
        seed, depth, policy = key
        if _source_summary(run, p, status="executed_cpu_tail_lr_control") != (seed, depth):
            raise ValueError("Run summary belongs to a different seed or trained depth.")
        config, timing = run["config"], run.get("timing", {})
        if (config.get("tail_policy") != policy or config.get("tail_learning_rate") != RATES[policy]
                or config.get("prefix_updates") != 2560 or timing.get("updates") != 3200
                or timing.get("new_updates_in_branch") != 640):
            raise ValueError("The branch LR or update schedule differs from the locked control.")
        history = run.get("development_history")
        if (not isinstance(history, list) or not all(isinstance(row, dict) for row in history)
                or [row.get("epoch") for row in history] != [5, 10, 15, 20, 25]):
            raise ValueError("Run summary does not record the full fixed evaluation schedule.")
        for row in history:
            endpoint = row.get("by_step", {}).get(str(depth), {})
            if endpoint.get("n_cases") != 159:
                raise ValueError("Run history has a changed development cohort.")
            for metric in ("brier", "auroc", "nll", "mean_case_dice"):
                _metric(endpoint, metric, "branch history")
        endpoint = run.get("development_subset", {}).get(str(depth), {})
        for metric in ("brier", "auroc", "nll", "mean_case_dice"):
            value = _metric(endpoint, metric, "final endpoint")
            if not math.isclose(value, history[-1]["by_step"][str(depth)][metric], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("Final metrics disagree with the recorded fixed endpoint.")
    regimes = analysis.get("regimes", {})
    if set(regimes) != {regime_name(depth, policy) for policy in POLICIES for depth in DEPTHS}:
        raise ValueError("Analysis must describe exactly the four policy/depth regimes.")
    for policy in POLICIES:
        for depth in DEPTHS:
            for metric in ("brier", "auroc", "nll", "mean_case_dice"):
                mean, sd = _mean_sd([runs[seed, depth, policy]["development_subset"][str(depth)][metric] for seed in SEEDS])
                declared = regimes[regime_name(depth, policy)].get("endpoint", {}).get(metric, {})
                for name, expected in (("mean", mean), ("std_across_training_seeds", sd)):
                    value = declared.get(name)
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(value) or not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-12)):
                        raise ValueError("Analysis endpoint means/SD disagree with the public run summaries.")
    gains = {policy: np.array([runs[seed, 2, policy]["development_subset"]["2"]["brier"]-
                               runs[seed, 4, policy]["development_subset"]["4"]["brier"] for seed in SEEDS])
             for policy in POLICIES}
    effects = [(analysis.get("primary_H", {}), gains["low_tail"]-gains["constant"])]
    effects.extend((analysis.get("depth_benefit_Brier", {}).get(policy, {}), gains[policy]) for policy in POLICIES)
    for claim, values in effects:
        mean, sd = _mean_sd(values)
        for key, expected in (("mean", mean), ("estimate", mean), ("std_across_training_seeds", sd)):
            value = claim.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-12)):
                raise ValueError("Analysis G/H effects disagree with the public endpoint metrics.")
        by_seed = claim.get("by_seed", [])
        if (not isinstance(by_seed, list) or len(by_seed) != 5
                or not all(isinstance(row, dict) for row in by_seed)
                or [row.get("seed") for row in by_seed] != list(SEEDS)):
            raise ValueError("Analysis G/H effects must preserve the five paired seeds.")
        for row, expected in zip(by_seed, values):
            value = row.get("value")
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-12)):
                raise ValueError("Analysis seed G/H effects disagree with endpoint metrics.")
    return runs


def _decorate(fig, axes, title, subtitle, *, legend_axes=None):
    fig.suptitle(title, fontsize=13, fontweight="bold", y=.975)
    fig.text(.5, .91, subtitle, ha="center", fontsize=9.5, color="#333333")
    handles, labels = [], []
    for axis in axes:
        axis.grid(alpha=.2, linewidth=.6)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in legend_axes or [axes[0]]:
        h, l = axis.get_legend_handles_labels(); handles.extend(h); labels.extend(l)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .19), ncol=2, frameon=False, fontsize=8.5)
    fig.text(.5, .025, DISCLAIMER, ha="center", va="bottom", fontsize=8, color="#333333", linespacing=1.45)
    fig.subplots_adjust(left=.075, right=.97, top=.81, bottom=.40, wspace=.30)


def _render_figures(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    endpoint, axes = plt.subplots(1, 3, figsize=(15, 7))
    for axis, metric, ylabel in zip(axes, ("brier", "auroc", "mean_case_dice"),
                                   ("Endpoint Brier (lower is better)", "Endpoint AUROC (higher is better)", "Mean Case Dice (higher is better)")):
        for depth in DEPTHS:
            values = [[runs[seed, depth, policy]["development_subset"][str(depth)][metric] for policy in POLICIES] for seed in SEEDS]
            mean, sd = _mean_sd(values)
            axis.errorbar((0, 1), mean, yerr=sd, color=DEPTH_COLORS[depth], marker="o" if depth == 2 else "s",
                          linewidth=1.8, markersize=6, capsize=4, label=f"Separately trained {depth}-step endpoint")
        axis.set(xlabel="LR in the last 640 updates", ylabel=ylabel, xticks=(0, 1),
                 xticklabels=("3e-4\nconstant", "3e-5\nlower tail LR"), xlim=(-.2, 1.2))
        axis.set_title("Mean +/- sample SD across five paired seeds", fontsize=10)
    _decorate(endpoint, axes, "Endpoint comparison | paired tail-LR control", "256 training / 159 development Cases | terminal-only supervision | final update 3,200")
    effects, axes = plt.subplots(1, 3, figsize=(15, 7))
    brier = {(policy, depth): np.array([runs[seed, depth, policy]["development_subset"][str(depth)]["brier"] for seed in SEEDS])
             for policy in POLICIES for depth in DEPTHS}
    gains = {policy: brier[policy, 2]-brier[policy, 4] for policy in POLICIES}
    x = np.arange(len(SEEDS))
    for offset, policy, color in ((-.17, "constant", "#666666"), (.17, "low_tail", "#0072B2")):
        axes[0].bar(x+offset, gains[policy], width=.32, color=color, alpha=.85, label=f"Depth benefit: {policy.replace('_', ' ')}")
    axes[0].set(ylabel="G = Brier2 - Brier4 (positive favors 4 steps)")
    axes[0].set_title(f"Absolute depth benefit by policy\nMean Gconstant: {gains['constant'].mean():+.4f}; Glow: {gains['low_tail'].mean():+.4f}", fontsize=10)
    H = gains["low_tail"]-gains["constant"]
    axes[1].bar(x, H, width=.6, color=["#0072B2" if v >= 0 else "#D55E00" for v in H], alpha=.85)
    axes[1].axhline(float(H.mean()), color="#111111", linestyle="--", linewidth=1.4)
    axes[1].set(ylabel="H = Glow - Gconstant")
    axes[1].set_title(f"Primary: tail policy x trained depth\nMean H: {H.mean():+.4f}", fontsize=10)
    for offset, depth in ((-.17, 2), (.17, 4)):
        values = brier["constant", depth]-brier["low_tail", depth]
        axes[2].bar(x+offset, values, width=.32, color=DEPTH_COLORS[depth], alpha=.85, label=f"LR benefit: {depth}-step endpoint")
    axes[2].set(ylabel="Brierconstant - Brierlow (positive favors low LR)")
    axes[2].set_title("Same-depth LR benefit", fontsize=10)
    for axis in axes:
        axis.axhline(0, color="#555555", linewidth=.9)
        axis.set(xlabel="Paired training seed", xticks=x, xticklabels=[str(seed) for seed in SEEDS]); axis.margins(y=.25)
    _decorate(effects, axes, "Paired seed effects | optimizer policy interaction", "H and absolute G answer different questions; a positive H alone does not prove positive depth benefit", legend_axes=[axes[0], axes[2]])
    curves, axes = plt.subplots(1, 3, figsize=(15, 7))
    for axis, metric, ylabel in zip(axes, ("brier", "auroc", "nll"),
                                   ("Development Brier (lower is better)", "Development AUROC (higher is better)", "Development NLL (lower is better)")):
        for policy in POLICIES:
            for depth in DEPTHS:
                values = [[entry["by_step"][str(depth)][metric] for entry in runs[seed, depth, policy]["development_history"] if entry["epoch"] in (20, 25)] for seed in SEEDS]
                mean, sd = _mean_sd(values)
                axis.plot((2560, 3200), mean, color=COLORS[depth, policy], marker="o" if depth == 2 else "s",
                          markersize=5, linewidth=1.8, linestyle="-" if policy == "constant" else "--", label=LABELS[depth, policy])
                axis.fill_between((2560, 3200), mean-sd, mean+sd, color=COLORS[depth, policy], alpha=.14, linewidth=0)
        axis.set(xlabel="Optimizer updates", ylabel=ylabel, xticks=(2560, 3200), xticklabels=("2,560\nshared prefix", "3,200\nfixed endpoint"))
        axis.set_title("Observed prefix/endpoint means +/- sample SD", fontsize=10)
    _decorate(curves, axes, "Recorded tail checkpoints | cloned full-state prefix", "Two observed checkpoints; connecting lines do not represent additional intermediate evaluations")
    payloads = {}
    figures = (("endpoint_comparison", endpoint), ("seed_effects", effects), ("tail_curves", curves))
    try:
        for name, figure in figures:
            for extension in ("png", "svg"):
                stream = BytesIO(); figure.savefig(stream, format=extension, dpi=180); payloads[f"{name}.{extension}"] = stream.getvalue()
    finally:
        for _, figure in figures: plt.close(figure)
    return payloads


def create_figures(report_dir, output_dir, *, protocol_path=DEFAULT_PROTOCOL):
    runs = load_reports(report_dir, protocol_path=protocol_path)
    output_dir = Path(output_dir)
    names = [f"{name}.{extension}" for name in ("endpoint_comparison", "seed_effects", "tail_curves") for extension in ("png", "svg")]
    if any((output_dir/name).exists() for name in names):
        raise ValueError("Figure outputs already exist; choose a fresh output directory.")
    saved_env = {key: os.environ.get(key) for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME")}
    try:
        with tempfile.TemporaryDirectory(prefix="loop-ultrasound-tail-lr-plot-") as cache:
            os.environ["MPLCONFIGDIR"] = cache; os.environ["XDG_CACHE_HOME"] = cache
            try: payloads = _render_figures(runs)
            except ModuleNotFoundError as error:
                if error.name == "matplotlib": raise ValueError("Optional plotting dependency missing; install matplotlib.") from error
                raise
    finally:
        for key, value in saved_env.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        try:
            with (output_dir/name).open("xb") as stream: stream.write(payload)
        except FileExistsError: raise ValueError("Figure outputs already exist; choose a fresh output directory.") from None
    return [str(output_dir/name) for name in names]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", default="reports/tail_lr256")
    parser.add_argument("--output-dir", default="reports/tail_lr256/figures")
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    args = parser.parse_args(argv)
    try: paths = create_figures(args.report_dir, args.output_dir, protocol_path=args.protocol)
    except ValueError as error: parser.error(str(error))
    print(json.dumps({"status": "aggregate_tail_lr_figures_created", "files": paths}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
