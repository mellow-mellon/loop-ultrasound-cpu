"""Aggregate-only scientific figures for shared/untied CPU refiners.

Reads the locked architecture protocol, aggregate analysis and forty public
branch summaries. No prediction CSV, checkpoint or Case/image identifier is
read. The comparison changes trainable parameter count and compute cost.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from .plot_seed_replication import _digest, _mean_sd, _metric, _read_json


ARMS = ("SJ", "UJ")
DEPTHS = (2, 4)
POLICIES = ("constant", "low_tail")
SEEDS = (53, 71, 89, 107, 131)
MODELS = (("UJ", 2), ("UJ", 4), ("SJ", 2), ("SJ", 4))
PARAMETERS = {"UJ2": 1159394, "UJ4": 2049122, "SJ2": 714530, "SJ4": 714530}
ENCODER_PARAMETERS = 5717416
RATES = {"constant": .0003, "low_tail": .00003}
SOURCE_KEYS = ("encoder_sha256", "manifest_sha256", "training_source_sha256",
               "launcher_source_sha256", "helper_source_sha256", "shared_launcher_source_sha256",
               "shared_protocol_sha256", "shared_reference_protocol_sha256")
DEFAULT_PROTOCOL = "configs/cpu_architecture256.json"
FIXED_CONFIG = {"epochs": 25, "batch_size": 2, "learning_rate": .0003,
                "segmentation_weight": 1., "cached_frozen_features": True,
                "augment": False, "selection_seed": 20261004, "sampling": "proportional",
                "eval_every": 5, "supervision": "terminal", "prefix_updates": 2560}
MODEL_COLORS = {("UJ", 2): "#0072B2", ("UJ", 4): "#56B4E9",
                ("SJ", 2): "#D55E00", ("SJ", 4): "#E69F00"}
POLICY_COLORS = {"constant": "#666666", "low_tail": "#0072B2"}
POLICY_LABELS = {"constant": "Constant tail LR (3e-4)", "low_tail": "Lower tail LR (3e-5)"}
DISCLAIMER = (
    "Observed tune: 159 Cases (51 malignant, 108 benign). Five paired seeds; Case-to-patient mapping unverified; no clinical validation.\n"
    "Only refiners/heads are trained; common image encoder frozen (5,717,416 parameters). No augmentation; final update 3,200 fixed.\n"
    "Trainable parameters: UJ2 1,159,394; UJ4 2,049,122; SJ2/SJ4 714,530. Matched updates do not equal parameters or FLOPs.\n"
    "Untied blocks start at shared initial values but own independent parameter storage; this is a controlled refiner contrast.\n"
    "Means are not ensembles; variability is sample SD across fitted seeds, not confidence intervals. Cross-model errors are not internal corrections."
)


def regime_name(depth, policy):
    return f"depth{depth}_{policy}"


def _number(value, context, *, minimum=0., maximum=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < minimum or (maximum is not None and value > maximum)):
        raise ValueError(f"Invalid/nonfinite aggregate value in {context}")
    return float(value)


def _count(value, context, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"Invalid Case count in {context}")
    return value


def _config(mapping, expected, context):
    if (not isinstance(mapping, dict)
            or any(mapping.get(key) != value or type(mapping.get(key)) is not type(value)
                   for key, value in expected.items())):
        raise ValueError(f"Configuration differs from the locked design in {context}")


def _stats_match(claim, values, context):
    values = np.asarray(values, dtype=np.float64)
    mean, sd = _mean_sd(values)
    expected = {"mean": mean, "std_across_training_seeds": sd, "min": values.min(), "max": values.max()}
    if not isinstance(claim, dict):
        raise ValueError(f"Missing aggregate statistics in {context}")
    for key, target in expected.items():
        value = _number(claim.get(key), context, minimum=-math.inf)
        if not math.isclose(value, target, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"Aggregate means/spread disagree with fitted summaries in {context}")


def _effect_match(claim, values, context):
    _stats_match(claim, values, context)
    if not math.isclose(_number(claim.get("estimate"), context, minimum=-math.inf), np.mean(values), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"Paired effect estimate disagrees with fitted summaries in {context}")
    rows = claim.get("by_seed")
    if (not isinstance(rows, list) or len(rows) != 5 or not all(isinstance(row, dict) for row in rows)
            or [row.get("seed") for row in rows] != list(SEEDS)):
        raise ValueError(f"Paired effects must preserve all five seed identities in {context}")
    for row, expected in zip(rows, values):
        if not math.isclose(_number(row.get("value"), context, minimum=-math.inf), expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"Paired seed effect disagrees with fitted summaries in {context}")


def _seed_effects(runs):
    brier = {(arm, depth, policy): np.array([runs[arm, depth, policy, seed]["development_subset"][str(depth)]["brier"]
                                           for seed in SEEDS]) for arm in ARMS for depth in DEPTHS for policy in POLICIES}
    sharing = {(depth, policy): brier["UJ", depth, policy]-brier["SJ", depth, policy]
               for depth in DEPTHS for policy in POLICIES}
    gains = {(arm, policy): brier[arm, 2, policy]-brier[arm, 4, policy] for arm in ARMS for policy in POLICIES}
    theta = {policy: gains["SJ", policy]-gains["UJ", policy] for policy in POLICIES}
    return sharing, gains, theta


def load_reports(report_dir, *, protocol_path=DEFAULT_PROTOCOL):
    report_dir = Path(report_dir)
    p = _read_json(Path(protocol_path))
    if (p.get("status") != "fixed_before_cpu_architecture_comparison"
            or p.get("phase") != "exploratory_sharing_depth_comparison_on_observed_tune"
            or p.get("arms") != list(ARMS) or p.get("depths") != list(DEPTHS)
            or p.get("policies") != list(POLICIES) or p.get("seeds") != list(SEEDS)
            or p.get("device") != "cpu" or p.get("clinical_validation") is not False
            or p.get("patient_mapping") != "unverified_Case_grouped_only"
            or p.get("calibration_and_test_predictions") is not False
            or p.get("model_selection") != "fixed_final_epoch_no_best_epoch_selection"
            or p.get("training_updates_per_run") != 3200 or p.get("prefix_updates") != 2560
            or p.get("tail_updates") != 640 or p.get("tail_learning_rates") != RATES
            or p.get("encoder_frozen_parameters") != ENCODER_PARAMETERS):
        raise ValueError("Protocol is not the five-seed frozen-refiner architecture comparison.")
    _config(p, {key: value for key, value in FIXED_CONFIG.items() if key != "supervision"}, "protocol")
    for key in SOURCE_KEYS:
        if not _digest(p.get(key)): raise ValueError("Protocol source/artifact digests are incomplete.")
    if p["shared_launcher_source_sha256"] != p["helper_source_sha256"]:
        raise ValueError("The reused shared launcher must match the common locked tail helper.")
    _config(p.get("error_diagnostics"), {"classification_threshold": .5,
                                        "high_confidence_false_negative_max": .05,
                                        "high_confidence_false_positive_min": .95,
                                        "bounds": "inclusive", "unit": "Case"}, "pathology cutoffs")
    for split, counts in (("train", {"cases": 256, "images": 445, "benign": 174, "malignant": 82}),
                          ("tune", {"cases": 159, "images": 281, "benign": 108, "malignant": 51})):
        _config(p.get(split), counts, f"{split} cohort")
        if not _digest(p[split].get("fingerprint")): raise ValueError("Missing fixed-cohort fingerprint.")
    for name, count in PARAMETERS.items():
        values = p.get("parameters_by_arm_depth", {}).get(name, {})
        if values.get("trainable") != count:
            raise ValueError("Protocol trainable parameter counts differ from the disclosed architecture.")
    common_init = p.get("common_initialization_sha256_by_seed", {})
    untied_init = p.get("initialization_sha256_by_seed_depth", {})
    if (set(common_init) != {str(seed) for seed in SEEDS}
            or set(untied_init) != {f"seed{seed}_depth{depth}" for seed in SEEDS for depth in DEPTHS}
            or not all(_digest(value) for value in (*common_init.values(), *untied_init.values()))):
        raise ValueError("Protocol initial parameter bindings are incomplete.")
    analysis = _read_json(report_dir/"architecture_analysis.json")
    if (analysis.get("status") != "executed_cpu_architecture_analysis"
            or analysis.get("clinical_validation") is not False
            or analysis.get("training_seeds") != list(SEEDS) or analysis.get("arms") != list(ARMS)
            or analysis.get("depths") != list(DEPTHS) or analysis.get("policies") != list(POLICIES)
            or analysis.get("n_training_cases") != 256 or analysis.get("n_development_cases") != 159
            or analysis.get("evaluation_split") != "tune_development"):
        raise ValueError("Aggregate analysis does not describe the complete locked architecture design.")
    sources = analysis.get("sources", {})
    if (any(sources.get(key) != p[key] for key in SOURCE_KEYS if key != "shared_launcher_source_sha256")
            or sources.get("shared_launcher_source_sha256", p["shared_launcher_source_sha256"]) != p["shared_launcher_source_sha256"]
            or sources.get("protocol_sha256") != sha256(Path(protocol_path).read_bytes()).hexdigest()
            or analysis.get("cohort_fingerprints") != {split: p[split]["fingerprint"] for split in ("train", "tune")}):
        raise ValueError("Analysis protocol/source/cohort bindings disagree.")
    runs = {(arm, depth, policy, seed): _read_json(report_dir/f"{arm}-depth{depth}-{policy}-seed{seed}.json")
            for arm in ARMS for depth in DEPTHS for policy in POLICIES for seed in SEEDS}
    for (arm, depth, policy, seed), run in runs.items():
        context = f"{arm}, depth {depth}, {policy}, seed {seed}"
        status = "executed_cpu_architecture_control" if arm == "UJ" else "executed_cpu_tail_lr_control"
        if (run.get("status") != status or run.get("clinical_validation") is not False
                or run.get("device") != "cpu" or run.get("evaluation_split") != "tune_development"
                or run.get("patient_mapping") != "unverified_Case_grouped_only"
                or run.get("model_selection") != p["model_selection"]
                or run.get("n_training_cases") != 256 or run.get("n_training_images") != 445):
            raise ValueError(f"Wrong completed refiner provenance in {context}")
        _config(run.get("config"), {**FIXED_CONFIG, "arm": arm, "steps": depth, "seed": seed,
                                   "tail_policy": policy, "tail_learning_rate": RATES[policy]}, context)
        for key in ("encoder_sha256", "manifest_sha256", "training_source_sha256"):
            if run.get(key) != p[key]: raise ValueError(f"Core source/encoder/manifest mismatch in {context}")
        launcher = p["launcher_source_sha256"] if arm == "UJ" else p["shared_launcher_source_sha256"]
        if run.get("launcher_source_sha256") != launcher:
            raise ValueError(f"Arm-specific runner provenance mismatch in {context}")
        expected_init = untied_init[f"seed{seed}_depth{depth}"] if arm == "UJ" else common_init[str(seed)]
        if run.get("initialization_sha256") != expected_init:
            raise ValueError(f"Initial parameter binding differs in {context}")
        if arm == "UJ" and (run.get("helper_source_sha256") != p["helper_source_sha256"]
                            or run.get("common_initialization_sha256") != common_init[str(seed)]):
            raise ValueError(f"Untied common-initialization/helper binding differs in {context}")
        selection = {"sampling": "proportional", "selection_seed": 20261004,
                     "train_fingerprint": p["train"]["fingerprint"], "tune_fingerprint": p["tune"]["fingerprint"]}
        if run.get("cohort_selection") != selection: raise ValueError(f"Fixed cohort differs in {context}")
        _config(run.get("timing"), {"updates": 3200, "new_updates_in_branch": 640}, context)
        _config(run.get("readout_supervision"), {"mode": "terminal", "directly_supervised_steps": [depth]}, context)
        parameters = run.get("parameters", {})
        if parameters.get("trainable") != PARAMETERS[f"{arm}{depth}"] or parameters.get("encoder_frozen") != ENCODER_PARAMETERS:
            raise ValueError(f"Actual trainable/frozen parameter counts differ in {context}")
        history = run.get("development_history")
        if (not isinstance(history, list) or not all(isinstance(entry, dict) for entry in history)
                or [entry.get("epoch") for entry in history] != [5, 10, 15, 20, 25]):
            raise ValueError(f"Incomplete fixed evaluation schedule in {context}")
        endpoint = run.get("development_subset", {}).get(str(depth), {})
        for value in [endpoint, *[entry.get("by_step", {}).get(str(depth), {}) for entry in history]]:
            _config(value, {"n_cases": 159, "n_positive": 51, "n_negative": 108, "threshold": .5}, context)
            for metric in ("brier", "auroc", "nll", "mean_case_dice"): _metric(value, metric, context)
        for metric in ("brier", "auroc", "nll", "mean_case_dice"):
            if not math.isclose(endpoint[metric], history[-1]["by_step"][str(depth)][metric], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"Final epoch/readout mismatch in {context}")
        fn, fp = _count(endpoint.get("fn"), context, 51), _count(endpoint.get("fp"), context, 108)
        if endpoint.get("tp") != 51-fn or endpoint.get("tn") != 108-fp:
            raise ValueError(f"Confusion counts disagree with fixed pathology labels in {context}")
    seed_rows = analysis.get("seed_results")
    if (not isinstance(seed_rows, list) or len(seed_rows) != 5 or not all(isinstance(row, dict) for row in seed_rows)
            or [row.get("seed") for row in seed_rows] != list(SEEDS)):
        raise ValueError("Analysis error diagnostics must preserve five fitted seeds.")
    indexed = {row["seed"]: row for row in seed_rows}
    regimes = analysis.get("regimes", {})
    if set(regimes) != set(ARMS): raise ValueError("Analysis must contain both shared and untied refiners.")
    for arm in ARMS:
        if set(regimes[arm]) != {regime_name(depth, policy) for depth in DEPTHS for policy in POLICIES}:
            raise ValueError("Analysis must contain all four depth/policy regimes for each arm.")
        for depth in DEPTHS:
            for policy in POLICIES:
                name = regime_name(depth, policy)
                aggregate = regimes[arm][name]
                for metric in ("brier", "auroc", "nll", "mean_case_dice"):
                    _stats_match(aggregate.get("endpoint", {}).get(metric),
                                 [runs[arm, depth, policy, seed]["development_subset"][str(depth)][metric] for seed in SEEDS],
                                 f"{arm} {name} {metric}")
                for truth, total, error_key, confident_key in (("malignant", 51, "fn", "high_confidence_false_negatives"),
                                                               ("benign", 108, "fp", "high_confidence_false_positives")):
                    values = {key: [] for key in ("error_count", "error_rate", "high_confidence_error_count", "high_confidence_error_rate")}
                    for seed in SEEDS:
                        row = indexed[seed].get("regimes", {}).get(arm, {}).get(name, {})
                        pathology, diagnostics = row.get("pathology", {}).get(truth, {}), row.get("endpoint_diagnostics", {})
                        endpoint = runs[arm, depth, policy, seed]["development_subset"][str(depth)]
                        error = _count(pathology.get("error_count"), f"{arm} pathology errors", total)
                        confident = _count(pathology.get("high_confidence_error_count"), f"{arm} high-confidence errors", error)
                        if (pathology.get("n_cases") != total or error != endpoint[error_key]
                                or confident != diagnostics.get(confident_key)):
                            raise ValueError("Per-seed pathology diagnostics disagree with summary/analysis error counts.")
                        for key, actual in (("error_rate", error/total), ("high_confidence_error_rate", confident/total)):
                            if not math.isclose(_number(pathology.get(key), key, maximum=1), actual, rel_tol=1e-9, abs_tol=1e-12):
                                raise ValueError("Truth-specific rates disagree with their Case-count denominators.")
                        values["error_count"].append(error); values["error_rate"].append(error/total)
                        values["high_confidence_error_count"].append(confident); values["high_confidence_error_rate"].append(confident/total)
                    claim = aggregate.get("pathology", {}).get(truth, {})
                    if claim.get("n_cases") != total: raise ValueError("Aggregate pathology counts disagree with fixed Cases.")
                    for key, numbers in values.items(): _stats_match(claim.get(key), numbers, f"{arm} {name} {truth} {key}")
    sharing, gains, theta = _seed_effects(runs)
    for policy in POLICIES:
        for depth in DEPTHS:
            _effect_match(analysis.get("sharing_benefit_by_depth_policy", {}).get(regime_name(depth, policy), {}).get("Brier_benefit"),
                          sharing[depth, policy], "sharing benefit")
        for arm in ARMS:
            _effect_match(analysis.get("depth_benefit_by_arm_policy", {}).get(arm, {}).get(policy, {}).get("Brier_benefit"),
                          gains[arm, policy], "depth benefit")
        _effect_match(analysis.get("theta_by_policy", {}).get(policy), theta[policy], "sharing x depth interaction")
    return runs, indexed


def _decorate(fig, axes, title, subtitle, *, legend_axes=None, rows=1):
    fig.suptitle(title, fontsize=14, fontweight="bold", y=.98)
    fig.text(.5, .925, subtitle, ha="center", fontsize=10, color="#333333")
    handles, labels = [], []
    for axis in axes:
        axis.grid(alpha=.2, linewidth=.6); axis.spines[["top", "right"]].set_visible(False)
    for axis in legend_axes or [axes[0]]:
        h, l = axis.get_legend_handles_labels(); handles.extend(h); labels.extend(l)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .16 if rows == 2 else .19),
               ncol=2 if len(labels) <= 4 else 4, frameon=False, fontsize=9)
    fig.text(.5, .02, DISCLAIMER, ha="center", va="bottom", fontsize=8.5, color="#333333", linespacing=1.45)
    fig.subplots_adjust(left=.075, right=.97, top=.84, bottom=.30 if rows == 2 else .40,
                        wspace=.30, hspace=.48 if rows == 2 else .2)


def _render_figures(runs, seed_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none", "savefig.facecolor": "white"})
    endpoint, axes = plt.subplots(1, 3, figsize=(16, 7.5))
    x = np.arange(4)
    for axis, metric, ylabel in zip(axes, ("brier", "auroc", "nll"),
                                   ("Case Brier (lower is better)", "Case AUROC (higher is better)", "Case NLL (lower is better)")):
        for policy in POLICIES:
            mean, sd = _mean_sd([[runs[arm, depth, policy, seed]["development_subset"][str(depth)][metric]
                                 for arm, depth in MODELS] for seed in SEEDS])
            axis.errorbar(x, mean, yerr=sd, color=POLICY_COLORS[policy], marker="o" if policy == "constant" else "s",
                          linestyle="-" if policy == "constant" else "--", linewidth=1.7, capsize=4, label=POLICY_LABELS[policy])
        axis.set(xlabel="Trained refiner architecture", ylabel=ylabel, xticks=x,
                 xticklabels=("UJ2\n1.159M", "UJ4\n2.049M", "SJ2\n0.715M", "SJ4\n0.715M"))
        axis.set_title("Endpoint mean +/- sample SD over five seeds", fontsize=10)
    _decorate(endpoint, axes, "Endpoint architecture comparison | frozen image encoder",
               "UJ: distinct block parameters; SJ: shared block | terminal-only supervision | connecting segments compare different fitted models")
    effects, axes = plt.subplots(2, 3, figsize=(16, 10))
    sharing, gains, theta = _seed_effects(runs); sx = np.arange(len(SEEDS))
    for row, policy in enumerate(POLICIES):
        for offset, depth, color in ((-.17, 2, "#0072B2"), (.17, 4, "#D55E00")):
            axes[row, 0].bar(sx+offset, sharing[depth, policy], width=.32, color=color, alpha=.85, label=f"Sharing at depth {depth}")
        axes[row, 0].set_title(f"{POLICY_LABELS[policy]}: sharing", fontsize=10)
        axes[row, 0].set_ylabel("A = BrierUJ - BrierSJ")
        for offset, arm, color in ((-.17, "UJ", "#666666"), (.17, "SJ", "#009E73")):
            axes[row, 1].bar(sx+offset, gains[arm, policy], width=.32, color=color, alpha=.85, label=f"Depth benefit: {arm}")
        axes[row, 1].set_title("Extra-depth benefit within each architecture", fontsize=10)
        axes[row, 1].set_ylabel("G = Brier2 - Brier4")
        axes[row, 2].bar(sx, theta[policy], width=.6, color=["#0072B2" if value >= 0 else "#D55E00" for value in theta[policy]], alpha=.85)
        axes[row, 2].axhline(float(theta[policy].mean()), color="#111111", linestyle="--", linewidth=1.3)
        axes[row, 2].set_title(f"Sharing x depth interaction\nMean Theta: {theta[policy].mean():+.4f}", fontsize=10)
        axes[row, 2].set_ylabel("Theta = G_SJ - G_UJ")
        for axis in axes[row]:
            axis.axhline(0, color="#555555", linewidth=.9)
            axis.set(xlabel="Paired training seed", xticks=sx, xticklabels=[str(seed) for seed in SEEDS]); axis.margins(y=.2)
    _decorate(effects, list(axes.flat), "Paired seed effects | sharing, depth and their interaction",
               "Positive A favors shared weights; positive G favors four steps; positive Theta favors the shared model's extra-depth benefit",
               legend_axes=[axes[0, 0], axes[0, 1]], rows=2)
    errors, axes = plt.subplots(2, 2, figsize=(13, 10))
    for column, policy in enumerate(POLICIES):
        for row, count_key in enumerate(("error_count", "high_confidence_error_count")):
            axis = axes[row, column]
            for arm, depth in MODELS:
                name = regime_name(depth, policy)
                fp = np.array([seed_rows[seed]["regimes"][arm][name]["pathology"]["benign"][count_key] for seed in SEEDS])
                fn = np.array([seed_rows[seed]["regimes"][arm][name]["pathology"]["malignant"][count_key] for seed in SEEDS])
                color = MODEL_COLORS[arm, depth]
                axis.scatter(fp, fn, color=color, s=16, alpha=.25, marker="o" if arm == "UJ" else "s")
                axis.errorbar(fp.mean(), fn.mean(), xerr=fp.std(ddof=1), yerr=fn.std(ddof=1), color=color,
                              marker="o" if arm == "UJ" else "s", markersize=7, capsize=3, linestyle="none", label=f"{arm}{depth}")
            axis.set_title(POLICY_LABELS[policy] + (" | threshold 0.5" if row == 0 else " | high-confidence errors"), fontsize=10)
            axis.set(xlabel="False positives (benign Cases)" if row == 0 else "Confident false positives (benign p >= 0.95)",
                     ylabel="False negatives\n(malignant Cases)" if row == 0 else "Confident false negatives\n(malignant p <= 0.05)")
    # Policies must use the same scales for a fair visual comparison. Include
    # zero counts and every plotted point/SD bar, including any negative SD
    # extent; SD describes fitted-seed spread rather than feasible counts.
    for row in range(2):
        x_limits = (min(0., *(axis.get_xlim()[0] for axis in axes[row])),
                    max(axis.get_xlim()[1] for axis in axes[row]))
        y_limits = (min(0., *(axis.get_ylim()[0] for axis in axes[row])),
                    max(axis.get_ylim()[1] for axis in axes[row]))
        for axis in axes[row]:
            axis.set_xlim(x_limits); axis.set_ylim(y_limits)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    _decorate(errors, list(axes.flat), "Pathology error tradeoffs | lower-left means fewer errors",
               "Small markers: five fitted seeds; large markers: means; horizontal/vertical bars: sample SD | inclusive probability cutoffs",
               rows=2)
    payloads = {}; figures = (("endpoint_comparison", endpoint), ("seed_effects", effects), ("pathology_errors", errors))
    try:
        for name, figure in figures:
            for extension in ("png", "svg"):
                buffer = BytesIO(); figure.savefig(buffer, format=extension, dpi=180); payloads[f"{name}.{extension}"] = buffer.getvalue()
    finally:
        for _, figure in figures: plt.close(figure)
    return payloads


def create_figures(report_dir, output_dir, *, protocol_path=DEFAULT_PROTOCOL):
    runs, seed_rows = load_reports(report_dir, protocol_path=protocol_path)
    output_dir = Path(output_dir)
    names = [f"{name}.{extension}" for name in ("endpoint_comparison", "seed_effects", "pathology_errors") for extension in ("png", "svg")]
    if any((output_dir/name).exists() for name in names): raise ValueError("Figure outputs already exist; choose a fresh output directory.")
    previous_env = {key: os.environ.get(key) for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME")}
    try:
        with tempfile.TemporaryDirectory(prefix="loop-ultrasound-architecture-plot-") as cache:
            os.environ["MPLCONFIGDIR"] = cache; os.environ["XDG_CACHE_HOME"] = cache
            try: payloads = _render_figures(runs, seed_rows)
            except ModuleNotFoundError as error:
                if error.name == "matplotlib": raise ValueError("Optional plotting dependency missing; install matplotlib.") from error
                raise
    finally:
        for key, value in previous_env.items():
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
    parser.add_argument("--report-dir", default="reports/architecture256")
    parser.add_argument("--output-dir", default="reports/architecture256/figures")
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    args = parser.parse_args(argv)
    try: paths = create_figures(args.report_dir, args.output_dir, protocol_path=args.protocol)
    except ValueError as error: parser.error(str(error))
    print(json.dumps({"status": "aggregate_architecture_figures_created", "files": paths}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
