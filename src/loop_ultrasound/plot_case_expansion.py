"""Aggregate-only figures comparing matched-update 128/256-Case CPU fits.

Reads a locked expansion protocol and explicitly named 256/128-Case summary
pairs for either the original three seeds or the five additional seeds.
Learning curves align optimizer updates, not epochs. Endpoints belong to
separately fitted two/four-step models; the two seed groups are never pooled.
Matplotlib is optional and imported only after all inputs are validated.
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

from .plot_seed_replication import _read_json, _digest, _metric, _mean_sd


DEPTHS = (2, 4)
SEEDS = (17, 29, 43)
REPLICATION_SEEDS = (53, 71, 89, 107, 131)
PROTOCOL_SEEDS = {"fixed_before_case_expansion_training": SEEDS,
                  "fixed_before_case_expansion_seed_replication_training": REPLICATION_SEEDS}
SIZES = (128, 256)
UPDATE_POINTS = (640, 1280, 1920, 2560, 3200)
SOURCE_KEYS = ("encoder_sha256", "manifest_sha256", "training_source_sha256")
FINAL_SELECTION = "fixed_final_epoch_no_best_epoch_selection"
DEFAULT_PROTOCOL = "configs/cpu_case_expansion256.json"
FIXED_CONFIG = {"arm": "SJ", "supervision": "terminal", "batch_size": 2,
                "learning_rate": .0003, "segmentation_weight": 1.,
                "cached_frozen_features": True, "augment": False,
                "selection_seed": 20261004, "sampling": "proportional", "eval_every": 5}
COUNTS = {128: {"cases": 128, "images": 227, "benign": 87, "malignant": 41},
          256: {"cases": 256, "images": 445, "benign": 174, "malignant": 82}}
EPOCHS = {128: 50, 256: 25}
COLORS = {(128, 2): "#0072B2", (256, 2): "#56B4E9",
          (128, 4): "#D55E00", (256, 4): "#E69F00"}
DEPTH_COLORS = {2: "#0072B2", 4: "#D55E00"}
DISCLAIMER = (
    "Exploratory follow-up on an already observed development cohort; Case-to-patient mapping unverified; no clinical validation.\n"
    "Nested fixed 128/256 training Cases; 3,200 updates each, but 50/25 sampled views per Case and different view/order trajectories.\n"
    "Frozen features; no augmentation; terminal-only supervision; four steps use more compute. Means are not ensembles; sample SD is not a CI.\n"
    "Fixed final epochs (128: 50; 256: 25), no best-epoch selection. Dice: 224 x 224 letterbox valid pixels."
)


def load_reports(report_dir, reference_report_dir, *, protocol_path=DEFAULT_PROTOCOL):
    """Validate both cohorts, paired fits, and the matched-update curve grid."""
    protocol = _read_json(Path(protocol_path))
    status = protocol.get("status")
    seeds = PROTOCOL_SEEDS.get(status) if isinstance(status, str) else None
    if (seeds is None or protocol.get("seeds") != list(seeds) or protocol.get("depths") != list(DEPTHS)
            or protocol.get("supervisions") != ["terminal"] or protocol.get("arms") != ["SJ"]
            or protocol.get("device") != "cpu" or protocol.get("clinical_validation") is not False
            or protocol.get("calibration_and_test_predictions") is not False
            or protocol.get("warm_starts") is not False
            or protocol.get("patient_mapping") != "unverified_Case_grouped_only"
            or protocol.get("model_selection") != FINAL_SELECTION
            or protocol.get("epochs") != 25 or protocol.get("training_updates_per_run") != 3200
            or protocol.get("reference_epochs") != 50 or protocol.get("updates_per_epoch") != 128
            or protocol.get("reference_updates_per_epoch") != 64
            or protocol.get("matched_update_points") != list(UPDATE_POINTS)
            or protocol.get("nested_cohort_verified") is not True
            or not _digest(protocol.get("reference_protocol_sha256"))
            or protocol.get("cached_frozen_features") is not True or protocol.get("augment") is not False
            or any(protocol.get(key) != value for key, value in FIXED_CONFIG.items()
                   if key not in {"arm", "supervision"})):
        raise ValueError("Protocol is not the fixed matched-update 256-Case CPU expansion.")
    if seeds == REPLICATION_SEEDS and not _digest(protocol.get("original_expansion_protocol_sha256")):
        raise ValueError("Replication protocol is missing the original expansion provenance digest.")
    nested = protocol.get("nested_cohort_check", {})
    if (not isinstance(nested, dict)
            or any(nested.get(key) is not True for key in ("reference_cases_subset", "reference_images_subset",
                                                          "shared_view_labels_match"))):
        raise ValueError("Protocol does not declare a verified nested training cohort.")
    initialization = protocol.get("initialization_sha256_by_seed", {})
    if (not isinstance(initialization, dict) or set(initialization) != {str(seed) for seed in seeds}
            or not all(_digest(value) for value in initialization.values())
            or not all(_digest(protocol.get(key)) for key in SOURCE_KEYS)):
        raise ValueError("Protocol is missing source or paired-initialization digests.")
    for split, counts in (("train", COUNTS[256]), ("reference_train", COUNTS[128]),
                          ("tune", {"cases": 159, "images": 281, "benign": 108, "malignant": 51})):
        cohort = protocol.get(split, {})
        if (not isinstance(cohort, dict) or not _digest(cohort.get("fingerprint"))
                or any(cohort.get(key) != count for key, count in counts.items())):
            raise ValueError(f"Protocol has an unexpected {split} cohort.")
    directories = {128: Path(reference_report_dir), 256: Path(report_dir)}
    runs = {(size, depth, seed): _read_json(
        directory / f"SJ-depth{depth}-terminal-seed{seed}.json")
        for size, directory in directories.items() for depth in DEPTHS for seed in seeds}
    common_config, fingerprints = None, {}
    for (size, depth, seed), run in runs.items():
        name = f"{size}-Case SJ-depth{depth}-terminal-seed{seed}.json"
        config, timing = run.get("config", {}), run.get("timing", {})
        if (not isinstance(config, dict)
                or any(config.get(key) != value for key, value in FIXED_CONFIG.items())
                or config.get("cached_frozen_features") is not True or config.get("augment") is not False
                or config.get("steps") != depth or config.get("seed") != seed
                or config.get("epochs") != EPOCHS[size]
                or run.get("status") != "executed_cpu_engineering_pilot"
                or run.get("clinical_validation") is not False or run.get("device") != "cpu"
                or run.get("patient_mapping") != "unverified_Case_grouped_only"
                or run.get("evaluation_split") != "tune_development" or run.get("model_selection") != FINAL_SELECTION
                or run.get("n_training_cases") != size or run.get("n_training_images") != COUNTS[size]["images"]
                or not isinstance(timing, dict) or timing.get("updates") != 3200):
            raise ValueError(f"Wrong fit configuration, provenance, or incomplete updates in {name}")
        comparable = {key: value for key, value in config.items() if key not in {"steps", "seed", "epochs"}}
        if common_config is not None and comparable != common_config:
            raise ValueError(f"Non-experimental configuration differs in {name}")
        common_config = comparable
        cohort = run.get("cohort_selection", {})
        if (not isinstance(cohort, dict) or not _digest(cohort.get("train_fingerprint"))
                or cohort.get("sampling") != protocol["sampling"]
                or cohort.get("selection_seed") != protocol["selection_seed"]
                or cohort.get("tune_fingerprint") != protocol["tune"]["fingerprint"]
                or any(run.get(key) != protocol[key] for key in SOURCE_KEYS)
                or run.get("initialization_sha256") != initialization[str(seed)]):
            raise ValueError(f"Cohort/source/paired initialization mismatch in {name}")
        if size in fingerprints and fingerprints[size] != cohort["train_fingerprint"]:
            raise ValueError(f"Training cohort changes between fits in {name}")
        fingerprints[size] = cohort["train_fingerprint"]
        if size == 256 and fingerprints[size] != protocol["train"]["fingerprint"]:
            raise ValueError(f"Expansion training cohort differs from the locked protocol in {name}")
        if size == 128 and fingerprints[size] != protocol["reference_train"]["fingerprint"]:
            raise ValueError(f"Reference training cohort differs from the locked protocol in {name}")
        readout = run.get("readout_supervision", {})
        if (not isinstance(readout, dict) or readout.get("mode") != "terminal"
                or readout.get("directly_supervised_steps") != [depth]
                or readout.get("intermediate_predictions_scope")
                != "diagnostic_only_no_direct_intermediate_readout_loss"):
            raise ValueError(f"Terminal-only readout metadata disagree in {name}")
        history = run.get("development_history")
        epochs = list(range(5, EPOCHS[size]+1, 5))
        if (not isinstance(history, list) or not all(isinstance(entry, dict) for entry in history)
                or [entry.get("epoch") for entry in history] != epochs):
            raise ValueError(f"Missing or misaligned development evaluation epochs in {name}")
        final = run.get("development_subset", {})
        snapshots = [entry.get("by_step", {}) for entry in history]
        for snapshot in snapshots + [final]:
            if not isinstance(snapshot, dict) or set(snapshot) != {str(step) for step in range(1, depth+1)}:
                raise ValueError(f"Unexpected recorded readout depths in {name}")
            endpoint = snapshot[str(depth)]
            if (not isinstance(endpoint, dict) or endpoint.get("n_cases") != 159
                    or endpoint.get("n_positive") != 51 or endpoint.get("n_negative") != 108):
                raise ValueError(f"Development cohort count changed in {name}")
            for metric in ("brier", "auroc", "nll", "mean_case_dice"):
                _metric(endpoint, metric, name)
        for metric in ("brier", "auroc", "nll", "mean_case_dice"):
            if not math.isclose(snapshots[-1][str(depth)][metric], final[str(depth)][metric],
                                rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"Recorded final epoch and endpoint metrics disagree in {name}")
        updates = [entry["epoch"] * size // config["batch_size"] for entry in history]
        if not set(UPDATE_POINTS).issubset(updates):
            raise ValueError(f"Matched optimizer-update curve points are unavailable in {name}")
    if fingerprints[128] == fingerprints[256]:
        raise ValueError("Distinct-sized training cohorts must have distinct fingerprints.")
    return runs


def _matched_history(run, size, depth, metric):
    by_update = {entry["epoch"] * size // run["config"]["batch_size"]:
                 entry["by_step"][str(depth)][metric] for entry in run["development_history"]}
    return [by_update[update] for update in UPDATE_POINTS]


def _seed_effects(runs, seeds):
    """Endpoint differences, paired by seed and with explicit effect signs."""
    brier = {size: np.asarray([[runs[size, depth, seed]["development_subset"][str(depth)]["brier"]
                               for depth in DEPTHS] for seed in seeds], dtype=np.float64)
             for size in SIZES}
    gains = {size: values[:, 0] - values[:, 1] for size, values in brier.items()}
    data_benefits = {depth: brier[128][:, index] - brier[256][:, index]
                     for index, depth in enumerate(DEPTHS)}
    return gains, data_benefits


def _decorate(figure, axes, title, subtitle, *, replication=False, legend_axes=None):
    figure.suptitle(title, fontsize=13, fontweight="bold", y=.975)
    figure.text(.5, .91, subtitle, ha="center", fontsize=9.5, color="#333333")
    for axis in axes:
        axis.grid(alpha=.20, linewidth=.6)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = [], []
    for axis in legend_axes or [axes[0]]:
        current_handles, current_labels = axis.get_legend_handles_labels()
        handles.extend(current_handles)
        labels.extend(current_labels)
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .20),
                  ncol=2, frameon=False, fontsize=8.5)
    disclaimer = DISCLAIMER
    if replication:
        disclaimer += "\nPrimary replication: only five additional seeds; original three are excluded from these figures."
    figure.text(.5, .025, disclaimer, ha="center", va="bottom", fontsize=8,
                color="#333333", linespacing=1.45)
    figure.subplots_adjust(left=.075, right=.97, top=.81, bottom=.40, wspace=.28)


def _render_figures(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    seeds = tuple(sorted({key[2] for key in runs}))
    replication = seeds == REPLICATION_SEEDS
    learning, axes = plt.subplots(1, 2, figsize=(12, 7))
    for axis, metric, ylabel in zip(axes, ("brier", "auroc"),
                                   ("Development Brier (lower is better)", "Development AUROC (higher is better)")):
        for depth in DEPTHS:
            for size in SIZES:
                means, sd = _mean_sd([_matched_history(runs[size, depth, seed], size, depth, metric)
                                     for seed in seeds])
                axis.plot(UPDATE_POINTS, means, color=COLORS[size, depth], linewidth=1.8,
                          marker="o" if depth == 2 else "s", markersize=4,
                          linestyle="-" if size == 128 else "--", label=f"{size} Cases: {depth}-step endpoint")
                axis.fill_between(UPDATE_POINTS, means-sd, means+sd,
                                  color=COLORS[size, depth], alpha=.14, linewidth=0)
        axis.set(xlabel="Completed optimizer updates (matched across sizes)", ylabel=ylabel, xticks=UPDATE_POINTS)
        axis.set_title(f"Endpoint mean +/- sample SD across {len(seeds)} paired seeds", fontsize=10)
    _decorate(learning, axes, "Matched-update learning curves | " + ("five additional seeds" if replication else "training cohort expansion"),
              "159 development Cases unchanged | 128-Case epochs 10/20/30/40/50 versus 256-Case epochs 5/10/15/20/25",
              replication=replication)

    endpoint, axes = plt.subplots(1, 3, figsize=(15, 7))
    for axis, metric, ylabel in zip(axes, ("brier", "auroc", "mean_case_dice"),
                                   ("Endpoint Brier (lower is better)", "Endpoint AUROC (higher is better)",
                                    "Mean Case Dice (higher is better)")):
        for depth in DEPTHS:
            means, sd = _mean_sd([[runs[size, depth, seed]["development_subset"][str(depth)][metric]
                                  for size in SIZES] for seed in seeds])
            axis.errorbar((0, 1), means, yerr=sd, color=DEPTH_COLORS[depth],
                          marker="o" if depth == 2 else "s", markersize=6, linewidth=1.8,
                          capsize=4, label=f"Separately trained {depth}-step endpoints")
        axis.set(xlabel="Number of training Cases", ylabel=ylabel, xticks=(0, 1),
                 xticklabels=("128 Cases\n50 epochs", "256 Cases\n25 epochs"), xlim=(-.2, 1.2))
        axis.set_title("Fixed 3,200 updates per fitted model", fontsize=10)
    _decorate(endpoint, axes, "Endpoint comparison | " + ("five additional seeds" if replication else "nested 128/256-Case training cohorts"),
              "159 development Cases unchanged | connecting lines compare separate fitted models, not intermediate readouts",
              replication=replication)
    figures = [("learning_curves", learning), ("endpoint_comparison", endpoint)]
    if replication:
        effects, axes = plt.subplots(1, 3, figsize=(15, 7))
        gains, data_benefits = _seed_effects(runs, seeds)
        x = np.arange(len(seeds))
        for offset, size, color in ((-.17, 128, "#666666"), (.17, 256, "#0072B2")):
            axes[0].bar(x+offset, gains[size], width=.32, color=color, alpha=.85,
                        label=f"{size} Cases: depth benefit")
        axes[0].set(ylabel="G = Brier2 - Brier4 (positive favors 4 steps)")
        axes[0].set_title(f"Depth benefit by paired seed\nMean G128: {np.mean(gains[128]):+.4f}; G256: {np.mean(gains[256]):+.4f}", fontsize=10)
        interaction = gains[256] - gains[128]
        axes[1].bar(x, interaction, width=.6,
                    color=["#0072B2" if value >= 0 else "#D55E00" for value in interaction], alpha=.85)
        axes[1].axhline(float(np.mean(interaction)), color="#111111", linestyle="--", linewidth=1.4)
        axes[1].set(ylabel="G256 - G128 (positive: greater depth benefit)")
        axes[1].set_title(f"Data size x depth interaction\nMean: {np.mean(interaction):+.4f}", fontsize=10)
        if np.max(np.abs(interaction)) < 1e-12:
            # Avoid presenting floating-point roundoff as a visible effect.
            axes[1].set_ylim(-1e-4, 1e-4)
        for offset, depth in ((-.17, 2), (.17, 4)):
            axes[2].bar(x+offset, data_benefits[depth], width=.32, color=DEPTH_COLORS[depth], alpha=.85,
                        label=f"{depth}-step data-size benefit")
        axes[2].set(ylabel="Brier128 - Brier256 (positive favors 256 Cases)")
        axes[2].set_title("Data-size benefit at each trained depth", fontsize=10)
        for axis in axes:
            axis.axhline(0, color="#555555", linewidth=.9)
            axis.set(xlabel="Additional training seed", xticks=x, xticklabels=[str(seed) for seed in seeds])
            axis.margins(y=.25)
        _decorate(effects, axes, "Paired seed effects | five additional seeds only",
                  "Separately trained endpoints at 3,200 updates | fixed nested 128/256-Case cohorts",
                  replication=True, legend_axes=[axes[0], axes[2]])
        figures.append(("seed_effects", effects))
    payloads = {}
    try:
        for name, figure in figures:
            for extension in ("png", "svg"):
                buffer = BytesIO()
                figure.savefig(buffer, format=extension, dpi=180)
                payloads[f"{name}.{extension}"] = buffer.getvalue()
    finally:
        for _, figure in figures:
            plt.close(figure)
    return payloads


def create_figures(report_dir, reference_report_dir, output_dir, *, protocol_path=DEFAULT_PROTOCOL):
    runs = load_reports(report_dir, reference_report_dir, protocol_path=protocol_path)
    output_dir = Path(output_dir)
    figure_names = ["endpoint_comparison", "learning_curves"]
    if tuple(sorted({key[2] for key in runs})) == REPLICATION_SEEDS:
        figure_names.append("seed_effects")
    names = [f"{name}.{extension}" for name in figure_names
             for extension in ("png", "svg")]
    if any((output_dir / name).exists() for name in names):
        raise ValueError("Figure output already exists; choose a fresh output directory.")
    previous_config = {key: os.environ.get(key) for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME")}
    try:
        with tempfile.TemporaryDirectory(prefix="loop-ultrasound-expansion-plot-") as config_dir:
            os.environ["MPLCONFIGDIR"] = config_dir
            os.environ["XDG_CACHE_HOME"] = config_dir
            try:
                payloads = _render_figures(runs)
            except ModuleNotFoundError as error:
                if error.name == "matplotlib":
                    raise ValueError("Optional plotting dependency missing; install matplotlib in the project environment.") from error
                raise
    finally:
        for key, value in previous_config.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        try:
            with (output_dir / name).open("xb") as stream:
                stream.write(payload)
        except FileExistsError as error:
            raise ValueError("Figure output already exists; choose a fresh output directory.") from error
    return [str(output_dir / name) for name in names]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", default="reports/case_expansion256")
    parser.add_argument("--reference-report-dir", default="reports/depth_supervision128")
    parser.add_argument("--output-dir", default="reports/case_expansion256/figures")
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    args = parser.parse_args(argv)
    try:
        paths = create_figures(args.report_dir, args.reference_report_dir, args.output_dir, protocol_path=args.protocol)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"status": "aggregate_case_expansion_figures_created", "files": paths}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
